// ===========================================================================
// API server — OpenAI and Anthropic compatible HTTP endpoints.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Sets up an HTTP server (axum + tokio), defines the shared types that
//   bridge HTTP handlers to the inference worker, and provides the shared
//   worker loop (run_worker_loop) that drives any InferenceEngine.
//
// Module layout:
//   mod.rs        — server setup, shared types, spawn_worker, run_worker_loop
//   openai.rs     — /v1/chat/completions, /v1/completions, /v1/models
//   anthropic.rs  — /v1/messages
//   auth/         — pluggable auth hooks (init, request, background)
//   tls.rs        — TLS support (manual certs, Let's Encrypt)
//
// Architecture:
//
//   HTTP clients ←→ [tokio async runtime / axum handlers]
//                         ↕ channels
//                    [worker thread: run_worker_loop(&mut dyn InferenceEngine)]
//                         ↕
//                    [Engine<B> or MultiGpuEngine — server doesn't know which]
//
// The worker loop is GPU-topology-agnostic.  It calls add_request / step /
// abort_sequence / has_work / tokenizer on the trait object.  spawn_worker()
// delegates to engine::loader::load_and_run() which handles single-GPU vs
// multi-GPU setup.
//
// Why a dedicated worker thread?
//   The model borrows the GPU backend and has mutable scratch buffers.
//   Rather than fighting Arc<Mutex<...>>, we keep ALL GPU state on one
//   std::thread and communicate via channels:
//
//   - Request channel (std::sync::mpsc):  HTTP handlers → worker
//   - Response channel (tokio::sync::mpsc, one per request):  worker → handler
//
//   The worker calls `blocking_send` on the tokio channel, which is safe
//   from synchronous code.  The handler calls `recv().await` on the async
//   side.  This cleanly bridges sync GPU code and async HTTP serving.
// ===========================================================================

pub(crate) mod anthropic;
pub(crate) mod auth;
pub(crate) mod openai;
pub(crate) mod tls;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use auth::AuthProvider;
use crate::ServeArgs;
use crate::engine;
use crate::engine::SeqId;
use crate::gpu;
use crate::model::{config, tokenizer};

// ---------------------------------------------------------------------------
// Shared types: the bridge between HTTP handlers and the inference worker.
// ---------------------------------------------------------------------------

/// Pre-tokenized request sent from an HTTP handler to the inference worker.
///
/// Tokenization happens on the async handler thread (CPU-only work) so the
/// worker's Engine step loop is never stalled by tokenization.
pub(crate) struct WorkerRequest {
    /// Pre-tokenized prompt (already includes BOS, chat template, etc.).
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Per-request channel to send token events back to the handler.
    pub response_tx: tokio::sync::mpsc::Sender<InferenceEvent>,
    /// Whether thinking (extended reasoning) was requested.
    /// Some(true) = enabled, Some(false) = explicitly disabled, None = not specified.
    pub thinking: Option<bool>,
    /// Preprocessed images for vision models.
    /// Decoded and normalised on the handler thread (CPU), passed to the
    /// worker for GPU encoding during prefill.
    pub images: Vec<crate::model::vision::ProcessedImage>,
    /// Authenticated user identity (None when auth is disabled).
    pub user: Option<auth::AuthUser>,
}

/// Events sent from the inference worker back to an HTTP handler.
pub(crate) enum InferenceEvent {
    /// A new token was generated.
    Token { text: String },
    /// Generation finished.
    Done {
        stop_reason: StopReason,
        prompt_tokens: usize,
        completion_tokens: usize,
        /// Number of prompt tokens served from the prefix cache.
        cached_tokens: usize,
    },
    /// An error occurred during inference.
    Error(String),
}

/// Why generation stopped.
#[derive(Clone, Copy)]
pub(crate) enum StopReason {
    /// Hit an end-of-sequence token (EOS/EOT).
    EndOfSequence,
    /// Reached the max_tokens limit.
    MaxTokens,
    /// Model produced tool calls (detected during post-processing).
    ToolCalls,
}

/// Shared state accessible by all axum handlers via `State(Arc<ServerState>)`.
pub(crate) struct ServerState {
    /// Channel to send pre-tokenized requests to the inference worker.
    pub request_tx: std::sync::mpsc::SyncSender<WorkerRequest>,
    /// Model name for API responses (derived from directory name).
    pub model_name: String,
    /// Shared tokenizer for handler-side tokenization.
    pub tokenizer: Arc<tokenizer::Tokenizer>,
    /// Model architecture (for chat template selection).
    pub arch: config::ModelArch,
    /// Vision config for image preprocessing (None for text-only models).
    pub vision_config: Option<crate::model::config::VisionConfig>,
    /// Auth provider (None variant when auth is disabled).
    pub auth: auth::AuthProviderKind,
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Preprocess images from the last user message for vision models.
///
/// Extracts images from the most recent user message, decodes and normalises
/// them on the CPU (handler thread), and returns a flat list of ProcessedImage
/// ready for GPU upload during prefill.
pub(crate) fn preprocess_images(
    messages: &[crate::model::chat::Message],
    vision_config: Option<&crate::model::config::VisionConfig>,
) -> Vec<crate::model::vision::ProcessedImage> {
    let Some(vc) = vision_config else {
        return Vec::new();
    };
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.images.as_ref())
        .map(|imgs| {
            imgs.iter()
                .filter_map(|img| crate::model::vision::preprocess_image(&img.data, vc).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Generate a random hex ID for API responses.
pub(crate) fn generate_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    format!("{:016x}", rng.random::<u64>())
}

/// Current Unix timestamp in seconds.
pub(crate) fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Server entry point.
// ---------------------------------------------------------------------------

/// Check if the host is a loopback address (127.0.0.1, ::1, localhost).
/// Traffic on loopback never leaves the machine, so TLS and auth are
/// unnecessary for single-user local development.
fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "::1" | "localhost")
}

/// Validate TLS-related CLI args before doing any heavy work.
///
/// Returns the resolved TlsMode on success, or a descriptive error
/// if the configuration is invalid (missing certs, unreadable files, etc.).
fn validate_tls_args(args: &ServeArgs) -> anyhow::Result<tls::TlsMode> {
    if args.letsencrypt {
        let domain = args.domain.clone().unwrap(); // guaranteed by clap `requires`
        return Ok(tls::TlsMode::LetsEncrypt {
            domain,
            email: args.letsencrypt_email.clone(),
            cache_dir: args.cert_cache_dir.clone(),
        });
    }

    if let (Some(cert), Some(key)) = (&args.cert, &args.private_key) {
        if !cert.exists() {
            anyhow::bail!("TLS certificate not found: {}", cert.display());
        }
        if !key.exists() {
            anyhow::bail!("TLS private key not found: {}", key.display());
        }
        return Ok(tls::TlsMode::Manual {
            cert: cert.clone(),
            key: key.clone(),
        });
    }

    if args.dangerous_no_tls || is_loopback(&args.host) {
        return Ok(tls::TlsMode::None);
    }

    anyhow::bail!(
        "no TLS configuration provided. Use --cert/--private-key or --letsencrypt \
         to enable TLS, or pass --dangerous-no-tls to serve over plain HTTP.\n\
         (localhost binds don't require TLS — this check only applies to external interfaces.)"
    )
}

/// Start the API server.  Called from `main()` when `rllm serve` is invoked.
///
/// Loads the model in a dedicated worker thread, then starts axum on the
/// tokio runtime.  This function blocks until the server shuts down.
pub(crate) fn serve(args: ServeArgs) -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. Validate CLI args before doing any heavy work (model loading).
    //    Fail fast on missing TLS config, bad cert paths, etc.
    // ------------------------------------------------------------------
    let tls_mode = validate_tls_args(&args)?;

    let model_name = args
        .model
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "rllm-model".into());

    if !args.model.join("config.json").exists() {
        anyhow::bail!(
            "model directory '{}' does not contain config.json",
            args.model.display()
        );
    }

    let scheme = if matches!(tls_mode, tls::TlsMode::None) {
        "http"
    } else {
        "https"
    };
    let addr = format!("{}:{}", args.host, args.port);

    // ------------------------------------------------------------------
    // 2. Load model.
    // ------------------------------------------------------------------
    eprintln!();
    eprintln!("  rllm — loading {}", model_name);
    eprintln!("  ----------------------------------------");
    // Detect pre-quantized models by checking safetensors metadata.
    let is_prequantized = crate::model::loader::is_prequantized_model(&args.model);
    if is_prequantized {
        eprintln!("  mode      : Q4 (pre-quantized)");
    } else {
        eprintln!("  mode      : bf16");
    }

    // Resolve --tp 0 → auto-detect available GPUs.
    let mut tp = args.tp;
    if tp == 0 {
        tp = gpu::device_count();
    }

    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    // On macOS (Metal), fall back to single GPU with a warning.
    #[cfg(not(feature = "cuda"))]
    if tp > 1 {
        eprintln!(
            "  warning   : --tp {} ignored (multi-GPU requires CUDA + NCCL), using single GPU",
            tp
        );
        tp = 1;
    }

    if tp > 1 {
        eprintln!("  tp        : {} GPUs", tp);
    }

    let WorkerHandle {
        request_tx,
        tokenizer,
        arch,
        vision_config: _,
    } = spawn_worker(args.model.clone(), args.stream_experts, tp)?;

    // Parse vision config directly from config.json for handler-side image preprocessing.
    let vision_config = config::ModelConfig::from_file(&args.model.join("config.json"))
        .ok()
        .and_then(|c| c.vision);

    // ------------------------------------------------------------------
    // 3. Start HTTP server.
    // ------------------------------------------------------------------

    // Build the tokio runtime early — auth init may need async I/O (OIDC
    // discovery, JWKS fetch).  The inference worker is a plain std::thread
    // and never needs the runtime.
    let rt = tokio::runtime::Runtime::new()?;

    // ------------------------------------------------------------------
    // 3a. Initialize auth provider (if configured).
    // ------------------------------------------------------------------
    let auth_provider = if let Some(path) = &args.auth_config {
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(path).map_err(|e| {
                anyhow::anyhow!("failed to open auth config '{}': {e}", path.display())
            })?)?;
        let provider_name = config
            .get("provider")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("auth config missing \"provider\" field"))?;
        match provider_name {
            "oidc" => {
                eprintln!("  auth      : oidc");
                let provider =
                    rt.block_on(auth::oidc::OidcProvider::init(&config))?;
                auth::AuthProviderKind::Oidc(Arc::new(provider))
            }
            other => anyhow::bail!("unknown auth provider: {other}"),
        }
    } else {
        // On external interfaces, require --dangerous-no-auth to run without auth.
        // On loopback, auth is unnecessary — you're the only user.
        if !is_loopback(&args.host) && !args.dangerous_no_auth {
            anyhow::bail!(
                "no auth configured on external interface (--host {}).\n\
                 Use --auth-config to enable authentication, or pass --dangerous-no-auth \
                 to serve without auth.\n\
                 (localhost binds don't require auth — this check only applies to external interfaces.)",
                args.host
            );
        }
        eprintln!("  auth      : none");
        auth::AuthProviderKind::None
    };

    // Warn if auth is enabled but TLS is not — tokens, prompts, and
    // completions are sent in plaintext and can be intercepted or modified
    // by an attacker with network access (man-in-the-middle).
    if !matches!(auth_provider, auth::AuthProviderKind::None)
        && matches!(tls_mode, tls::TlsMode::None)
    {
        eprintln!();
        eprintln!("  WARNING: auth is enabled but TLS is disabled.");
        eprintln!("  Bearer tokens, prompts, and completions are sent in plaintext.");
        eprintln!("  An attacker with network access can intercept tokens to");
        eprintln!("  impersonate users, read prompts and completions, or modify");
        eprintln!("  requests and responses in transit (man-in-the-middle).");
        eprintln!("  This is safe over localhost or an SSH tunnel, but dangerous");
        eprintln!("  on any network an attacker can observe.");
        eprintln!("  To fix: add --cert/--private-key or --letsencrypt.");
        eprintln!();
    }

    eprintln!("  ----------------------------------------");
    eprintln!("  endpoint  : {scheme}://{addr}/v1/chat/completions");
    eprintln!("  health    : {scheme}://{addr}/health");
    eprintln!();

    let state = Arc::new(ServerState {
        request_tx,
        model_name,
        tokenizer,
        arch,
        vision_config,
        auth: auth_provider,
    });

    // Build axum router with all API endpoints.
    let app = axum::Router::new()
        // OpenAI-compatible endpoints.
        .route(
            "/v1/chat/completions",
            axum::routing::post(openai::chat_completions),
        )
        .route("/v1/completions", axum::routing::post(openai::completions))
        .route("/v1/models", axum::routing::get(openai::list_models))
        // Anthropic-compatible endpoint.
        .route("/v1/messages", axum::routing::post(anthropic::messages))
        // Health check.
        .route("/health", axum::routing::get(|| async { "ok" }))
        // Auth middleware — inside CORS so preflight OPTIONS get CORS headers
        // even when auth denies them.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state.clone());

    // Spawn auth provider's background task (JWKS refresh, etc.).
    rt.block_on(async {
        state.auth.spawn_background();
    });

    rt.block_on(async {
        match tls_mode {
            tls::TlsMode::None => {
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                axum::serve(listener, app).await?;
            }
            tls::TlsMode::Manual { cert, key } => {
                tls::serve_manual_tls(app, &addr, &cert, &key).await?;
            }
            tls::TlsMode::LetsEncrypt {
                domain,
                email,
                cache_dir,
            } => {
                tls::serve_letsencrypt(app, &addr, &domain, email.as_deref(), &cache_dir).await?;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Inference worker thread.
// ---------------------------------------------------------------------------

/// Result of spawning the worker thread: the request channel plus shared
/// tokenizer and architecture for handler-side tokenization.
struct WorkerHandle {
    request_tx: std::sync::mpsc::SyncSender<WorkerRequest>,
    tokenizer: Arc<tokenizer::Tokenizer>,
    arch: config::ModelArch,
    vision_config: Option<crate::model::config::VisionConfig>,
}

/// Spawn the inference worker thread.
///
/// Delegates to engine::loader::load_and_run() for model loading and engine
/// construction, providing callbacks for the ready signal and the worker loop.
fn spawn_worker(
    model_dir: std::path::PathBuf,
    stream_experts: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    type ReadyPayload = (Arc<tokenizer::Tokenizer>, config::ModelArch, Option<crate::model::config::VisionConfig>);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<ReadyPayload, String>,
    >(1);

    std::thread::spawn(move || {
        let max_active = 32;

        let result = engine::loader::load_and_run_ext(
            &model_dir,
            stream_experts,
            tp,
            max_active,
            |tok, arch| {
                let _ = ready_tx.send(Ok((Arc::new(tok.clone()), arch, None)));
            },
            |eng| run_worker_loop(eng, request_rx),
        );

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

    match ready_rx.recv() {
        Ok(Ok((tokenizer, arch, vision_config))) => Ok(WorkerHandle {
            request_tx,
            tokenizer,
            arch,
            vision_config,
        }),
        Ok(Err(e)) => anyhow::bail!("worker failed to start: {e}"),
        Err(_) => anyhow::bail!("worker thread died during startup"),
    }
}

/// Per-request context tracked by the worker's request registry.
/// Maps an Engine SeqId to the HTTP response channel.
struct RequestContext {
    response_tx: tokio::sync::mpsc::Sender<InferenceEvent>,
    prompt_token_count: usize,
    generated_count: usize,
    /// Running buffer of all generated token IDs for this sequence.
    /// Used for incremental decoding: we decode the full buffer each step
    /// and emit only the new characters.  This avoids SentencePiece Strip
    /// decoder issues where single-token decode strips leading spaces
    /// (affects Mistral/Mixtral).
    token_ids: Vec<u32>,
    /// Number of characters already sent to the client.
    prev_text_len: usize,
    /// Number of prompt tokens served from the prefix cache (0 if miss).
    cached_tokens: usize,
    /// When this request was submitted to the worker (for TTFT / tok/s logging).
    created_at: Instant,
    /// When the first token was generated (None until the first token event).
    first_token_at: Option<Instant>,
    /// Whether thinking was requested for this sequence.
    /// Used to inject literal `<think>`/`</think>` markers when the model
    /// generates these as special tokens (which `decode(skip_special=true)`
    /// would otherwise strip).
    thinking: bool,
    /// Suffix to inject before the next decode.
    ///
    /// When we see a `<think>` or `</think>` special token, we can't just
    /// append the marker text — we need to inject it at the right position
    /// in the incremental decode buffer.  This field accumulates markers
    /// that were seen since the last decode.
    inject_marker: Option<&'static str>,
    /// Authenticated user identity (for per-user logging).
    user: Option<auth::AuthUser>,
}

// ---------------------------------------------------------------------------
// Shared worker loop — drives any InferenceEngine (single- or multi-GPU).
//
// This is the continuous batching loop that bridges HTTP requests (via the
// request channel) to the inference engine.  It drains requests, calls
// engine.step(), streams tokens back to clients, and cleans up finished
// or disconnected sequences.
//
// spawn_worker() (above) delegates to this function via
// engine::loader::load_and_run() after constructing the appropriate engine.
// ---------------------------------------------------------------------------

/// Run the continuous batching loop for any InferenceEngine.
///
/// Blocks the calling thread until the request channel closes (server shutdown).
/// The engine must already be fully initialized (model loaded, KV cache allocated).
fn run_worker_loop(
    engine: &mut dyn engine::InferenceEngine,
    request_rx: std::sync::mpsc::Receiver<WorkerRequest>,
) -> anyhow::Result<()> {
    let mut registry: HashMap<SeqId, RequestContext> = HashMap::new();

    loop {
        // 1. Drain all pending requests (non-blocking).
        while let Ok(req) = request_rx.try_recv() {
            let prompt_token_count = req.prompt_tokens.len();
            let thinking = req.thinking.unwrap_or(false);
            let user = req.user;
            let seq_id = engine.add_request(
                req.prompt_tokens,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.images,
            );
            registry.insert(
                seq_id,
                RequestContext {
                    response_tx: req.response_tx,
                    prompt_token_count,
                    generated_count: 0,
                    cached_tokens: 0,
                    token_ids: Vec::new(),
                    prev_text_len: 0,
                    created_at: Instant::now(),
                    first_token_at: None,
                    thinking,
                    inject_marker: None,
                    user,
                },
            );
        }

        // 2. If no work, block until a new request arrives.
        if !engine.has_work() {
            match request_rx.recv() {
                Ok(req) => {
                    let prompt_token_count = req.prompt_tokens.len();
                    let thinking = req.thinking.unwrap_or(false);
                    let user = req.user;
                    let seq_id = engine.add_request(
                        req.prompt_tokens,
                        req.max_tokens,
                        req.temperature,
                        req.top_p,
                        req.images,
                    );
                    registry.insert(
                        seq_id,
                        RequestContext {
                            response_tx: req.response_tx,
                            prompt_token_count,
                            generated_count: 0,
                            cached_tokens: 0,
                            token_ids: Vec::new(),
                            prev_text_len: 0,
                            created_at: Instant::now(),
                            first_token_at: None,
                            thinking,
                            inject_marker: None,
                            user,
                        },
                    );
                }
                Err(_) => break, // Channel closed — server shutting down.
            }
        }

        // 3. Run one engine step (prefill + decode + sample).
        let step_output = match engine.step() {
            Ok(output) => output,
            Err(e) => {
                let error_msg = format!("{e:#}");
                for (_, ctx) in registry.drain() {
                    let _ = ctx
                        .response_tx
                        .blocking_send(InferenceEvent::Error(error_msg.clone()));
                }
                continue;
            }
        };

        // 4. Stream tokens to response channels.
        let mut to_abort: Vec<SeqId> = Vec::new();
        let tokenizer = engine.tokenizer();

        for &(seq_id, token_id) in &step_output.tokens {
            if let Some(ctx) = registry.get_mut(&seq_id) {
                // Detect thinking special tokens.  Models like Qwen 3.5 emit
                // <think> (248068) and </think> (248069) as special tokens that
                // `decode(skip_special=true)` strips.  We record a pending
                // marker so we can inject the literal text after decoding.
                if ctx.thinking && tokenizer.is_think_start(token_id) {
                    // Flush any pending marker first, then store the new one.
                    if let Some(prev) = ctx.inject_marker.take() {
                        let _ = ctx
                            .response_tx
                            .blocking_send(InferenceEvent::Token { text: prev.to_string() });
                    }
                    ctx.inject_marker = Some("<think>");
                    ctx.generated_count += 1;
                    if ctx.first_token_at.is_none() {
                        ctx.first_token_at = Some(Instant::now());
                    }
                    continue;
                }
                if ctx.thinking && tokenizer.is_think_end(token_id) {
                    if let Some(prev) = ctx.inject_marker.take() {
                        let _ = ctx
                            .response_tx
                            .blocking_send(InferenceEvent::Token { text: prev.to_string() });
                    }
                    ctx.inject_marker = Some("</think>");
                    ctx.generated_count += 1;
                    continue;
                }

                ctx.token_ids.push(token_id);
                let full_text = tokenizer.decode(&ctx.token_ids).unwrap_or_default();
                let mut text = full_text[ctx.prev_text_len..].to_string();
                ctx.prev_text_len = full_text.len();

                // Prepend any pending thinking marker to the decoded text.
                if let Some(marker) = ctx.inject_marker.take() {
                    text = format!("{marker}{text}");
                }

                ctx.generated_count += 1;
                if ctx.first_token_at.is_none() {
                    ctx.first_token_at = Some(Instant::now());
                }
                if ctx
                    .response_tx
                    .blocking_send(InferenceEvent::Token { text })
                    .is_err()
                {
                    to_abort.push(seq_id);
                }
            }
        }

        // 5. Handle finished sequences.
        for finished in &step_output.finished {
            if let Some(mut ctx) = registry.remove(&finished.id) {
                // Propagate cached_tokens count from engine.
                ctx.cached_tokens = finished.cached_tokens;
                // Flush any pending thinking marker (e.g. </think> was the
                // last token before EOS).
                if let Some(marker) = ctx.inject_marker.take() {
                    let _ = ctx
                        .response_tx
                        .blocking_send(InferenceEvent::Token { text: marker.to_string() });
                }

                let stop_reason = match finished.reason {
                    engine::FinishReason::Eos => StopReason::EndOfSequence,
                    engine::FinishReason::MaxTokens => StopReason::MaxTokens,
                };

                // Log TTFT and tok/s for benchmarking.
                let now = Instant::now();
                let ttft_ms = ctx
                    .first_token_at
                    .map(|t| t.duration_since(ctx.created_at).as_secs_f64() * 1000.0);
                let total_secs = now.duration_since(ctx.created_at).as_secs_f64();
                let decode_tokens = ctx.generated_count.saturating_sub(1);
                let decode_secs = ctx
                    .first_token_at
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(0.0);
                let tok_per_sec = if decode_secs > 0.0 {
                    decode_tokens as f64 / decode_secs
                } else {
                    0.0
                };
                let stop_label = match stop_reason {
                    StopReason::EndOfSequence => "eos",
                    StopReason::MaxTokens => "max_tokens",
                    StopReason::ToolCalls => "tool_calls",
                };
                let cache_label = if ctx.cached_tokens > 0 {
                    format!(" ({} cached)", ctx.cached_tokens)
                } else {
                    String::new()
                };
                // Include user identity in the log line when auth is enabled.
                let user_label = ctx.user.as_ref().map(|u| {
                    format!("  {}  |", u.sub)
                }).unwrap_or_default();
                eprintln!(
                    "  seq {:>3}  |{} {} prompt{} + {} gen  |  TTFT {:.0} ms  |  {:.1} tok/s  |  {:.2}s  |  {}",
                    finished.id,
                    user_label,
                    ctx.prompt_token_count,
                    cache_label,
                    ctx.generated_count,
                    ttft_ms.unwrap_or(0.0),
                    tok_per_sec,
                    total_secs,
                    stop_label,
                );

                let _ = ctx.response_tx.blocking_send(InferenceEvent::Done {
                    stop_reason,
                    prompt_tokens: ctx.prompt_token_count,
                    completion_tokens: ctx.generated_count,
                    cached_tokens: ctx.cached_tokens,
                });
            }
        }

        // 6. Abort disconnected sequences + proactive disconnect check.
        for &id in &to_abort {
            engine.abort_sequence(id);
            registry.remove(&id);
        }

        let disconnected: Vec<SeqId> = registry
            .iter()
            .filter(|(_, ctx)| ctx.response_tx.is_closed())
            .map(|(&id, _)| id)
            .collect();
        for id in disconnected {
            engine.abort_sequence(id);
            registry.remove(&id);
        }
    }

    Ok(())
}
