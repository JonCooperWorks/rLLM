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
// Worker loop features:
//   - Stop sequences:  checked after each token decode; when matched, the
//     stop string is excluded and a Done(EndOfSequence) is sent.
//   - Seeded RNG:  when a request includes a seed, the engine creates a
//     per-sequence SmallRng for deterministic sampling.
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
pub(crate) mod metrics;
pub(crate) mod openai;
pub(crate) mod tls;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::response::IntoResponse;
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
    /// Unique request ID generated at the handler, used for correlation
    /// across logs, metrics, and API responses.
    pub request_id: String,
    /// Pre-tokenized prompt (already includes BOS, chat template, etc.).
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling parameters (temperature, top_p, top_k, penalties, logprobs, etc.).
    pub params: crate::model::sampler::SampleParams,
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
    /// Optional seed for deterministic sampling.
    /// When provided, the RNG is seeded with this value so the same prompt
    /// produces the same output (assuming single-sequence batching).
    pub seed: Option<u64>,
    /// Stop sequences — generation stops when any of these strings appear
    /// in the output.  The stop sequence itself is excluded from the response.
    pub stop: Vec<String>,
    /// Precompiled grammar for structured output (None for unconstrained generation).
    /// Built on the async handler thread from a JSON schema, passed to the engine
    /// for per-token constraint enforcement.
    pub grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
    /// Per-token logit bias (token_id → additive adjustment).
    pub logit_bias: HashMap<u32, f32>,
}

/// Events sent from the inference worker back to an HTTP handler.
#[derive(Debug)]
pub(crate) enum InferenceEvent {
    /// A new token was generated.
    Token {
        text: String,
        /// Log-probability information (only when logprobs was requested).
        logprob_info: Option<engine::TokenLogprobInfo>,
    },
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

/// Add standard headers to an API response.
///
/// Sets `x-request-id` for tracing, and `Connection: close` for non-streaming
/// responses so clients like `curl -N` don't hang waiting for the keep-alive
/// connection to close.  Streaming (SSE) responses leave the connection open
/// naturally — the chunked stream terminates cleanly when the last event is sent.
pub(crate) fn finalize_response(
    response: &mut axum::response::Response,
    request_id: &str,
    close_connection: bool,
) {
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("x-request-id"),
        axum::http::HeaderValue::from_str(request_id).unwrap(),
    );
    if close_connection {
        response.headers_mut().insert(
            axum::http::header::CONNECTION,
            axum::http::HeaderValue::from_static("close"),
        );
    }
}

/// Collect all tokens from an inference event channel into a single string.
///
/// Used by non-streaming (blocking) handlers across OpenAI and Anthropic
/// endpoints.  Returns immediately after the `Done` event — does NOT wait
/// for the sender to be dropped (which would cause the HTTP response to
/// hang until the worker cleans up the request context).
pub(crate) async fn collect_response(
    mut rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Result<(String, usize, usize, StopReason), String> {
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut stop_reason = StopReason::MaxTokens;

    while let Some(event) = rx.recv().await {
        match event {
            InferenceEvent::Token { text: t, .. } => text.push_str(&t),
            InferenceEvent::Done {
                stop_reason: sr,
                prompt_tokens: pt,
                completion_tokens: ct,
                cached_tokens: _,
            } => {
                prompt_tokens = pt;
                completion_tokens = ct;
                stop_reason = sr;
                break;
            }
            InferenceEvent::Error(e) => return Err(e),
        }
    }

    Ok((text, prompt_tokens, completion_tokens, stop_reason))
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug)]
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
    /// Image placeholder token ID for vision scatter (e.g., <|image_pad|> for Qwen).
    pub image_token_id: Option<u32>,
    /// Auth provider (None variant when auth is disabled).
    pub auth: auth::AuthProviderKind,
    /// Prometheus metrics (shared with the worker thread via Arc).
    pub metrics: Arc<metrics::Metrics>,
    /// Set to true when a shutdown signal (SIGTERM/SIGINT) is received.
    /// The health endpoint returns 503 during shutdown so load balancers
    /// stop sending traffic before the server stops accepting connections.
    pub shutting_down: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// GET /health — returns 200 when ready, 503 during shutdown.
///
/// Returns a JSON body with queue depth for observability.  Load balancers
/// can use the status code alone; dashboards can parse the body for details.
///
/// Since the server only binds after model loading completes (spawn_worker
/// blocks until the engine is ready), a reachable /health endpoint implies
/// the model is loaded and serving.  During shutdown, it returns 503 so
/// load balancers stop routing traffic before connections are closed.
async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> axum::response::Response {
    let status = if state.shutting_down.load(Ordering::SeqCst) {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    } else {
        axum::http::StatusCode::OK
    };
    let body = serde_json::json!({
        "status": if status == axum::http::StatusCode::OK { "ok" } else { "shutting_down" },
        "active_sequences": state.metrics.active_sequences.get(),
        "waiting_sequences": state.metrics.waiting_sequences.get(),
    });
    (status, axum::Json(body)).into_response()
}

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

    anyhow::ensure!(
        args.max_pending >= 1,
        "--max-pending must be at least 1 (got {})",
        args.max_pending,
    );

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
    // Detect pre-quantized models by checking safetensors metadata.
    let is_prequantized = crate::model::loader::is_prequantized_model(&args.model);
    let mode = if is_prequantized { "Q4 (pre-quantized)" } else { "bf16" };
    tracing::info!(model = %model_name, mode, "loading model");

    // Resolve --tp 0 → auto-detect available GPUs.
    let mut tp = args.tp;
    if tp == 0 {
        tp = gpu::device_count();
    }

    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    // On macOS (Metal), fall back to single GPU with a warning.
    #[cfg(not(feature = "cuda"))]
    if tp > 1 {
        tracing::warn!(requested_tp = tp, "multi-GPU requires CUDA + NCCL, falling back to single GPU");
        tp = 1;
    }

    if tp > 1 {
        tracing::info!(gpus = tp, "tensor parallelism enabled");
    }

    let kv_quant = crate::model::turboquant::KvQuantMode::from_str(&args.kv_quant)
        .unwrap_or_else(|| {
            tracing::error!(value = %args.kv_quant, "invalid --kv-quant value, expected: turbo4, turbo3, turbo2, none");
            std::process::exit(1);
        });

    let server_metrics = Arc::new(metrics::Metrics::new());

    let request_timeout = std::time::Duration::from_secs(args.request_timeout);

    let WorkerHandle {
        request_tx,
        tokenizer,
        arch,
        vision_config: _,
    } = spawn_worker(
        args.model.clone(), args.stream_experts, tp, kv_quant,
        server_metrics.clone(), args.max_pending, args.max_active, request_timeout,
    )?;

    // Parse vision config + image token ID from config.json for handler-side preprocessing.
    let parsed_config = config::ModelConfig::from_file(&args.model.join("config.json")).ok();
    let vision_config = parsed_config.as_ref().and_then(|c| c.vision.clone());
    let image_token_id = parsed_config.as_ref().and_then(|c| c.image_token_id);

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
                tracing::info!(provider = "oidc", "auth enabled");
                let provider =
                    rt.block_on(auth::oidc::OidcProvider::init(&config))?;
                auth::AuthProviderKind::Oidc(Arc::new(provider))
            }
            "static_api_key" => {
                tracing::info!(provider = "static_api_key", "auth enabled");
                // Inject the config file path so the background task can
                // re-read it for hot reload of the key hash.
                let mut config = config;
                config["_config_path"] =
                    serde_json::Value::String(path.display().to_string());
                let provider = rt.block_on(
                    auth::static_api_key::StaticApiKeyProvider::init(&config),
                )?;
                auth::AuthProviderKind::StaticApiKey(Arc::new(provider))
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
        tracing::info!(provider = "none", "auth disabled");
        auth::AuthProviderKind::None
    };

    // Warn if auth is enabled but TLS is not — tokens, prompts, and
    // completions are sent in plaintext and can be intercepted or modified
    // by an attacker with network access (man-in-the-middle).
    if auth_provider.is_enabled()
        && matches!(tls_mode, tls::TlsMode::None)
    {
        tracing::warn!(
            "auth is enabled but TLS is disabled — tokens, prompts, and completions are sent \
             in plaintext. Add --cert/--private-key or --letsencrypt to fix."
        );
    }

    tracing::info!(
        endpoint = %format_args!("{scheme}://{addr}/v1/chat/completions"),
        health = %format_args!("{scheme}://{addr}/health"),
        metrics = %format_args!("{scheme}://{addr}/metrics"),
        max_pending = args.max_pending,
        max_active = args.max_active,
        request_timeout_secs = args.request_timeout,
        "server ready",
    );

    let shutting_down = Arc::new(AtomicBool::new(false));

    let state = Arc::new(ServerState {
        request_tx,
        model_name,
        tokenizer,
        arch,
        vision_config,
        image_token_id,
        auth: auth_provider,
        metrics: server_metrics,
        shutting_down: shutting_down.clone(),
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
        // Health check: returns 200 when healthy, 503 during shutdown so
        // load balancers drain traffic before the server stops.
        .route("/health", axum::routing::get(health_handler))
        .route("/metrics", axum::routing::get(metrics::metrics_handler))
        // Auth middleware — inside CORS so preflight OPTIONS get CORS headers
        // even when auth denies them.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(tower_http::cors::CorsLayer::permissive())
        // Reject request bodies larger than 50 MB to prevent memory
        // exhaustion from oversized payloads (e.g. huge base64 images).
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            50 * 1024 * 1024,
        ))
        .with_state(state.clone());

    // Spawn auth provider's background task (JWKS refresh, etc.).
    rt.block_on(async {
        state.auth.spawn_background();
    });

    rt.block_on(async {
        // Shutdown signal: SIGINT (Ctrl-C) or SIGTERM (container orchestrators).
        // A watch channel broadcasts the signal so all server paths (plain HTTP,
        // manual TLS, Let's Encrypt) can share the same shutdown trigger.
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let shutdown_flag = shutting_down.clone();

        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to install SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c.await.ok();
            }
            tracing::info!("shutdown signal received, draining in-flight requests");
            shutdown_flag.store(true, Ordering::SeqCst);
            let _ = shutdown_tx.send(true);
        });

        /// Create a future that resolves when the shutdown watch fires.
        fn shutdown_from_watch(mut rx: tokio::sync::watch::Receiver<bool>) -> impl std::future::Future<Output = ()> + Send + 'static {
            async move {
                while !*rx.borrow_and_update() {
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            }
        }

        match tls_mode {
            tls::TlsMode::None => {
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                axum::serve(listener, app)
                    .with_graceful_shutdown(shutdown_from_watch(shutdown_rx))
                    .await?;
            }
            tls::TlsMode::Manual { cert, key } => {
                tls::serve_manual_tls(app, &addr, &cert, &key, shutdown_from_watch(shutdown_rx))
                    .await?;
            }
            tls::TlsMode::LetsEncrypt {
                domain,
                email,
                cache_dir,
            } => {
                tls::serve_letsencrypt(
                    app, &addr, &domain, email.as_deref(), &cache_dir,
                    shutdown_from_watch(shutdown_rx),
                ).await?;
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
    #[allow(dead_code)] // populated by worker thread; will be used for handler-side vision preprocessing
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
    kv_quant: crate::model::turboquant::KvQuantMode,
    metrics: Arc<metrics::Metrics>,
    max_pending: usize,
    max_active: usize,
    request_timeout: std::time::Duration,
) -> anyhow::Result<WorkerHandle> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(max_pending);
    type ReadyPayload = (Arc<tokenizer::Tokenizer>, config::ModelArch, Option<crate::model::config::VisionConfig>);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<ReadyPayload, String>,
    >(1);

    std::thread::spawn(move || {
        let run = || -> anyhow::Result<()> {
            engine::loader::load_and_run_ext(
                &model_dir,
                stream_experts,
                tp,
                kv_quant,
                max_active,
                |tok, arch| {
                    let _ = ready_tx.send(Ok((Arc::new(tok.clone()), arch, None)));
                },
                |eng| run_worker_loop(eng, request_rx, metrics, request_timeout),
            )
        };

        // catch_unwind prevents a panic in the GPU backend or model code from
        // silently killing the worker thread.  Without this, a panic leaves
        // all in-flight requests hanging forever and the server appears alive
        // but cannot process any new work.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = ready_tx.send(Err(format!("{e:#}")));
                tracing::error!(error = %e, "worker thread exited with error");
            }
            Err(panic_payload) => {
                let msg = match panic_payload.downcast_ref::<&str>() {
                    Some(s) => s.to_string(),
                    None => match panic_payload.downcast_ref::<String>() {
                        Some(s) => s.clone(),
                        None => "unknown panic".to_string(),
                    },
                };
                let _ = ready_tx.send(Err(format!("worker panicked: {msg}")));
                tracing::error!(panic = %msg, "worker thread panicked");
            }
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
    /// Request ID for log correlation (generated at the HTTP handler).
    request_id: String,
    response_tx: tokio::sync::mpsc::Sender<InferenceEvent>,
    prompt_token_count: usize,
    generated_count: usize,
    /// Absolute deadline after which this request is aborted.
    /// None means no timeout (request_timeout was 0).
    deadline: Option<Instant>,
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
    #[allow(dead_code)] // stored for future per-sequence thinking control
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
    /// Stop sequences — generation stops when any of these strings appear.
    /// Checked after each token decode against the accumulated output text.
    /// When a match is found, the stop sequence itself is excluded: the
    /// engine is told to abort, and a Done event with EndOfSequence is sent.
    stop: Vec<String>,
    /// Accumulated full decoded text for stop-sequence matching.
    /// Kept separately from the token-by-token SSE output so we can check
    /// the full string for stop sequence substrings each step.
    full_text: String,
    /// Whether this request wants log-probability information.
    logprobs_requested: bool,
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

/// Record Prometheus metrics for a completed request.
///
/// Called from both the normal completion path (section 5) and the
/// stop-sequence path (section 4) to avoid missing metrics for
/// sequences that finish via stop strings.
fn record_completion_metrics(metrics: &metrics::Metrics, ctx: &RequestContext) {
    let now = Instant::now();
    let total_secs = now.duration_since(ctx.created_at).as_secs_f64();
    metrics.request_duration.observe(total_secs);

    if let Some(first) = ctx.first_token_at {
        let ttft_secs = first.duration_since(ctx.created_at).as_secs_f64();
        metrics.time_to_first_token.observe(ttft_secs);

        let decode_secs = now.duration_since(first).as_secs_f64();
        let decode_tokens = ctx.generated_count.saturating_sub(1);
        if decode_secs > 0.0 {
            metrics
                .decode_tokens_per_second
                .observe(decode_tokens as f64 / decode_secs);
        }
    }

    metrics.prompt_tokens.inc_by(ctx.prompt_token_count as u64);
    metrics.completion_tokens.inc_by(ctx.generated_count as u64);
    if ctx.cached_tokens > 0 {
        metrics.prefix_cache_hits.inc();
    }
}

/// Run the continuous batching loop for any InferenceEngine.
///
/// Blocks the calling thread until the request channel closes (server shutdown).
/// The engine must already be fully initialized (model loaded, KV cache allocated).
fn run_worker_loop(
    engine: &mut dyn engine::InferenceEngine,
    request_rx: std::sync::mpsc::Receiver<WorkerRequest>,
    metrics: Arc<metrics::Metrics>,
    request_timeout: std::time::Duration,
) -> anyhow::Result<()> {
    let mut registry: HashMap<SeqId, RequestContext> = HashMap::new();
    let has_timeout = !request_timeout.is_zero();

    /// Helper: register a WorkerRequest with the engine and request registry.
    fn register_request(
        engine: &mut dyn engine::InferenceEngine,
        registry: &mut HashMap<SeqId, RequestContext>,
        req: WorkerRequest,
        deadline: Option<Instant>,
    ) {
        let prompt_token_count = req.prompt_tokens.len();
        let thinking = req.thinking.unwrap_or(false);
        let user = req.user;
        let stop = req.stop;
        let logprobs_requested = req.params.logprobs;
        let seq_id = engine.add_request(
            req.prompt_tokens,
            req.max_tokens,
            req.params,
            req.images,
            req.seed,
            req.grammar,
            req.logit_bias,
        );
        registry.insert(
            seq_id,
            RequestContext {
                request_id: req.request_id,
                response_tx: req.response_tx,
                prompt_token_count,
                generated_count: 0,
                deadline,
                cached_tokens: 0,
                token_ids: Vec::new(),
                prev_text_len: 0,
                created_at: Instant::now(),
                first_token_at: None,
                thinking,
                inject_marker: None,
                user,
                stop,
                full_text: String::new(),
                logprobs_requested,
            },
        );
    }

    loop {
        let deadline = if has_timeout { Some(Instant::now() + request_timeout) } else { None };

        // 1. Drain all pending requests (non-blocking).
        while let Ok(req) = request_rx.try_recv() {
            register_request(engine, &mut registry, req, deadline);
        }

        // 2. If no work, block until a new request arrives.
        if !engine.has_work() {
            match request_rx.recv() {
                Ok(req) => {
                    register_request(engine, &mut registry, req, deadline);
                }
                Err(_) => break, // Channel closed — server shutting down.
            }
        }

        // 3. Run one engine step (prefill + decode + sample).
        let step_output = match engine.step() {
            Ok(output) => output,
            Err(e) => {
                metrics.errors.inc();
                // Log the full error chain for debugging, but send a generic
                // message to the client to avoid leaking internal details
                // (file paths, GPU memory addresses, Metal/CUDA error codes).
                tracing::error!(error = %format_args!("{e:#}"), "engine step failed");
                let seq_ids: Vec<SeqId> = registry.keys().copied().collect();
                for (_, ctx) in registry.drain() {
                    let _ = ctx
                        .response_tx
                        .blocking_send(InferenceEvent::Error(
                            "internal inference error".to_string(),
                        ));
                }
                for id in seq_ids {
                    engine.abort_sequence(id);
                }
                continue;
            }
        };

        // Update sequence gauges after each step.
        metrics.active_sequences.set(engine.active_count() as i64);
        metrics.waiting_sequences.set(engine.waiting_count() as i64);

        // 4. Stream tokens to response channels.
        let mut to_abort: Vec<SeqId> = Vec::new();
        let tokenizer = engine.tokenizer();

        for (seq_id, token_id, logprob_info) in step_output.tokens {
            if let Some(ctx) = registry.get_mut(&seq_id) {
                // Detect thinking special tokens.  Models like Qwen 3.5 emit
                // <think> (248068) and </think> (248069) as special tokens that
                // `decode(skip_special=true)` strips.  We always intercept these
                // so thinking content is visible even when the user didn't
                // explicitly request thinking — Qwen 3.5 thinks by default.
                if tokenizer.is_think_start(token_id) {
                    // Flush any pending marker first, then store the new one.
                    if let Some(prev) = ctx.inject_marker.take() {
                        let _ = ctx
                            .response_tx
                            .blocking_send(InferenceEvent::Token { text: prev.to_string(), logprob_info: None });
                    }
                    ctx.inject_marker = Some("<think>");
                    ctx.generated_count += 1;
                    if ctx.first_token_at.is_none() {
                        ctx.first_token_at = Some(Instant::now());
                    }
                    continue;
                }
                if tokenizer.is_think_end(token_id) {
                    if let Some(prev) = ctx.inject_marker.take() {
                        let _ = ctx
                            .response_tx
                            .blocking_send(InferenceEvent::Token { text: prev.to_string(), logprob_info: None });
                    }
                    ctx.inject_marker = Some("</think>");
                    ctx.generated_count += 1;
                    continue;
                }

                ctx.token_ids.push(token_id);
                let full_text = tokenizer.decode(&ctx.token_ids).unwrap_or_default();
                // When a multi-byte character (e.g. emoji) is split across tokens,
                // the previous decode's byte length may land inside that character
                // in the new, longer decode.  Snap to a valid char boundary.
                let start = full_text.floor_char_boundary(ctx.prev_text_len);
                let mut text = full_text[start..].to_string();
                ctx.prev_text_len = full_text.len();

                // Prepend any pending thinking marker to the decoded text.
                if let Some(marker) = ctx.inject_marker.take() {
                    text = format!("{marker}{text}");
                }

                ctx.generated_count += 1;
                if ctx.first_token_at.is_none() {
                    ctx.first_token_at = Some(Instant::now());
                }

                // Accumulate decoded text for stop-sequence matching.
                ctx.full_text.push_str(&text);

                // Check stop sequences against the accumulated output.
                // When a match is found, truncate the stop string from the
                // final token event and signal completion.
                let mut hit_stop = false;
                for stop_str in &ctx.stop {
                    if let Some(pos) = ctx.full_text.find(stop_str.as_str()) {
                        // The stop string may span this token and previous ones.
                        // Calculate how much of `text` to keep: everything before
                        // the stop string's start minus what was already sent.
                        let already_sent = ctx.full_text.len() - text.len();
                        if pos >= already_sent {
                            // Stop string starts within this token's text.
                            let keep = pos - already_sent;
                            text.truncate(keep);
                        } else {
                            // Stop string started in previously sent text — don't
                            // emit this token at all.
                            text.clear();
                        }
                        hit_stop = true;
                        break;
                    }
                }

                if !text.is_empty() {
                    if ctx
                        .response_tx
                        .blocking_send(InferenceEvent::Token { text, logprob_info })
                        .is_err()
                    {
                        to_abort.push(seq_id);
                        continue;
                    }
                }

                if hit_stop {
                    // Record metrics for stop-sequence completions (these
                    // bypass section 5 because the engine aborts them).
                    record_completion_metrics(&metrics, ctx);
                    let _ = ctx.response_tx.blocking_send(InferenceEvent::Done {
                        stop_reason: StopReason::EndOfSequence,
                        prompt_tokens: ctx.prompt_token_count,
                        completion_tokens: ctx.generated_count,
                        cached_tokens: ctx.cached_tokens,
                    });
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
                        .blocking_send(InferenceEvent::Token { text: marker.to_string(), logprob_info: None });
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
                let user_sub = ctx.user.as_ref().map(|u| u.sub.as_str()).unwrap_or("-");
                tracing::info!(
                    request_id = %ctx.request_id,
                    seq = finished.id,
                    user = user_sub,
                    prompt_tokens = ctx.prompt_token_count,
                    cached_tokens = ctx.cached_tokens,
                    generated_tokens = ctx.generated_count,
                    ttft_ms = format_args!("{:.0}", ttft_ms.unwrap_or(0.0)),
                    tok_per_sec = format_args!("{:.1}", tok_per_sec),
                    total_secs = format_args!("{:.2}", total_secs),
                    stop = stop_label,
                    "request complete",
                );

                record_completion_metrics(&metrics, &ctx);

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

        // 7. Abort timed-out sequences.
        if has_timeout {
            let now = Instant::now();
            let timed_out: Vec<SeqId> = registry
                .iter()
                .filter(|(_, ctx)| ctx.deadline.is_some_and(|d| now >= d))
                .map(|(&id, _)| id)
                .collect();
            for id in timed_out {
                if let Some(ctx) = registry.remove(&id) {
                    tracing::warn!(
                        request_id = %ctx.request_id,
                        generated = ctx.generated_count,
                        timeout_secs = request_timeout.as_secs(),
                        "request timed out",
                    );
                    metrics.request_timeouts.inc();
                    let _ = ctx.response_tx.blocking_send(InferenceEvent::Error(
                        format!("request timed out after {}s", request_timeout.as_secs()),
                    ));
                    engine.abort_sequence(id);
                }
            }
        }
    }

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt; // for .oneshot()

    /// Build a minimal RequestContext for testing metrics recording.
    fn test_ctx(prompt_tokens: usize, generated: usize, cached: usize) -> RequestContext {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let created = Instant::now() - std::time::Duration::from_millis(100);
        RequestContext {
            request_id: "test-id".into(),
            response_tx: tx,
            prompt_token_count: prompt_tokens,
            generated_count: generated,
            deadline: None,
            token_ids: Vec::new(),
            prev_text_len: 0,
            cached_tokens: cached,
            created_at: created,
            first_token_at: Some(created + std::time::Duration::from_millis(20)),
            thinking: false,
            inject_marker: None,
            user: None,
            stop: Vec::new(),
            full_text: String::new(),
            logprobs_requested: false,
        }
    }

    #[test]
    fn record_completion_metrics_increments_counters() {
        let m = metrics::Metrics::new();
        let ctx = test_ctx(100, 50, 10);

        record_completion_metrics(&m, &ctx);

        assert_eq!(m.prompt_tokens.get(), 100);
        assert_eq!(m.completion_tokens.get(), 50);
        assert_eq!(m.prefix_cache_hits.get(), 1);
        assert_eq!(m.request_duration.get_sample_count(), 1);
        assert_eq!(m.time_to_first_token.get_sample_count(), 1);
        assert_eq!(m.decode_tokens_per_second.get_sample_count(), 1);
    }

    #[test]
    fn record_completion_metrics_no_cache_hit() {
        let m = metrics::Metrics::new();
        let ctx = test_ctx(50, 10, 0);

        record_completion_metrics(&m, &ctx);

        assert_eq!(m.prefix_cache_hits.get(), 0);
        assert_eq!(m.prompt_tokens.get(), 50);
    }

    #[test]
    fn record_completion_metrics_no_first_token() {
        let m = metrics::Metrics::new();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let ctx = RequestContext {
            request_id: "test".into(),
            response_tx: tx,
            prompt_token_count: 10,
            generated_count: 0,
            deadline: None,
            token_ids: Vec::new(),
            prev_text_len: 0,
            cached_tokens: 0,
            created_at: Instant::now(),
            first_token_at: None, // No tokens generated.
            thinking: false,
            inject_marker: None,
            user: None,
            stop: Vec::new(),
            full_text: String::new(),
            logprobs_requested: false,
        };

        record_completion_metrics(&m, &ctx);

        // TTFT and decode throughput should NOT be recorded.
        assert_eq!(m.time_to_first_token.get_sample_count(), 0);
        assert_eq!(m.decode_tokens_per_second.get_sample_count(), 0);
        // Duration and counters should still be recorded.
        assert_eq!(m.request_duration.get_sample_count(), 1);
        assert_eq!(m.prompt_tokens.get(), 10);
    }

    #[test]
    fn request_timeout_counter_is_separate_from_errors() {
        let m = metrics::Metrics::new();
        // Simulate a timeout and an error — they go to different counters.
        m.request_timeouts.inc();
        m.errors.inc();
        m.errors.inc();
        assert_eq!(m.request_timeouts.get(), 1);
        assert_eq!(m.errors.get(), 2);
    }

    #[test]
    fn deadline_computed_from_timeout() {
        // Zero timeout → no deadline.
        let timeout = std::time::Duration::ZERO;
        let has_timeout = !timeout.is_zero();
        assert!(!has_timeout);

        // Non-zero timeout → deadline is in the future.
        let timeout = std::time::Duration::from_secs(30);
        let has_timeout = !timeout.is_zero();
        assert!(has_timeout);
        let deadline = Instant::now() + timeout;
        assert!(deadline > Instant::now());
    }

    #[test]
    fn expired_deadline_detected() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let ctx = RequestContext {
            request_id: "timeout-test".into(),
            response_tx: tx,
            prompt_token_count: 10,
            generated_count: 5,
            // Set deadline to 50ms ago — should be expired.
            deadline: Some(Instant::now() - std::time::Duration::from_millis(50)),
            token_ids: Vec::new(),
            prev_text_len: 0,
            cached_tokens: 0,
            created_at: Instant::now() - std::time::Duration::from_secs(1),
            first_token_at: None,
            thinking: false,
            inject_marker: None,
            user: None,
            stop: Vec::new(),
            full_text: String::new(),
            logprobs_requested: false,
        };
        let now = Instant::now();
        assert!(ctx.deadline.is_some_and(|d| now >= d));
    }

    #[test]
    fn no_deadline_never_expires() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let ctx = RequestContext {
            request_id: "no-timeout".into(),
            response_tx: tx,
            prompt_token_count: 10,
            generated_count: 5,
            deadline: None,
            token_ids: Vec::new(),
            prev_text_len: 0,
            cached_tokens: 0,
            created_at: Instant::now(),
            first_token_at: None,
            thinking: false,
            inject_marker: None,
            user: None,
            stop: Vec::new(),
            full_text: String::new(),
            logprobs_requested: false,
        };
        let now = Instant::now();
        assert!(!ctx.deadline.is_some_and(|d| now >= d));
    }

    #[test]
    fn timeout_metric_in_prometheus_output() {
        let m = metrics::Metrics::new();
        m.request_timeouts.inc_by(3);
        let output = m.encode();
        assert!(output.contains("rllm_request_timeouts_total 3"));
    }

    // -----------------------------------------------------------------------
    // Health handler
    // -----------------------------------------------------------------------

    /// Build a minimal ServerState for testing (no real tokenizer or engine).
    fn test_server_state() -> Arc<ServerState> {
        // Channel that we never read — just need a valid SyncSender.
        let (tx, _rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(1);
        Arc::new(ServerState {
            request_tx: tx,
            model_name: "test".into(),
            // Safety: tokenizer is never called in these tests — we just
            // need a valid Arc to satisfy the struct.  Using a zeroed pointer
            // would be UB, so we use a real but empty-ish construction.
            tokenizer: Arc::new(tokenizer::Tokenizer::for_test(vec![0])),
            arch: config::ModelArch::Llama,
            vision_config: None,
            image_token_id: None,
            auth: auth::AuthProviderKind::None,
            metrics: Arc::new(metrics::Metrics::new()),
            shutting_down: Arc::new(AtomicBool::new(false)),
        })
    }

    #[tokio::test]
    async fn health_returns_200_when_healthy() {
        let state = test_server_state();
        let app = axum::Router::new()
            .route("/health", axum::routing::get(health_handler))
            .with_state(state);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["active_sequences"], 0);
        assert_eq!(json["waiting_sequences"], 0);
    }

    #[tokio::test]
    async fn health_returns_503_during_shutdown() {
        let state = test_server_state();
        state.shutting_down.store(true, Ordering::SeqCst);

        let app = axum::Router::new()
            .route("/health", axum::routing::get(health_handler))
            .with_state(state);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "shutting_down");
    }

    #[tokio::test]
    async fn health_reflects_sequence_gauges() {
        let state = test_server_state();
        state.metrics.active_sequences.set(5);
        state.metrics.waiting_sequences.set(12);

        let app = axum::Router::new()
            .route("/health", axum::routing::get(health_handler))
            .with_state(state);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["active_sequences"], 5);
        assert_eq!(json["waiting_sequences"], 12);
    }

    // -----------------------------------------------------------------------
    // Shutdown watch channel
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn shutdown_watch_resolves_on_signal() {
        let (tx, rx) = tokio::sync::watch::channel(false);

        // shutdown_from_watch is defined inside serve(), so replicate its logic.
        let fut = {
            let mut rx = rx;
            async move {
                while !*rx.borrow_and_update() {
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            }
        };

        // Signal shutdown after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            tx.send(true).unwrap();
        });

        // The future should resolve within 100ms.
        let result = tokio::time::timeout(std::time::Duration::from_millis(100), fut).await;
        assert!(result.is_ok(), "shutdown watch did not resolve in time");
    }

    #[tokio::test]
    async fn shutdown_watch_does_not_resolve_without_signal() {
        let (_tx, rx) = tokio::sync::watch::channel(false);

        let fut = {
            let mut rx = rx;
            async move {
                while !*rx.borrow_and_update() {
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            }
        };

        // Should NOT resolve within 50ms.
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), fut).await;
        assert!(result.is_err(), "shutdown watch resolved without a signal");
    }

    // -----------------------------------------------------------------------
    // Mock engine + worker loop integration tests
    // -----------------------------------------------------------------------

    use std::collections::{HashSet, VecDeque};
    use crate::engine::{FinishReason, FinishedSequence, StepOutput};

    /// A mock InferenceEngine that returns pre-programmed step results.
    struct MockEngine {
        tokenizer: tokenizer::Tokenizer,
        next_id: SeqId,
        active: HashSet<SeqId>,
        /// Pre-programmed results for each step() call.
        step_results: VecDeque<Box<dyn FnMut(&mut MockEngine) -> anyhow::Result<StepOutput> + Send>>,
        aborted: Vec<SeqId>,
    }

    impl MockEngine {
        fn new() -> Self {
            Self {
                tokenizer: tokenizer::Tokenizer::for_test(vec![2]), // EOS = 2
                next_id: 1,
                active: HashSet::new(),
                step_results: VecDeque::new(),
                aborted: Vec::new(),
            }
        }

        /// Queue a step that emits one token for each active sequence.
        fn push_token_step(&mut self, token_id: u32) {
            self.step_results.push_back(Box::new(move |eng: &mut MockEngine| {
                let tokens: Vec<_> = eng.active.iter().map(|&id| (id, token_id, None)).collect();
                Ok(StepOutput { tokens, finished: Vec::new() })
            }));
        }

        /// Queue a step that finishes all active sequences.
        fn push_finish_step(&mut self) {
            self.step_results.push_back(Box::new(|eng: &mut MockEngine| {
                let finished: Vec<_> = eng.active.drain().map(|id| FinishedSequence {
                    id,
                    tokens: Vec::new(),
                    text: String::new(),
                    reason: FinishReason::Eos,
                    cached_tokens: 0,
                }).collect();
                Ok(StepOutput { tokens: Vec::new(), finished })
            }));
        }
    }

    impl engine::InferenceEngine for MockEngine {
        fn add_request(
            &mut self,
            _prompt_tokens: Vec<u32>,
            _max_gen_tokens: usize,
            _params: crate::model::sampler::SampleParams,
            _images: Vec<crate::model::vision::ProcessedImage>,
            _seed: Option<u64>,
            _grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
            _logit_bias: HashMap<u32, f32>,
        ) -> SeqId {
            let id = self.next_id;
            self.next_id += 1;
            self.active.insert(id);
            id
        }

        fn step(&mut self) -> anyhow::Result<StepOutput> {
            if let Some(mut f) = self.step_results.pop_front() {
                f(self)
            } else {
                // No more programmed steps — return empty (no tokens, no finished).
                Ok(StepOutput { tokens: Vec::new(), finished: Vec::new() })
            }
        }

        fn abort_sequence(&mut self, id: SeqId) {
            self.active.remove(&id);
            self.aborted.push(id);
        }

        fn has_work(&self) -> bool {
            !self.active.is_empty() || !self.step_results.is_empty()
        }

        fn tokenizer(&self) -> &tokenizer::Tokenizer {
            &self.tokenizer
        }

        fn active_count(&self) -> usize { self.active.len() }
        fn waiting_count(&self) -> usize { 0 }
    }

    /// Helper: build a WorkerRequest with a response channel.
    fn mock_worker_request() -> (WorkerRequest, tokio::sync::mpsc::Receiver<InferenceEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let req = WorkerRequest {
            request_id: "test-req".into(),
            prompt_tokens: vec![1],
            max_tokens: 10,
            params: crate::model::sampler::SampleParams {
                temperature: 0.0,
                ..crate::model::sampler::SampleParams::default()
            },
            response_tx: tx,
            thinking: None,
            images: Vec::new(),
            user: None,
            seed: None,
            stop: Vec::new(),
            grammar: None,
            logit_bias: HashMap::new(),
        };
        (req, rx)
    }

    #[test]
    fn worker_loop_generates_tokens_and_finishes() {
        let mut engine = MockEngine::new();
        engine.push_token_step(42); // step 1: emit token (decode may produce empty text)
        engine.push_finish_step();  // step 2: finish

        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
        let m = Arc::new(metrics::Metrics::new());

        let (req, mut response_rx) = mock_worker_request();
        request_tx.send(req).unwrap();
        // Drop sender so the loop exits after processing.
        drop(request_tx);

        run_worker_loop(
            &mut engine,
            request_rx,
            m.clone(),
            std::time::Duration::from_secs(300),
        ).unwrap();

        // Collect all events.
        let mut events = Vec::new();
        while let Ok(ev) = response_rx.try_recv() {
            events.push(ev);
        }

        // Should receive a Done event (Token events depend on tokenizer vocabulary).
        assert!(
            events.iter().any(|e| matches!(e, InferenceEvent::Done { .. })),
            "expected Done event, got: {events:?}",
        );

        // Metrics should be recorded.
        assert!(m.prompt_tokens.get() > 0);
        assert!(m.completion_tokens.get() > 0);
    }

    #[test]
    fn worker_loop_aborts_timed_out_request() {
        let mut engine = MockEngine::new();
        // Queue enough steps that the timeout fires during processing.
        for _ in 0..5 {
            engine.push_token_step(42);
        }
        engine.push_finish_step(); // in case timeout doesn't fire

        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
        let m = Arc::new(metrics::Metrics::new());

        let (req, mut response_rx) = mock_worker_request();
        request_tx.send(req).unwrap();
        drop(request_tx);

        // Zero timeout — deadline is already expired on first check.
        run_worker_loop(
            &mut engine,
            request_rx,
            m.clone(),
            std::time::Duration::from_nanos(1),
        ).unwrap();

        // Collect all events — should contain an Error about timeout.
        let mut events = Vec::new();
        while let Ok(ev) = response_rx.try_recv() {
            events.push(ev);
        }

        assert!(
            events.iter().any(|e| matches!(e, InferenceEvent::Error(msg) if msg.contains("timed out"))),
            "expected a timeout error event, got: {events:?}",
        );
        assert_eq!(m.request_timeouts.get(), 1);
        // The sequence should have been aborted in the engine.
        assert!(engine.aborted.len() >= 1);
    }

    #[test]
    fn worker_loop_handles_engine_step_error() {
        let mut engine = MockEngine::new();
        engine.step_results.push_back(Box::new(|_eng: &mut MockEngine| {
            anyhow::bail!("GPU exploded")
        }));

        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
        let m = Arc::new(metrics::Metrics::new());

        let (req, mut response_rx) = mock_worker_request();
        request_tx.send(req).unwrap();
        drop(request_tx);

        run_worker_loop(
            &mut engine,
            request_rx,
            m.clone(),
            std::time::Duration::from_secs(300),
        ).unwrap();

        let mut events = Vec::new();
        while let Ok(ev) = response_rx.try_recv() {
            events.push(ev);
        }

        assert!(
            events.iter().any(|e| matches!(e, InferenceEvent::Error(msg) if msg.contains("internal inference error"))),
            "expected sanitized engine error event, got: {events:?}",
        );
        // The raw engine error must NOT leak to the client.
        assert!(
            !events.iter().any(|e| matches!(e, InferenceEvent::Error(msg) if msg.contains("GPU exploded"))),
            "raw engine error leaked to client: {events:?}",
        );
        assert_eq!(m.errors.get(), 1);
        // All active sequences should be aborted in the engine after a step error.
        assert!(!engine.aborted.is_empty(), "engine sequences not aborted after step error");
    }

    #[test]
    fn worker_loop_aborts_disconnected_client() {
        let mut engine = MockEngine::new();
        engine.push_token_step(42);
        engine.push_token_step(42);
        engine.push_token_step(42);
        engine.push_finish_step();

        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
        let m = Arc::new(metrics::Metrics::new());

        let (req, response_rx) = mock_worker_request();
        request_tx.send(req).unwrap();
        drop(request_tx);
        // Drop the receiver — simulates client disconnect.
        drop(response_rx);

        run_worker_loop(
            &mut engine,
            request_rx,
            m.clone(),
            std::time::Duration::from_secs(300),
        ).unwrap();

        // The sequence should have been aborted.
        assert!(!engine.aborted.is_empty());
    }

    #[test]
    fn worker_loop_aborts_all_sequences_on_engine_error() {
        // Regression test: engine step error must abort ALL active sequences,
        // not just drain the registry (which left orphans looping forever).
        let mut engine = MockEngine::new();
        // First step succeeds (both sequences get tokens), second step errors.
        engine.push_token_step(42);
        engine.step_results.push_back(Box::new(|_eng: &mut MockEngine| {
            anyhow::bail!("GPU memory corruption")
        }));

        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
        let m = Arc::new(metrics::Metrics::new());

        // Submit two requests so multiple sequences are active.
        let (req1, _rx1) = mock_worker_request();
        let (req2, _rx2) = mock_worker_request();
        request_tx.send(req1).unwrap();
        request_tx.send(req2).unwrap();
        drop(request_tx);

        run_worker_loop(
            &mut engine,
            request_rx,
            m.clone(),
            std::time::Duration::from_secs(300),
        ).unwrap();

        // Both sequences must have been aborted in the engine.
        assert_eq!(
            engine.aborted.len(), 2,
            "expected 2 aborted sequences, got: {:?}", engine.aborted,
        );
        assert!(engine.active.is_empty(), "engine still has active sequences");
    }

    #[test]
    fn shutdown_flag_uses_correct_ordering() {
        // Verify the shutdown flag is read/written with SeqCst (not Relaxed).
        // We can't test memory ordering directly, but we can verify the flag
        // is visible immediately from the same thread.
        let flag = Arc::new(AtomicBool::new(false));
        flag.store(true, Ordering::SeqCst);
        assert!(flag.load(Ordering::SeqCst));
    }

    // -----------------------------------------------------------------------
    // collect_response() unit tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn collect_response_returns_on_done_not_sender_drop() {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let handle = tokio::spawn(collect_response(rx));

        tx.send(InferenceEvent::Token { text: "hello ".into(), logprob_info: None }).await.unwrap();
        tx.send(InferenceEvent::Token { text: "world".into(), logprob_info: None }).await.unwrap();
        tx.send(InferenceEvent::Done {
            stop_reason: StopReason::EndOfSequence,
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 0,
        }).await.unwrap();

        // collect_response must complete while tx is still alive — if it waits
        // for the sender to drop (the original bug), this timeout will fire.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            handle,
        ).await;
        assert!(result.is_ok(), "collect_response hung waiting for sender drop");

        let (text, prompt_tokens, completion_tokens, stop_reason) =
            result.unwrap().unwrap().unwrap();
        assert_eq!(text, "hello world");
        assert_eq!(prompt_tokens, 5);
        assert_eq!(completion_tokens, 2);
        assert!(matches!(stop_reason, StopReason::EndOfSequence));

        // tx is still alive — proves we returned on Done, not sender drop.
        drop(tx);
    }

    #[tokio::test]
    async fn collect_response_returns_error() {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tx.send(InferenceEvent::Token { text: "partial".into(), logprob_info: None }).await.unwrap();
        tx.send(InferenceEvent::Error("engine exploded".into())).await.unwrap();
        drop(tx);

        let result = collect_response(rx).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "engine exploded");
    }
}
