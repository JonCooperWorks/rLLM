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
pub(crate) mod openai;
pub(crate) mod tls;

use std::collections::HashMap;
use std::sync::Arc;

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
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

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

/// Start the API server.  Called from `main()` when `rllm serve` is invoked.
///
/// Loads the model in a dedicated worker thread, then starts axum on the
/// tokio runtime.  This function blocks until the server shuts down.
pub(crate) fn serve(args: ServeArgs) -> anyhow::Result<()> {
    eprintln!("loading model from {}...", args.model.display());

    let model_name = args
        .model
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "rllm-model".into());

    // Spawn the inference worker thread.
    // The worker loads backend/tokenizer/weights/model and runs the Engine
    // step loop.  We get back the request sender plus shared tokenizer/arch
    // for handler-side tokenization.
    // Resolve --tp 0 → auto-detect available GPUs.
    let mut tp = args.tp;
    if tp == 0 {
        tp = gpu::device_count();
        eprintln!("auto-detected {} GPU(s)", tp);
    }

    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    // On macOS (Metal), fall back to single GPU with a warning.
    #[cfg(not(feature = "cuda"))]
    if tp > 1 {
        eprintln!(
            "warning: --tp {} ignored (multi-GPU requires CUDA + NCCL), using single GPU",
            tp
        );
        tp = 1;
    }

    let WorkerHandle {
        request_tx,
        tokenizer,
        arch,
    } = spawn_worker(args.model.clone(), args.quantize, tp)?;

    eprintln!("model ready: {}", model_name);

    let state = Arc::new(ServerState {
        request_tx,
        model_name,
        tokenizer,
        arch,
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
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);

    // Determine TLS mode from CLI args.
    let tls_mode = if args.letsencrypt {
        let domain = args.domain.clone().unwrap(); // guaranteed by clap `requires`
        eprintln!(
            "TLS: Let's Encrypt for {domain} (cache: {})",
            args.cert_cache_dir.display()
        );
        tls::TlsMode::LetsEncrypt {
            domain,
            email: args.letsencrypt_email.clone(),
            cache_dir: args.cert_cache_dir.clone(),
        }
    } else if let (Some(cert), Some(key)) = (&args.cert, &args.private_key) {
        eprintln!("TLS: manual certs ({}, {})", cert.display(), key.display());
        tls::TlsMode::Manual {
            cert: cert.clone(),
            key: key.clone(),
        }
    } else if args.dangerous_no_tls {
        eprintln!("WARNING: running without TLS — traffic is unencrypted");
        tls::TlsMode::None
    } else {
        anyhow::bail!(
            "no TLS configuration provided. Use --cert/--private-key or --letsencrypt \
             to enable TLS, or pass --dangerous-no-tls to serve over plain HTTP."
        );
    };

    let scheme = if matches!(tls_mode, tls::TlsMode::None) {
        "http"
    } else {
        "https"
    };
    eprintln!("serving on {scheme}://{addr}");

    // Build the tokio runtime here (not in main) so the rest of the binary
    // stays synchronous.  The inference worker is a plain std::thread —
    // it never needs async I/O.
    let rt = tokio::runtime::Runtime::new()?;
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
}

/// Spawn the inference worker thread.
///
/// Delegates to engine::loader::load_and_run() for model loading and engine
/// construction, providing callbacks for the ready signal and the worker loop.
fn spawn_worker(
    model_dir: std::path::PathBuf,
    quantize: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>,
    >(1);

    std::thread::spawn(move || {
        let max_active = 32;

        let result = engine::loader::load_and_run(
            &model_dir,
            quantize,
            tp,
            max_active,
            |tok, arch| {
                let _ = ready_tx.send(Ok((Arc::new(tok.clone()), arch)));
            },
            |eng| run_worker_loop(eng, request_rx),
        );

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

    match ready_rx.recv() {
        Ok(Ok((tokenizer, arch))) => Ok(WorkerHandle {
            request_tx,
            tokenizer,
            arch,
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
            let seq_id = engine.add_request(
                req.prompt_tokens,
                req.max_tokens,
                req.temperature,
                req.top_p,
            );
            registry.insert(
                seq_id,
                RequestContext {
                    response_tx: req.response_tx,
                    prompt_token_count,
                    generated_count: 0,
                    token_ids: Vec::new(),
                    prev_text_len: 0,
                },
            );
        }

        // 2. If no work, block until a new request arrives.
        if !engine.has_work() {
            match request_rx.recv() {
                Ok(req) => {
                    let prompt_token_count = req.prompt_tokens.len();
                    let seq_id = engine.add_request(
                        req.prompt_tokens,
                        req.max_tokens,
                        req.temperature,
                        req.top_p,
                    );
                    registry.insert(
                        seq_id,
                        RequestContext {
                            response_tx: req.response_tx,
                            prompt_token_count,
                            generated_count: 0,
                            token_ids: Vec::new(),
                            prev_text_len: 0,
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

        for &(seq_id, token_id) in &step_output.tokens {
            if let Some(ctx) = registry.get_mut(&seq_id) {
                ctx.token_ids.push(token_id);
                let full_text = engine
                    .tokenizer()
                    .decode(&ctx.token_ids)
                    .unwrap_or_default();
                let text = full_text[ctx.prev_text_len..].to_string();
                ctx.prev_text_len = full_text.len();
                ctx.generated_count += 1;
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
            if let Some(ctx) = registry.remove(&finished.id) {
                let stop_reason = match finished.reason {
                    engine::FinishReason::Eos => StopReason::EndOfSequence,
                    engine::FinishReason::MaxTokens => StopReason::MaxTokens,
                };
                let _ = ctx.response_tx.blocking_send(InferenceEvent::Done {
                    stop_reason,
                    prompt_tokens: ctx.prompt_token_count,
                    completion_tokens: ctx.generated_count,
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
