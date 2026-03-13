// ===========================================================================
// API server — OpenAI and Anthropic compatible HTTP endpoints.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Sets up an HTTP server (axum + tokio) and bridges async HTTP handlers
//   to the synchronous GPU inference pipeline via a dedicated worker thread.
//
// Architecture:
//
//   HTTP clients ←→ [tokio async runtime / axum handlers]
//                         ↕ channels
//                    [inference worker thread]
//                         ↕ GPU
//                    [Metal/CUDA backend]
//
// Why a dedicated worker thread?
//   The model (`Model<'a, B>`) borrows the GPU backend, has mutable scratch
//   buffers, and GPU operations are single-threaded.  Rather than fighting
//   Rust's borrow checker with Arc<Mutex<...>>, we keep ALL GPU state on
//   one std::thread and communicate via channels:
//
//   - Request channel (std::sync::mpsc):  HTTP handlers → worker
//   - Response channel (tokio::sync::mpsc, one per request):  worker → handler
//
//   The worker calls `blocking_send` on the tokio channel, which is safe
//   from synchronous code.  The handler calls `recv().await` on the async
//   side.  This cleanly bridges sync GPU code and async HTTP serving.
//
// Streaming:
//   For SSE (Server-Sent Events), the handler reads token events from the
//   response channel and yields them as SSE frames.  The worker sends one
//   event per generated token, so the client sees output in real-time.
//
//   If the client disconnects mid-stream, the handler drops its receiver.
//   The worker detects this when `blocking_send` returns Err, and immediately
//   stops generating and frees KV cache blocks — no wasted GPU time.
// ===========================================================================

pub(crate) mod anthropic;
pub(crate) mod openai;
pub(crate) mod tls;

use std::path::PathBuf;
use std::sync::Arc;

use crate::ServeArgs;
use crate::gpu::{self, GpuBackend};
use crate::model;
use crate::model::loader;
use crate::model::{chat, config, kv_cache, sampler, tokenizer};

// ---------------------------------------------------------------------------
// Shared types: the bridge between HTTP handlers and the inference worker.
// ---------------------------------------------------------------------------

/// Request sent from an HTTP handler to the inference worker thread.
pub(crate) struct InferenceRequest {
    /// Chat messages (for chat/messages endpoints).
    pub messages: Vec<chat::Message>,
    /// Raw prompt text (for completions endpoint — bypasses chat template).
    pub raw_prompt: Option<String>,
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
}

/// Shared state accessible by all axum handlers via `State(Arc<ServerState>)`.
pub(crate) struct ServerState {
    /// Channel to send requests to the inference worker.
    pub request_tx: std::sync::mpsc::SyncSender<InferenceRequest>,
    /// Model name for API responses (derived from directory name).
    pub model_name: String,
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
    // The worker loads backend/tokenizer/weights/model and runs the blocking
    // inference loop.  We get back a channel sender to submit requests.
    let request_tx = spawn_worker(args.model.clone(), args.quantize)?;

    eprintln!("model ready: {}", model_name);

    let state = Arc::new(ServerState {
        request_tx,
        model_name,
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
    } else {
        tls::TlsMode::None
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

/// Spawn the inference worker thread that owns all GPU state.
///
/// Returns the request sender.  The worker loads the model inside the thread
/// (so all GPU resources have the thread's lifetime) and blocks on the
/// request channel.
fn spawn_worker(
    model_dir: PathBuf,
    quantize: bool,
) -> anyhow::Result<std::sync::mpsc::SyncSender<InferenceRequest>> {
    // Channel for incoming requests.  Capacity 8 gives a small queue;
    // if full, handlers get backpressure (try_send returns Err → 503).
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<InferenceRequest>(8);

    // We need to report load errors to the caller.  Use a oneshot channel
    // so the worker can signal success/failure before we return.
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);

    std::thread::spawn(move || {
        // All model state is created inside this thread so lifetimes work out.
        // The backend is on the stack, model borrows it — both live until
        // the thread exits (which is when the request channel closes).
        let result = (|| -> anyhow::Result<()> {
            let backend = gpu::create_backend()?;
            eprintln!("gpu: {}", backend.device_name());

            let loader::LoadedModel {
                config,
                arch,
                tokenizer,
                weights,
            } = loader::load_model(&backend, &model_dir, quantize)?;

            let mut model = model::Model::new(config.clone(), weights, &backend)?;

            // KV pool: 8192 blocks × 16 tokens/block = 131072 (128K) max context.
            // Memory: num_blocks * BLOCK_SIZE * kv_dim * 2 bytes * 2 (K+V) * num_layers.
            // For Llama 3.1 8B (kv_dim=1024, 32 layers): 8192 * 16 * 1024 * 2 * 2 * 32 = ~16 GB.
            let num_blocks = 8192;
            let kv_dim = config.num_key_value_heads * config.head_dim;
            let mut kv_pool =
                kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_hidden_layers);
            let max_prefill = 4096;
            let prefill_bufs = model::PrefillBuffers::new(&backend, &config, max_prefill);

            eprintln!(
                "KV cache: {} blocks ({} max tokens), prefill up to {} tokens",
                num_blocks,
                num_blocks * kv_cache::BLOCK_SIZE,
                max_prefill
            );

            // Signal success to the main thread.
            let _ = ready_tx.send(Ok(()));

            let mut rng = rand::rng();

            // Blocking request loop — runs until the channel closes (server shutdown).
            while let Ok(req) = request_rx.recv() {
                let result = process_request(
                    &req,
                    &mut model,
                    &mut kv_pool,
                    &prefill_bufs,
                    &tokenizer,
                    &backend,
                    arch,
                    &mut rng,
                );
                if let Err(e) = result {
                    let _ = req
                        .response_tx
                        .blocking_send(InferenceEvent::Error(format!("{e:#}")));
                }
            }

            Ok(())
        })();

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

    // Wait for the worker to finish loading (or fail).
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(request_tx),
        Ok(Err(e)) => anyhow::bail!("worker failed to start: {e}"),
        Err(_) => anyhow::bail!("worker thread died during startup"),
    }
}

/// Process a single inference request on the worker thread.
///
/// This mirrors the single-sequence path from `run()` in main.rs:
///   1. Tokenize the prompt (chat template or raw).
///   2. Allocate a KV sequence from the pool.
///   3. Prefill the entire prompt via GEMM forward pass.
///   4. Generate tokens one at a time, streaming each via the response channel.
///   5. Free the KV sequence blocks back to the pool.
fn process_request<B: GpuBackend>(
    req: &InferenceRequest,
    model: &mut model::Model<'_, B>,
    kv_pool: &mut kv_cache::KvPool<B>,
    prefill_bufs: &model::PrefillBuffers<B>,
    tokenizer: &tokenizer::Tokenizer,
    backend: &B,
    arch: config::ModelArch,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<()> {
    // 1. Tokenize: raw prompt or chat-formatted messages.
    let prompt_tokens = if let Some(ref raw) = req.raw_prompt {
        tokenizer.encode(raw)?
    } else {
        tokenizer.encode_messages(&req.messages, arch)?
    };
    let prompt_token_count = prompt_tokens.len();

    // 2. Allocate KV sequence from the shared pool.
    let mut seq_state = kv_pool.new_sequence(backend);

    // 3. Prefill: process entire prompt in one GEMM forward pass.
    seq_state.ensure_slots(kv_pool, prompt_tokens.len())?;
    seq_state.sync_block_table(backend);
    model.forward_prefill_paged(&prompt_tokens, kv_pool, &seq_state, prefill_bufs)?;
    seq_state.advance_by(prompt_tokens.len());

    // 4. Generation loop: sample → send token → forward → repeat.
    let mut gen_count: usize = 0;
    let mut next_token = sampler::sample(backend, model.logits(), req.temperature, req.top_p, rng)?;

    let mut stop_reason = StopReason::MaxTokens;

    for _ in 0..req.max_tokens {
        if tokenizer.is_eos(next_token) {
            stop_reason = StopReason::EndOfSequence;
            break;
        }
        gen_count += 1;

        let text = tokenizer.decode(&[next_token])?;

        // Send token to the HTTP handler.  If the handler dropped its
        // receiver (client disconnected), stop generating immediately.
        if req
            .response_tx
            .blocking_send(InferenceEvent::Token { text })
            .is_err()
        {
            // Client disconnected — clean up and return.
            kv_pool.free_sequence(&seq_state);
            return Ok(());
        }

        // Forward pass for the next token.
        seq_state.ensure_slot(kv_pool)?;
        seq_state.sync_block_table(backend);
        model.forward_single_paged(next_token, kv_pool, &seq_state)?;
        seq_state.advance();

        next_token = sampler::sample(backend, model.logits(), req.temperature, req.top_p, rng)?;
    }

    // 5. Send completion event.
    let _ = req.response_tx.blocking_send(InferenceEvent::Done {
        stop_reason,
        prompt_tokens: prompt_token_count,
        completion_tokens: gen_count,
    });

    // 6. Free KV blocks back to the pool for the next request.
    kv_pool.free_sequence(&seq_state);

    Ok(())
}
