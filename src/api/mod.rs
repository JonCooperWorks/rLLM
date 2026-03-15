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
//                    [inference worker thread (Engine + Scheduler)]
//                         ↕ GPU
//                    [Metal/CUDA backend]
//
// Continuous batching:
//   Multiple concurrent HTTP requests are batched together through the model.
//   The worker thread runs an Engine step loop that processes all active
//   sequences each step — prefilling new prompts via GEMM and decoding one
//   token per active sequence.  The decode phase is memory-bandwidth-bound,
//   so batching N sequences costs almost nothing extra per token.
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
// Tokenization:
//   Tokenization (text → token IDs) is CPU-only work.  It happens on the
//   async handler threads using the shared Arc<Tokenizer>, so the worker
//   thread's step loop is never blocked by tokenization.
//
// Streaming:
//   For SSE (Server-Sent Events), the handler reads token events from the
//   response channel and yields them as SSE frames.  The worker sends one
//   event per generated token, so the client sees output in real-time.
//
//   If the client disconnects mid-stream, the handler drops its receiver.
//   The worker detects this when `blocking_send` returns Err, and aborts
//   the sequence via the Engine — KV cache blocks are freed immediately.
// ===========================================================================

pub(crate) mod anthropic;
pub(crate) mod openai;
pub(crate) mod tls;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ServeArgs;
use crate::engine;
use crate::engine::scheduler::{self, SeqId};
use crate::gpu::{self, GpuCore};
use crate::model;
use crate::model::loader;
use crate::model::{config, kv_cache, tokenizer};

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
    let WorkerHandle {
        request_tx,
        tokenizer,
        arch,
    } = spawn_worker(args.model.clone(), args.quantize)?;

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

/// Per-request context tracked by the worker's request registry.
/// Maps an Engine SeqId to the HTTP response channel.
struct RequestContext {
    response_tx: tokio::sync::mpsc::Sender<InferenceEvent>,
    prompt_token_count: usize,
    generated_count: usize,
}

/// Spawn the inference worker thread that owns all GPU state.
///
/// Returns the request sender, shared tokenizer, and model architecture.
/// The worker loads the model inside the thread (so all GPU resources have
/// the thread's lifetime) and runs the Engine continuous batching loop.
fn spawn_worker(model_dir: PathBuf, quantize: bool) -> anyhow::Result<WorkerHandle> {
    // Channel for incoming requests.  Capacity 8 gives a small queue;
    // if full, handlers get backpressure (try_send returns Err → 503).
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);

    // Ready channel: worker sends back the tokenizer and arch (or an error).
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>>(1);

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

            // Share the tokenizer with HTTP handlers for async tokenization.
            let tokenizer = Arc::new(tokenizer);
            let _ = ready_tx.send(Ok((Arc::clone(&tokenizer), arch)));

            // Unwrap the Arc for the Engine — the worker thread keeps its own
            // reference.  The Engine takes ownership of a Tokenizer, so we
            // need to make a clone for it.
            let engine_tokenizer = (*tokenizer).clone();

            let model = model::Model::new(config.clone(), weights, &backend)?;

            // Dynamic KV cache sizing: fit as many blocks as possible into
            // the remaining GPU memory after weights.  Large models (e.g. 32B
            // with 64 layers) need much more KV cache per block, so a fixed
            // count can over-allocate and cause memory pressure / swap.
            let gpu_budget = backend.recommended_max_memory();
            let num_blocks = config.recommended_kv_blocks(gpu_budget, quantize);
            let kv_dim = config.num_key_value_heads * config.head_dim;
            let kv_pool =
                kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_kv_layers());

            // Maximum 32 concurrent sequences (can tune based on memory).
            let max_active = 32;
            let sched = scheduler::Scheduler::new(kv_pool, max_active);

            let weight_mb = config.estimate_weight_bytes(quantize) as f64 / (1024.0 * 1024.0);
            let kv_bytes_per_block =
                2 * config.num_hidden_layers * kv_cache::BLOCK_SIZE * kv_dim * 2;
            let kv_mb = (num_blocks * kv_bytes_per_block) as f64 / (1024.0 * 1024.0);
            eprintln!(
                "memory: {:.0} MB weights, {:.0} MB KV cache ({} blocks, {} max tokens), {:.0} MB GPU budget",
                weight_mb,
                kv_mb,
                num_blocks,
                num_blocks * kv_cache::BLOCK_SIZE,
                gpu_budget as f64 / (1024.0 * 1024.0),
            );
            eprintln!("max {} concurrent sequences", max_active);

            let mut eng = engine::Engine::new(
                model,
                sched,
                engine_tokenizer,
                &backend,
            );

            // Request registry: maps Engine sequence IDs to HTTP response channels.
            let mut registry: HashMap<SeqId, RequestContext> = HashMap::new();

            // ---------------------------------------------------------------------------
            // Continuous batching loop.
            //
            // The loop has three phases:
            //   1. Drain new requests from the channel (non-blocking).
            //   2. If no work, block on recv() until a request arrives.
            //   3. Run one Engine step, stream tokens, handle disconnects.
            // ---------------------------------------------------------------------------
            loop {
                // 1. Drain all pending requests (non-blocking).
                //    Batches up everything that arrived since the last step.
                while let Ok(req) = request_rx.try_recv() {
                    let prompt_token_count = req.prompt_tokens.len();
                    let seq_id = eng.add_request(req.prompt_tokens, req.max_tokens, req.temperature, req.top_p);
                    registry.insert(
                        seq_id,
                        RequestContext {
                            response_tx: req.response_tx,
                            prompt_token_count,
                            generated_count: 0,
                        },
                    );
                }

                // 2. If no work, block until a new request arrives.
                if !eng.has_work() {
                    match request_rx.recv() {
                        Ok(req) => {
                            let prompt_token_count = req.prompt_tokens.len();
                            let seq_id = eng.add_request(req.prompt_tokens, req.max_tokens, req.temperature, req.top_p);
                            registry.insert(
                                seq_id,
                                RequestContext {
                                    response_tx: req.response_tx,
                                    prompt_token_count,
                                    generated_count: 0,
                                },
                            );
                        }
                        Err(_) => break, // Channel closed — server shutting down.
                    }
                }

                // 3. Run one Engine step (prefill + decode + sample).
                let step_output = match eng.step() {
                    Ok(output) => output,
                    Err(e) => {
                        // Engine error — notify all active requests and drain.
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
                //    Collect aborts separately to avoid borrowing registry while iterating.
                let mut to_abort: Vec<SeqId> = Vec::new();

                for &(seq_id, token_id) in &step_output.tokens {
                    if let Some(ctx) = registry.get_mut(&seq_id) {
                        let text = eng.tokenizer.decode(&[token_id]).unwrap_or_default();
                        ctx.generated_count += 1;
                        if ctx
                            .response_tx
                            .blocking_send(InferenceEvent::Token { text })
                            .is_err()
                        {
                            // Client disconnected — schedule abort.
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
                    eng.abort_sequence(id);
                    registry.remove(&id);
                }

                let disconnected: Vec<SeqId> = registry
                    .iter()
                    .filter(|(_, ctx)| ctx.response_tx.is_closed())
                    .map(|(&id, _)| id)
                    .collect();
                for id in disconnected {
                    eng.abort_sequence(id);
                    registry.remove(&id);
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
        Ok(Ok((tokenizer, arch))) => Ok(WorkerHandle {
            request_tx,
            tokenizer,
            arch,
        }),
        Ok(Err(e)) => anyhow::bail!("worker failed to start: {e}"),
        Err(_) => anyhow::bail!("worker thread died during startup"),
    }
}
