// ===========================================================================
// Multi-GPU inference worker for the API server (--tp > 1).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Provides a tensor-parallel inference worker that serves concurrent HTTP
//   requests across multiple GPUs.  It mirrors the single-GPU Engine step
//   loop (see engine/mod.rs) but drives MultiGpuInference directly — each
//   sequence gets its own per-rank KV state set, and forward passes fan out
//   to all GPUs via thread::scope + NCCL.
//
// Why a separate module (not integrated into Engine)?
//   The Engine uses a single Model<B> with one backend.  Multi-GPU inference
//   has N models, N backends, and N KV pools — one per rank.  Fanning out at
//   the forward-pass level (1 thread::scope per token) is efficient; fanning
//   out at the kernel level (1 thread::scope per matmul/attention/norm) would
//   add ~1600 thread dispatches per token for a 70B model.  So we keep the
//   existing N-models-N-backends pattern and build the batching loop here.
//
// Architecture:
//
//   HTTP handlers → [request channel] → multi-GPU worker thread
//                                             ↓
//                                  MultiGpuInference (N GPUs)
//                                     ├── rank 0: model + KV pool
//                                     ├── rank 1: model + KV pool
//                                     └── ... (all connected via NCCL)
//
// Continuous batching:
//   Each step: drain new requests → prefill prompts → decode one token per
//   active sequence → stream tokens to clients → clean up finished sequences.
//   Sequences are independent — each has its own per-rank KV state allocated
//   from the shared block pools.
//
// Related files:
//   - gpu/multi_gpu.rs — MultiGpuInference: fan-out, NCCL, per-rank state
//   - api/mod.rs       — spawn_worker (single-GPU path), serve(), shared types
//   - engine/mod.rs    — Engine: single-GPU continuous batching (same pattern)
// ===========================================================================

use std::path::PathBuf;

use super::WorkerHandle;

// ---------------------------------------------------------------------------
// Per-sequence state.
// ---------------------------------------------------------------------------

/// Tracks a single active sequence in the multi-GPU batching loop.
///
/// Each sequence has N KV states (one per GPU rank) plus the HTTP response
/// channel and generation state.  This is the multi-GPU equivalent of the
/// `RequestContext` + `Sequence` used in the single-GPU Engine path.
#[cfg(feature = "cuda")]
struct MultiGpuSeq {
    /// Per-rank KV cache state (one SeqKvState per GPU).
    kv_states: Vec<crate::model::kv_cache::SeqKvState<crate::gpu::cuda::CudaBackend>>,
    /// Prompt tokens still needing prefill.
    pending_prefill: VecDeque<u32>,
    /// Last generated token (used as input for next decode step).
    last_token: u32,
    /// All generated token IDs (for incremental text decode).
    token_ids: Vec<u32>,
    /// Characters already sent to client.
    prev_text_len: usize,
    /// Total generated token count.
    generated: usize,
    /// Max tokens to generate.
    max_tokens: usize,
    /// Sampling parameters.
    temperature: f32,
    top_p: f32,
    /// Channel back to the HTTP handler.
    response_tx: tokio::sync::mpsc::Sender<super::InferenceEvent>,
    /// Prompt token count (for Done event).
    prompt_token_count: usize,
}

// ---------------------------------------------------------------------------
// Worker entry point.
// ---------------------------------------------------------------------------

/// Spawn the multi-GPU inference worker thread.
///
/// Returns the request channel, shared tokenizer, and model architecture —
/// the same WorkerHandle as the single-GPU path, so the HTTP handlers don't
/// know or care which backend is running.
#[cfg(feature = "cuda")]
pub(crate) fn spawn_worker(
    model_dir: PathBuf,
    quantize: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    use std::collections::HashMap;
    use std::sync::Arc;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;
    use crate::model::{config, sampler, tokenizer};
    use super::{InferenceEvent, StopReason, WorkerRequest};

    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>>(1);

    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<()> {
            eprintln!("tensor parallelism: {} GPUs", tp);

            let config = config::ModelConfig::from_file(&model_dir.join("config.json"))?;
            let arch = config.arch()?;
            let tokenizer = tokenizer::Tokenizer::from_file(
                &model_dir.join("tokenizer.json"),
                arch,
            )?;
            let tokenizer = Arc::new(tokenizer);

            let num_blocks = 256;
            let mut multi = MultiGpuInference::new(
                &model_dir, config.clone(), quantize, tp, num_blocks,
            )?;

            let max_active: usize = 32;
            eprintln!(
                "multi-GPU inference ready ({} ranks, max {} concurrent sequences)",
                tp, max_active,
            );

            let _ = ready_tx.send(Ok((Arc::clone(&tokenizer), arch)));

            // ---------------------------------------------------------------
            // Continuous batching loop.
            //
            // Each iteration:
            //   1. Drain new requests from the channel (non-blocking)
            //   2. If no work, block until a request arrives
            //   3. Prefill: process all pending prompts
            //   4. Decode: generate one token per active sequence
            //   5. Clean up finished and disconnected sequences
            // ---------------------------------------------------------------
            let mut rng = rand::rng();
            let mut next_id: u64 = 0;
            let mut active: HashMap<u64, MultiGpuSeq> = HashMap::new();

            loop {
                // 1. Drain new requests (non-blocking).
                while let Ok(req) = request_rx.try_recv() {
                    if active.len() >= max_active {
                        let _ = req.response_tx.blocking_send(
                            InferenceEvent::Error("server at capacity".into()),
                        );
                        continue;
                    }
                    let id = next_id;
                    next_id += 1;
                    active.insert(id, new_seq(&multi, req));
                }

                // 2. If no work, block until a request arrives.
                if active.is_empty() {
                    match request_rx.recv() {
                        Ok(req) => {
                            let id = next_id;
                            next_id += 1;
                            active.insert(id, new_seq(&multi, req));
                        }
                        Err(_) => break, // Channel closed — server shutting down.
                    }
                }

                // 3. Prefill phase: process all sequences with pending prompt tokens.
                let prefill_ids: Vec<u64> = active.iter()
                    .filter(|(_, seq)| !seq.pending_prefill.is_empty())
                    .map(|(&id, _)| id)
                    .collect();

                for id in &prefill_ids {
                    let seq = active.get_mut(id).unwrap();
                    let tokens: Vec<u32> = seq.pending_prefill.drain(..).collect();
                    let prompt_len = tokens.len();
                    seq.prompt_token_count = prompt_len;

                    // Allocate KV slots + run prefill forward pass.
                    if let Err(e) = multi.ensure_slots_for(&mut seq.kv_states, prompt_len) {
                        let _ = seq.response_tx.blocking_send(
                            InferenceEvent::Error(format!("{e:#}")),
                        );
                        remove_seq(&mut multi, &mut active, *id);
                        continue;
                    }
                    if let Err(e) = multi.forward_prefill_paged_with(&tokens, &seq.kv_states) {
                        let _ = seq.response_tx.blocking_send(
                            InferenceEvent::Error(format!("{e:#}")),
                        );
                        remove_seq(&mut multi, &mut active, *id);
                        continue;
                    }
                    MultiGpuInference::advance_by_for(&mut seq.kv_states, prompt_len);

                    // Sample first token from prefill logits.
                    match sampler::sample(
                        multi.backend(), multi.logits(),
                        seq.temperature, seq.top_p, &mut rng,
                    ) {
                        Ok(t) => seq.last_token = t,
                        Err(e) => {
                            let _ = seq.response_tx.blocking_send(
                                InferenceEvent::Error(format!("{e:#}")),
                            );
                            remove_seq(&mut multi, &mut active, *id);
                        }
                    }
                }

                // 4. Decode phase: one token per active sequence.
                let decode_ids: Vec<u64> = active.iter()
                    .filter(|(_, seq)| seq.pending_prefill.is_empty())
                    .map(|(&id, _)| id)
                    .collect();

                let mut to_remove: Vec<u64> = Vec::new();

                for id in &decode_ids {
                    let seq = active.get_mut(id).unwrap();
                    let token = seq.last_token;

                    // Check EOS.
                    if tokenizer.is_eos(token) {
                        let _ = seq.response_tx.blocking_send(InferenceEvent::Done {
                            stop_reason: StopReason::EndOfSequence,
                            prompt_tokens: seq.prompt_token_count,
                            completion_tokens: seq.generated,
                        });
                        to_remove.push(*id);
                        continue;
                    }

                    // Check max tokens.
                    if seq.generated >= seq.max_tokens {
                        let _ = seq.response_tx.blocking_send(InferenceEvent::Done {
                            stop_reason: StopReason::MaxTokens,
                            prompt_tokens: seq.prompt_token_count,
                            completion_tokens: seq.generated,
                        });
                        to_remove.push(*id);
                        continue;
                    }

                    seq.generated += 1;
                    seq.token_ids.push(token);

                    // Incremental decode + stream token to client.
                    let full_text = tokenizer.decode(&seq.token_ids).unwrap_or_default();
                    let text = full_text[seq.prev_text_len..].to_string();
                    seq.prev_text_len = full_text.len();

                    if seq.response_tx.blocking_send(InferenceEvent::Token { text }).is_err() {
                        // Client disconnected.
                        to_remove.push(*id);
                        continue;
                    }

                    // Forward next token through the model.
                    if multi.ensure_slot_for(&mut seq.kv_states).is_err()
                        || multi.forward_single_paged_with(token, &seq.kv_states).is_err()
                    {
                        let _ = seq.response_tx.blocking_send(
                            InferenceEvent::Error("forward pass error".into()),
                        );
                        to_remove.push(*id);
                        continue;
                    }
                    MultiGpuInference::advance_for(&mut seq.kv_states);

                    // Sample next token.
                    match sampler::sample(
                        multi.backend(), multi.logits(),
                        seq.temperature, seq.top_p, &mut rng,
                    ) {
                        Ok(t) => seq.last_token = t,
                        Err(_) => {
                            let _ = seq.response_tx.blocking_send(
                                InferenceEvent::Error("sampling error".into()),
                            );
                            to_remove.push(*id);
                        }
                    }
                }

                // 5. Clean up finished/disconnected sequences.
                for (&id, seq) in &active {
                    if seq.response_tx.is_closed() && !to_remove.contains(&id) {
                        to_remove.push(id);
                    }
                }
                for id in to_remove {
                    remove_seq(&mut multi, &mut active, id);
                }
            }

            Ok(())
        })();

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

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Create a new MultiGpuSeq from a worker request.
#[cfg(feature = "cuda")]
fn new_seq(
    multi: &crate::gpu::multi_gpu::tp::MultiGpuInference,
    req: super::WorkerRequest,
) -> MultiGpuSeq {
    MultiGpuSeq {
        kv_states: multi.new_sequence(),
        pending_prefill: req.prompt_tokens.into(),
        last_token: 0,
        token_ids: Vec::new(),
        prev_text_len: 0,
        generated: 0,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        response_tx: req.response_tx,
        prompt_token_count: 0,
    }
}

/// Remove a sequence and free its KV blocks across all ranks.
#[cfg(feature = "cuda")]
fn remove_seq(
    multi: &mut crate::gpu::multi_gpu::tp::MultiGpuInference,
    active: &mut HashMap<u64, MultiGpuSeq>,
    id: u64,
) {
    if let Some(seq) = active.remove(&id) {
        if !seq.kv_states.is_empty() {
            multi.free_sequence(&seq.kv_states);
        }
    }
}

/// Non-CUDA stub — multi-GPU requires CUDA + NCCL.
#[cfg(not(feature = "cuda"))]
pub(crate) fn spawn_worker(
    _model_dir: PathBuf,
    _quantize: bool,
    _tp: usize,
) -> anyhow::Result<WorkerHandle> {
    anyhow::bail!("multi-GPU tensor parallelism requires the cuda feature")
}
