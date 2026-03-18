// ===========================================================================
// Multi-GPU inference engine — wraps MultiGpuInference in the InferenceEngine
// trait so the API worker loop can drive multi-GPU batching without knowing
// the GPU topology.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Adapts the N-rank MultiGpuInference (which fans forward passes across
//   GPUs via thread::scope + NCCL) to the same InferenceEngine interface
//   that the single-GPU Engine uses.  The API server's worker loop calls
//   add_request / step / abort_sequence / has_work — identical to the
//   single-GPU path.
//
// Why a separate wrapper (not integrated into Engine)?
//   Engine<'a, B> owns one Model<B> with one KV pool.  Multi-GPU has N
//   models, N backends, N KV pools — one per rank.  Rather than threading
//   rank arrays through the existing Engine, we build a thin adapter that
//   manages per-sequence per-rank KV states and delegates forward passes
//   to MultiGpuInference.
//
// Sequence lifecycle:
//   add_request → queued with prompt tokens
//   step():
//     1. Admit queued requests (check KV capacity on rank 0 as heuristic)
//     2. Prefill: drain prompt, allocate KV slots on all ranks, fan-out
//        GEMM forward pass, sample first token
//     3. Decode: allocate one KV slot on all ranks, fan-out single-token
//        forward pass, sample next token
//     4. Collect finished sequences (EOS or max_tokens)
//   abort_sequence → free KV blocks on all ranks
//
// Related files:
//   - engine/mod.rs       — InferenceEngine trait, single-GPU Engine
//   - gpu/multi_gpu.rs    — MultiGpuInference: fan-out, NCCL, per-rank state
//   - api/mod.rs          — worker loop (shared between single- and multi-GPU)
// ===========================================================================

#[cfg(feature = "cuda")]
pub(crate) use imp::MultiGpuEngine;

#[cfg(feature = "cuda")]
mod imp {
    use std::collections::{HashMap, VecDeque};

    use crate::engine::scheduler::SeqId;
    use crate::engine::{FinishReason, FinishedSequence, InferenceEngine, StepOutput};
    use crate::gpu::cuda::CudaBackend;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;
    use crate::model::kv_cache::SeqKvState;
    use crate::model::sampler;
    use crate::model::tokenizer::Tokenizer;

    /// A queued request waiting for admission.
    struct PendingRequest {
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    }

    /// Per-sequence state in the multi-GPU engine.
    struct MultiGpuSeq {
        /// Per-rank KV cache state (one SeqKvState per GPU).
        kv_states: Vec<SeqKvState<CudaBackend>>,
        /// Prompt tokens still needing prefill.
        pending_prefill: VecDeque<u32>,
        /// Generated tokens so far.
        generated_tokens: Vec<u32>,
        /// Maximum tokens to generate after prefill.
        max_gen_tokens: usize,
        /// Sampling temperature.
        temperature: f32,
        /// Top-p (nucleus) sampling threshold.
        top_p: f32,
        /// Whether this sequence has finished.
        finished: bool,
    }

    /// Multi-GPU inference engine implementing InferenceEngine.
    ///
    /// Manages multiple concurrent sequences across N GPU ranks, providing
    /// the same add_request/step/abort/has_work interface as the single-GPU
    /// Engine.
    pub(crate) struct MultiGpuEngine {
        multi: MultiGpuInference,
        tokenizer: Tokenizer,
        /// Waiting queue (FCFS).
        waiting: VecDeque<(SeqId, PendingRequest)>,
        /// Active sequences being processed.
        active: HashMap<SeqId, MultiGpuSeq>,
        next_id: SeqId,
        max_active: usize,
    }

    impl MultiGpuEngine {
        pub fn new(
            multi: MultiGpuInference,
            tokenizer: Tokenizer,
            max_active: usize,
        ) -> Self {
            Self {
                multi,
                tokenizer,
                waiting: VecDeque::new(),
                active: HashMap::new(),
                next_id: 0,
                max_active,
            }
        }

        /// Admit waiting requests into the active set.
        fn schedule(&mut self) -> Vec<SeqId> {
            let mut admitted = Vec::new();

            while !self.waiting.is_empty() && self.active.len() < self.max_active {
                // Use rank 0's free block count as an admission heuristic.
                // All ranks have identical block counts.
                let (_, req) = self.waiting.front().unwrap();
                let prompt_blocks =
                    crate::model::kv_cache::blocks_needed_for(req.prompt_tokens.len());
                if self.multi.ranks[0].kv_pool.free_block_count() < prompt_blocks {
                    break;
                }

                let (id, req) = self.waiting.pop_front().unwrap();
                let kv_states = self.multi.new_sequence();
                let seq = MultiGpuSeq {
                    kv_states,
                    pending_prefill: req.prompt_tokens.into(),
                    generated_tokens: Vec::new(),
                    max_gen_tokens: req.max_gen_tokens,
                    temperature: req.temperature,
                    top_p: req.top_p,
                    finished: false,
                };
                self.active.insert(id, seq);
                admitted.push(id);
            }

            admitted
        }
    }

    impl InferenceEngine for MultiGpuEngine {
        fn add_request(
            &mut self,
            prompt_tokens: Vec<u32>,
            max_gen_tokens: usize,
            temperature: f32,
            top_p: f32,
        ) -> SeqId {
            let id = self.next_id;
            self.next_id += 1;
            self.waiting.push_back((
                id,
                PendingRequest {
                    prompt_tokens,
                    max_gen_tokens,
                    temperature,
                    top_p,
                },
            ));
            id
        }

        fn step(&mut self) -> anyhow::Result<StepOutput> {
            let mut rng = rand::rng();
            let mut step_tokens: Vec<(SeqId, u32)> = Vec::new();

            // 1. Admit waiting requests.
            self.schedule();

            // 2. Prefill: drain all pending tokens for each prefilling sequence.
            let prefilling_ids: Vec<SeqId> = self
                .active
                .iter()
                .filter(|(_, seq)| !seq.pending_prefill.is_empty() && !seq.finished)
                .map(|(&id, _)| id)
                .collect();

            for id in prefilling_ids {
                let seq = self.active.get_mut(&id).unwrap();
                let tokens: Vec<u32> = seq.pending_prefill.drain(..).collect();
                let chunk_size = tokens.len();
                let temperature = seq.temperature;
                let top_p = seq.top_p;

                // Allocate KV slots on all ranks.
                self.multi
                    .ensure_slots_for(&mut seq.kv_states, chunk_size)?;

                // Run batched prefill forward pass across all ranks.
                self.multi
                    .forward_prefill_paged_with(&tokens, &seq.kv_states)?;
                MultiGpuInference::advance_by_for(&mut seq.kv_states, chunk_size);

                // Sample first token from rank 0's logits.
                let next_token = sampler::sample(
                    self.multi.backend(),
                    self.multi.logits(),
                    temperature,
                    top_p,
                    &mut rng,
                )?;
                seq.generated_tokens.push(next_token);
                step_tokens.push((id, next_token));

                if self.tokenizer.is_eos(next_token)
                    || seq.generated_tokens.len() >= seq.max_gen_tokens
                {
                    seq.finished = true;
                }
            }

            // 3. Decode: one token per active sequence.
            let decoding_ids: Vec<SeqId> = self
                .active
                .iter()
                .filter(|(_, seq)| seq.pending_prefill.is_empty() && !seq.finished)
                .map(|(&id, _)| id)
                .collect();

            for id in decoding_ids {
                let seq = self.active.get_mut(&id).unwrap();
                let token = *seq.generated_tokens.last().unwrap();
                let temperature = seq.temperature;
                let top_p = seq.top_p;

                // Allocate one KV slot on all ranks.
                self.multi.ensure_slot_for(&mut seq.kv_states)?;

                // Run single-token forward pass across all ranks.
                self.multi
                    .forward_single_paged_with(token, &seq.kv_states)?;
                MultiGpuInference::advance_for(&mut seq.kv_states);

                // Sample next token from rank 0's logits.
                let next_token = sampler::sample(
                    self.multi.backend(),
                    self.multi.logits(),
                    temperature,
                    top_p,
                    &mut rng,
                )?;
                seq.generated_tokens.push(next_token);
                step_tokens.push((id, next_token));

                if self.tokenizer.is_eos(next_token)
                    || seq.generated_tokens.len() >= seq.max_gen_tokens
                {
                    seq.finished = true;
                }
            }

            // 4. Collect finished sequences.
            let finished_ids: Vec<SeqId> = self
                .active
                .iter()
                .filter(|(_, seq)| seq.finished)
                .map(|(&id, _)| id)
                .collect();

            let mut finished = Vec::new();
            for id in finished_ids {
                if let Some(seq) = self.active.remove(&id) {
                    self.multi.free_sequence(&seq.kv_states);
                    let text = self
                        .tokenizer
                        .decode(&seq.generated_tokens)
                        .unwrap_or_default();
                    let reason = if seq
                        .generated_tokens
                        .last()
                        .map_or(false, |&t| self.tokenizer.is_eos(t))
                    {
                        FinishReason::Eos
                    } else {
                        FinishReason::MaxTokens
                    };
                    finished.push(FinishedSequence {
                        id,
                        tokens: seq.generated_tokens,
                        text,
                        reason,
                    });
                }
            }

            Ok(StepOutput {
                tokens: step_tokens,
                finished,
            })
        }

        fn abort_sequence(&mut self, id: SeqId) {
            // Remove from waiting queue.
            self.waiting.retain(|(wid, _)| *wid != id);
            // Remove from active set and free KV blocks on all ranks.
            if let Some(seq) = self.active.remove(&id) {
                self.multi.free_sequence(&seq.kv_states);
            }
        }

        fn has_work(&self) -> bool {
            !self.waiting.is_empty() || !self.active.is_empty()
        }

        fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }
    }
}
