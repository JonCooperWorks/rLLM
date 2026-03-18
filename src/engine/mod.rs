// ===========================================================================
// Inference engine: drives the continuous batching loop.
//
// LEARNING OVERVIEW
//
// What this file does:
//   The engine owns the model, scheduler, and tokenizer, and drives the
//   generation loop for MULTIPLE concurrent sequences.  Each step:
//     1. Admit new requests from the waiting queue.
//     2. Prefill: for sequences still processing their prompt, run the
//        ENTIRE prompt through the model in one batched forward pass (GEMM).
//     3. Decode: for sequences in the generation phase, run one token per
//        sequence through the single-token forward pass.
//     4. Sample next tokens for each sequence.
//     5. Advance state: mark finished sequences, collect results.
//
// Batched prefill:
//   The engine drains ALL pending prefill tokens for a sequence in one call
//   to forward_prefill_paged().  This uses GEMM (mat-mat) instead of mat-vec
//   for all projections, shifting from bandwidth-bound to compute-bound.
//
//   The flow for each prefilling sequence:
//     1. Drain all pending tokens from the queue
//     2. Pre-allocate KV blocks for the whole prompt (ensure_slots)
//     3. Upload block table to GPU (sync_block_table)
//     4. Run the GEMM forward pass (forward_prefill_paged)
//     5. Record the positions as filled (advance_by)
//     6. Sample the first generated token from the logits
//
// Continuous batching:
//   Multiple sequences can be in flight simultaneously: some prefilling,
//   some decoding, some finishing.  The scheduler manages the KV block pool
//   across all sequences, and finished sequences' blocks are returned to
//   the free list for reuse.
// ===========================================================================

pub(crate) mod multi_gpu;
pub(crate) mod scheduler;

use self::scheduler::{Scheduler, SeqId, SequenceRequest};
use crate::gpu::GpuBackend;
use crate::model::tokenizer::Tokenizer;
use crate::model::{Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// InferenceEngine trait — unified interface for single- and multi-GPU.
//
// Both the single-GPU Engine and multi-GPU MultiGpuEngine implement this
// trait.  The API worker loop operates on `&mut dyn InferenceEngine`, so
// it never needs to know how many GPUs are involved.
//
// The trait is object-safe (no generics, no Self by value) so it can be
// used as a trait object.  The lifetime and backend type parameters of
// the concrete impls are erased behind the `dyn` pointer.
// ---------------------------------------------------------------------------

/// Unified interface for inference engines.
///
/// Abstracts over single-GPU (`Engine<B>`) and multi-GPU (`MultiGpuEngine`)
/// so the API server's worker loop doesn't need to know the GPU topology.
pub(crate) trait InferenceEngine {
    /// Submit a new completion request.  Returns a sequence ID for tracking.
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId;

    /// Run one engine step: admit → prefill → decode → sample → collect.
    fn step(&mut self) -> anyhow::Result<StepOutput>;

    /// Abort a sequence, freeing its KV cache blocks.
    fn abort_sequence(&mut self, id: SeqId);

    /// Whether the engine has pending or active work.
    fn has_work(&self) -> bool;

    /// Access the tokenizer (for incremental text decoding in the worker loop).
    fn tokenizer(&self) -> &Tokenizer;
}

/// Why a sequence stopped generating.
#[derive(Clone, Copy)]
pub(crate) enum FinishReason {
    /// Hit an end-of-sequence token (EOS/EOT).
    Eos,
    /// Reached the max_tokens limit.
    MaxTokens,
}

/// A finished sequence with its generated text.
pub(crate) struct FinishedSequence {
    pub id: SeqId,
    pub tokens: Vec<u32>,
    pub text: String,
    pub reason: FinishReason,
}

/// Output from a single engine step.
pub(crate) struct StepOutput {
    /// Token generated this step for each active sequence (id, token_id).
    pub tokens: Vec<(SeqId, u32)>,
    /// Sequences that finished this step.
    pub finished: Vec<FinishedSequence>,
}

/// The inference engine.
pub(crate) struct Engine<'a, B: GpuBackend> {
    pub model: Model<'a, B>,
    pub scheduler: Scheduler<B>,
    pub tokenizer: Tokenizer,
    backend: &'a B,
    prefill_bufs: PrefillBuffers<B>,
}

impl<'a, B: GpuBackend> Engine<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        scheduler: Scheduler<B>,
        tokenizer: Tokenizer,
        backend: &'a B,
    ) -> Self {
        // Allocate prefill buffers for batched prompt processing.
        // Max chunk of 1024 supports prompts up to 1024 tokens in one pass.
        let prefill_bufs = PrefillBuffers::new(backend, model.config(), 1024);

        Self {
            model,
            scheduler,
            tokenizer,
            backend,
            prefill_bufs,
        }
    }

    /// Submit a new completion request.  Returns the sequence ID.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId {
        self.scheduler.add_request(SequenceRequest {
            prompt_tokens,
            max_gen_tokens,
            temperature,
            top_p,
        })
    }

    /// Abort a sequence, removing it from the waiting queue or active set.
    pub fn abort_sequence(&mut self, id: SeqId) {
        self.scheduler.abort_sequence(id);
    }

    /// Run one engine step.  Returns per-step tokens and newly finished sequences.
    ///
    /// Each step processes:
    ///   1. Admit waiting requests.
    ///   2. Batched prefill: process entire prompt via GEMM forward pass.
    ///   3. Decode: run single-token forward per decoding sequence.
    ///   4. Sample and advance.
    pub fn step(&mut self) -> anyhow::Result<StepOutput> {
        let mut rng = rand::rng();
        let mut step_tokens: Vec<(SeqId, u32)> = Vec::new();

        // 1. Admit waiting requests.
        self.scheduler.schedule(self.backend);

        // 2. Batched prefill: drain all pending tokens for each prefilling sequence.
        //    Each sequence's entire prompt is processed in one forward pass using
        //    GEMM (mat-mat) instead of individual mat-vec calls per token.
        //
        //    Why collect IDs first?  We need mutable access to seq.kv_state and
        //    seq.pending_prefill, but immutable access to self.scheduler.kv_pool
        //    and self.model.  Collecting IDs avoids borrow-checker conflicts.
        let prefilling_ids: Vec<SeqId> = self
            .scheduler
            .active
            .iter()
            .filter(|(_, seq)| !seq.pending_prefill.is_empty() && !seq.finished)
            .map(|(&id, _)| id)
            .collect();

        for id in prefilling_ids {
            let seq = self.scheduler.active.get_mut(&id).unwrap();

            // Drain all pending prefill tokens — the full prompt in one shot.
            let tokens: Vec<u32> = seq.pending_prefill.drain(..).collect();
            let chunk_size = tokens.len();
            let temperature = seq.temperature;
            let top_p = seq.top_p;

            // Pre-allocate all KV blocks needed for this prompt.
            // For a 100-token prompt: ceil(100/16) = 7 blocks from the pool.
            seq.kv_state
                .ensure_slots(&mut self.scheduler.kv_pool, chunk_size)?;
            seq.kv_state.sync_block_table(self.backend);

            // Run batched GEMM forward pass — the entire prompt in one call.
            // All Q/K/V projections use mat-mat, attention uses causal kernel.
            self.model.forward_prefill_paged(
                &tokens,
                &self.scheduler.kv_pool,
                &seq.kv_state,
                &self.prefill_bufs,
            )?;
            seq.kv_state.advance_by(chunk_size);

            // Sample the first generated token from the last token's logits.
            let next_token = crate::model::sampler::sample(
                self.backend,
                self.model.logits(),
                temperature,
                top_p,
                &mut rng,
            )?;
            seq.generated_tokens.push(next_token);
            step_tokens.push((id, next_token));

            if self.tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens
            {
                seq.finished = true;
            }
        }

        // 3. Decode: run one token per decoding sequence.
        let decoding_ids: Vec<SeqId> = self
            .scheduler
            .active
            .iter()
            .filter(|(_, seq)| seq.pending_prefill.is_empty() && !seq.finished)
            .map(|(&id, _)| id)
            .collect();

        for id in decoding_ids {
            let seq = self.scheduler.active.get_mut(&id).unwrap();
            let token = *seq.generated_tokens.last().unwrap();
            let temperature = seq.temperature;
            let top_p = seq.top_p;

            seq.kv_state.ensure_slot(&mut self.scheduler.kv_pool)?;
            seq.kv_state.sync_block_table(self.backend);

            self.model
                .forward_single_paged(token, &self.scheduler.kv_pool, &seq.kv_state)?;
            seq.kv_state.advance();
            crate::model::profile::tick();

            let next_token = crate::model::sampler::sample(
                self.backend,
                self.model.logits(),
                temperature,
                top_p,
                &mut rng,
            )?;
            seq.generated_tokens.push(next_token);
            step_tokens.push((id, next_token));

            if self.tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens
            {
                seq.finished = true;
            }
        }

        // 4. Collect finished sequences.
        let finished_pairs = self.scheduler.collect_finished();
        let mut finished = Vec::new();
        for (id, seq) in finished_pairs {
            let text = self.tokenizer.decode(&seq.generated_tokens).unwrap_or_default();
            let reason = if seq.generated_tokens.last().map_or(false, |&t| self.tokenizer.is_eos(t)) {
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

        Ok(StepOutput {
            tokens: step_tokens,
            finished,
        })
    }

    /// Whether the engine has any work remaining.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }
}

impl<'a, B: GpuBackend> InferenceEngine for Engine<'a, B> {
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId {
        Engine::add_request(self, prompt_tokens, max_gen_tokens, temperature, top_p)
    }

    fn step(&mut self) -> anyhow::Result<StepOutput> {
        Engine::step(self)
    }

    fn abort_sequence(&mut self, id: SeqId) {
        Engine::abort_sequence(self, id)
    }

    fn has_work(&self) -> bool {
        Engine::has_work(self)
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
