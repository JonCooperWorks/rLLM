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

pub(crate) mod scheduler;

use self::scheduler::{Scheduler, SeqId, SequenceRequest};
use crate::gpu::GpuBackend;
use crate::model::tokenizer::Tokenizer;
use crate::model::{Model, PrefillBuffers};

/// A finished sequence with its generated text.
pub(crate) struct FinishedSequence {
    pub id: SeqId,
    pub tokens: Vec<u32>,
    pub text: String,
}

/// The inference engine.
pub(crate) struct Engine<'a, B: GpuBackend> {
    pub model: Model<'a, B>,
    pub scheduler: Scheduler<B>,
    pub tokenizer: Tokenizer,
    backend: &'a B,
    temperature: f32,
    top_p: f32,
    prefill_bufs: PrefillBuffers<B>,
}

impl<'a, B: GpuBackend> Engine<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        scheduler: Scheduler<B>,
        tokenizer: Tokenizer,
        backend: &'a B,
        temperature: f32,
        top_p: f32,
    ) -> Self {
        // Allocate prefill buffers for batched prompt processing.
        // Max chunk of 1024 supports prompts up to 1024 tokens in one pass.
        let prefill_bufs = PrefillBuffers::new(backend, model.config(), 1024);

        Self {
            model,
            scheduler,
            tokenizer,
            backend,
            temperature,
            top_p,
            prefill_bufs,
        }
    }

    /// Submit a new completion request.
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_gen_tokens: usize) {
        self.scheduler.add_request(SequenceRequest {
            prompt_tokens,
            max_gen_tokens,
        });
    }

    /// Run one engine step.  Returns newly finished sequences.
    ///
    /// Each step processes:
    ///   1. Admit waiting requests.
    ///   2. Batched prefill: process entire prompt via GEMM forward pass.
    ///   3. Decode: run single-token forward per decoding sequence.
    ///   4. Sample and advance.
    pub fn step(&mut self) -> anyhow::Result<Vec<FinishedSequence>> {
        let mut rng = rand::rng();

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
                self.temperature,
                self.top_p,
                &mut rng,
            )?;
            seq.generated_tokens.push(next_token);

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

            seq.kv_state.ensure_slot(&mut self.scheduler.kv_pool)?;
            seq.kv_state.sync_block_table(self.backend);

            self.model
                .forward_single_paged(token, &self.scheduler.kv_pool, &seq.kv_state)?;
            seq.kv_state.advance();

            let next_token = crate::model::sampler::sample(
                self.backend,
                self.model.logits(),
                self.temperature,
                self.top_p,
                &mut rng,
            )?;
            seq.generated_tokens.push(next_token);

            if self.tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens
            {
                seq.finished = true;
            }
        }

        // 4. Collect finished sequences.
        let finished_pairs = self.scheduler.collect_finished();
        let mut results = Vec::new();
        for (id, tokens) in finished_pairs {
            let text = self.tokenizer.decode(&tokens).unwrap_or_default();
            results.push(FinishedSequence { id, tokens, text });
        }

        Ok(results)
    }

    /// Whether the engine has any work remaining.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }
}
