// ===========================================================================
// Multi-GPU inference engine — wraps MultiGpuInference behind the Dispatch
// trait so the shared run_step() loop can drive multi-GPU batching without
// knowing the GPU topology.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements Dispatch for multi-GPU inference.  MultiGpuDispatch wraps
//   MultiGpuInference (which fans forward passes across GPUs via thread::scope
//   + NCCL) and provides the same Dispatch interface as SingleGpuDispatch.
//
//   The engine step loop (run_step in engine/mod.rs) is shared between
//   single-GPU and multi-GPU — this module only provides the GPU-specific
//   dispatch operations.
//
// SeqState = Vec<SeqKvState<CudaBackend>>:
//   Multi-GPU needs one KV state per rank per sequence, because each GPU
//   has its own KV pool with its own block allocator.  The Dispatch trait's
//   associated SeqState type abstracts this — run_step() doesn't know it's
//   a Vec.
//
// Related files:
//   - engine/dispatch.rs  — Dispatch trait
//   - engine/mod.rs       — run_step(), InferenceEngine trait, Engine
//   - gpu/multi_gpu.rs    — MultiGpuInference: fan-out, NCCL, per-rank state
//   - engine/loader.rs    — load_and_run_multi_gpu() constructs MultiGpuEngine
// ===========================================================================

#[cfg(feature = "cuda")]
pub(crate) use imp::MultiGpuEngine;

#[cfg(feature = "cuda")]
mod imp {
    use crate::engine::dispatch::Dispatch;
    use crate::engine::{InferenceEngine, Scheduler, SeqId, SequenceRequest, StepOutput, run_step};
    use crate::gpu::cuda::CudaBackend;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;
    use crate::model::kv_cache::SeqKvState;
    use crate::model::tokenizer::Tokenizer;

    // -----------------------------------------------------------------------
    // MultiGpuDispatch — Dispatch implementation for multi-GPU inference.
    //
    // Each method fans out to N ranks via MultiGpuInference.  The SeqState
    // is Vec<SeqKvState<CudaBackend>> — one KV state per rank.
    // -----------------------------------------------------------------------

    /// Multi-GPU dispatch: wraps MultiGpuInference for N-rank fan-out.
    pub(crate) struct MultiGpuDispatch {
        pub multi: MultiGpuInference,
        pub tokenizer_vocab_size: usize,
    }

    impl Dispatch for MultiGpuDispatch {
        type SeqState = Vec<SeqKvState<CudaBackend>>;

        fn new_seq_state(&self) -> Self::SeqState {
            self.multi.new_sequence()
        }

        fn free_seq_state(&mut self, state: &Self::SeqState) {
            self.multi.free_sequence(state);
        }

        fn free_block_count(&self) -> usize {
            // Use rank 0's free block count as an admission heuristic.
            // All ranks have identical block counts.
            self.multi.ranks[0].kv_pool.free_block_count()
        }

        fn prepare_prefill(
            &mut self,
            state: &mut Self::SeqState,
            token_count: usize,
        ) -> anyhow::Result<()> {
            self.multi.ensure_slots_for(state, token_count)
        }

        fn forward_prefill(
            &self,
            tokens: &[u32],
            state: &Self::SeqState,
            images: &[crate::model::vision::ProcessedImage],
        ) -> anyhow::Result<()> {
            self.multi.forward_prefill_paged_with(tokens, state, images)
        }

        fn finish_prefill(state: &mut Self::SeqState, token_count: usize) {
            MultiGpuInference::advance_by_for(state, token_count);
        }

        fn seq_len(state: &Self::SeqState) -> usize {
            // All ranks have the same seq_len — use rank 0.
            state[0].seq_len
        }

        fn prepare_decode(&mut self, state: &mut Self::SeqState) -> anyhow::Result<()> {
            self.multi.ensure_slot_for(state)
        }

        fn forward_decode(&self, token: u32, state: &Self::SeqState) -> anyhow::Result<()> {
            self.multi.forward_single_paged_with(token, state)
        }

        fn finish_decode(state: &mut Self::SeqState) {
            MultiGpuInference::advance_for(state);
        }

        fn sample(
            &self,
            temperature: f32,
            top_p: f32,
            rng: &mut impl rand::Rng,
            allowed_tokens: Option<&[u32]>,
        ) -> anyhow::Result<u32> {
            crate::model::sampler::sample(
                self.multi.backend(),
                self.multi.logits(),
                temperature,
                top_p,
                rng,
                self.tokenizer_vocab_size,
                allowed_tokens,
            )
        }
    }

    // -----------------------------------------------------------------------
    // MultiGpuEngine — InferenceEngine impl using MultiGpuDispatch.
    // -----------------------------------------------------------------------

    /// Multi-GPU inference engine implementing InferenceEngine.
    ///
    /// Manages multiple concurrent sequences across N GPU ranks, providing
    /// the same add_request/step/abort/has_work interface as the single-GPU
    /// Engine.  Uses the shared run_step() loop with MultiGpuDispatch.
    pub(crate) struct MultiGpuEngine {
        dispatch: MultiGpuDispatch,
        scheduler: Scheduler<Vec<SeqKvState<CudaBackend>>>,
        tokenizer: Tokenizer,
    }

    impl MultiGpuEngine {
        pub fn new(multi: MultiGpuInference, tokenizer: Tokenizer, max_active: usize) -> Self {
            let tokenizer_vocab_size = tokenizer.vocab_size();
            Self {
                dispatch: MultiGpuDispatch { multi, tokenizer_vocab_size },
                scheduler: Scheduler::new(max_active),
                tokenizer,
            }
        }
    }

    impl InferenceEngine for MultiGpuEngine {
        fn add_request(
            &mut self,
            prompt_tokens: Vec<u32>,
            max_gen_tokens: usize,
            temperature: f32,
            top_p: f32,
            images: Vec<crate::model::vision::ProcessedImage>,
            seed: Option<u64>,
            grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
        ) -> SeqId {
            self.scheduler.add_request(SequenceRequest {
                prompt_tokens,
                max_gen_tokens,
                temperature,
                top_p,
                images,
                seed,
                grammar,
            })
        }

        fn step(&mut self) -> anyhow::Result<StepOutput> {
            run_step(&mut self.dispatch, &mut self.scheduler, &self.tokenizer)
        }

        fn abort_sequence(&mut self, id: SeqId) {
            self.scheduler.abort_sequence(id, &mut self.dispatch);
        }

        fn has_work(&self) -> bool {
            self.scheduler.has_work()
        }

        fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        fn active_count(&self) -> usize {
            self.scheduler.active_count()
        }

        fn waiting_count(&self) -> usize {
            self.scheduler.waiting_count()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Compile-time verification that MultiGpuDispatch implements Dispatch
        /// with the expected associated type.
        fn _assert_dispatch_impl() {
            fn assert_dispatch<T: Dispatch>() {}
            assert_dispatch::<MultiGpuDispatch>();
        }

        /// Compile-time verification that the SeqState associated type is
        /// Vec<SeqKvState<CudaBackend>>, matching the per-rank KV state design.
        fn _assert_seq_state_is_vec() {
            fn check<T: Dispatch<SeqState = Vec<SeqKvState<CudaBackend>>>>() {}
            check::<MultiGpuDispatch>();
        }

        /// Compile-time verification that MultiGpuEngine implements InferenceEngine.
        fn _assert_engine_impl() {
            fn assert_engine<T: InferenceEngine>() {}
            assert_engine::<MultiGpuEngine>();
        }
    }
}
