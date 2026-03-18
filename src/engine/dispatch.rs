// ===========================================================================
// Dispatch trait — abstracts over single-GPU and multi-GPU inference backends.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the Dispatch trait that encapsulates all GPU-specific operations
//   the engine step loop needs: KV cache management, forward passes, and
//   sampling.  This eliminates duplication between the single-GPU Engine
//   and multi-GPU MultiGpuEngine — both use the same run_step() function
//   with different Dispatch implementations.
//
// Why a trait instead of duplicated loops?
//   The single-GPU and multi-GPU engines had nearly identical step() methods
//   (admit → prefill → decode → sample → collect).  The only differences
//   were HOW KV slots are allocated (one pool vs N pools across ranks) and
//   HOW forward passes are dispatched (one model vs fan-out across GPUs).
//   This trait captures exactly those differences, letting the loop logic
//   live in one place.
//
// Associated type SeqState:
//   Single-GPU uses SeqKvState<B> (one KV state per sequence).
//   Multi-GPU uses Vec<SeqKvState<CudaBackend>> (one per rank per sequence).
//   The engine step loop doesn't need to know which — it just passes the
//   opaque SeqState through to the Dispatch methods.
//
// Related files:
//   - engine/mod.rs       — run_step() uses this trait, Engine holds SingleGpuDispatch
//   - engine/multi_gpu.rs — MultiGpuDispatch wraps MultiGpuInference
//   - engine/mod.rs       — Scheduler<S> is generic over the same SeqState
// ===========================================================================

/// Abstracts over single-GPU and multi-GPU inference dispatch.
///
/// The engine step loop calls these methods without knowing the GPU topology.
/// Each implementation handles KV cache allocation, forward passes, and
/// sampling for its specific backend configuration.
pub(crate) trait Dispatch {
    /// Per-sequence KV cache state.
    ///
    /// - Single-GPU: `SeqKvState<B>` (one KV state)
    /// - Multi-GPU: `Vec<SeqKvState<CudaBackend>>` (one per rank)
    type SeqState;

    /// Create a new empty per-sequence KV state (no blocks allocated yet).
    fn new_seq_state(&self) -> Self::SeqState;

    /// Free all KV cache blocks belonging to a sequence.
    fn free_seq_state(&mut self, state: &Self::SeqState);

    /// Number of free KV cache blocks (used for admission heuristic).
    fn free_block_count(&self) -> usize;

    /// Prepare for prefill: allocate KV blocks for `token_count` tokens
    /// and sync the block table to the GPU.
    fn prepare_prefill(
        &mut self,
        state: &mut Self::SeqState,
        token_count: usize,
    ) -> anyhow::Result<()>;

    /// Run the batched (GEMM) forward pass for prefill.
    fn forward_prefill(&self, tokens: &[u32], state: &Self::SeqState) -> anyhow::Result<()>;

    /// Advance KV state after prefill (marks positions as filled).
    fn finish_prefill(state: &mut Self::SeqState, token_count: usize);

    /// Prepare for decode: allocate one KV slot and sync block table.
    fn prepare_decode(&mut self, state: &mut Self::SeqState) -> anyhow::Result<()>;

    /// Run single-token forward pass for decode.
    fn forward_decode(&self, token: u32, state: &Self::SeqState) -> anyhow::Result<()>;

    /// Advance KV state by one position after decode.
    fn finish_decode(state: &mut Self::SeqState);

    /// Sample a token from the current logits.
    fn sample(&self, temperature: f32, top_p: f32, rng: &mut impl rand::Rng)
    -> anyhow::Result<u32>;
}
