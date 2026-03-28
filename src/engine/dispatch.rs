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

    /// Maximum number of tokens that can be prefilled in a single forward pass.
    ///
    /// Prompts longer than this are chunked into multiple prefill passes.
    /// Returns usize::MAX by default (no chunking).
    fn max_prefill_chunk(&self) -> usize {
        usize::MAX
    }

    /// Prepare for prefill: allocate KV blocks for `token_count` tokens
    /// and sync the block table to the GPU.
    fn prepare_prefill(
        &mut self,
        state: &mut Self::SeqState,
        token_count: usize,
    ) -> anyhow::Result<()>;

    /// Run the batched (GEMM) forward pass for prefill.
    /// `images` contains preprocessed vision data (empty for text-only).
    fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &Self::SeqState,
        images: &[crate::model::vision::ProcessedImage],
    ) -> anyhow::Result<()>;

    /// Advance KV state after prefill (marks positions as filled).
    fn finish_prefill(state: &mut Self::SeqState, token_count: usize);

    /// Current sequence length (number of tokens in KV cache).
    ///
    /// Needed by the batched decode loop to collect per-sequence positions.
    fn seq_len(state: &Self::SeqState) -> usize;

    /// Prepare for decode: allocate one KV slot and sync block table.
    fn prepare_decode(&mut self, state: &mut Self::SeqState) -> anyhow::Result<()>;

    /// Run single-token forward pass for decode.
    fn forward_decode(&self, token: u32, state: &Self::SeqState) -> anyhow::Result<()>;

    /// Advance KV state by one position after decode.
    fn finish_decode(state: &mut Self::SeqState);

    /// Sample a token from the current logits.
    fn sample(&self, temperature: f32, top_p: f32, rng: &mut impl rand::Rng)
    -> anyhow::Result<u32>;

    /// GPU-resident greedy sampling: argmax entirely on device.
    ///
    /// Avoids copying the full logits tensor to CPU — only 4 bytes (one u32
    /// token ID) are transferred.  Used when temperature == 0.0.
    ///
    /// Inspired by rvLLM (Andy Norris / m0at).
    fn sample_greedy_gpu(&self) -> anyhow::Result<u32> {
        anyhow::bail!("GPU greedy sampling not supported by this Dispatch implementation")
    }

    /// GPU-resident batched greedy sampling: argmax for N sequences at once.
    fn sample_batch_greedy_gpu(&self, _batch_size: usize) -> anyhow::Result<Vec<u32>> {
        anyhow::bail!("GPU batched greedy sampling not supported by this Dispatch implementation")
    }

    // -----------------------------------------------------------------------
    // Prefix caching — reusing KV blocks across requests with shared prefixes.
    //
    // Default implementations are no-ops (caching disabled).  SingleGpuDispatch
    // provides the real implementation.
    // -----------------------------------------------------------------------

    /// Look up a cached prefix for the given prompt tokens.
    ///
    /// Returns `(block_handles, token_count)` for the longest matching prefix,
    /// or None if no match found.  The handles carry generation counters for
    /// stale-reference detection.
    fn prefix_cache_lookup(
        &mut self,
        _prompt_tokens: &[u32],
    ) -> Option<(Vec<crate::model::kv_cache::BlockHandle>, usize)> {
        None
    }

    /// Register a new prefix after prefill completes.
    ///
    /// `tokens` are the full prompt tokens; the implementation extracts the
    /// block-aligned prefix and records the block table entries.
    /// `state` is mutable so the implementation can mark the prefix blocks
    /// as shared (preventing them from being freed when this sequence finishes).
    fn prefix_cache_register(&mut self, _tokens: &[u32], _state: &mut Self::SeqState) {}

    /// Link a sequence's KV state to a cached prefix.
    ///
    /// Copies the prefix's block handles into the sequence's block table
    /// and advances seq_len past the already-computed positions.  Handles
    /// carry generation counters for stale-reference detection.
    fn link_prefix(
        &self,
        _state: &mut Self::SeqState,
        _prefix_handles: &[crate::model::kv_cache::BlockHandle],
        _prefix_token_count: usize,
        _prefix_tokens: Vec<u32>,
    ) {
    }


    // -----------------------------------------------------------------------
    // Batched decode — processing N decoding sequences in one forward pass.
    //
    // These methods turn N separate mat-vec decode passes into one GEMM pass.
    // The default implementations fall back to serial per-sequence decode
    // (no batching), so existing Dispatch impls work without changes.
    // -----------------------------------------------------------------------

    /// Whether this dispatch supports batched decode.
    ///
    /// Returns false by default.  Implementations return true when the
    /// model architecture supports batched decode AND the necessary
    /// buffers (logits_batch) are allocated.
    fn supports_batched_decode(&self) -> bool {
        false
    }

    /// Run a batched forward pass for N decoding sequences.
    ///
    /// `tokens[i]` is the last generated token for sequence i.
    /// `positions[i]` is the KV cache position for sequence i.
    /// `states[i]` is the KV state for sequence i (block table already synced).
    ///
    /// Produces [N, vocab_size] logits in an internal buffer, ready for
    /// `sample_batch()`.
    fn forward_decode_batch(
        &self,
        _tokens: &[u32],
        _positions: &[u32],
        _states: &[&Self::SeqState],
    ) -> anyhow::Result<()> {
        anyhow::bail!("batched decode not supported by this Dispatch implementation")
    }

    /// Sample N tokens from the batched logits produced by `forward_decode_batch`.
    ///
    /// Each sequence gets its own temperature and top_p — different concurrent
    /// requests can have different sampling parameters.
    fn sample_batch(
        &self,
        _temperatures: &[f32],
        _top_ps: &[f32],
        _rng: &mut impl rand::Rng,
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::bail!("batched sampling not supported by this Dispatch implementation")
    }
}
