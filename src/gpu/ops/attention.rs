// ---------------------------------------------------------------------------
// GpuAttention — attention and KV cache kernels.
//
// Covers the full attention lifecycle: writing K/V vectors into the cache
// (flat or paged), computing softmax(Q·K^T/scale)·V (single-token or
// prefill), and reading through block-table indirection for paged KV.
//
// The paged variants are the primary code path; flat cache methods exist
// for reference but are currently unused (#[allow(dead_code)]).
//
// Metal shaders: shaders/attention.metal
// Metal impl:    gpu/metal/kernels/attention.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuAttention: GpuCore {
    /// Grouped-Query Attention using flat KV cache.
    #[allow(dead_code)]
    fn attention(
        &self,
        q: &Self::Tensor,
        k_cache: &Self::Tensor,
        v_cache: &Self::Tensor,
        out: &Self::Tensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    );

    /// Write K/V vector into flat KV cache at position `pos`.
    #[allow(dead_code)]
    fn copy_to_kv_cache(
        &self,
        src: &Self::Tensor,
        cache: &Self::Tensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Write K/V vector into a paged KV cache pool.
    fn copy_to_paged_kv_cache(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Paged attention: softmax(Q·K^T/scale)·V through block table indirection.
    ///
    /// `sinks`: optional per-head attention sink logits [num_heads].  When Some,
    /// each head's sink value participates in the softmax as an extra entry that
    /// absorbs probability mass but has no associated V vector.  Models without
    /// sinks pass None.
    fn paged_attention(
        &self,
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_table: &Self::Tensor,
        out: &Self::Tensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&Self::Tensor>,
    );

    /// Batched paged KV cache write: N vectors at different positions.
    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        positions: &Self::Tensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Fused KV cache write + paged attention in a single dispatch.
    ///
    /// Writes K and V into the paged cache and computes attention in one call.
    /// The default implementation calls `copy_to_paged_kv_cache` (×2) then
    /// `paged_attention` — three separate kernel launches.  Backends can
    /// override this with a single fused kernel that:
    ///   1. Loads the block table once (not three times)
    ///   2. Keeps the current token's K/V in fast memory (threadgroup/shared)
    ///      instead of writing to global memory and immediately re-reading
    ///   3. Eliminates 2 kernel launch overheads per layer per token
    ///
    /// The fused path is a pure performance optimisation — the default gives
    /// correct results on any backend without additional implementation work.
    fn paged_attention_fused(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_table: &Self::Tensor,
        out: &Self::Tensor,
        pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&Self::Tensor>,
    ) {
        self.copy_to_paged_kv_cache(k, k_pool, block_table, pos, num_kv_heads, head_dim);
        self.copy_to_paged_kv_cache(v, v_pool, block_table, pos, num_kv_heads, head_dim);
        self.paged_attention(
            q,
            k_pool,
            v_pool,
            block_table,
            out,
            pos + 1,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size,
            attn_scale,
            sinks,
        );
    }

    /// Fused QKV prefill attention for vision encoders.
    ///
    /// Takes a single interleaved QKV buffer [chunk_size, 3 * num_heads * head_dim]
    /// instead of separate Q, K, V tensors.  Within each row, the layout is:
    ///   [Q₀..Q_{hd-1}, K₀..K_{hd-1}, V₀..V_{hd-1}]
    ///
    /// This eliminates 2 of the 3 matmul dispatches for QKV projection — one
    /// fused matmul with a [3*hd, hd] weight matrix produces the entire QKV
    /// output, and this method reads Q/K/V at the correct strides.
    ///
    /// Always bidirectional (no causal mask) — designed for vision encoders.
    fn prefill_attention_fused_qkv(
        &self,
        qkv: &Self::Tensor,
        out: &Self::Tensor,
        chunk_size: u32,
        num_heads: u32,
        head_dim: u32,
        attn_scale: f32,
    ) {
        // Default implementation: fall back to separate prefill_attention by
        // splitting the QKV buffer conceptually. Backends can override with
        // a dedicated fused kernel for better performance.
        //
        // Note: this default ONLY works if the backend can read from the same
        // buffer at different offsets. Since we pass the same tensor for Q/K/V,
        // the prefill_attention kernel will read Q from offset 0 with stride 3*hd,
        // which is WRONG for the non-fused kernel. So backends MUST override this
        // or use the separate-tensor path.
        //
        // The Metal backend provides a dedicated fused kernel.
        let _ = (qkv, out, chunk_size, num_heads, head_dim, attn_scale);
        unimplemented!("prefill_attention_fused_qkv requires a dedicated kernel implementation");
    }

    /// Prefill attention on dense Q/K/V tensors.
    ///
    /// When `causal` is true (default for LLM text), applies a causal mask so
    /// each position only attends to itself and earlier positions.  When false
    /// (vision encoder), allows bidirectional attention across all positions.
    fn prefill_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        out: &Self::Tensor,
        chunk_size: u32,
        start_pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&Self::Tensor>,
        causal: bool,
    );
}
