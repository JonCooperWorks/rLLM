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
    ) {
        self.copy_to_paged_kv_cache(k, k_pool, block_table, pos, num_kv_heads, head_dim);
        self.copy_to_paged_kv_cache(v, v_pool, block_table, pos, num_kv_heads, head_dim);
        self.paged_attention(
            q, k_pool, v_pool, block_table, out,
            pos + 1, num_heads, num_kv_heads, head_dim,
            window_size, attn_scale,
        );
    }

    /// Causal prefill attention on dense Q/K/V tensors.
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
    );
}
