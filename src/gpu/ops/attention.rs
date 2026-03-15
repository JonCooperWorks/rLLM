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
