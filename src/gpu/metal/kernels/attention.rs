// ---------------------------------------------------------------------------
// Metal impl: GpuAttention — attention and KV cache kernels.
//
// Trait contract: gpu/ops/attention.rs
// Metal shader:   metal/shaders/attention.metal
//
// Paged attention is the primary path: KV vectors are stored in fixed-size
// blocks (BLOCK_SIZE from model::kv_cache), and a block table maps logical
// positions to physical block slots.  This avoids copying the entire cache
// when it grows.
//
// Flat-cache variants (attention, copy_to_kv_cache) exist for reference
// but are currently unused — marked #[allow(dead_code)] on the trait side.
//
// Attention dispatch: one 256-thread threadgroup per query head.  Each
// threadgroup computes softmax(Q·K^T/scale)·V across the full sequence
// (or sliding window).  Prefill uses one threadgroup per (token, head) pair.
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuAttention;

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct AttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct CopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedCopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedAttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    window_size: u32,
    attn_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedCopyKvBatchParams {
    batch_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PrefillAttentionParams {
    chunk_size: u32,
    start_pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
}

impl GpuAttention for MetalBackend {
    fn attention(
        &self,
        q: &MetalTensor,
        k_cache: &MetalTensor,
        v_cache: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    ) {
        let params = AttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size,
            attn_scale,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k_cache.buffer, 2),
                (&v_cache.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn copy_to_kv_cache(
        &self,
        src: &MetalTensor,
        cache: &MetalTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = CopyKvParams {
            pos,
            num_kv_heads,
            head_dim,
        };
        let size = num_kv_heads * head_dim;
        self.dispatch_async(
            &self.pipeline_copy_kv,
            &params,
            &[(&src.buffer, 1), (&cache.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn copy_to_paged_kv_cache(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PagedCopyKvParams {
            pos,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
        };
        let size = num_kv_heads * head_dim;
        self.dispatch_async(
            &self.pipeline_paged_copy_kv,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
            ],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn paged_attention(
        &self,
        q: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_table: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    ) {
        let params = PagedAttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            window_size,
            attn_scale,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_paged_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k_pool.buffer, 2),
                (&v_pool.buffer, 3),
                (&block_table.buffer, 4),
                (&out.buffer, 5),
            ],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        positions: &MetalTensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PagedCopyKvBatchParams {
            batch_size,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
        };
        let kv_dim = num_kv_heads * head_dim;
        let total = batch_size as u64 * kv_dim as u64;
        self.dispatch_async(
            &self.pipeline_paged_copy_kv_batch,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
                (&positions.buffer, 4),
            ],
            MTLSize::new(total, 1, 1),
            MTLSize::new(256.min(total), 1, 1),
        );
    }

    fn prefill_attention(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        out: &MetalTensor,
        chunk_size: u32,
        start_pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    ) {
        let params = PrefillAttentionParams {
            chunk_size,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size,
            attn_scale,
        };
        let threads_per_group: u64 = 256;
        let num_threadgroups = chunk_size as u64 * num_heads as u64;
        self.dispatch_async(
            &self.pipeline_prefill_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k.buffer, 2),
                (&v.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }
}
