// ---------------------------------------------------------------------------
// CUDA impl: GpuAttention — attention and KV cache kernels.
//
// Trait contract: gpu/ops/attention.rs
// CUDA shader:    cuda/shaders/attention.cu
//
// Paged attention is the primary path: KV vectors are stored in fixed-size
// blocks (BLOCK_SIZE from model::kv_cache), and a block table maps logical
// positions to physical block slots.
//
// Attention dispatch: one 256-thread block per query head.  Each block
// computes softmax(Q·K^T/scale)·V across the full sequence (or sliding
// window).  Prefill uses one block per (token, head) pair.
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
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
unsafe impl DeviceRepr for AttentionParams {}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct CopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
}
unsafe impl DeviceRepr for CopyKvParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedCopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}
unsafe impl DeviceRepr for PagedCopyKvParams {}

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
unsafe impl DeviceRepr for PagedAttentionParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedAttentionFusedParams {
    pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    window_size: u32,
    attn_scale: f32,
}
unsafe impl DeviceRepr for PagedAttentionFusedParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedCopyKvBatchParams {
    batch_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}
unsafe impl DeviceRepr for PagedCopyKvBatchParams {}

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
unsafe impl DeviceRepr for PrefillAttentionParams {}

impl GpuAttention for CudaBackend {
    fn attention(
        &self,
        q: &CudaTensor,
        k_cache: &CudaTensor,
        v_cache: &CudaTensor,
        out: &CudaTensor,
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
        let func = if head_dim > 128 { &self.fn_attention_hd256 } else { &self.fn_attention };
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream.launch_builder(func)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k_cache.buf)
                .arg(&v_cache.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("attention launch failed");
    }

    fn copy_to_kv_cache(
        &self,
        src: &CudaTensor,
        cache: &CudaTensor,
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
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream.launch_builder(&self.fn_copy_kv)
                .arg(&params)
                .arg(&src.buf)
                .arg(&cache.buf)
                .launch(cfg)
        }.expect("copy_to_kv_cache launch failed");
    }

    fn copy_to_paged_kv_cache(
        &self,
        src: &CudaTensor,
        pool: &CudaTensor,
        block_table: &CudaTensor,
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
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream.launch_builder(&self.fn_paged_copy_kv)
                .arg(&params)
                .arg(&src.buf)
                .arg(&pool.buf)
                .arg(&block_table.buf)
                .launch(cfg)
        }.expect("copy_to_paged_kv_cache launch failed");
    }

    fn paged_attention(
        &self,
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_table: &CudaTensor,
        out: &CudaTensor,
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
        let func = if head_dim > 128 { &self.fn_paged_attention_hd256 } else { &self.fn_paged_attention };
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream.launch_builder(func)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k_pool.buf)
                .arg(&v_pool.buf)
                .arg(&block_table.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("paged_attention launch failed");
    }

    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &CudaTensor,
        pool: &CudaTensor,
        block_table: &CudaTensor,
        positions: &CudaTensor,
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
        let total = batch_size * kv_dim;
        let block = 256.min(total);
        let cfg = CudaBackend::cfg_1d(total, block);
        unsafe {
            self.stream.launch_builder(&self.fn_paged_copy_kv_batch)
                .arg(&params)
                .arg(&src.buf)
                .arg(&pool.buf)
                .arg(&block_table.buf)
                .arg(&positions.buf)
                .launch(cfg)
        }.expect("copy_to_paged_kv_cache_batch launch failed");
    }

    fn paged_attention_fused(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_table: &CudaTensor,
        out: &CudaTensor,
        pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    ) {
        let params = PagedAttentionFusedParams {
            pos,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            window_size,
            attn_scale,
        };
        let func = if head_dim > 128 {
            &self.fn_paged_attention_fused_hd256
        } else {
            &self.fn_paged_attention_fused
        };
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream.launch_builder(func)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&v.buf)
                .arg(&k_pool.buf)
                .arg(&v_pool.buf)
                .arg(&block_table.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("paged_attention_fused launch failed");
    }

    fn prefill_attention(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        out: &CudaTensor,
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
        let num_blocks = chunk_size * num_heads;
        let func = if head_dim > 128 {
            &self.fn_prefill_attention_hd256
        } else {
            &self.fn_prefill_attention
        };
        let cfg = CudaBackend::cfg_blocks(num_blocks, 256);
        unsafe {
            self.stream.launch_builder(func)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&v.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("prefill_attention launch failed");
    }
}
