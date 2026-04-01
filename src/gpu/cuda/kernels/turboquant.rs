// ---------------------------------------------------------------------------
// CUDA impl: GpuTurboQuant — TurboQuant KV cache quantization kernels.
//
// Trait contract: gpu/ops/turboquant.rs
// CUDA shader:    cuda/shaders/turboquant.cu
//
// Five kernels:
//   turbo_quantize_paged        — rotate + quantize one K/V vector
//   turbo_quantize_paged_batch  — batched version for prefill
//   turbo_rotate_q              — pre-rotate query
//   turbo_paged_attention       — paged attention with inline dequant
//   turbo_paged_attention_v_only — asymmetric: BF16 K + TurboQuant V
//
// Related files:
//   Metal shader:  metal/shaders/turboquant.metal
//   Metal bridge:  metal/kernels/turboquant.rs
//   Algorithm:     model/turboquant.rs
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use crate::gpu::cuda::CudaBackend;
use crate::gpu::cuda::CudaTensor;
use crate::gpu::GpuTurboQuant;

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboQuantizeParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
}
unsafe impl DeviceRepr for TurboQuantizeParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboQuantizeBatchParams {
    batch_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
}
unsafe impl DeviceRepr for TurboQuantizeBatchParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboRotateQParams {
    num_heads: u32,
    head_dim: u32,
}
unsafe impl DeviceRepr for TurboRotateQParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboPagedAttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    window_size: u32,
    attn_scale: f32,
    has_sinks: u32,
}
unsafe impl DeviceRepr for TurboPagedAttentionParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboPagedAttentionVOnlyParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    kv_dim: u32,
    v_bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    window_size: u32,
    attn_scale: f32,
    has_sinks: u32,
}
unsafe impl DeviceRepr for TurboPagedAttentionVOnlyParams {}

impl GpuTurboQuant for CudaBackend {
    fn turbo_quantize_to_paged(
        &self,
        src: &CudaTensor,
        pool: &CudaTensor,
        block_table: &CudaTensor,
        pi: &CudaTensor,
        centroids: &CudaTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
    ) {
        let params = TurboQuantizeParams {
            pos,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            num_centroids: centroids.shape[0] as u32,
        };
        let cfg = CudaBackend::cfg_blocks(num_kv_heads, head_dim);
        unsafe {
            self.stream
                .launch_builder(&self.fn_turbo_quantize_paged)
                .arg(&params)
                .arg(&src.buf)
                .arg(&pool.buf)
                .arg(&block_table.buf)
                .arg(&pi.buf)
                .arg(&centroids.buf)
                .launch(cfg)
        }
        .expect("turbo_quantize_paged launch failed");
    }

    fn turbo_quantize_to_paged_batch(
        &self,
        src: &CudaTensor,
        pool: &CudaTensor,
        block_table: &CudaTensor,
        positions: &CudaTensor,
        pi: &CudaTensor,
        centroids: &CudaTensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
    ) {
        let params = TurboQuantizeBatchParams {
            batch_size,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            num_centroids: centroids.shape[0] as u32,
        };
        let num_blocks = batch_size * num_kv_heads;
        let cfg = CudaBackend::cfg_blocks(num_blocks, head_dim);
        unsafe {
            self.stream
                .launch_builder(&self.fn_turbo_quantize_paged_batch)
                .arg(&params)
                .arg(&src.buf)
                .arg(&pool.buf)
                .arg(&block_table.buf)
                .arg(&positions.buf)
                .arg(&pi.buf)
                .arg(&centroids.buf)
                .launch(cfg)
        }
        .expect("turbo_quantize_paged_batch launch failed");
    }

    fn turbo_rotate_q(
        &self,
        q: &CudaTensor,
        q_rot: &CudaTensor,
        pi: &CudaTensor,
        num_heads: u32,
        head_dim: u32,
    ) {
        let params = TurboRotateQParams {
            num_heads,
            head_dim,
        };
        let cfg = CudaBackend::cfg_blocks(num_heads, head_dim);
        unsafe {
            self.stream
                .launch_builder(&self.fn_turbo_rotate_q)
                .arg(&params)
                .arg(&q.buf)
                .arg(&q_rot.buf)
                .arg(&pi.buf)
                .launch(cfg)
        }
        .expect("turbo_rotate_q launch failed");
    }

    fn turbo_paged_attention(
        &self,
        q_rot: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_table: &CudaTensor,
        pi_t: &CudaTensor,
        centroids: &CudaTensor,
        out: &CudaTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&CudaTensor>,
    ) {
        let params = TurboPagedAttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            num_centroids: centroids.shape[0] as u32,
            window_size,
            attn_scale,
            has_sinks: if sinks.is_some() { 1 } else { 0 },
        };
        let sinks_buf = sinks.map(|s| &s.buf).unwrap_or(&out.buf);
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_turbo_paged_attention)
                .arg(&params)
                .arg(&q_rot.buf)
                .arg(&k_pool.buf)
                .arg(&v_pool.buf)
                .arg(&block_table.buf)
                .arg(&pi_t.buf)
                .arg(&centroids.buf)
                .arg(&out.buf)
                .arg(sinks_buf)
                .launch(cfg)
        }
        .expect("turbo_paged_attention launch failed");
    }

    fn turbo_paged_attention_v_only(
        &self,
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_table: &CudaTensor,
        pi_t: &CudaTensor,
        centroids: &CudaTensor,
        out: &CudaTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        kv_dim: u32,
        v_bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&CudaTensor>,
    ) {
        let params = TurboPagedAttentionVOnlyParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            bits,
            kv_dim,
            v_bytes_per_head_pos,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
            num_centroids: 1 << bits,
            window_size,
            attn_scale,
            has_sinks: if sinks.is_some() { 1 } else { 0 },
        };
        let sinks_buf = sinks.map(|s| &s.buf).unwrap_or(&out.buf);
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_turbo_paged_attention_v_only)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k_pool.buf)
                .arg(&v_pool.buf)
                .arg(&block_table.buf)
                .arg(&pi_t.buf)
                .arg(&centroids.buf)
                .arg(&out.buf)
                .arg(sinks_buf)
                .launch(cfg)
        }
        .expect("turbo_paged_attention_v_only launch failed");
    }
}
