// ---------------------------------------------------------------------------
// CUDA stub for GpuTurboQuant — TurboQuant KV cache quantization.
//
// Not yet implemented for CUDA.  Will panic at runtime if TurboQuant is
// enabled on a CUDA backend.  Use --kv-quant none to disable.
// ---------------------------------------------------------------------------

use crate::gpu::cuda::CudaBackend;
use crate::gpu::cuda::CudaTensor;
use crate::gpu::GpuTurboQuant;

impl GpuTurboQuant for CudaBackend {
    fn turbo_quantize_to_paged(
        &self, _src: &CudaTensor, _pool: &CudaTensor, _block_table: &CudaTensor,
        _pi: &CudaTensor, _centroids: &CudaTensor, _pos: u32, _num_kv_heads: u32,
        _head_dim: u32, _bits: u32, _bytes_per_head_pos: u32,
    ) {
        todo!("TurboQuant not yet implemented for CUDA backend")
    }

    fn turbo_quantize_to_paged_batch(
        &self, _src: &CudaTensor, _pool: &CudaTensor, _block_table: &CudaTensor,
        _positions: &CudaTensor, _pi: &CudaTensor, _centroids: &CudaTensor,
        _batch_size: u32, _num_kv_heads: u32, _head_dim: u32, _bits: u32,
        _bytes_per_head_pos: u32,
    ) {
        todo!("TurboQuant not yet implemented for CUDA backend")
    }

    fn turbo_rotate_q(
        &self, _q: &CudaTensor, _q_rot: &CudaTensor, _pi: &CudaTensor,
        _num_heads: u32, _head_dim: u32,
    ) {
        todo!("TurboQuant not yet implemented for CUDA backend")
    }

    fn turbo_paged_attention(
        &self, _q_rot: &CudaTensor, _k_pool: &CudaTensor, _v_pool: &CudaTensor,
        _block_table: &CudaTensor, _pi_t: &CudaTensor, _centroids: &CudaTensor,
        _out: &CudaTensor, _seq_len: u32, _num_heads: u32, _num_kv_heads: u32,
        _head_dim: u32, _bits: u32, _bytes_per_head_pos: u32, _window_size: u32,
        _attn_scale: f32, _sinks: Option<&CudaTensor>,
    ) {
        todo!("TurboQuant not yet implemented for CUDA backend")
    }
}
