// CUDA backend — Phase 4 implementation
//
// This module will implement GpuBackend for NVIDIA GPUs via cudarc.
// For now it's a placeholder to keep the module structure compilable.

use super::ops::*;
use super::TensorDtype;

pub(crate) struct CudaBackend;

impl CudaBackend {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!("CUDA backend not yet implemented")
    }
}

pub(crate) struct CudaTensor;

impl GpuCore for CudaBackend {
    type Tensor = CudaTensor;
    fn device_name(&self) -> &str { "" }
    fn recommended_max_memory(&self) -> u64 { unreachable!() }
    fn flush(&self) { unreachable!() }
    fn submit(&self) { unreachable!() }
    fn alloc_tensor(&self, _: &[usize], _: TensorDtype) -> CudaTensor { unreachable!() }
    fn upload_tensor(&self, _: &[u8], _: &[usize], _: TensorDtype) -> CudaTensor { unreachable!() }
    fn copy_to_host(&self, _: &CudaTensor, _: &mut [u8]) { unreachable!() }
    fn copy_to_tensor(&self, _: &CudaTensor, _: &[u8]) { unreachable!() }
    fn tensor_byte_count(&self, _: &CudaTensor) -> usize { unreachable!() }
}

impl GpuNorm for CudaBackend {
    fn rms_norm(&self, _: &CudaTensor, _: &CudaTensor, _: f32, _: &CudaTensor) { unreachable!() }
    fn rms_norm_batch(&self, _: &CudaTensor, _: &CudaTensor, _: f32, _: &CudaTensor, _: u32) { unreachable!() }
}

impl GpuMatmul for CudaBackend {
    fn matmul(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
    fn matmul_batch(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32) { unreachable!() }
}

impl GpuRope for CudaBackend {
    fn rope(&self, _: &CudaTensor, _: &CudaTensor, _: u32, _: f32, _: u32, _: u32, _: u32) { unreachable!() }
    fn rope_batch(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: f32, _: u32, _: u32, _: u32, _: u32) { unreachable!() }
    fn rope_partial(&self, _: &CudaTensor, _: &CudaTensor, _: u32, _: f32, _: u32, _: u32, _: u32, _: u32) { unreachable!() }
}

impl GpuAttention for CudaBackend {
    fn attention(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32, _: u32, _: u32, _: f32) { unreachable!() }
    fn copy_to_kv_cache(&self, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32) { unreachable!() }
    fn copy_to_paged_kv_cache(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32) { unreachable!() }
    fn paged_attention(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32, _: u32, _: u32, _: f32) { unreachable!() }
    fn copy_to_paged_kv_cache_batch(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32) { unreachable!() }
    fn prefill_attention(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: f32) { unreachable!() }
}

impl GpuElementwise for CudaBackend {
    fn silu_mul(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn gelu_mul(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn scalar_mul(&self, _: &CudaTensor, _: &CudaTensor, _: f32, _: u32) { unreachable!() }
    fn add(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn scale_add(&self, _: &CudaTensor, _: &CudaTensor, _: f32, _: u32) { unreachable!() }
    fn fill_zero(&self, _: &CudaTensor, _: u32) { unreachable!() }
    fn bias_add_batch(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
    fn sigmoid(&self, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn sigmoid_bf16(&self, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn silu(&self, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn mul(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn top_k_softmax(&self, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
}

impl GpuEmbed for CudaBackend {
    fn embed_lookup(&self, _: &CudaTensor, _: u32, _: &CudaTensor, _: u32) { unreachable!() }
    fn embed_lookup_batch(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
}

impl GpuDeltaNet for CudaBackend {
    fn conv1d_depthwise_single(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
    fn conv1d_shift_history(&self, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32) { unreachable!() }
    fn l2_normalize_heads(&self, _: &CudaTensor, _: u32, _: u32, _: u32) { unreachable!() }
    fn deltanet_decay_gate(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32) { unreachable!() }
    fn deltanet_step(&self, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: &CudaTensor, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32) { unreachable!() }
}
