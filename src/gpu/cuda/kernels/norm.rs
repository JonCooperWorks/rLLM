// ---------------------------------------------------------------------------
// CUDA impl: GpuNorm — RMS normalization kernels.
//
// Trait contract: gpu/ops/norm.rs
// CUDA shader:    cuda/shaders/rms_norm.cu
//
// Two variants: weighted (per-layer) and batched (prefill). Both use
// 256-thread blocks for the parallel reduction (sum of squares →
// rsqrt → scale).
//
// Param structs are #[repr(C)] and must match the CUDA shader's struct
// layout byte-for-byte.
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuNorm;

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormParams {
    hidden_size: u32,
    eps: f32,
}
unsafe impl DeviceRepr for RmsNormParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormBatchParams {
    hidden_size: u32,
    eps: f32,
    batch_size: u32,
}
unsafe impl DeviceRepr for RmsNormBatchParams {}

impl GpuNorm for CudaBackend {
    fn rms_norm(&self, input: &CudaTensor, weight: &CudaTensor, eps: f32, out: &CudaTensor) {
        let hidden_size = input.shape[0] as u32;
        let params = RmsNormParams { hidden_size, eps };
        // Single vector: 1 block × 256 threads.
        let cfg = CudaBackend::cfg_blocks(1, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_rms_norm)
                .arg(&params)
                .arg(&input.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("rms_norm launch failed");
    }

    fn rms_norm_batch(
        &self,
        input: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
        out: &CudaTensor,
        batch_size: u32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = RmsNormBatchParams {
            hidden_size,
            eps,
            batch_size,
        };
        // One block per row, 256 threads per block.
        let cfg = CudaBackend::cfg_blocks(batch_size, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_rms_norm_batch)
                .arg(&params)
                .arg(&input.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("rms_norm_batch launch failed");
    }
}
