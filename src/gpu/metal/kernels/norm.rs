// ---------------------------------------------------------------------------
// Metal impl: GpuNorm — RMS normalization kernels.
//
// Trait contract: gpu/ops/norm.rs
// Metal shader:   metal/shaders/rms_norm.metal
//
// Three variants: weighted (per-layer), batched (prefill), and unweighted
// (standalone normalize). All use 256-thread threadgroups for the parallel
// reduction (sum of squares → rsqrt → scale).
//
// Param structs are #[repr(C)] and must match the Metal shader's argument
// buffer layout byte-for-byte.
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuNorm;

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormParams {
    hidden_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormBatchParams {
    hidden_size: u32,
    eps: f32,
    batch_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormNoWeightParams {
    size: u32,
    eps: f32,
}

impl GpuNorm for MetalBackend {
    fn rms_norm(&self, input: &MetalTensor, weight: &MetalTensor, eps: f32, out: &MetalTensor) {
        let hidden_size = input.shape[0] as u32;
        let params = RmsNormParams { hidden_size, eps };
        self.dispatch_async(
            &self.pipeline_rms_norm,
            &params,
            &[(&input.buffer, 1), (&weight.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn rms_norm_batch(
        &self,
        input: &MetalTensor,
        weight: &MetalTensor,
        eps: f32,
        out: &MetalTensor,
        batch_size: u32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = RmsNormBatchParams {
            hidden_size,
            eps,
            batch_size,
        };
        self.dispatch_async(
            &self.pipeline_rms_norm_batch,
            &params,
            &[(&input.buffer, 1), (&weight.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(batch_size as u64 * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn rms_norm_no_weight(&self, input: &MetalTensor, out: &MetalTensor, size: u32, eps: f32) {
        let params = RmsNormNoWeightParams { size, eps };
        self.dispatch_async(
            &self.pipeline_rms_norm_no_weight,
            &params,
            &[(&input.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
