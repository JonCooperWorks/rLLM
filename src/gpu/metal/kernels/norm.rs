// ---------------------------------------------------------------------------
// Metal impl: GpuNorm — RMS normalization kernels.
//
// Trait contract: gpu/ops/norm.rs
// Metal shader:   metal/shaders/rms_norm.metal
//
// Two variants: weighted (per-layer) and batched (prefill). Both use
// 256-thread threadgroups for the parallel reduction (sum of squares →
// rsqrt → scale).
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
struct LayerNormBatchParams {
    hidden_size: u32,
    eps: f32,
    batch_size: u32,
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

    fn layer_norm_batch(
        &self,
        input: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        eps: f32,
        out: &MetalTensor,
        batch_size: u32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = LayerNormBatchParams {
            hidden_size,
            eps,
            batch_size,
        };
        self.dispatch_async(
            &self.pipeline_layer_norm_batch,
            &params,
            &[
                (&input.buffer, 1),
                (&weight.buffer, 2),
                (&bias.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(batch_size as u64 * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn fused_residual_rms_norm(
        &self,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        weight: &MetalTensor,
        out: &MetalTensor,
        _hidden_size: u32,
        eps: f32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = RmsNormParams { hidden_size, eps };
        self.dispatch_async(
            &self.pipeline_fused_residual_rms_norm,
            &params,
            &[
                (&hidden.buffer, 1),
                (&residual.buffer, 2),
                (&weight.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn fused_residual_rms_norm_batch(
        &self,
        hidden: &MetalTensor,
        residual: &MetalTensor,
        weight: &MetalTensor,
        out: &MetalTensor,
        _hidden_size: u32,
        eps: f32,
        batch_size: u32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = RmsNormBatchParams {
            hidden_size,
            eps,
            batch_size,
        };
        self.dispatch_async(
            &self.pipeline_fused_residual_rms_norm_batch,
            &params,
            &[
                (&hidden.buffer, 1),
                (&residual.buffer, 2),
                (&weight.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(batch_size as u64 * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
