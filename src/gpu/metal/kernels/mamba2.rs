// ---------------------------------------------------------------------------
// Metal impl: GpuMamba2 — Mamba-2 Selective State Space Model kernels.
//
// Trait contract: gpu/ops/mamba2.rs
// Metal shader:   metal/shaders/mamba2.metal
//
// Two kernels for Nemotron-H's Mamba-2 layers:
//
//   mamba2_conv1d_silu — depthwise conv1d with bias + SiLU activation
//     (one thread per channel, same pattern as DeltaNet's conv1d but with bias)
//   mamba2_ssm_step — state update + output readout + fused RMSNorm
//     (one threadgroup of 256 threads per head, cooperative reduction for norm)
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuMamba2;

#[repr(C)]
#[derive(Clone, Copy)]
struct MambaConv1dParams {
    dim: u32,
    kernel_size: u32,
    input_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mamba2SsmStepParams {
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    b_offset: u32,
    c_offset: u32,
    dt_offset: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MambaGatedNormParams {
    d_inner: u32,
    group_size: u32,
    z_offset: u32,
    eps: f32,
}

impl GpuMamba2 for MetalBackend {
    fn mamba2_gated_rms_norm(
        &self,
        y: &MetalTensor,
        z_buf: &MetalTensor,
        weight: &MetalTensor,
        out: &MetalTensor,
        d_inner: u32,
        group_size: u32,
        z_offset: u32,
        eps: f32,
    ) {
        let params = MambaGatedNormParams { d_inner, group_size, z_offset, eps };
        let num_groups = (d_inner + group_size - 1) / group_size;
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_mamba2_gated_rms_norm,
            &params,
            &[
                (&y.buffer, 1),
                (&z_buf.buffer, 2),
                (&weight.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(num_groups as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn mamba2_conv1d_silu(
        &self,
        input: &MetalTensor,
        history: &MetalTensor,
        weight: &MetalTensor,
        bias: &MetalTensor,
        out: &MetalTensor,
        dim: u32,
        kernel_size: u32,
        input_offset: u32,
    ) {
        let params = MambaConv1dParams { dim, kernel_size, input_offset };
        self.dispatch_async(
            &self.pipeline_mamba2_conv1d_silu,
            &params,
            &[
                (&input.buffer, 1),
                (&history.buffer, 2),
                (&weight.buffer, 3),
                (&bias.buffer, 4),
                (&out.buffer, 5),
            ],
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(256.min(dim as u64), 1, 1),
        );
    }

    fn mamba2_ssm_step(
        &self,
        state: &MetalTensor,
        x: &MetalTensor,
        bc_buf: &MetalTensor,
        dt_buf: &MetalTensor,
        a_log: &MetalTensor,
        d_skip: &MetalTensor,
        dt_bias: &MetalTensor,
        norm_weight: &MetalTensor,
        out: &MetalTensor,
        num_heads: u32,
        head_dim: u32,
        state_size: u32,
        n_groups: u32,
        b_offset: u32,
        c_offset: u32,
        dt_offset: u32,
        eps: f32,
    ) {
        let params = Mamba2SsmStepParams {
            num_heads,
            head_dim,
            state_size,
            n_groups,
            b_offset,
            c_offset,
            dt_offset,
            eps,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_mamba2_ssm_step,
            &params,
            &[
                (&state.buffer, 1),
                (&x.buffer, 2),
                (&bc_buf.buffer, 3),
                (&dt_buf.buffer, 4),
                (&a_log.buffer, 5),
                (&d_skip.buffer, 6),
                (&dt_bias.buffer, 7),
                (&norm_weight.buffer, 8),
                (&out.buffer, 9),
            ],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }
}
