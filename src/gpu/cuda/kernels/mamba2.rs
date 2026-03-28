// ---------------------------------------------------------------------------
// CUDA impl: GpuMamba2 — Mamba-2 Selective State Space Model kernels.
//
// Trait contract: gpu/ops/mamba2.rs
// CUDA shader:    cuda/shaders/mamba2.cu
//
// Three kernels:
//   mamba2_conv1d_silu    — depthwise conv1d + bias + SiLU
//   mamba2_ssm_step       — SSM state update + output
//   mamba2_gated_rms_norm — gated grouped RMSNorm
//
// Related files:
//   Metal shader:     metal/shaders/mamba2.metal
//   Metal bridge:     metal/kernels/mamba2.rs
//   CPU reference:    cpu/mod.rs
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuMamba2;

#[repr(C)]
#[derive(Clone, Copy)]
struct MambaConv1dParams {
    dim: u32,
    kernel_size: u32,
    input_offset: u32,
}
unsafe impl DeviceRepr for MambaConv1dParams {}

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
unsafe impl DeviceRepr for Mamba2SsmStepParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct MambaGatedNormParams {
    d_inner: u32,
    group_size: u32,
    z_offset: u32,
    eps: f32,
}
unsafe impl DeviceRepr for MambaGatedNormParams {}

impl GpuMamba2 for CudaBackend {
    fn mamba2_conv1d_silu(
        &self,
        input: &CudaTensor,
        history: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        out: &CudaTensor,
        dim: u32,
        kernel_size: u32,
        input_offset: u32,
    ) {
        let params = MambaConv1dParams {
            dim,
            kernel_size,
            input_offset,
        };
        let block = 256.min(dim);
        let cfg = CudaBackend::cfg_1d(dim, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_mamba2_conv1d_silu)
                .arg(&params)
                .arg(&input.buf)
                .arg(&history.buf)
                .arg(&weight.buf)
                .arg(&bias.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("mamba2_conv1d_silu launch failed");
    }

    fn mamba2_ssm_step(
        &self,
        state: &CudaTensor,
        x: &CudaTensor,
        bc_buf: &CudaTensor,
        dt_buf: &CudaTensor,
        a_log: &CudaTensor,
        d_skip: &CudaTensor,
        dt_bias: &CudaTensor,
        norm_weight: &CudaTensor,
        out: &CudaTensor,
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
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_mamba2_ssm_step)
                .arg(&params)
                .arg(&state.buf)
                .arg(&x.buf)
                .arg(&bc_buf.buf)
                .arg(&dt_buf.buf)
                .arg(&a_log.buf)
                .arg(&d_skip.buf)
                .arg(&dt_bias.buf)
                .arg(&norm_weight.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("mamba2_ssm_step launch failed");
    }

    fn mamba2_gated_rms_norm(
        &self,
        y: &CudaTensor,
        z_buf: &CudaTensor,
        weight: &CudaTensor,
        out: &CudaTensor,
        d_inner: u32,
        group_size: u32,
        z_offset: u32,
        eps: f32,
    ) {
        let params = MambaGatedNormParams {
            d_inner,
            group_size,
            z_offset,
            eps,
        };
        let num_groups = d_inner / group_size;
        let cfg = CudaBackend::cfg_blocks(num_groups, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_mamba2_gated_rms_norm)
                .arg(&params)
                .arg(&y.buf)
                .arg(&z_buf.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("mamba2_gated_rms_norm launch failed");
    }
}
