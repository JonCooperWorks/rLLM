// ---------------------------------------------------------------------------
// CUDA impl: GpuDeltaNet — Gated DeltaNet linear attention kernels.
//
// Trait contract: gpu/ops/deltanet.rs
// CUDA shader:    cuda/shaders/deltanet.cu
//
// Model-specific to Qwen 3.5 which uses DeltaNet (recurrent state matrix)
// in 75% of its layers instead of softmax attention.  Five kernels:
//
//   conv1d_depthwise_single — causal depthwise conv1d + SiLU for local
//     context mixing (one thread per channel)
//   conv1d_shift_history — FIFO shift of the conv history buffer
//   l2_normalize_heads — per-head L2 normalization of Q/K vectors
//     (256-thread reduction per head)
//   deltanet_decay_gate — Mamba-style decay: exp(softplus(x+bias) * -exp(A))
//   deltanet_step — core recurrent update: state' = alpha*state + beta*k*v^T,
//     output = q * state (256-thread reduction per head)
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuDeltaNet;

#[repr(C)]
#[derive(Clone, Copy)]
struct Conv1dParams {
    dim: u32,
    kernel_size: u32,
}
unsafe impl DeviceRepr for Conv1dParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct L2NormParams {
    num_heads: u32,
    head_dim: u32,
    elem_offset: u32,
}
unsafe impl DeviceRepr for L2NormParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct DecayGateParams {
    size: u32,
}
unsafe impl DeviceRepr for DecayGateParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaNetStepParams {
    num_qk_heads: u32,
    num_v_heads: u32,
    head_dim: u32,
    q_elem_offset: u32,
    k_elem_offset: u32,
    v_elem_offset: u32,
}
unsafe impl DeviceRepr for DeltaNetStepParams {}

impl GpuDeltaNet for CudaBackend {
    fn conv1d_depthwise_single(
        &self,
        input: &CudaTensor,
        history: &CudaTensor,
        weight: &CudaTensor,
        out: &CudaTensor,
        dim: u32,
        kernel_size: u32,
    ) {
        let params = Conv1dParams { dim, kernel_size };
        let block = 256.min(dim);
        let cfg = CudaBackend::cfg_1d(dim, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_conv1d_depthwise)
                .arg(&params)
                .arg(&input.buf)
                .arg(&history.buf)
                .arg(&weight.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("conv1d_depthwise_single launch failed");
    }

    fn conv1d_shift_history(
        &self,
        history: &CudaTensor,
        input: &CudaTensor,
        dim: u32,
        kernel_size: u32,
    ) {
        let params = Conv1dParams { dim, kernel_size };
        let block = 256.min(dim);
        let cfg = CudaBackend::cfg_1d(dim, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_conv1d_shift)
                .arg(&params)
                .arg(&history.buf)
                .arg(&input.buf)
                .launch(cfg)
        }
        .expect("conv1d_shift_history launch failed");
    }

    fn l2_normalize_heads(
        &self,
        data: &CudaTensor,
        num_heads: u32,
        head_dim: u32,
        elem_offset: u32,
    ) {
        let params = L2NormParams {
            num_heads,
            head_dim,
            elem_offset,
        };
        // One block per head, 256 threads per block.
        let cfg = CudaBackend::cfg_blocks(num_heads, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_l2_normalize)
                .arg(&params)
                .arg(&data.buf)
                .launch(cfg)
        }
        .expect("l2_normalize_heads launch failed");
    }

    fn deltanet_decay_gate(
        &self,
        x: &CudaTensor,
        dt_bias: &CudaTensor,
        a_log: &CudaTensor,
        out: &CudaTensor,
        size: u32,
    ) {
        let params = DecayGateParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_decay_gate)
                .arg(&params)
                .arg(&x.buf)
                .arg(&dt_bias.buf)
                .arg(&a_log.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("deltanet_decay_gate launch failed");
    }

    fn deltanet_step(
        &self,
        state: &CudaTensor,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        alpha: &CudaTensor,
        beta: &CudaTensor,
        out: &CudaTensor,
        num_qk_heads: u32,
        num_v_heads: u32,
        head_dim: u32,
        q_offset: u32,
        k_offset: u32,
        v_offset: u32,
    ) {
        let params = DeltaNetStepParams {
            num_qk_heads,
            num_v_heads,
            head_dim,
            q_elem_offset: q_offset,
            k_elem_offset: k_offset,
            v_elem_offset: v_offset,
        };
        // One block per QK-head, 256 threads per block.
        let cfg = CudaBackend::cfg_blocks(num_qk_heads, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_deltanet_step)
                .arg(&params)
                .arg(&state.buf)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&v.buf)
                .arg(&alpha.buf)
                .arg(&beta.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("deltanet_step launch failed");
    }
}
