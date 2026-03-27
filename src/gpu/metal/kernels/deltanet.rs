// ---------------------------------------------------------------------------
// Metal impl: GpuDeltaNet — Gated DeltaNet linear attention kernels.
//
// Trait contract: gpu/ops/deltanet.rs
// Metal shader:   metal/shaders/deltanet.metal
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

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuDeltaNet;

#[repr(C)]
#[derive(Clone, Copy)]
struct Conv1dParams {
    dim: u32,
    kernel_size: u32,
    input_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct L2NormParams {
    num_heads: u32,
    head_dim: u32,
    elem_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DecayGateParams {
    size: u32,
}

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

impl GpuDeltaNet for MetalBackend {
    fn conv1d_depthwise_single(
        &self,
        input: &MetalTensor,
        history: &MetalTensor,
        weight: &MetalTensor,
        out: &MetalTensor,
        dim: u32,
        kernel_size: u32,
    ) {
        let params = Conv1dParams { dim, kernel_size, input_offset: 0 };
        self.dispatch_async(
            &self.pipeline_conv1d_depthwise,
            &params,
            &[
                (&input.buffer, 1),
                (&history.buffer, 2),
                (&weight.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(256.min(dim as u64), 1, 1),
        );
    }

    fn conv1d_shift_history(
        &self,
        history: &MetalTensor,
        input: &MetalTensor,
        dim: u32,
        kernel_size: u32,
        input_offset: u32,
    ) {
        let params = Conv1dParams { dim, kernel_size, input_offset };
        self.dispatch_async(
            &self.pipeline_conv1d_shift,
            &params,
            &[(&history.buffer, 1), (&input.buffer, 2)],
            MTLSize::new(dim as u64, 1, 1),
            MTLSize::new(256.min(dim as u64), 1, 1),
        );
    }

    fn l2_normalize_heads(
        &self,
        data: &MetalTensor,
        num_heads: u32,
        head_dim: u32,
        elem_offset: u32,
    ) {
        let params = L2NormParams {
            num_heads,
            head_dim,
            elem_offset,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_l2_normalize,
            &params,
            &[(&data.buffer, 1)],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn deltanet_decay_gate(
        &self,
        x: &MetalTensor,
        dt_bias: &MetalTensor,
        a_log: &MetalTensor,
        out: &MetalTensor,
        size: u32,
    ) {
        let params = DecayGateParams { size };
        self.dispatch_async(
            &self.pipeline_decay_gate,
            &params,
            &[
                (&x.buffer, 1),
                (&dt_bias.buffer, 2),
                (&a_log.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn deltanet_step(
        &self,
        state: &MetalTensor,
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        alpha: &MetalTensor,
        beta: &MetalTensor,
        out: &MetalTensor,
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
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_deltanet_step,
            &params,
            &[
                (&state.buffer, 1),
                (&q.buffer, 2),
                (&k.buffer, 3),
                (&v.buffer, 4),
                (&alpha.buffer, 5),
                (&beta.buffer, 6),
                (&out.buffer, 7),
            ],
            MTLSize::new(num_qk_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }
}
