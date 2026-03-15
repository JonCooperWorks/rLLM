// ---------------------------------------------------------------------------
// GpuDeltaNet — Gated DeltaNet linear attention kernels (Qwen 3.5).
//
// DeltaNet replaces softmax attention with a recurrent state matrix in 75%
// of Qwen 3.5's layers.  The ops here implement the single-token decode
// path: causal conv1d for local context, L2 normalization for Q/K,
// Mamba-style decay gating, and the core state-update step.
//
// These ops are model-specific to Qwen 3.5 today but the trait boundary
// keeps them isolated — backends that don't need DeltaNet can stub it out.
//
// Metal shaders: shaders/deltanet.metal
// Metal impl:    gpu/metal/kernels/deltanet.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuDeltaNet: GpuCore {
    /// Causal depthwise Conv1D with SiLU for single-token decode.
    fn conv1d_depthwise_single(
        &self,
        input: &Self::Tensor,
        history: &Self::Tensor,
        weight: &Self::Tensor,
        out: &Self::Tensor,
        dim: u32,
        kernel_size: u32,
    );

    /// Shift Conv1D history buffer: discard oldest entry, append current input.
    fn conv1d_shift_history(
        &self,
        history: &Self::Tensor,
        input: &Self::Tensor,
        dim: u32,
        kernel_size: u32,
    );

    /// L2-normalize each head's vector in place.
    fn l2_normalize_heads(
        &self,
        data: &Self::Tensor,
        num_heads: u32,
        head_dim: u32,
        elem_offset: u32,
    );

    /// Mamba-style decay gate: g = exp(softplus(x + dt_bias) * (-exp(A_log))).
    fn deltanet_decay_gate(
        &self,
        x: &Self::Tensor,
        dt_bias: &Self::Tensor,
        a_log: &Self::Tensor,
        out: &Self::Tensor,
        size: u32,
    );

    /// DeltaNet recurrent state update + output computation (single token).
    fn deltanet_step(
        &self,
        state: &Self::Tensor,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        alpha: &Self::Tensor,
        beta: &Self::Tensor,
        out: &Self::Tensor,
        num_qk_heads: u32,
        num_v_heads: u32,
        head_dim: u32,
        q_offset: u32,
        k_offset: u32,
        v_offset: u32,
    );
}
