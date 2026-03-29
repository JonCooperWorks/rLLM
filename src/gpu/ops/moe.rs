// ---------------------------------------------------------------------------
// GpuMoe — fused MoE (Mixture of Experts) kernels.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines fused GPU kernel contracts for MoE expert FFN computation.
//   These combine multiple sequential operations into single kernels,
//   reducing dispatch overhead and memory traffic.
//
// Why fusion matters for MoE:
//   A standard MoE layer with K=8 experts runs 8 × 4 = 32 kernel dispatches
//   per token (gate matmul + up matmul + silu_mul + down matmul per expert).
//   Each dispatch has Metal/CUDA overhead (~2-5μs) and the gate/up matmuls
//   both read the same input vector from GPU memory.
//
//   fused_gate_up_swiglu combines steps 1-3 into one dispatch: each SIMD
//   group computes two dot products simultaneously (gate and up), then
//   applies SiLU.  This halves dispatches (8 × 2 = 16) and reads the
//   input once instead of twice.
//
//   moe_combine_residual replaces K scale_add calls + 1 add with a single
//   kernel that reads all K expert outputs in one pass.
//
// Inspired by flash-moe (github.com/danveloper/flash-moe).
//
// Metal shaders: shaders/moe.metal
// Metal impl:    gpu/metal/kernels/moe.rs
// CUDA impl:     cuda/kernels/moe.rs
// CPU impl:      cpu/mod.rs (reference implementation for testing)
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuMoe: GpuCore {
    /// Fused gate+up projection with SwiGLU activation.
    ///
    /// Computes: out[i] = silu(dot(W_gate[i], x)) * dot(W_up[i], x)
    ///
    /// Fuses three separate operations (gate matmul, up matmul, silu_mul)
    /// into one kernel that reads the input vector once instead of twice.
    /// Each SIMD group computes two dot products simultaneously.
    ///
    /// Supports both bf16 and Q4 weights (dispatched by weight dtype,
    /// same pattern as `matmul`).
    fn fused_gate_up_swiglu(
        &self,
        w_gate: &Self::Tensor,
        w_up: &Self::Tensor,
        input: &Self::Tensor,
        output: &Self::Tensor,
        m: u32,
        k: u32,
    );

    /// Fused MoE combine + residual add.
    ///
    /// Combines k expert outputs with routing weights and adds the residual
    /// in a single pass:
    ///   output[j] = residual[j] + sum_i(weights[i] * expert_outs[i * hidden + j])
    ///
    /// Replaces k separate scale_add calls + 1 add call with a single kernel.
    #[allow(dead_code)] // trait method implemented by Metal; callers use explicit loops for now
    fn moe_combine_residual(
        &self,
        residual: &Self::Tensor,
        expert_outputs: &Self::Tensor,
        weights: &[f32],
        output: &Self::Tensor,
        hidden_size: u32,
        k: u32,
    );
}
