// ---------------------------------------------------------------------------
// GpuMamba2 — Mamba-2 Selective State Space Model kernels (Nemotron-H).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines GPU kernel contracts for the Mamba-2 (SSD — State Space Duality)
//   recurrent sequence model, used by NVIDIA's Nemotron-H architecture.
//
// How Mamba-2 works:
//   Mamba-2 is a selective state space model where each head maintains a
//   fixed-size [head_dim, state_size] recurrent state matrix.  At each
//   token, the state is updated via:
//
//     dt    = softplus(dt_raw + dt_bias)
//     dA    = exp(dt × (-exp(A_log)))          — discretized decay
//     state = dA × state + dt × outer(x, B)    — state update
//     y     = state @ C + D × x                — output readout
//
//   The B (state input) and C (state output) vectors are computed from the
//   input at each step — this makes the model "selective" (it decides what
//   to remember based on the current input).  The dt parameter controls
//   how much of the previous state to retain vs. replace.
//
// Grouped B/C:
//   Instead of per-head B and C vectors (which would be expensive), Mamba-2
//   shares them across groups of heads — similar to how GQA shares KV heads.
//   With n_groups=8 and num_heads=64, each group of 8 heads shares one B/C.
//
// How it differs from DeltaNet (Qwen 3.5):
//   - State shape: [head_dim, state_size] vs DeltaNet's [head_dim, head_dim]
//   - Update rule: simple additive (outer(x, B)) vs DeltaNet's delta rule
//   - Gate: Mamba-2 uses dt-driven discretization; DeltaNet uses explicit
//     alpha/beta gates
//   - Conv1D: Mamba-2 needs bias support; DeltaNet's conv has no bias
//
// Metal shaders: shaders/mamba2.metal
// Metal impl:    gpu/metal/kernels/mamba2.rs
// CUDA impl:     cuda/kernels/mamba2.rs (stub)
// CPU impl:      cpu/mod.rs (reference implementation for testing)
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuMamba2: GpuCore {
    /// Causal depthwise Conv1D with bias and SiLU activation for Mamba-2.
    ///
    /// Computes for each channel i:
    ///   acc = bias[i]
    ///   acc += sum_j(weight[i, j] × history[j × dim + i])  for j in 0..kernel_size-1
    ///   acc += weight[i, kernel_size-1] × input[i]
    ///   out[i] = silu(acc)
    ///
    /// This is identical to DeltaNet's `conv1d_depthwise_single` but with an
    /// additive bias BEFORE the SiLU activation.  Nemotron-H config has
    /// `use_conv_bias: true` while DeltaNet has no conv bias.
    ///
    /// Note: the history shift is handled separately by reusing DeltaNet's
    /// `conv1d_shift_history` — same FIFO operation, no bias involved.
    fn mamba2_conv1d_silu(
        &self,
        input: &Self::Tensor,   // [d_inner] — current token's SSM input
        history: &Self::Tensor, // [(kernel_size-1) × d_inner] — prior tokens
        weight: &Self::Tensor,  // [d_inner, kernel_size] — per-channel weights
        bias: &Self::Tensor,    // [d_inner] — per-channel bias
        out: &Self::Tensor,     // [d_inner] — conv1d + silu output
        dim: u32,               // d_inner
        kernel_size: u32,       // typically 4
    );

    /// Mamba-2 selective SSM step: state update + output with grouped B/C.
    ///
    /// For each head h (with group g = h / (num_heads / n_groups)):
    ///   1. Discretize: dt_h = softplus(dt_raw[h] + dt_bias[h])
    ///      dA_h = exp(dt_h × (-exp(A_log[h])))
    ///   2. State update: state[h] = dA_h × state[h] + dt_h × outer(x[h], B[g])
    ///      where x[h] is [head_dim] and B[g] is [state_size], producing a
    ///      [head_dim, state_size] outer product added to the state.
    ///   3. Output: y[h] = state[h] @ C[g] + D[h] × x[h]
    ///      The [head_dim, state_size] state is contracted with [state_size]
    ///      C vector to produce [head_dim] output, plus skip connection.
    ///   4. Norm: y = rmsnorm(y, norm_weight) with learned per-element weight.
    ///
    /// State layout: [num_heads × head_dim × state_size] in f32 for precision.
    /// This is persistent across tokens (the "recurrent state").
    fn mamba2_ssm_step(
        &self,
        state: &Self::Tensor,       // [num_heads × head_dim × state_size] f32 (persistent)
        x: &Self::Tensor,           // [num_heads × head_dim] bf16 — SSM input (after conv1d + silu)
        b: &Self::Tensor,           // [n_groups × state_size] bf16 — state input vector
        c: &Self::Tensor,           // [n_groups × state_size] bf16 — state output vector
        dt: &Self::Tensor,          // [num_heads] bf16 — raw time steps (before softplus)
        a_log: &Self::Tensor,       // [num_heads] f32 — log of diagonal decay rates
        d_skip: &Self::Tensor,      // [num_heads] f32 — skip connection scalars
        dt_bias: &Self::Tensor,     // [num_heads] f32 — learned time step bias
        norm_weight: &Self::Tensor, // [num_heads × head_dim] bf16 — RMSNorm weight
        out: &Self::Tensor,         // [num_heads × head_dim] bf16 — output
        num_heads: u32,
        head_dim: u32,
        state_size: u32,
        n_groups: u32,
        eps: f32,                   // RMSNorm epsilon
    );
}
