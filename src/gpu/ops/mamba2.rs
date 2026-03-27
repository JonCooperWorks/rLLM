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
// Conv1d on [x, B, C]:
//   Unlike a naive implementation, Mamba-2 applies the causal conv1d to the
//   concatenation of [x, B, C] — not just x.  This means B and C also get
//   temporal smoothing from the convolution before being used in the SSM.
//   conv_dim = d_inner + 2 × n_groups × state_size.
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
//   - Conv1D: Mamba-2 convolves [x, B, C] with bias; DeltaNet convolves
//     only x with no bias
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
    /// Operates on the [x, B, C] concatenation (conv_dim elements) from the
    /// in_proj output.  Each channel i independently computes:
    ///   acc = bias[i]
    ///   acc += sum_j(weight[i, j] × history[j × dim + i])  for j in 0..kernel_size-1
    ///   acc += weight[i, kernel_size-1] × input[i]
    ///   out[i] = silu(acc)
    ///
    /// After this, out contains [x_conv(d_inner), B_conv(ngs), C_conv(ngs)]
    /// where ngs = n_groups × state_size.
    ///
    /// Note: the history shift is handled separately by reusing DeltaNet's
    /// `conv1d_shift_history` — same FIFO operation, no bias involved.
    fn mamba2_conv1d_silu(
        &self,
        input: &Self::Tensor,   // buffer containing xBC at `input_offset`
        history: &Self::Tensor, // [(kernel_size-1) × conv_dim] — prior tokens
        weight: &Self::Tensor,  // [conv_dim, kernel_size] — per-channel weights
        bias: &Self::Tensor,    // [conv_dim] bf16 — per-channel bias
        out: &Self::Tensor,     // [conv_dim] — conv1d + silu output
        dim: u32,               // conv_dim (= d_inner + 2*n_groups*state_size)
        kernel_size: u32,       // typically 4
        input_offset: u32,      // element offset into `input` where xBC starts
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
    ///
    /// B/C come from `bc_buf` (the conv_out containing [x, B, C] after conv1d).
    /// dt comes from `dt_buf` (the in_proj_buf, where dt was not convolved).
    /// Gated grouped RMSNorm for Mamba-2 output.
    ///
    /// Computes: out[i] = rmsnorm_grouped(y[i] * silu(z[i])) * weight[i]
    ///
    /// The normalization operates on non-overlapping groups of `group_size`
    /// elements.  Each group is normalized independently, matching the
    /// Mamba-2 `MambaRMSNormGated` with `norm_before_gate=False`.
    ///
    /// For Nemotron-H: d_inner=4096, group_size=512 (= d_inner / n_groups).
    fn mamba2_gated_rms_norm(
        &self,
        y: &Self::Tensor,          // [d_inner] bf16 — raw SSM output
        z_buf: &Self::Tensor,      // buffer containing z at z_offset
        weight: &Self::Tensor,     // [d_inner] bf16 — norm weight
        out: &Self::Tensor,        // [d_inner] bf16 — output
        d_inner: u32,
        group_size: u32,           // d_inner / n_groups
        z_offset: u32,             // element offset in z_buf where z starts
        eps: f32,
    );

    fn mamba2_ssm_step(
        &self,
        state: &Self::Tensor,       // [num_heads × head_dim × state_size] f32 (persistent)
        x: &Self::Tensor,           // [d_inner] bf16 — SSM input (after conv1d + silu)
        bc_buf: &Self::Tensor,      // buffer containing B and C at specified offsets
        dt_buf: &Self::Tensor,      // buffer containing dt at specified offset
        a_log: &Self::Tensor,       // [num_heads] f32 — log of diagonal decay rates
        d_skip: &Self::Tensor,      // [num_heads] f32 — skip connection scalars
        dt_bias: &Self::Tensor,     // [num_heads] f32 — learned time step bias
        norm_weight: &Self::Tensor, // [d_inner] bf16 — RMSNorm weight
        out: &Self::Tensor,         // [d_inner] bf16 — output
        num_heads: u32,
        head_dim: u32,
        state_size: u32,
        n_groups: u32,
        b_offset: u32,              // element offset in bc_buf where B starts
        c_offset: u32,              // element offset in bc_buf where C starts
        dt_offset: u32,             // element offset in dt_buf where dt starts
        eps: f32,                   // RMSNorm epsilon
    );
}
