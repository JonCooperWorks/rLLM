// ===========================================================================
// Mamba-2 GPU kernels — Selective State Space Model for Nemotron-H.
//
// LEARNING OVERVIEW
//
// Mamba-2 (SSD — State Space Duality) is a recurrent sequence model where
// each head maintains a [head_dim, state_size] state matrix.  Unlike
// softmax attention (O(seq_len) KV cache), the recurrent state is O(1) —
// fixed memory regardless of sequence length.
//
// The core recurrence per head per token:
//   dt    = softplus(dt_raw + dt_bias)       — learned step size
//   dA    = exp(dt × (-exp(A_log)))          — discretized decay ∈ (0, 1)
//   state = dA × state + dt × outer(x, B)   — state update
//   y     = state @ C + D × x               — output readout
//
// B (state input) and C (state output) are grouped across heads — similar
// to GQA sharing KV heads.  With n_groups=8, num_heads=64, each group of
// 8 heads shares one B/C vector.
//
// This file contains two kernels:
//   1. mamba2_conv1d_silu — Depthwise Conv1D + bias + SiLU (one thread/channel)
//   2. mamba2_ssm_step — State update + output + RMSNorm (one threadgroup/head)
//
// Trait contract: gpu/ops/mamba2.rs
// Metal impl:    gpu/metal/kernels/mamba2.rs
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Depthwise Conv1D with bias and SiLU activation.
//
// One thread per channel.  For channel i:
//   acc = bias[i]
//   acc += sum(weight[i, j] * history[j * dim + i])  for j in 0..kernel_size-1
//   acc += weight[i, kernel_size-1] * input[i]
//   out[i] = silu(acc)
//
// This is identical to DeltaNet's conv1d but with an additive bias BEFORE
// the SiLU activation.  Nemotron-H sets `use_conv_bias: true` while
// DeltaNet omits it.  The bias allows each channel to learn a different
// baseline activation level — useful for SSM models where the conv1d
// provides local mixing before the state space recurrence.
// ===========================================================================

struct MambaConv1dParams {
    uint dim;          // Number of channels (d_inner).
    uint kernel_size;  // Convolution window (typically 4).
    uint input_offset; // Element offset into input buffer where x starts.
};

kernel void mamba2_conv1d_silu(
    constant MambaConv1dParams& params [[buffer(0)]],
    device const bfloat* input         [[buffer(1)]],
    device const bfloat* history       [[buffer(2)]],
    device const bfloat* weight        [[buffer(3)]],
    device const float* bias           [[buffer(4)]],
    device bfloat* out                 [[buffer(5)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= params.dim) return;

    // Start with bias — the key difference from DeltaNet's conv1d.
    // Bias is f32 (loaded as f32 by the weight loader), no cast needed.
    float acc = bias[gid];

    // Convolve over the history buffer (kernel_size - 1 prior tokens).
    // History layout: [time_step, dim] — each row is one prior token's
    // full channel vector.  Row 0 is the oldest, row kernel_size-2 is
    // the most recent prior token.
    for (uint j = 0; j < params.kernel_size - 1; j++) {
        acc += float(weight[gid * params.kernel_size + j]) *
               float(history[j * params.dim + gid]);
    }

    // The current token occupies the last weight tap.
    acc += float(weight[gid * params.kernel_size + params.kernel_size - 1]) *
           float(input[params.input_offset + gid]);

    // SiLU activation: silu(x) = x / (1 + exp(-x)).
    out[gid] = bfloat(acc / (1.0f + exp(-acc)));
}

// ===========================================================================
// Mamba-2 SSM step: state update + output readout + RMSNorm.
//
// One threadgroup (256 threads) per head.  The 256 threads cooperatively
// process all head_dim rows of the state matrix, then reduce for RMSNorm.
//
// Algorithm per head h (group g = h * n_groups / num_heads):
//   1. Thread 0 computes shared scalars: dt, dA, D
//   2. All threads update their assigned state rows and compute partial
//      output values
//   3. Cooperative reduction for RMSNorm (sum of squares → normalize)
//
// State layout: [num_heads, head_dim, state_size] in f32 for precision.
// The f32 state avoids accumulation drift across thousands of tokens —
// critical because the state is never reset during inference.
//
// Why RMSNorm inside the kernel?
//   The output must be normalized before the output projection matmul.
//   Fusing it here avoids a separate kernel dispatch and an extra global
//   memory round-trip for the [num_heads * head_dim] output vector.
// ===========================================================================

struct Mamba2SsmStepParams {
    uint num_heads;   // Number of SSM heads.
    uint head_dim;    // Dimension per head.
    uint state_size;  // SSM state size (N in the paper).
    uint n_groups;    // Number of B/C groups (shared across heads).
    uint b_offset;    // Element offset in bcdt buffer where B starts.
    uint c_offset;    // Element offset where C starts.
    uint dt_offset;   // Element offset where dt starts.
    float eps;        // RMSNorm epsilon.
};

kernel void mamba2_ssm_step(
    constant Mamba2SsmStepParams& params [[buffer(0)]],
    device float* state                  [[buffer(1)]],
    device const bfloat* x               [[buffer(2)]],
    device const bfloat* bcdt            [[buffer(3)]],
    device const float* a_log            [[buffer(4)]],
    device const float* d_skip           [[buffer(5)]],
    device const float* dt_bias          [[buffer(6)]],
    device const bfloat* norm_weight     [[buffer(7)]],
    device bfloat* out                   [[buffer(8)]],
    uint tgid                            [[threadgroup_position_in_grid]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]]
) {
    // One threadgroup per head.
    uint h = tgid;
    if (h >= params.num_heads) return;

    uint hd = params.head_dim;
    uint ss = params.state_size;

    // Which B/C group does this head belong to?
    // heads_per_group = num_heads / n_groups, so group = h / heads_per_group
    // = h * n_groups / num_heads.
    uint g = h * params.n_groups / params.num_heads;

    // --- Step 1: Shared scalars (computed by thread 0, broadcast via shared memory) ---

    threadgroup float shared_dt_val;
    threadgroup float shared_dA;
    threadgroup float shared_D_val;

    if (tid == 0) {
        // Softplus: log(1 + exp(x)).  dt_bias is in f32 (learned parameter).
        float dt_raw = float(bcdt[params.dt_offset + h]) + dt_bias[h];
        float dt_val = log(1.0f + exp(dt_raw));

        // Discretised decay: dA = exp(dt * (-exp(A_log))).
        // A_log stores log(-A) where A is the diagonal decay rate.
        // exp(A_log) recovers |A|, negated to make decay < 1.
        float dA = exp(dt_val * (-exp(a_log[h])));

        shared_dt_val = dt_val;
        shared_dA = dA;
        shared_D_val = d_skip[h];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float dt_val = shared_dt_val;
    float dA = shared_dA;
    float D_val = shared_D_val;

    // --- Step 2: State update + output computation ---
    // Each thread handles rows [tid, tid+stride, ...] of the state matrix.
    // For head_dim=64 with 256 threads, each thread does at most 1 row.
    // For head_dim=128 with 256 threads, still at most 1 row per thread.

    // We need a per-thread partial output that we'll later use for RMSNorm.
    // Use threadgroup memory to collect all y values for the RMSNorm reduction.
    threadgroup float y_shared[256];  // Max head_dim we'll ever see.
    y_shared[tid] = 0.0f;

    for (uint i = tid; i < hd; i += tg_size) {
        float x_val = float(x[h * hd + i]);

        // Base index into the state: state[h, i, :] is a contiguous [state_size] slice.
        uint state_base = h * hd * ss + i * ss;

        // Compute output for this row: y[i] = sum_s(state[h,i,s] * C[g,s]) + D * x[i].
        // We update the state first, THEN read it for the output.
        float y_val = 0.0f;

        for (uint s = 0; s < ss; s++) {
            uint idx = state_base + s;
            float b_val = float(bcdt[params.b_offset + g * ss + s]);

            // State update: state[h,i,s] = dA * state[h,i,s] + dt * x[i] * B[g,s]
            // The outer(x, B) term is rank-1: each element is x[i] * B[s].
            state[idx] = dA * state[idx] + dt_val * x_val * b_val;

            // Accumulate output: y[i] += state[h,i,s] * C[g,s]
            float c_val = float(bcdt[params.c_offset + g * ss + s]);
            y_val += state[idx] * c_val;
        }

        // Skip connection: y[i] += D * x[i]
        y_val += D_val * x_val;

        y_shared[i] = y_val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 3: RMSNorm on y[0..head_dim] ---
    // RMSNorm(y) = y / sqrt(mean(y²) + eps) * weight
    //
    // Cooperative reduction: each thread sums squares for its assigned rows,
    // then we reduce across threads.

    // Sum of squares — each thread accumulates its rows.
    float local_ss = 0.0f;
    for (uint i = tid; i < hd; i += tg_size) {
        float v = y_shared[i];
        local_ss += v * v;
    }

    // Reduction across threadgroup using shared memory.
    threadgroup float reduce_buf[256];
    reduce_buf[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(reduce_buf[0] / float(hd) + params.eps);

    // Normalize and write output.
    for (uint i = tid; i < hd; i += tg_size) {
        float normed = y_shared[i] * rms;
        float w = float(norm_weight[h * hd + i]);
        out[h * hd + i] = bfloat(normed * w);
    }
}
