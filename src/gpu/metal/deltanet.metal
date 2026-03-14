// ===========================================================================
// DeltaNet GPU kernels — Gated DeltaNet linear attention for Qwen 3.5.
//
// LEARNING OVERVIEW
//
// Gated DeltaNet replaces softmax attention with a linear recurrence.  Instead
// of maintaining a growing KV cache (O(seq_len) memory), each DeltaNet head
// maintains a fixed-size [head_dim, head_dim] state matrix — O(1) memory
// regardless of sequence length.
//
// The core recurrence per head per token (HF reference implementation):
//   g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))  — decay gate ∈ (0, 1)
//   β_t = sigmoid(beta_t)                              — update gate ∈ (0, 1)
//   q_t = L2_norm(q_t) / sqrt(head_dim)                — scaled query
//   k_t = L2_norm(k_t)                                 — normalized key
//   S_t = g_t * S_{t-1}                                — decay state
//   r_t = S_t^T @ k_t                                  — retrieve with plain k
//   S_t = S_t + k_t ⊗ β_t*(v_t - r_t)                 — delta rule update
//   o_t = S_t^T @ q_t                                  — output
//
// The delta rule (v_t - S^T @ k_t) is the key innovation: it computes the
// error between the desired value and what the state currently retrieves for
// this key, then uses that error to update the state.  This is analogous to
// online gradient descent on an associative memory.
//
// DeltaNet layers also use:
//   - Causal depthwise Conv1D (kernel=4) for local positional context
//   - L2 normalization on Q and K for numerical stability
//   - SiLU-gated output (in_proj_z) before the output projection
//
// This file contains kernels for:
//   1. conv1d_depthwise_single — Conv1D for single-token decode
//   2. conv1d_shift_history — Update conv history buffer
//   3. l2_normalize_heads — Per-head L2 normalization
//   4. sigmoid_kernel — Element-wise sigmoid
//   5. silu_elementwise — Element-wise SiLU (for output gate)
//   6. mul_elementwise — Element-wise multiply (for output gating)
//   7. deltanet_step — Recurrent state update + output (core kernel)
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Causal depthwise Conv1D — single-token decode.
//
// For each element i of the output vector:
//   out[i] = weight[i*kernel_size + 0] * history[0*dim + i]   (oldest)
//          + weight[i*kernel_size + 1] * history[1*dim + i]
//          + ...
//          + weight[i*kernel_size + kernel_size-2] * history[(kernel_size-2)*dim + i]
//          + weight[i*kernel_size + kernel_size-1] * input[i]   (current)
//
// After convolution, SiLU activation is applied: out[i] = silu(out[i]).
//
// Conv1D provides local positional context for DeltaNet layers, which don't
// use RoPE.  With kernel_size=4, each element "sees" itself plus 3 prior tokens.
// ===========================================================================

struct Conv1dParams {
    uint dim;           // total dimension (qk_dim + qk_dim + v_dim = 10240 for 27B)
    uint kernel_size;   // convolution kernel size (4)
};

kernel void conv1d_depthwise_single(
    constant Conv1dParams& params  [[buffer(0)]],
    device const bfloat* input     [[buffer(1)]],  // [dim] — current token's projection
    device const bfloat* history   [[buffer(2)]],  // [(kernel_size-1), dim] — previous tokens
    device const bfloat* weight    [[buffer(3)]],  // [dim, kernel_size] — depthwise conv weights
    device bfloat* output          [[buffer(4)]],  // [dim] — convolved output
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.dim) return;

    const uint ks = params.kernel_size;
    const uint dim = params.dim;

    // Depthwise convolution: each element has its own kernel_size weights.
    // weight layout: [dim, kernel_size] — weight[gid * ks + k] for kernel tap k.
    float acc = 0.0f;

    // Process history taps (kernel positions 0..kernel_size-2).
    for (uint k = 0; k < ks - 1; k++) {
        acc += float(weight[gid * ks + k]) * float(history[k * dim + gid]);
    }
    // Current token (kernel position kernel_size-1).
    acc += float(weight[gid * ks + ks - 1]) * float(input[gid]);

    // SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    float silu_val = acc / (1.0f + exp(-acc));
    output[gid] = bfloat(silu_val);
}

// ===========================================================================
// Conv1D history shift — discard oldest, append current token.
//
// History layout: [(kernel_size-1), dim].
// After shift: history[t] = history[t+1] for t in 0..ks-3,
//              history[ks-2] = input.
// ===========================================================================

kernel void conv1d_shift_history(
    constant Conv1dParams& params  [[buffer(0)]],
    device bfloat* history         [[buffer(1)]],  // [(kernel_size-1), dim]
    device const bfloat* input     [[buffer(2)]],  // [dim] — new token to append
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.dim) return;

    const uint ks = params.kernel_size;
    const uint dim = params.dim;

    // Shift left: copy history[1..ks-2] to history[0..ks-3].
    for (uint k = 0; k < ks - 2; k++) {
        history[k * dim + gid] = history[(k + 1) * dim + gid];
    }
    // Append current input at the end.
    history[(ks - 2) * dim + gid] = input[gid];
}

// ===========================================================================
// L2 normalization — per head, no learned weights.
//
// For each head h (one threadgroup per head):
//   norm = sqrt(sum_i(data[h*head_dim + i]^2))
//   data[h*head_dim + i] /= max(norm, eps)
//
// Unlike RMSNorm, L2 norm has no learned weight vector — it just normalizes
// each head's vector to unit length.  This stabilizes the DeltaNet recurrence
// by preventing the state matrix from growing unboundedly.
// ===========================================================================

struct L2NormParams {
    uint num_heads;
    uint head_dim;
    uint elem_offset;  // element offset from start of buffer (for sub-buffer access)
};

kernel void l2_normalize_heads(
    constant L2NormParams& params [[buffer(0)]],
    device bfloat* data           [[buffer(1)]],  // [num_heads * head_dim], modified in place
    uint head_id                  [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (head_id >= params.num_heads) return;

    const uint hd = params.head_dim;
    device bfloat* head_data = data + params.elem_offset + head_id * hd;

    // Phase 1: accumulate sum of squares (strided across threads).
    float sum_sq = 0.0f;
    for (uint i = tid; i < hd; i += tg_size) {
        float val = float(head_data[i]);
        sum_sq += val * val;
    }

    // Phase 2: SIMD reduction.
    sum_sq = simd_sum(sum_sq);

    // Phase 3: cross-SIMD reduction via shared memory.
    threadgroup float shared[32];
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

    if (simd_lane_id == 0) shared[simd_group_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_simd_groups) ? shared[simd_lane_id] : 0.0f;
        val = simd_sum(val);
        if (simd_lane_id == 0) shared[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: normalize.  Use rsqrt with small epsilon to avoid division by zero.
    float inv_norm = rsqrt(shared[0] + 1e-12f);
    for (uint i = tid; i < hd; i += tg_size) {
        head_data[i] = bfloat(float(head_data[i]) * inv_norm);
    }
}

// ===========================================================================
// Element-wise sigmoid: out[i] = 1 / (1 + exp(-input[i]))
//
// Used for computing decay gate (alpha) and update gate (beta) in DeltaNet.
// Input is f32, output is f32 (gates are small per-head scalars that stay
// in f32 for precision).
// ===========================================================================

struct SigmoidParams {
    uint size;
};

kernel void sigmoid_kernel(
    constant SigmoidParams& params [[buffer(0)]],
    device const bfloat* input     [[buffer(1)]],  // bf16 input (from matmul output)
    device float* output           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    float x = input[gid];
    output[gid] = 1.0f / (1.0f + exp(-x));
}

// ===========================================================================
// DeltaNet decay gate: Mamba-style discretized decay.
//
// For each head i:
//   dt = softplus(x[i] + dt_bias[i])       — positive timestep
//   A  = -exp(A_log[i])                     — negative continuous decay rate
//   g  = exp(dt * A)                        — discrete decay ∈ (0, 1)
//
// x is bf16 (from matmul), dt_bias and A_log are f32, output is f32.
// ===========================================================================

struct DecayGateParams {
    uint size;
};

kernel void deltanet_decay_gate(
    constant DecayGateParams& params [[buffer(0)]],
    device const bfloat* x           [[buffer(1)]],  // [size] bf16 — projected hidden
    device const float* dt_bias      [[buffer(2)]],   // [size] f32
    device const float* A_log        [[buffer(3)]],   // [size] f32
    device float* output             [[buffer(4)]],   // [size] f32
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    // softplus(x + dt_bias) = log(1 + exp(x + dt_bias))
    float val = float(x[gid]) + dt_bias[gid];
    float dt = log(1.0f + exp(val));
    // A = -exp(A_log), so dt * A = -dt * exp(A_log)
    float decay = exp(-dt * exp(A_log[gid]));
    output[gid] = decay;
}

// Shared params struct for element-wise kernels (SiLU, multiply).
struct ElemParams {
    uint size;
};

// ===========================================================================
// Element-wise sigmoid (bfloat16→bfloat16): out[i] = sigmoid(input[i])
//
// Used for the GQA output gate: attn_out *= sigmoid(z).
// Unlike sigmoid_kernel (bf16→f32), this keeps everything in bf16.
// ===========================================================================

kernel void sigmoid_bf16(
    constant ElemParams& params [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    float x = float(input[gid]);
    output[gid] = bfloat(1.0f / (1.0f + exp(-x)));
}

// ===========================================================================
// Element-wise SiLU on bfloat16: out[i] = silu(input[i])
//
// Used for the output gate: z_gate = silu(in_proj_z(x)).
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ===========================================================================

kernel void silu_elementwise(
    constant ElemParams& params [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    float x = float(input[gid]);
    output[gid] = bfloat(x / (1.0f + exp(-x)));
}

// ===========================================================================
// Element-wise multiply (bfloat16): out[i] = a[i] * b[i]
//
// Used for output gating: output = rmsnorm(deltanet_out) * silu(z).
// ===========================================================================

kernel void mul_elementwise(
    constant ElemParams& params [[buffer(0)]],
    device const bfloat* a      [[buffer(1)]],
    device const bfloat* b      [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    output[gid] = bfloat(float(a[gid]) * float(b[gid]));
}

// ===========================================================================
// DeltaNet recurrent state update + output — single token, one threadgroup
// per QK-head.
//
// LEARNING OVERVIEW
//
// This is the core DeltaNet kernel.  For each QK-head h:
//
//   1. Compute retrieval: r = S^T @ k_h                [head_dim → head_dim]
//      (What the state currently "remembers" for this key.)
//
//   2. Compute delta for each associated V-head:
//      For each v in v_heads mapped to this qk_head:
//        delta_v = beta * (v_v - r)                    [head_dim]
//        (Error between desired value and current retrieval.)
//
//   3. Update state for each V-head's submatrix:
//        S_v = alpha * S_v + k_h ⊗ delta_v            [head_dim × head_dim]
//        (Rank-1 update: decay old state, add new association.)
//
//   4. Compute output for each V-head:
//        o_v = S_v^T @ q_h                             [head_dim]
//
// The state S is [num_qk_heads, v_per_qk, head_dim, head_dim] in f32.
// Q/K/V inputs are bf16, output is bf16.
//
// Thread strategy: one threadgroup (256 threads) per QK-head.
// Each thread handles multiple rows of the head_dim × head_dim state matrix.
// With head_dim=128 and 256 threads, each thread handles ~64 state elements
// for each of the v_per_qk=3 V-heads (for 27B: 48V/16QK = 3).
// ===========================================================================

struct DeltaNetStepParams {
    uint num_qk_heads;   // 16 for 27B
    uint num_v_heads;    // 48 for 27B
    uint head_dim;       // 128
    uint q_elem_offset;  // element offset for Q in the Q buffer
    uint k_elem_offset;  // element offset for K in the K buffer
    uint v_elem_offset;  // element offset for V in the V buffer
};

kernel void deltanet_step(
    constant DeltaNetStepParams& params [[buffer(0)]],
    device float* state          [[buffer(1)]],  // [num_qk_heads, v_per_qk, head_dim, head_dim] f32
    device const bfloat* q       [[buffer(2)]],  // [num_qk_heads * head_dim] bf16
    device const bfloat* k       [[buffer(3)]],  // [num_qk_heads * head_dim] bf16
    device const bfloat* v       [[buffer(4)]],  // [num_v_heads * head_dim] bf16
    device const float* alpha    [[buffer(5)]],  // [num_v_heads] f32 — per-head decay gates
    device const float* beta     [[buffer(6)]],  // [num_v_heads] f32 — per-head update gates
    device bfloat* output        [[buffer(7)]],  // [num_v_heads * head_dim] bf16
    uint qk_head_id              [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    if (qk_head_id >= params.num_qk_heads) return;

    const uint hd = params.head_dim;
    const uint v_per_qk = params.num_v_heads / params.num_qk_heads;

    // Pointers to this QK-head's Q and K vectors (with element offsets for sub-buffer access).
    device const bfloat* q_h = q + params.q_elem_offset + qk_head_id * hd;
    device const bfloat* k_h = k + params.k_elem_offset + qk_head_id * hd;

    // Load Q and K for this head into shared memory so ALL threads can read.
    // Thread-local arrays won't work here because each thread only loads its
    // own element but the inner loop needs all head_dim values.
    //
    // Query scaling: HF applies query *= 1/sqrt(head_dim) after L2 norm.
    // This prevents the output magnitude from growing with head_dim.
    threadgroup float shared_q[128];  // max head_dim
    threadgroup float shared_k[128];
    const float q_scale = rsqrt(float(hd));
    for (uint i = tid; i < hd; i += tg_size) {
        shared_q[i] = float(q_h[i]) * q_scale;
        shared_k[i] = float(k_h[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each V-head associated with this QK-head.
    for (uint vi = 0; vi < v_per_qk; vi++) {
        uint v_head_idx = qk_head_id * v_per_qk + vi;
        device const bfloat* v_h = v + params.v_elem_offset + v_head_idx * hd;

        // State matrix for this (qk_head, v_head) pair.
        // Layout: state[qk_head * v_per_qk * hd * hd + vi * hd * hd + row * hd + col]
        device float* S = state + (qk_head_id * v_per_qk + vi) * hd * hd;

        float a = alpha[v_head_idx];  // decay gate for this v-head
        float b = beta[v_head_idx];   // update gate for this v-head

        // Step 1: Decay state and compute retrieval.
        //   S_decayed = a * S
        //   r = S_decayed^T @ k = a * (S^T @ k)
        //
        // HF reference (torch_recurrent_gated_delta_rule):
        //   last_recurrent_state = last_recurrent_state * g_t   (decay)
        //   kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)      (retrieve with plain k)
        //   delta = (v - kv_mem) * beta                          (error * learning rate)
        //   state += k.unsqueeze(-1) * delta.unsqueeze(-2)       (rank-1 update)
        threadgroup float shared_r[128];  // max head_dim

        for (uint j = tid; j < hd; j += tg_size) {
            float r_j = 0.0f;
            for (uint i = 0; i < hd; i++) {
                r_j += S[i * hd + j] * shared_k[i];
            }
            shared_r[j] = a * r_j;  // retrieval from DECAYED state, plain k
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Steps 2-4: delta, state update, and output combined.
        // delta = beta * (v - r)    (error * learning rate)
        // S = a * S + k ⊗ delta    (decay + rank-1 update)
        // o = S^T @ q              (output with scaled query)
        for (uint j = tid; j < hd; j += tg_size) {
            float v_j = float(v_h[j]);
            float delta_j = b * (v_j - shared_r[j]);

            float o_j = 0.0f;
            for (uint i = 0; i < hd; i++) {
                float s_ij = a * S[i * hd + j] + shared_k[i] * delta_j;
                S[i * hd + j] = s_ij;
                o_j += s_ij * shared_q[i];
            }
            output[v_head_idx * hd + j] = bfloat(o_j);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// RMSNorm without learned weight — used for DeltaNet output normalization.
//
// out[i] = input[i] / sqrt(mean(input^2) + eps)
//
// Unlike the standard rms_norm kernel, this one does NOT multiply by a
// learned weight vector.  Used in the DeltaNet output path where the gated
// output is normalized before multiplying by the SiLU gate.
// ===========================================================================

struct RmsNormNoWeightParams {
    uint size;
    float eps;
};

kernel void rms_norm_no_weight(
    constant RmsNormNoWeightParams& params [[buffer(0)]],
    device const bfloat* input             [[buffer(1)]],
    device bfloat* output                  [[buffer(2)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint tg_size                           [[threads_per_threadgroup]]
) {
    const uint size = params.size;

    // Phase 1: sum of squares.
    float sum_sq = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = float(input[i]);
        sum_sq += val * val;
    }

    // Phase 2: SIMD reduction.
    sum_sq = simd_sum(sum_sq);

    // Phase 3: cross-SIMD reduction.
    threadgroup float shared[32];
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

    if (simd_lane_id == 0) shared[simd_group_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_simd_groups) ? shared[simd_lane_id] : 0.0f;
        val = simd_sum(val);
        if (simd_lane_id == 0) shared[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: normalize (no weight multiplication).
    float mean_sq = shared[0] / float(size);
    float scale = rsqrt(mean_sq + params.eps);

    for (uint i = tid; i < size; i += tg_size) {
        output[i] = bfloat(float(input[i]) * scale);
    }
}
