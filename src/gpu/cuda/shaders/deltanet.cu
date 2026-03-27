// ===========================================================================
// DeltaNet CUDA kernels — Gated DeltaNet linear attention for Qwen 3.5.
//
// LEARNING OVERVIEW
//
// Port of the Metal deltanet.metal kernels to CUDA for NVIDIA GPUs.
//
// Gated DeltaNet replaces softmax attention with a linear recurrence.
// Each DeltaNet head maintains a fixed-size [head_dim, head_dim] state
// matrix — O(1) memory regardless of sequence length.
//
// Kernels in this file:
//   conv1d_depthwise_single — causal depthwise conv + SiLU
//   conv1d_shift_history    — FIFO shift of conv history buffer
//   l2_normalize_heads      — per-head L2 normalization (warp reduction)
//   sigmoid_kernel          — element-wise sigmoid (bf16→f32)
//   sigmoid_bf16            — element-wise sigmoid (bf16→bf16)
//   deltanet_decay_gate     — Mamba-style decay: exp(softplus(x+bias) * -exp(A))
//   silu_elementwise        — element-wise SiLU on bf16
//   mul_elementwise         — element-wise multiply on bf16
//   deltanet_step           — core recurrent state update + output
//   rms_norm_no_weight      — RMSNorm without learned weight
//
// Related files:
//   Metal shader:  metal/shaders/deltanet.metal
//   CUDA bridge:   cuda/kernels/deltanet.rs
//   Trait contract: gpu/ops/deltanet.rs
// ===========================================================================

#include <cuda_bf16.h>

__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ===========================================================================
// Causal depthwise Conv1D — single-token decode with SiLU activation.
// ===========================================================================

struct Conv1dParams {
    unsigned int dim;
    unsigned int kernel_size;
    unsigned int input_offset;
};

extern "C" __global__ void conv1d_depthwise_single(
    const Conv1dParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ history,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.dim) return;

    const unsigned int ks = params.kernel_size;
    const unsigned int dim = params.dim;

    float acc = 0.0f;
    for (unsigned int k_idx = 0; k_idx < ks - 1; k_idx++) {
        acc += __bfloat162float(weight[gid * ks + k_idx]) * __bfloat162float(history[k_idx * dim + gid]);
    }
    acc += __bfloat162float(weight[gid * ks + ks - 1]) * __bfloat162float(input[gid]);

    // SiLU activation.
    float silu_val = acc / (1.0f + expf(-acc));
    output[gid] = __float2bfloat16(silu_val);
}

// ===========================================================================
// Conv1D history shift — discard oldest, append current token.
// ===========================================================================

extern "C" __global__ void conv1d_shift_history(
    const Conv1dParams params,
    __nv_bfloat16* __restrict__ history,
    const __nv_bfloat16* __restrict__ input
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.dim) return;

    const unsigned int ks = params.kernel_size;
    const unsigned int dim = params.dim;

    for (unsigned int k_idx = 0; k_idx < ks - 2; k_idx++) {
        history[k_idx * dim + gid] = history[(k_idx + 1) * dim + gid];
    }
    history[(ks - 2) * dim + gid] = input[params.input_offset + gid];
}

// ===========================================================================
// L2 normalization — per head, no learned weights.
// One block (256 threads) per head, warp reduction for sum-of-squares.
// ===========================================================================

struct L2NormParams {
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int elem_offset;
};

extern "C" __global__ void l2_normalize_heads(
    const L2NormParams params,
    __nv_bfloat16* __restrict__ data
) {
    const unsigned int head_id = blockIdx.x;
    if (head_id >= params.num_heads) return;

    const unsigned int hd = params.head_dim;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    __nv_bfloat16* head_data = data + params.elem_offset + head_id * hd;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hd; i += tg_size) {
        float val = __bfloat162float(head_data[i]);
        sum_sq += val * val;
    }

    sum_sq = warp_sum(sum_sq);

    __shared__ float shared[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_sum(val);
        if (lane_id == 0) shared[0] = val;
    }
    __syncthreads();

    float inv_norm = rsqrtf(shared[0] + 1e-12f);
    for (unsigned int i = tid; i < hd; i += tg_size) {
        head_data[i] = __float2bfloat16(__bfloat162float(head_data[i]) * inv_norm);
    }
}

// ===========================================================================
// Element-wise sigmoid (bf16→f32).
// ===========================================================================

struct SigmoidParams {
    unsigned int size;
};

extern "C" __global__ void sigmoid_kernel(
    const SigmoidParams params,
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float x = __bfloat162float(input[gid]);
    output[gid] = 1.0f / (1.0f + expf(-x));
}

// ===========================================================================
// Element-wise sigmoid (bf16→bf16).
// ===========================================================================

struct ElemParams {
    unsigned int size;
};

extern "C" __global__ void sigmoid_bf16(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float x = __bfloat162float(input[gid]);
    output[gid] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
}

// ===========================================================================
// DeltaNet decay gate: g = exp(softplus(x+dt_bias) * -exp(A_log)).
// ===========================================================================

struct DecayGateParams {
    unsigned int size;
};

extern "C" __global__ void deltanet_decay_gate(
    const DecayGateParams params,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    float* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float val = __bfloat162float(x[gid]) + dt_bias[gid];
    float dt = logf(1.0f + expf(val));
    float decay = expf(-dt * expf(A_log[gid]));
    output[gid] = decay;
}

// ===========================================================================
// Element-wise SiLU on bf16.
// ===========================================================================

extern "C" __global__ void silu_elementwise(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float x = __bfloat162float(input[gid]);
    output[gid] = __float2bfloat16(x / (1.0f + expf(-x)));
}

// ===========================================================================
// Element-wise multiply (bf16).
// ===========================================================================

extern "C" __global__ void mul_elementwise(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    output[gid] = __float2bfloat16(__bfloat162float(a[gid]) * __bfloat162float(b[gid]));
}

// ===========================================================================
// DeltaNet recurrent state update + output — one block per QK-head.
// ===========================================================================

struct DeltaNetStepParams {
    unsigned int num_qk_heads;
    unsigned int num_v_heads;
    unsigned int head_dim;
    unsigned int q_elem_offset;
    unsigned int k_elem_offset;
    unsigned int v_elem_offset;
};

extern "C" __global__ void deltanet_step(
    const DeltaNetStepParams params,
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int qk_head_id = blockIdx.x;
    if (qk_head_id >= params.num_qk_heads) return;

    const unsigned int hd = params.head_dim;
    const unsigned int v_per_qk = params.num_v_heads / params.num_qk_heads;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    const __nv_bfloat16* q_h = q + params.q_elem_offset + qk_head_id * hd;
    const __nv_bfloat16* k_h = k + params.k_elem_offset + qk_head_id * hd;

    // Load Q and K into shared memory.
    __shared__ float shared_q[128];
    __shared__ float shared_k[128];
    float q_scale = rsqrtf((float)hd);
    for (unsigned int i = tid; i < hd; i += tg_size) {
        shared_q[i] = __bfloat162float(q_h[i]) * q_scale;
        shared_k[i] = __bfloat162float(k_h[i]);
    }
    __syncthreads();

    // Process each V-head associated with this QK-head.
    for (unsigned int vi = 0; vi < v_per_qk; vi++) {
        unsigned int v_head_idx = qk_head_id * v_per_qk + vi;
        const __nv_bfloat16* v_h = v + params.v_elem_offset + v_head_idx * hd;
        float* S = state + (qk_head_id * v_per_qk + vi) * hd * hd;

        float a = alpha[v_head_idx];
        float b = beta[v_head_idx];

        // Step 1: Decay state and compute retrieval.
        __shared__ float shared_r[128];

        for (unsigned int j = tid; j < hd; j += tg_size) {
            float r_j = 0.0f;
            for (unsigned int i = 0; i < hd; i++) {
                r_j += S[i * hd + j] * shared_k[i];
            }
            shared_r[j] = a * r_j;
        }
        __syncthreads();

        // Steps 2-4: delta, state update, and output.
        for (unsigned int j = tid; j < hd; j += tg_size) {
            float v_j = __bfloat162float(v_h[j]);
            float delta_j = b * (v_j - shared_r[j]);

            float o_j = 0.0f;
            for (unsigned int i = 0; i < hd; i++) {
                float s_ij = a * S[i * hd + j] + shared_k[i] * delta_j;
                S[i * hd + j] = s_ij;
                o_j += s_ij * shared_q[i];
            }
            output[v_head_idx * hd + j] = __float2bfloat16(o_j);
        }
        __syncthreads();
    }
}

// ===========================================================================
// RMSNorm without learned weight — for DeltaNet output normalization.
// ===========================================================================

struct RmsNormNoWeightParams {
    unsigned int size;
    float eps;
};

extern "C" __global__ void rms_norm_no_weight(
    const RmsNormNoWeightParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int size = params.size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < size; i += tg_size) {
        float val = __bfloat162float(input[i]);
        sum_sq += val * val;
    }

    sum_sq = warp_sum(sum_sq);

    __shared__ float shared[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_sum(val);
        if (lane_id == 0) shared[0] = val;
    }
    __syncthreads();

    float mean_sq = shared[0] / (float)size;
    float scale = rsqrtf(mean_sq + params.eps);

    for (unsigned int i = tid; i < size; i += tg_size) {
        output[i] = __float2bfloat16(__bfloat162float(input[i]) * scale);
    }
}
