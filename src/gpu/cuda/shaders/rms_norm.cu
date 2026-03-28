// ===========================================================================
// RMSNorm (Root Mean Square Layer Normalisation) CUDA kernel.
//
// LEARNING OVERVIEW
//
// Port of the Metal rms_norm.metal kernel to CUDA for NVIDIA GPUs.
//
// What this kernel does:
//   Normalises a hidden-state vector so its root-mean-square magnitude is ~1,
//   then scales each element by a learned weight:
//
//     out[i] = weight[i] * input[i] / sqrt(mean(input²) + eps)
//
//   RMSNorm is used 2×layers+1 times per token (before attention, before FFN,
//   and once before the LM head).  Llama and most modern LLMs prefer it over
//   LayerNorm because it requires only one reduction (sum-of-squares) instead
//   of two (mean + variance).
//
// CUDA vs Metal differences:
//   - Metal uses `simd_sum` (32-lane warp intrinsic); CUDA uses `__shfl_xor_sync`
//     for the same warp-level reduction.
//   - Metal `threadgroup` memory → CUDA `__shared__` memory.
//   - Metal `threadgroup_barrier` → CUDA `__syncthreads`.
//   - CUDA natively supports __nv_bfloat16 via cuda_bf16.h.
//
// Dispatch model:
//   Single: 1 block × 256 threads.
//   Batch:  batch_size blocks × 256 threads/block.
//
// Related files:
//   Metal shader:  metal/shaders/rms_norm.metal
//   CUDA bridge:   cuda/kernels/norm.rs
//   Trait contract: gpu/ops/norm.rs
// ===========================================================================

#include <cuda_bf16.h>

// -----------------------------------------------------------------------
// Warp-level sum reduction using butterfly shuffle.
//
// CUDA equivalent of Metal's `simd_sum`.  Uses `__shfl_xor_sync` to
// exchange values between lanes in a warp (32 threads).  After 5 rounds
// of XOR shuffles (offsets 16, 8, 4, 2, 1), every lane holds the total
// sum of all 32 lanes.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// -----------------------------------------------------------------------
// Warp-level max reduction using butterfly shuffle.
// Same pattern as warp_sum but with fmaxf instead of addition.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Host → GPU parameter block.  Must match Rust `RmsNormParams`.
struct RmsNormParams {
    unsigned int hidden_size;
    float eps;
};

extern "C" __global__ void rms_norm(
    const RmsNormParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int hidden = params.hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    // Phase 1: Each thread accumulates sum-of-squares for its strided elements.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float val = __bfloat162float(input[i]);
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction via shuffle.
    sum_sq = warp_sum(sum_sq);

    // Phase 3: Cross-warp reduction via shared memory.
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

    // All threads compute the scale factor.
    float mean_sq = shared[0] / (float)hidden;
    float scale = rsqrtf(mean_sq + params.eps);

    // Phase 4: Normalise and multiply by learned weight.
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float val = __bfloat162float(input[i]);
        output[i] = __float2bfloat16(val * scale * __bfloat162float(weight[i]));
    }
}

// ===========================================================================
// Batched RMSNorm — one block per row of [batch_size, hidden_size].
// Same algorithm as single-vector, but each block processes one row.
// ===========================================================================

struct RmsNormBatchParams {
    unsigned int hidden_size;
    float eps;
    unsigned int batch_size;
};

extern "C" __global__ void rms_norm_batch(
    const RmsNormBatchParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int row_id = blockIdx.x;
    if (row_id >= params.batch_size) return;

    const unsigned int hidden = params.hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    const __nv_bfloat16* row_in  = input  + row_id * hidden;
    __nv_bfloat16*       row_out = output + row_id * hidden;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float val = __bfloat162float(row_in[i]);
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

    float mean_sq = shared[0] / (float)hidden;
    float scale = rsqrtf(mean_sq + params.eps);

    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float val = __bfloat162float(row_in[i]);
        row_out[i] = __float2bfloat16(val * scale * __bfloat162float(weight[i]));
    }
}

// ===========================================================================
// Fused residual-add + RMSNorm — single vector.
//
// hidden[i] += residual[i], then out[i] = weight[i] * hidden[i] / sqrt(mean(hidden²) + eps)
//
// Saves one full read of the hidden tensor vs separate add() + rms_norm().
// For bandwidth-bound decode steps, this roughly halves memory traffic for
// the two most common consecutive operations in each transformer layer.
//
// Inspired by rvLLM (Andy Norris / m0at): fusing residual + norm eliminates
// redundant memory traffic.  See: https://github.com/m0at/rvllm
// ===========================================================================

extern "C" __global__ void fused_residual_rms_norm(
    const RmsNormParams params,
    __nv_bfloat16* __restrict__ hidden,          // in-place: hidden += residual
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int hidden_size = params.hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    // Phase 1: Add residual in-place and accumulate sum-of-squares.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += tg_size) {
        float h = __bfloat162float(hidden[i]) + __bfloat162float(residual[i]);
        hidden[i] = __float2bfloat16(h);
        sum_sq += h * h;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_sum(sum_sq);

    // Phase 3: Cross-warp reduction.
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

    float mean_sq = shared[0] / (float)hidden_size;
    float scale = rsqrtf(mean_sq + params.eps);

    // Phase 4: Normalise and multiply by learned weight.
    for (unsigned int i = tid; i < hidden_size; i += tg_size) {
        float val = __bfloat162float(hidden[i]);
        output[i] = __float2bfloat16(val * scale * __bfloat162float(weight[i]));
    }
}

// ===========================================================================
// Fused residual-add + RMSNorm — batched: one block per row.
// ===========================================================================

extern "C" __global__ void fused_residual_rms_norm_batch(
    const RmsNormBatchParams params,
    __nv_bfloat16* __restrict__ hidden,          // [batch_size, hidden_size] in-place
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int row_id = blockIdx.x;
    if (row_id >= params.batch_size) return;

    const unsigned int hidden_size = params.hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    __nv_bfloat16* row_hidden    = hidden   + row_id * hidden_size;
    const __nv_bfloat16* row_res = residual + row_id * hidden_size;
    __nv_bfloat16* row_out       = output   + row_id * hidden_size;

    // Phase 1: Add residual in-place and accumulate sum-of-squares.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += tg_size) {
        float h = __bfloat162float(row_hidden[i]) + __bfloat162float(row_res[i]);
        row_hidden[i] = __float2bfloat16(h);
        sum_sq += h * h;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_sum(sum_sq);

    // Phase 3: Cross-warp reduction.
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

    float mean_sq = shared[0] / (float)hidden_size;
    float scale = rsqrtf(mean_sq + params.eps);

    // Phase 4: Normalise and multiply by learned weight.
    for (unsigned int i = tid; i < hidden_size; i += tg_size) {
        float val = __bfloat162float(row_hidden[i]);
        row_out[i] = __float2bfloat16(val * scale * __bfloat162float(weight[i]));
    }
}

// ===========================================================================
// LayerNorm — one block per row of [batch_size, hidden_size].
// Unlike RMSNorm, LayerNorm has two reductions (mean and variance) and
// includes both weight and bias parameters.  Used by SigLIP ViT encoder.
// ===========================================================================

struct LayerNormBatchParams {
    unsigned int hidden_size;
    float eps;
    unsigned int batch_size;
};

extern "C" __global__ void layer_norm_batch(
    const LayerNormBatchParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int row_id = blockIdx.x;
    if (row_id >= params.batch_size) return;

    const unsigned int hidden = params.hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    const __nv_bfloat16* row_in  = input  + row_id * hidden;
    __nv_bfloat16*       row_out = output + row_id * hidden;

    // Phase 1: Compute mean via sum reduction.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        local_sum += __bfloat162float(row_in[i]);
    }

    local_sum = warp_sum(local_sum);

    __shared__ float shared[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_sum(val);
        if (lane_id == 0) shared[0] = val;
    }
    __syncthreads();

    float mean = shared[0] / (float)hidden;

    // Phase 2: Compute variance.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float d = __bfloat162float(row_in[i]) - mean;
        sum_sq += d * d;
    }

    sum_sq = warp_sum(sum_sq);

    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_sum(val);
        if (lane_id == 0) shared[0] = val;
    }
    __syncthreads();

    float scale = rsqrtf(shared[0] / (float)hidden + params.eps);

    // Phase 3: Normalise, scale by weight, add bias.
    for (unsigned int i = tid; i < hidden; i += tg_size) {
        float val = (__bfloat162float(row_in[i]) - mean) * scale;
        row_out[i] = __float2bfloat16(val * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}
