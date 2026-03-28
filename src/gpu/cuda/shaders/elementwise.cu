// ===========================================================================
// Element-wise CUDA kernels: gated activations and tensor arithmetic.
//
// LEARNING OVERVIEW
//
// Port of the Metal elementwise.metal kernels to CUDA for NVIDIA GPUs.
//
// These are one-thread-per-element kernels: no shared memory, no barriers,
// no reductions.  Each thread reads its inputs, computes, and writes output.
//
// Gated activations (SwiGLU and GeGLU) are the FFN activation functions:
//   - SwiGLU (Llama, Qwen, Phi): silu(x) = x * sigmoid(x)
//   - GeGLU  (Gemma 3): gelu(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
//
// On H100 these are pure bandwidth-bound — the GPU reads/writes data faster
// than the small per-element compute cost.
//
// Related files:
//   Metal shader:  metal/shaders/elementwise.metal
//   CUDA bridge:   cuda/kernels/elementwise.rs
//   Trait contract: gpu/ops/elementwise.rs
// ===========================================================================

#include <cuda_bf16.h>

// NVRTC doesn't provide math.h; define INFINITY for top-k selection.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

struct ElemParams {
    unsigned int size;
};

// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
extern "C" __global__ void silu_mul(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float g = __bfloat162float(gate[gid]);
    float u = __bfloat162float(up[gid]);
    float silu = g / (1.0f + expf(-g));
    output[gid] = __float2bfloat16(silu * u);
}

// GeGLU activation: out[i] = gelu(gate[i]) * up[i]
extern "C" __global__ void gelu_mul(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float g = __bfloat162float(gate[gid]);
    float u = __bfloat162float(up[gid]);
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + tanhf(inner));
    output[gid] = __float2bfloat16(gelu * u);
}

// Scalar multiply: out[i] = scalar * input[i]
struct ScalarMulParams {
    unsigned int size;
    float scalar;
};

extern "C" __global__ void scalar_mul(
    const ScalarMulParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    output[gid] = __float2bfloat16(params.scalar * __bfloat162float(input[gid]));
}

// Element-wise addition: out[i] = a[i] + b[i]
extern "C" __global__ void add_tensors(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    output[gid] = __float2bfloat16(__bfloat162float(a[gid]) + __bfloat162float(b[gid]));
}

// Broadcast bias-add: out[i] = input[i] + bias[i % dim]
struct BiasAddParams {
    unsigned int total;
    unsigned int dim;
};

extern "C" __global__ void bias_add(
    const BiasAddParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.total) return;
    output[gid] = __float2bfloat16(__bfloat162float(input[gid]) + __bfloat162float(bias[gid % params.dim]));
}

// Scaled accumulate: dst[i] += scale * src[i]
struct ScaleAddParams {
    unsigned int size;
    float scale;
};

extern "C" __global__ void scale_add(
    const ScaleAddParams params,
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    dst[gid] = __float2bfloat16(__bfloat162float(dst[gid]) + params.scale * __bfloat162float(src[gid]));
}

// Fill tensor with zeros.
extern "C" __global__ void fill_zero(
    const ElemParams params,
    __nv_bfloat16* __restrict__ dst
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    dst[gid] = __float2bfloat16(0.0f);
}

// Clamped SwiGLU: out[i] = clamp(silu(gate[i]) * up[i], -limit, limit)
struct SiluMulClampParams {
    unsigned int size;
    float limit;
};

extern "C" __global__ void silu_mul_clamp(
    const SiluMulClampParams params,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float g = __bfloat162float(gate[gid]);
    float u = __bfloat162float(up[gid]);
    float silu = g / (1.0f + expf(-g));
    float result = silu * u;
    result = fminf(fmaxf(result, -params.limit), params.limit);
    output[gid] = __float2bfloat16(result);
}

// GPU-side top-k selection with softmax for MoE routing.
struct TopKParams {
    unsigned int num_experts;
    unsigned int k;
};

extern "C" __global__ void top_k_softmax(
    const TopKParams params,
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int n = params.num_experts;
    unsigned int k = params.k;

    // Convert bf16 logits to f32 (max 1024 experts).
    float vals[1024];
    for (unsigned int i = 0; i < n && i < 1024; i++) {
        vals[i] = __bfloat162float(logits[i]);
    }

    // Find top-k by iteratively selecting the maximum.
    unsigned int top_indices[32];
    float top_logits[32];
    for (unsigned int j = 0; j < k; j++) {
        float best_val = -INFINITY;
        unsigned int best_idx = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = i;
            }
        }
        top_indices[j] = best_idx;
        top_logits[j] = best_val;
        vals[best_idx] = -INFINITY;
    }

    // Softmax over the top-k logits.
    float max_logit = top_logits[0];
    for (unsigned int j = 1; j < k; j++) {
        max_logit = fmaxf(max_logit, top_logits[j]);
    }
    float exp_sum = 0.0f;
    for (unsigned int j = 0; j < k; j++) {
        top_logits[j] = expf(top_logits[j] - max_logit);
        exp_sum += top_logits[j];
    }

    for (unsigned int j = 0; j < k; j++) {
        output[2 * j]     = (float)top_indices[j];
        output[2 * j + 1] = top_logits[j] / exp_sum;
    }
}

// GPT-OSS gated activation: (clamp(up,-lim,lim) + 1) * clamp(gate,max=lim) * sigmoid(gate*alpha)
//
// NOT standard SwiGLU — uses scaled sigmoid, gate-only upper clamp, and (up + 1) offset.
struct GptOssGatedActParams {
    unsigned int size;
    float alpha;
    float limit;
};

extern "C" __global__ void gpt_oss_gated_act(
    const GptOssGatedActParams params,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float g = __bfloat162float(gate[gid]);
    float u = __bfloat162float(up[gid]);

    // Separate clamping: gate has upper-only clamp, up has both bounds.
    float g_c = fminf(g, params.limit);
    float u_c = fminf(fmaxf(u, -params.limit), params.limit);

    // Gated activation: gate * sigmoid(gate * alpha).
    float glu = g_c / (1.0f + expf(-g_c * params.alpha));

    output[gid] = __float2bfloat16((u_c + 1.0f) * glu);
}

// Plain GELU activation: out[i] = gelu(input[i])
// Used by vision encoder FFN (SigLIP ViT).
extern "C" __global__ void gelu_act(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float x = __bfloat162float(input[gid]);
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
    output[gid] = __float2bfloat16(0.5f * x * (1.0f + tanhf(inner)));
}

// ReLU-squared activation: out[i] = max(0, input[i])²
// Used by Nemotron-H MoE experts instead of SwiGLU.
extern "C" __global__ void relu_squared(
    const ElemParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.size) return;
    float v = __bfloat162float(input[gid]);
    float r = v > 0.0f ? v * v : 0.0f;
    output[gid] = __float2bfloat16(r);
}

// GPU-side top-k selection with sigmoid routing for Nemotron-H MoE.
// DeepSeek-V3 style routing with correction bias for load balancing.
struct TopKSigmoidParams {
    unsigned int num_experts;
    unsigned int k;
    float scaling_factor;
    unsigned int norm_topk_prob;
};

// ===========================================================================
// GPU-side argmax for greedy decoding.
//
// One block per row (batch element), block_size = min(vocab_size, 1024).
// Each thread strides across its row finding a local max, then a shared-memory
// tree reduction produces the per-row argmax.
//
// Output: int32[batch_size] — one token ID per sequence.
//
// This kernel eliminates the dominant DtoH bottleneck in greedy decoding:
// instead of copying [batch_size, vocab_size] bf16 logits to CPU (~37 MB at
// batch=128, vocab=152K), only batch_size × 4 bytes of token IDs transfer.
//
// Inspired by rvLLM (Andy Norris / m0at): GPU-resident greedy decoding.
// See: https://github.com/m0at/rvllm
// ===========================================================================

struct ArgmaxParams {
    unsigned int vocab_size;
    unsigned int batch_size;
};

extern "C" __global__ void argmax_gpu(
    const ArgmaxParams params,
    const __nv_bfloat16* __restrict__ logits,   // [batch_size, vocab_size]
    unsigned int* __restrict__ output            // [batch_size] token IDs
) {
    const unsigned int row = blockIdx.x;
    if (row >= params.batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;
    const unsigned int vocab = params.vocab_size;
    const __nv_bfloat16* row_logits = logits + row * vocab;

    // Phase 1: Each thread finds its local max via strided scan.
    float best_val = -INFINITY;
    unsigned int best_idx = 0;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        float v = __bfloat162float(row_logits[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Phase 2: Warp-level reduction — find max within each 32-thread warp.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Phase 3: Cross-warp reduction via shared memory.
    __shared__ float smem_vals[32];
    __shared__ unsigned int smem_idxs[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        smem_vals[warp_id] = best_val;
        smem_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    // Final reduction in warp 0.
    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        best_val = (lane_id < num_warps) ? smem_vals[lane_id] : -INFINITY;
        best_idx = (lane_id < num_warps) ? smem_idxs[lane_id] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
            unsigned int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            output[row] = best_idx;
        }
    }
}

extern "C" __global__ void top_k_sigmoid(
    const TopKSigmoidParams params,
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ correction_bias,
    float* __restrict__ output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int n = params.num_experts;
    unsigned int k = params.k;

    // Step 1: Compute sigmoid scores from bf16 logits.
    float scores[1024];
    float adjusted[1024];
    for (unsigned int i = 0; i < n && i < 1024; i++) {
        float s = 1.0f / (1.0f + expf(-__bfloat162float(logits[i])));
        scores[i] = s;
        // Step 2: Bias shifts selection without affecting final weights.
        adjusted[i] = s + correction_bias[i];
    }

    // Step 3: Select top-k by adjusted scores.
    unsigned int top_indices[32];
    float top_weights[32];
    for (unsigned int j = 0; j < k; j++) {
        float best_val = -INFINITY;
        unsigned int best_idx = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (adjusted[i] > best_val) {
                best_val = adjusted[i];
                best_idx = i;
            }
        }
        top_indices[j] = best_idx;
        // Step 4: Use ORIGINAL sigmoid score as the routing weight.
        top_weights[j] = scores[best_idx];
        adjusted[best_idx] = -INFINITY;
    }

    // Step 5: Optionally normalize weights to sum to 1.
    if (params.norm_topk_prob != 0) {
        float weight_sum = 0.0f;
        for (unsigned int j = 0; j < k; j++) {
            weight_sum += top_weights[j];
        }
        if (weight_sum > 0.0f) {
            for (unsigned int j = 0; j < k; j++) {
                top_weights[j] /= weight_sum;
            }
        }
    }

    // Step 6: Write (index, weight) pairs with scaling factor applied.
    for (unsigned int j = 0; j < k; j++) {
        output[2 * j]     = (float)top_indices[j];
        output[2 * j + 1] = top_weights[j] * params.scaling_factor;
    }
}
