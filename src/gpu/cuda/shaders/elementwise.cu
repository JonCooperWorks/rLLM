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

    // Convert bf16 logits to f32 (max 256 experts).
    float vals[256];
    for (unsigned int i = 0; i < n && i < 256; i++) {
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
