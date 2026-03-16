// ===========================================================================
// Rotary Positional Embeddings (RoPE) CUDA kernels.
//
// LEARNING OVERVIEW
//
// Port of the Metal rope.metal kernels to CUDA for NVIDIA GPUs.
//
// Applies position-dependent 2D rotations to Query and Key vectors,
// encoding absolute position information into the attention computation.
// Uses the HALVED pairing convention (HuggingFace compatible):
//   element i pairs with element i + D/2 within each head.
//
// Three variants:
//   rotary_embedding        — single-token decode, fixed position
//   rotary_embedding_batch  — prefill with per-token position array
//   rotary_embedding_partial — Qwen 3.5 GQA partial rotation
//
// CUDA vs Metal differences:
//   - One thread per (head, pair) — identical dispatch model.
//   - CUDA `__bfloat162float`/`__float2bfloat16` replace Metal casts.
//   - CUDA `sinf`/`cosf`/`powf` replace Metal `sin`/`cos`/`pow`.
//
// Related files:
//   Metal shader:  metal/shaders/rope.metal
//   CUDA bridge:   cuda/kernels/rope.rs
//   Trait contract: gpu/ops/rope.rs
// ===========================================================================

#include <cuda_bf16.h>

struct RopeParams {
    unsigned int pos;
    float rope_theta;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
};

extern "C" __global__ void rotary_embedding(
    const RopeParams params,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k
) {
    const unsigned int half_dim = params.head_dim / 2;
    const unsigned int q_pairs = params.num_heads * half_dim;
    const unsigned int total_pairs = q_pairs + params.num_kv_heads * half_dim;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= total_pairs) return;

    __nv_bfloat16* data;
    unsigned int pair_within;
    if (gid < q_pairs) {
        data = q;
        pair_within = gid;
    } else {
        data = k;
        pair_within = gid - q_pairs;
    }

    unsigned int head_idx = pair_within / half_dim;
    unsigned int pair_in_head = pair_within % half_dim;

    float freq_exp = 2.0f * (float)pair_in_head / (float)params.head_dim;
    float inv_freq = 1.0f / powf(params.rope_theta, freq_exp);
    float angle = (float)params.pos * inv_freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int head_offset = head_idx * params.head_dim;
    unsigned int idx_a = head_offset + pair_in_head;
    unsigned int idx_b = head_offset + pair_in_head + half_dim;
    float a = __bfloat162float(data[idx_a]);
    float b = __bfloat162float(data[idx_b]);
    data[idx_a] = __float2bfloat16(a * cos_val - b * sin_val);
    data[idx_b] = __float2bfloat16(a * sin_val + b * cos_val);
}

// ===========================================================================
// Batched RoPE — per-token positions from a positions buffer.
// ===========================================================================

struct RopeBatchParams {
    unsigned int batch_size;
    float rope_theta;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
};

extern "C" __global__ void rotary_embedding_batch(
    const RopeBatchParams params,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const unsigned int* __restrict__ positions
) {
    const unsigned int half_dim = params.head_dim / 2;
    const unsigned int q_pairs = params.num_heads * half_dim;
    const unsigned int k_pairs = params.num_kv_heads * half_dim;
    const unsigned int pairs_per_token = q_pairs + k_pairs;
    const unsigned int total = params.batch_size * pairs_per_token;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= total) return;

    unsigned int batch = gid / pairs_per_token;
    unsigned int within = gid % pairs_per_token;
    unsigned int pos = positions[batch];

    __nv_bfloat16* data;
    unsigned int pair_within;
    unsigned int q_dim = params.num_heads * params.head_dim;
    unsigned int k_dim = params.num_kv_heads * params.head_dim;

    if (within < q_pairs) {
        data = q + batch * q_dim;
        pair_within = within;
    } else {
        data = k + batch * k_dim;
        pair_within = within - q_pairs;
    }

    unsigned int head_idx = pair_within / half_dim;
    unsigned int pair_in_head = pair_within % half_dim;

    float freq_exp = 2.0f * (float)pair_in_head / (float)params.head_dim;
    float inv_freq = 1.0f / powf(params.rope_theta, freq_exp);
    float angle = (float)pos * inv_freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int head_offset = head_idx * params.head_dim;
    unsigned int idx_a = head_offset + pair_in_head;
    unsigned int idx_b = head_offset + pair_in_head + half_dim;
    float a = __bfloat162float(data[idx_a]);
    float b = __bfloat162float(data[idx_b]);
    data[idx_a] = __float2bfloat16(a * cos_val - b * sin_val);
    data[idx_b] = __float2bfloat16(a * sin_val + b * cos_val);
}

// ===========================================================================
// Partial RoPE — only rotate first `rotary_dim` dimensions per head.
// Used by Qwen 3.5 GQA layers (partial_rotary_factor=0.25).
// ===========================================================================

struct RopePartialParams {
    unsigned int pos;
    float rope_theta;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int rotary_dim;
};

extern "C" __global__ void rotary_embedding_partial(
    const RopePartialParams params,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k
) {
    const unsigned int half_rotary = params.rotary_dim / 2;
    const unsigned int q_pairs = params.num_heads * half_rotary;
    const unsigned int total_pairs = q_pairs + params.num_kv_heads * half_rotary;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= total_pairs) return;

    __nv_bfloat16* data;
    unsigned int pair_within;
    if (gid < q_pairs) {
        data = q;
        pair_within = gid;
    } else {
        data = k;
        pair_within = gid - q_pairs;
    }

    unsigned int head_idx = pair_within / half_rotary;
    unsigned int pair_in_head = pair_within % half_rotary;

    float freq_exp = 2.0f * (float)pair_in_head / (float)params.rotary_dim;
    float inv_freq = 1.0f / powf(params.rope_theta, freq_exp);
    float angle = (float)params.pos * inv_freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int head_offset = head_idx * params.head_dim;
    unsigned int idx_a = head_offset + pair_in_head;
    unsigned int idx_b = head_offset + pair_in_head + half_rotary;
    float a = __bfloat162float(data[idx_a]);
    float b = __bfloat162float(data[idx_b]);
    data[idx_a] = __float2bfloat16(a * cos_val - b * sin_val);
    data[idx_b] = __float2bfloat16(a * sin_val + b * cos_val);
}
