// ===========================================================================
// TurboQuant CUDA Kernels — KV cache vector quantization for attention.
//
// Port of the Metal turboquant.metal kernels to CUDA for NVIDIA GPUs.
//
// Five kernels:
//   turbo_quantize_paged        — rotate + quantize one K/V vector into paged pool
//   turbo_quantize_paged_batch  — batched version for prefill
//   turbo_rotate_q              — pre-rotate query for quantized attention
//   turbo_paged_attention       — decode attention with inline dequantization
//   turbo_paged_attention_v_only — asymmetric: BF16 K + TurboQuant V
//
// Related files:
//   Metal shader:    metal/shaders/turboquant.metal
//   CUDA bridge:     cuda/kernels/turboquant.rs
//   Trait contract:  gpu/ops/turboquant.rs
//   Algorithm:       model/turboquant.rs
// ===========================================================================

#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// -----------------------------------------------------------------------
// Warp-level reductions.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// -----------------------------------------------------------------------
// Helper: extract a b-bit code from a packed byte array.
// -----------------------------------------------------------------------
__device__ __forceinline__ unsigned int extract_code(
    const unsigned char* packed, unsigned int idx, unsigned int bits
) {
    unsigned int bit_offset = idx * bits;
    unsigned int byte_idx = bit_offset / 8;
    unsigned int bit_within_byte = bit_offset % 8;

    unsigned int val = packed[byte_idx];
    if (bit_within_byte + bits > 8) {
        val |= ((unsigned int)packed[byte_idx + 1] << 8);
    }
    return (val >> bit_within_byte) & ((1u << bits) - 1u);
}

// -----------------------------------------------------------------------
// Helper: pack codes from shared memory into packed byte array.
// Each thread packs one complete byte — no concurrent writes.
// -----------------------------------------------------------------------
__device__ void pack_codes_shared(
    const unsigned int* codes,
    unsigned char* packed,
    unsigned int tid,
    unsigned int hd,
    unsigned int bits
) {
    unsigned int total_bytes = (hd * bits + 7) / 8;
    if (tid >= total_bytes) return;

    unsigned char val = 0;
    unsigned int byte_bit_start = tid * 8;
    for (unsigned int b = 0; b < 8; b++) {
        unsigned int global_bit = byte_bit_start + b;
        unsigned int code_idx = global_bit / bits;
        unsigned int bit_in_code = global_bit % bits;
        if (code_idx < hd) {
            unsigned int bit_val = (codes[code_idx] >> bit_in_code) & 1u;
            val |= (unsigned char)(bit_val << b);
        }
    }
    packed[tid] = val;
}

// -----------------------------------------------------------------------
// Param structs — must match #[repr(C)] Rust structs byte-for-byte.
// -----------------------------------------------------------------------

struct TurboQuantizeParams {
    unsigned int pos;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int bits;
    unsigned int bytes_per_head_pos;
    unsigned int block_size;
    unsigned int num_centroids;
};

struct TurboQuantizeBatchParams {
    unsigned int batch_size;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int bits;
    unsigned int bytes_per_head_pos;
    unsigned int block_size;
    unsigned int num_centroids;
};

struct TurboRotateQParams {
    unsigned int num_heads;
    unsigned int head_dim;
};

struct TurboPagedAttentionParams {
    unsigned int seq_len;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int bits;
    unsigned int bytes_per_head_pos;
    unsigned int block_size;
    unsigned int num_centroids;
    unsigned int window_size;
    float attn_scale;
    unsigned int has_sinks;
};

// ===========================================================================
// turbo_quantize_paged — rotate + quantize one K/V vector into paged pool.
//
// Dispatch: cfg_blocks(num_kv_heads, head_dim)
// ===========================================================================

extern "C" __global__ void turbo_quantize_paged(
    const TurboQuantizeParams params,
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ pool,
    const unsigned int* __restrict__ block_table,
    const float* __restrict__ pi,
    const float* __restrict__ centroids
) {
    unsigned int kv_head = blockIdx.x;
    if (kv_head >= params.num_kv_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    unsigned int hd = params.head_dim;

    const __nv_bfloat16* x = src + kv_head * hd;

    // --- Step 1: Load input + compute L2 norm. ---
    __shared__ float shared_x[256];
    __shared__ float shared_norm;

    float val = (tid < hd) ? __bfloat162float(x[tid]) : 0.0f;
    shared_x[tid] = val;

    float sq = val * val;
    sq = warp_sum(sq);

    __shared__ float norm_partials[8];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) norm_partials[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float norm_sq = 0.0f;
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 0; i < num_groups; i++) norm_sq += norm_partials[i];
        shared_norm = sqrtf(fmaxf(norm_sq, 1e-12f));
    }
    __syncthreads();

    float inv_norm = 1.0f / shared_norm;

    // --- Step 2: Rotate: y_j = sum_i Pi[j,i] * (x[i] / norm) ---
    float y = 0.0f;
    if (tid < hd) {
        for (unsigned int i = 0; i < hd; i++) {
            y += pi[tid * hd + i] * (shared_x[i] * inv_norm);
        }
    }

    // --- Step 3: Find nearest centroid. ---
    unsigned int best_idx = 0;
    if (tid < hd) {
        float best_dist = INFINITY;
        for (unsigned int c = 0; c < params.num_centroids; c++) {
            float d = fabsf(y - centroids[c]);
            if (d < best_dist) { best_dist = d; best_idx = c; }
        }
    }

    // --- Step 4: Write to paged pool. ---
    unsigned int logical_block = params.pos / params.block_size;
    unsigned int offset_in_block = params.pos % params.block_size;
    unsigned int physical_block = block_table[logical_block];
    unsigned int pool_pos = physical_block * params.block_size + offset_in_block;

    unsigned int bytes_per_pos = params.num_kv_heads * params.bytes_per_head_pos;
    unsigned char* pos_base = pool + pool_pos * bytes_per_pos;
    unsigned char* head_base = pos_base + kv_head * params.bytes_per_head_pos;

    if (tid == 0) {
        __nv_bfloat16 norm_bf16 = __float2bfloat16(shared_norm);
        *((__nv_bfloat16*)head_base) = norm_bf16;
    }

    // Collect codes in shared memory for race-free packing.
    __shared__ unsigned int shared_codes[256];
    shared_codes[tid] = (tid < hd) ? best_idx : 0;
    __syncthreads();

    pack_codes_shared(shared_codes, head_base + 2, tid, hd, params.bits);
}

// ===========================================================================
// turbo_quantize_paged_batch — batched version for prefill.
//
// Dispatch: cfg_blocks(batch_size * num_kv_heads, head_dim)
// ===========================================================================

extern "C" __global__ void turbo_quantize_paged_batch(
    const TurboQuantizeBatchParams params,
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ pool,
    const unsigned int* __restrict__ block_table,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ pi,
    const float* __restrict__ centroids
) {
    unsigned int batch_idx = blockIdx.x / params.num_kv_heads;
    unsigned int kv_head = blockIdx.x % params.num_kv_heads;
    if (batch_idx >= params.batch_size) return;

    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    unsigned int hd = params.head_dim;
    unsigned int kv_dim = params.num_kv_heads * hd;

    const __nv_bfloat16* x = src + batch_idx * kv_dim + kv_head * hd;

    // Load + compute norm.
    __shared__ float shared_x[256];
    __shared__ float shared_norm;
    __shared__ float norm_partials[8];

    float val = (tid < hd) ? __bfloat162float(x[tid]) : 0.0f;
    shared_x[tid] = val;

    float sq = val * val;
    sq = warp_sum(sq);
    unsigned int lane_id = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane_id == 0) norm_partials[warp_id] = sq;
    __syncthreads();

    if (tid == 0) {
        float norm_sq = 0.0f;
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 0; i < num_groups; i++) norm_sq += norm_partials[i];
        shared_norm = sqrtf(fmaxf(norm_sq, 1e-12f));
    }
    __syncthreads();
    float inv_norm = 1.0f / shared_norm;

    // Rotate.
    float y = 0.0f;
    if (tid < hd) {
        for (unsigned int i = 0; i < hd; i++) {
            y += pi[tid * hd + i] * (shared_x[i] * inv_norm);
        }
    }

    // Nearest centroid.
    unsigned int best_idx = 0;
    if (tid < hd) {
        float best_dist = INFINITY;
        for (unsigned int c = 0; c < params.num_centroids; c++) {
            float d = fabsf(y - centroids[c]);
            if (d < best_dist) { best_dist = d; best_idx = c; }
        }
    }

    // Write to paged pool.
    unsigned int pos = positions[batch_idx];
    unsigned int logical_block = pos / params.block_size;
    unsigned int offset_in_block = pos % params.block_size;
    unsigned int physical_block = block_table[logical_block];
    unsigned int pool_pos = physical_block * params.block_size + offset_in_block;

    unsigned int bytes_per_pos = params.num_kv_heads * params.bytes_per_head_pos;
    unsigned char* pos_base = pool + pool_pos * bytes_per_pos;
    unsigned char* head_base = pos_base + kv_head * params.bytes_per_head_pos;

    if (tid == 0) {
        *((__nv_bfloat16*)head_base) = __float2bfloat16(shared_norm);
    }

    __shared__ unsigned int shared_codes[256];
    shared_codes[tid] = (tid < hd) ? best_idx : 0;
    __syncthreads();

    pack_codes_shared(shared_codes, head_base + 2, tid, hd, params.bits);
}

// ===========================================================================
// turbo_rotate_q — pre-rotate query for quantized attention.
//
// Dispatch: cfg_blocks(num_heads, head_dim)
// ===========================================================================

extern "C" __global__ void turbo_rotate_q(
    const TurboRotateQParams params,
    const __nv_bfloat16* __restrict__ q,
    float* __restrict__ q_rot,
    const float* __restrict__ pi
) {
    unsigned int head = blockIdx.x;
    if (head >= params.num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int hd = params.head_dim;

    const __nv_bfloat16* q_head = q + head * hd;
    float* out_head = q_rot + head * hd;

    __shared__ float shared_q[256];
    if (tid < hd) shared_q[tid] = __bfloat162float(q_head[tid]);
    __syncthreads();

    if (tid < hd) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < hd; i++) {
            sum += pi[tid * hd + i] * shared_q[i];
        }
        out_head[tid] = sum;
    }
}

// ===========================================================================
// turbo_paged_attention — paged attention with inline TurboQuant dequantization.
//
// Mirrors the BF16 paged_attention kernel structure:
//   - 256 threads per query head, strided position loop
//   - Online softmax
//   - Q is f32 (pre-rotated by Pi)
//   - K/V read as packed codes + bf16 norm, dequantized via centroid lookup
//   - Pi^T inverse rotation applied at end
//
// Dispatch: cfg_blocks(num_heads, 256)
// ===========================================================================

extern "C" __global__ void turbo_paged_attention(
    const TurboPagedAttentionParams params,
    const float* __restrict__ q_rot,
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const unsigned int* __restrict__ block_table,
    const float* __restrict__ pi_t,
    const float* __restrict__ centroids,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ sinks
) {
    unsigned int head_id = blockIdx.x;
    if (head_id >= params.num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    unsigned int hd = params.head_dim;
    unsigned int seq_len = params.seq_len;
    unsigned int num_kv_heads = params.num_kv_heads;
    unsigned int bits = params.bits;
    unsigned int bytes_per_head_pos = params.bytes_per_head_pos;
    unsigned int block_size = params.block_size;
    unsigned int num_centroids = params.num_centroids;

    // GQA mapping.
    unsigned int heads_per_kv = params.num_heads / num_kv_heads;
    unsigned int kv_head = head_id / heads_per_kv;

    // Load centroids into shared memory.
    __shared__ float shared_centroids[16];
    if (tid < num_centroids) {
        shared_centroids[tid] = centroids[tid];
    }

    // Load q_rot into shared memory.
    __shared__ float q_shared[256];
    if (tid < hd) {
        q_shared[tid] = q_rot[head_id * hd + tid];
    }
    __syncthreads();

    // Per-thread V accumulator + online softmax state.
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;
    float v_acc[256] = {};

    unsigned int bytes_per_pos = num_kv_heads * bytes_per_head_pos;
    float scale = (params.attn_scale != 0.0f) ? params.attn_scale : rsqrtf((float)hd);

    // Block table caching.
    unsigned int prev_logical_block = ~0u;
    unsigned int physical_base = 0;

    // Attention window.
    unsigned int start = 0;
    if (params.window_size > 0 && seq_len > params.window_size) {
        start = seq_len - params.window_size;
    }

    // --- Main position loop ---
    for (unsigned int pos = start + tid; pos < seq_len; pos += tg_size) {
        unsigned int logical_block = pos / block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        unsigned int offset_in_block = pos % block_size;
        unsigned int pool_pos = physical_base + offset_in_block;

        const unsigned char* k_pos_base = k_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;

        float k_norm = __bfloat162float(*(const __nv_bfloat16*)k_pos_base);
        const unsigned char* k_codes = k_pos_base + 2;

        // Score = Q_rot · dequant(K)
        float score = 0.0f;
        for (unsigned int j = 0; j < hd; j++) {
            unsigned int code = extract_code(k_codes, j, bits);
            score += q_shared[j] * shared_centroids[code] * k_norm;
        }
        score *= scale;

        // Online softmax update.
        if (score > local_max) {
            float rescale = expf(local_max - score);
            for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += expf(score - local_max);
        }

        float weight = expf(score - local_max);

        // Accumulate weighted V in rotated space.
        const unsigned char* v_pos_base = v_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;
        float v_norm = __bfloat162float(*(const __nv_bfloat16*)v_pos_base);
        const unsigned char* v_codes = v_pos_base + 2;

        for (unsigned int d = 0; d < hd; d++) {
            unsigned int v_code = extract_code(v_codes, d, bits);
            v_acc[d] += weight * shared_centroids[v_code] * v_norm;
        }
    }

    // Handle attention sinks.
    if (params.has_sinks && tid == 0) {
        float sink_score = __bfloat162float(sinks[head_id]);
        if (sink_score > local_max) {
            float rescale = expf(local_max - sink_score);
            for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += expf(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax + V reduction. ---
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    // Reduce max.
    float smax = warp_max(local_max);
    if (lane_id == 0) shared_max[warp_id] = smax;
    __syncthreads();

    float global_max;
    if (tid == 0) {
        global_max = shared_max[0];
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 1; i < num_groups; i++) {
            global_max = fmaxf(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    __syncthreads();
    global_max = shared_max[0];

    // Rescale.
    float rescale = expf(local_max - global_max);
    local_sum_exp *= rescale;
    for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;

    // Reduce sum_exp.
    float ssum = warp_sum(local_sum_exp);
    if (lane_id == 0) shared_sum[warp_id] = ssum;
    __syncthreads();

    float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 0; i < num_groups; i++) total_sum += shared_sum[i];
        shared_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = shared_sum[0];

    // Normalise V.
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (unsigned int d = 0; d < hd; d++) v_acc[d] *= inv_sum;

    // --- Cross-thread V reduction + Pi^T inverse rotation. ---
    unsigned int num_groups = (tg_size + 31) / 32;

    // SIMD-level reduction.
    for (unsigned int d = 0; d < hd; d++) {
        v_acc[d] = warp_sum(v_acc[d]);
    }

    // Lane 0 of each warp writes partial sums.
    __shared__ float shared_reduce[8 * 256];
    if (lane_id == 0) {
        for (unsigned int d = 0; d < hd; d++) {
            shared_reduce[warp_id * hd + d] = v_acc[d];
        }
    }
    __syncthreads();

    // Final reduction + Pi^T rotation.
    if (tid < hd) {
        float v_out = 0.0f;
        for (unsigned int j = 0; j < hd; j++) {
            float v_rot_j = 0.0f;
            for (unsigned int g = 0; g < num_groups; g++) {
                v_rot_j += shared_reduce[g * hd + j];
            }
            v_out += pi_t[tid * hd + j] * v_rot_j;
        }
        output[head_id * hd + tid] = __float2bfloat16(v_out);
    }
}

// ===========================================================================
// turbo_paged_attention_v_only — asymmetric attention: BF16 K + TurboQuant V.
//
// For models with QKV bias (Qwen2, GPT-OSS), K quantization produces
// correlated errors that softmax amplifies.  V is tolerant because errors
// average out in weighted sums.  This kernel scores Q against BF16 K
// (standard dot product, no rotation) and accumulates turbo-quantized V
// (centroid dequant in rotated space, Pi^T inverse rotation at the end).
//
// Dispatch: cfg_blocks(num_heads, 256)
// ===========================================================================

struct TurboPagedAttentionVOnlyParams {
    unsigned int seq_len;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int bits;              // V quantization bits
    unsigned int kv_dim;            // num_kv_heads * head_dim (for BF16 K addressing)
    unsigned int v_bytes_per_head_pos;  // bytes per V head per position (quantized)
    unsigned int block_size;
    unsigned int num_centroids;
    unsigned int window_size;
    float attn_scale;
    unsigned int has_sinks;
};

extern "C" __global__ void turbo_paged_attention_v_only(
    const TurboPagedAttentionVOnlyParams params,
    const __nv_bfloat16* __restrict__ q,        // [num_heads, head_dim] bf16 (NOT rotated)
    const __nv_bfloat16* __restrict__ k_pool,   // BF16 paged K pool
    const unsigned char* __restrict__ v_pool,    // quantized paged V pool
    const unsigned int* __restrict__ block_table,
    const float* __restrict__ pi_t,             // [head_dim, head_dim] f32
    const float* __restrict__ centroids,        // [num_centroids] f32
    __nv_bfloat16* __restrict__ output,         // [num_heads, head_dim] bf16
    const __nv_bfloat16* __restrict__ sinks     // [num_heads] or dummy
) {
    unsigned int head_id = blockIdx.x;
    if (head_id >= params.num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    unsigned int hd = params.head_dim;
    unsigned int seq_len = params.seq_len;
    unsigned int num_kv_heads = params.num_kv_heads;
    unsigned int bits = params.bits;
    unsigned int kv_dim = params.kv_dim;
    unsigned int v_bytes_per_head_pos = params.v_bytes_per_head_pos;
    unsigned int block_size = params.block_size;
    unsigned int num_centroids = params.num_centroids;

    // GQA: map query head to KV head.
    unsigned int heads_per_kv = params.num_heads / num_kv_heads;
    unsigned int kv_head = head_id / heads_per_kv;

    // Load centroids into shared memory (for V dequant).
    __shared__ float shared_centroids[16];
    if (tid < num_centroids) {
        shared_centroids[tid] = centroids[tid];
    }

    // Load Q into shared memory (bf16, NOT rotated — K is BF16).
    __shared__ float q_shared[256];
    if (tid < hd) {
        q_shared[tid] = __bfloat162float(q[head_id * hd + tid]);
    }
    __syncthreads();

    // Per-thread V accumulator in rotated space + online softmax state.
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;
    float v_acc[256] = {};  // max head_dim; zero-initialised

    // V pool addressing: bytes per position across all V KV heads.
    unsigned int v_bytes_per_pos = num_kv_heads * v_bytes_per_head_pos;

    // Attention scale.
    float scale = (params.attn_scale != 0.0f) ? params.attn_scale : rsqrtf((float)hd);

    // Block table caching.
    unsigned int prev_logical_block = ~0u;
    unsigned int physical_base = 0;

    // Determine attention window.
    unsigned int start = 0;
    if (params.window_size > 0 && seq_len > params.window_size) {
        start = seq_len - params.window_size;
    }

    // --- Main position loop (strided across 256 threads). ---
    for (unsigned int pos = start + tid; pos < seq_len; pos += tg_size) {
        // Block table lookup with caching.
        unsigned int logical_block = pos / block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        unsigned int offset_in_block = pos % block_size;
        unsigned int pool_pos = physical_base + offset_in_block;

        // --- K scoring: standard BF16 dot product (same as paged_attention). ---
        const __nv_bfloat16* k_vec = k_pool + pool_pos * kv_dim + kv_head * hd;
        float score = 0.0f;
        for (unsigned int j = 0; j < hd; j++) {
            score += q_shared[j] * __bfloat162float(k_vec[j]);
        }
        score *= scale;

        // Online softmax update — rescale ALL V dimensions on new max.
        if (score > local_max) {
            float rescale = expf(local_max - score);
            for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += expf(score - local_max);
        }

        float weight = expf(score - local_max);

        // --- V accumulation: turbo-quantized dequant in rotated space. ---
        const unsigned char* v_pos_base = v_pool + pool_pos * v_bytes_per_pos + kv_head * v_bytes_per_head_pos;
        float v_norm = __bfloat162float(*(const __nv_bfloat16*)v_pos_base);
        const unsigned char* v_codes = v_pos_base + 2;

        for (unsigned int d = 0; d < hd; d++) {
            unsigned int v_code = extract_code(v_codes, d, bits);
            v_acc[d] += weight * shared_centroids[v_code] * v_norm;
        }
    }

    // Handle attention sinks.
    if (params.has_sinks && tid == 0) {
        float sink_score = __bfloat162float(sinks[head_id]);
        if (sink_score > local_max) {
            float rescale = expf(local_max - sink_score);
            for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += expf(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax + V reduction (same as turbo_paged_attention). ---
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    // Reduce max.
    float smax = warp_max(local_max);
    if (lane_id == 0) shared_max[warp_id] = smax;
    __syncthreads();

    float global_max;
    if (tid == 0) {
        global_max = shared_max[0];
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 1; i < num_groups; i++) {
            global_max = fmaxf(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    __syncthreads();
    global_max = shared_max[0];

    // Rescale.
    float rescale = expf(local_max - global_max);
    local_sum_exp *= rescale;
    for (unsigned int d = 0; d < hd; d++) v_acc[d] *= rescale;

    // Reduce sum_exp.
    float ssum = warp_sum(local_sum_exp);
    if (lane_id == 0) shared_sum[warp_id] = ssum;
    __syncthreads();

    float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        unsigned int num_groups = (tg_size + 31) / 32;
        for (unsigned int i = 0; i < num_groups; i++) total_sum += shared_sum[i];
        shared_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = shared_sum[0];

    // Normalise V.
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (unsigned int d = 0; d < hd; d++) v_acc[d] *= inv_sum;

    // --- Cross-thread V reduction + Pi^T inverse rotation. ---
    unsigned int num_groups = (tg_size + 31) / 32;

    // Warp-level reduction.
    for (unsigned int d = 0; d < hd; d++) {
        v_acc[d] = warp_sum(v_acc[d]);
    }

    // Lane 0 of each warp writes partial sums.
    __shared__ float shared_reduce[8 * 256];
    if (lane_id == 0) {
        for (unsigned int d = 0; d < hd; d++) {
            shared_reduce[warp_id * hd + d] = v_acc[d];
        }
    }
    __syncthreads();

    // Final reduction + Pi^T rotation.
    if (tid < hd) {
        float v_out = 0.0f;
        for (unsigned int j = 0; j < hd; j++) {
            float v_rot_j = 0.0f;
            for (unsigned int g = 0; g < num_groups; g++) {
                v_rot_j += shared_reduce[g * hd + j];
            }
            v_out += pi_t[tid * hd + j] * v_rot_j;
        }
        output[head_id * hd + tid] = __float2bfloat16(v_out);
    }
}
