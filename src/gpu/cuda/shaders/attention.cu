// ===========================================================================
// Grouped-Query Attention (GQA) CUDA kernels — decode + prefill paths.
//
// LEARNING OVERVIEW
//
// Port of the Metal attention.metal kernels to CUDA for NVIDIA GPUs.
//
// Implements the fused single-pass softmax + V accumulation algorithm:
//   - All 256 threads cooperate over KV positions (strided by block size)
//   - Each thread maintains head_dim float accumulators in registers
//   - Online softmax rescales V accumulators when new max is found
//   - After the position loop, cross-thread reduction combines results
//
// CUDA vs Metal differences:
//   - CUDA `__shared__` replaces Metal `threadgroup` memory.
//   - CUDA `__syncthreads()` replaces Metal `threadgroup_barrier`.
//   - CUDA `__shfl_xor_sync` replaces Metal `simd_sum`/`simd_max`.
//   - H100 has 80GB HBM3 at 3.35 TB/s — long-context attention benefits
//     enormously from the higher bandwidth.
//
// Kernel variants:
//   attention              — flat KV cache (reference, unused in production)
//   copy_to_kv_cache       — flat KV cache write
//   copy_to_paged_kv_cache — paged KV cache write (single token)
//   paged_attention        — paged KV cache attention
//   paged_attention_fused  — fused KV write + paged attention
//   copy_to_paged_kv_cache_batch — batched paged KV write (prefill)
//   prefill_attention      — causal self-attention for dense Q/K/V
//
// Related files:
//   Metal shader:  metal/shaders/attention.metal
//   CUDA bridge:   cuda/kernels/attention.rs
//   Trait contract: gpu/ops/attention.rs
// ===========================================================================

#include <cuda_bf16.h>

// NVRTC doesn't provide math.h; define INFINITY for -INFINITY sentinels.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// MAX_HEAD_DIM is injected at compile time via string replacement in backend.rs.
#ifndef MAX_HEAD_DIM
#define MAX_HEAD_DIM 128
#endif

static constexpr unsigned int MAX_HD = MAX_HEAD_DIM;
static constexpr unsigned int MAX_HD_VEC4 = MAX_HD / 4;
static constexpr unsigned int NUM_WARPS = 8;  // = 256 threads / 32 lanes

// -----------------------------------------------------------------------
// Warp-level reductions — CUDA equivalents of Metal simd_sum/simd_max.
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
// Helper: vectorised Q·K dot product.
// Q from shared memory (float), K from global memory (bf16).
// -----------------------------------------------------------------------
__device__ float dot_q_k(
    const float* __restrict__ q_shared,
    const __nv_bfloat16* __restrict__ k_vec,
    unsigned int head_dim
) {
    float acc = 0.0f;
    for (unsigned int i = 0; i < head_dim; i++) {
        acc += q_shared[i] * __bfloat162float(k_vec[i]);
    }
    return acc;
}

// -----------------------------------------------------------------------
// Helper: cross-thread (max, sum_exp) reduction for online softmax.
// Returns (global_max, total_sum) via shared memory.
// -----------------------------------------------------------------------
__device__ float2 reduce_softmax(
    float local_max,
    float local_sum_exp,
    unsigned int tid,
    unsigned int tg_size,
    float* shared_reduce
) {
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    // Reduce max across warps.
    float smax = warp_max(local_max);
    if (lane_id == 0) shared_reduce[warp_id] = smax;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared_reduce[lane_id] : -INFINITY;
        val = warp_max(val);
        if (lane_id == 0) shared_reduce[0] = val;
    }
    __syncthreads();
    float global_max = shared_reduce[0];

    // Adjust and reduce sum_exp.
    local_sum_exp *= expf(local_max - global_max);
    float ssum = warp_sum(local_sum_exp);
    if (lane_id == 0) shared_reduce[32 + warp_id] = ssum;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float val = (lane_id < num_warps) ? shared_reduce[32 + lane_id] : 0.0f;
        val = warp_sum(val);
        if (lane_id == 0) shared_reduce[0] = val;
    }
    __syncthreads();
    float total_sum = shared_reduce[0];

    return make_float2(global_max, total_sum);
}

// -----------------------------------------------------------------------
// Helper: reduce V accumulators across threads and write output.
// -----------------------------------------------------------------------
__device__ void reduce_v_and_write(
    float* v_acc,  // [head_dim] per thread
    float local_max,
    float global_max,
    float total_sum,
    unsigned int tid,
    unsigned int tg_size,
    unsigned int head_dim,
    float* shared_reduce,
    __nv_bfloat16* out_head
) {
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int num_warps = (tg_size + 31) / 32;

    float rescale = expf(local_max - global_max) / total_sum;
    for (unsigned int i = 0; i < head_dim; i++) {
        v_acc[i] *= rescale;
    }

    // Warp-level reduction of V accumulators.
    for (unsigned int i = 0; i < head_dim; i++) {
        v_acc[i] = warp_sum(v_acc[i]);
    }

    // Lane 0 of each warp writes partial sums to shared.
    if (lane_id == 0) {
        for (unsigned int i = 0; i < head_dim; i++) {
            shared_reduce[warp_id * head_dim + i] = v_acc[i];
        }
    }
    __syncthreads();

    // Thread 0..head_dim-1 sums partials and writes output.
    if (tid < head_dim) {
        float sum = 0.0f;
        for (unsigned int g = 0; g < num_warps; g++) {
            sum += shared_reduce[g * head_dim + tid];
        }
        out_head[tid] = __float2bfloat16(sum);
    }
}

// ===========================================================================
// Flat KV cache attention — fused single-pass (reference, kept for testing).
// ===========================================================================

struct AttentionParams {
    unsigned int seq_len;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int window_size;
    float attn_scale;
};

extern "C" __global__ void attention(
    const AttentionParams params,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int seq_len = params.seq_len;
    const unsigned int head_dim = params.head_dim;
    const unsigned int num_kv_heads = params.num_kv_heads;
    const unsigned int heads_per_kv = params.num_heads / num_kv_heads;
    const unsigned int head_id = blockIdx.x;
    const unsigned int kv_head = head_id / heads_per_kv;
    const unsigned int kv_stride = num_kv_heads * head_dim;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrtf((float)head_dim);
    const unsigned int start = (params.window_size > 0 && seq_len > params.window_size)
                               ? (seq_len - params.window_size) : 0;

    const __nv_bfloat16* q_ptr = q + head_id * head_dim;

    // Load Q into shared memory.
    __shared__ float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = __bfloat162float(q_ptr[tid]);
    }
    __syncthreads();

    // Per-thread V accumulators and online softmax state.
    float v_acc[MAX_HD] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    // Fused position loop.
    for (unsigned int pos = start + tid; pos < seq_len; pos += tg_size) {
        const __nv_bfloat16* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        float weight;
        if (score > local_max) {
            float correction = expf(local_max - score);
            for (unsigned int i = 0; i < head_dim; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
            weight = 1.0f;
        } else {
            weight = expf(score - local_max);
            local_sum_exp += weight;
        }
        const __nv_bfloat16* v_vec = v_cache + pos * kv_stride + kv_head * head_dim;
        for (unsigned int i = 0; i < head_dim; i++) {
            v_acc[i] += weight * __bfloat162float(v_vec[i]);
        }
    }

    // Cross-thread reduction.
    __shared__ float shared_reduce[NUM_WARPS * MAX_HD];
    float2 sr = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);

    __nv_bfloat16* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, sr.x, sr.y, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// Flat KV cache write.
// ===========================================================================

struct CopyKvParams {
    unsigned int pos;
    unsigned int num_kv_heads;
    unsigned int head_dim;
};

extern "C" __global__ void copy_to_kv_cache(
    const CopyKvParams params,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ cache
) {
    const unsigned int kv_size = params.num_kv_heads * params.head_dim;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= kv_size) return;
    cache[params.pos * kv_size + gid] = src[gid];
}

// ===========================================================================
// Paged KV cache write (single token).
// ===========================================================================

struct PagedCopyKvParams {
    unsigned int pos;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int block_size;
};

extern "C" __global__ void copy_to_paged_kv_cache(
    const PagedCopyKvParams params,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ pool,
    const unsigned int* __restrict__ block_table
) {
    const unsigned int kv_dim = params.num_kv_heads * params.head_dim;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= kv_dim) return;

    unsigned int logical_block = params.pos / params.block_size;
    unsigned int offset_in_block = params.pos % params.block_size;
    unsigned int physical_block = block_table[logical_block];

    unsigned int pool_idx = (physical_block * params.block_size + offset_in_block) * kv_dim + gid;
    pool[pool_idx] = src[gid];
}

// ===========================================================================
// Paged attention — fused single-pass softmax + V accumulation.
// ===========================================================================

struct PagedAttentionParams {
    unsigned int seq_len;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int block_size;
    unsigned int window_size;
    float attn_scale;
};

extern "C" __global__ void paged_attention(
    const PagedAttentionParams params,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const unsigned int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int seq_len = params.seq_len;
    const unsigned int head_dim = params.head_dim;
    const unsigned int num_kv_heads = params.num_kv_heads;
    const unsigned int heads_per_kv = params.num_heads / num_kv_heads;
    const unsigned int head_id = blockIdx.x;
    const unsigned int kv_head = head_id / heads_per_kv;
    const unsigned int kv_dim = num_kv_heads * head_dim;
    const unsigned int block_size = params.block_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrtf((float)head_dim);
    const unsigned int start = (params.window_size > 0 && seq_len > params.window_size)
                               ? (seq_len - params.window_size) : 0;

    const __nv_bfloat16* q_ptr = q + head_id * head_dim;

    __shared__ float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = __bfloat162float(q_ptr[tid]);
    }
    __syncthreads();

    float v_acc[MAX_HD] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    // Block table lookup caching.
    unsigned int prev_logical_block = 0xFFFFFFFF;
    unsigned int physical_base = 0;

    for (unsigned int pos = start + tid; pos < seq_len; pos += tg_size) {
        unsigned int logical_block = pos / block_size;
        unsigned int offset_in_block = pos % block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        unsigned int pool_row = physical_base + offset_in_block;

        const __nv_bfloat16* k_vec = k_pool + pool_row * kv_dim + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        if (score > local_max) {
            float correction = expf(local_max - score);
            for (unsigned int i = 0; i < head_dim; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += expf(score - local_max);
        }

        float weight = expf(score - local_max);
        const __nv_bfloat16* v_vec = v_pool + pool_row * kv_dim + kv_head * head_dim;
        for (unsigned int i = 0; i < head_dim; i++) {
            v_acc[i] += weight * __bfloat162float(v_vec[i]);
        }
    }

    __shared__ float shared_reduce[NUM_WARPS * MAX_HD];
    float2 sr = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);

    __nv_bfloat16* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, sr.x, sr.y, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// Fused paged KV write + paged attention.
// ===========================================================================

struct PagedAttentionFusedParams {
    unsigned int pos;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int block_size;
    unsigned int window_size;
    float attn_scale;
};

extern "C" __global__ void paged_attention_fused(
    const PagedAttentionFusedParams params,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    const unsigned int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int pos = params.pos;
    const unsigned int seq_len = pos + 1;
    const unsigned int head_dim = params.head_dim;
    const unsigned int num_kv_heads = params.num_kv_heads;
    const unsigned int heads_per_kv = params.num_heads / num_kv_heads;
    const unsigned int kv_head = blockIdx.x / heads_per_kv;
    const unsigned int head_id = blockIdx.x;
    const unsigned int kv_dim = num_kv_heads * head_dim;
    const unsigned int block_size = params.block_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    // Phase 1: Write K and V into paged cache.
    {
        unsigned int logical_block = pos / block_size;
        unsigned int offset_in_block = pos % block_size;
        unsigned int physical_block = block_table[logical_block];
        unsigned int pool_base = (physical_block * block_size + offset_in_block) * kv_dim;

        for (unsigned int d = tid; d < kv_dim; d += tg_size) {
            k_pool[pool_base + d] = k_in[d];
            v_pool[pool_base + d] = v_in[d];
        }
    }
    __threadfence();
    __syncthreads();

    // Phase 2: Fused paged attention.
    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrtf((float)head_dim);
    const unsigned int start = (params.window_size > 0 && seq_len > params.window_size)
                               ? (seq_len - params.window_size) : 0;

    const __nv_bfloat16* q_ptr = q + head_id * head_dim;

    __shared__ float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = __bfloat162float(q_ptr[tid]);
    }
    __syncthreads();

    float v_acc[MAX_HD] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    unsigned int prev_logical_block = 0xFFFFFFFF;
    unsigned int physical_base = 0;

    for (unsigned int pos_i = start + tid; pos_i < seq_len; pos_i += tg_size) {
        unsigned int logical_block = pos_i / block_size;
        unsigned int offset_in_block = pos_i % block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        unsigned int pool_row = physical_base + offset_in_block;

        const __nv_bfloat16* k_vec = k_pool + pool_row * kv_dim + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        float weight;
        if (score > local_max) {
            float correction = expf(local_max - score);
            for (unsigned int i = 0; i < head_dim; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
            weight = 1.0f;
        } else {
            weight = expf(score - local_max);
            local_sum_exp += weight;
        }

        const __nv_bfloat16* v_vec = v_pool + pool_row * kv_dim + kv_head * head_dim;
        for (unsigned int i = 0; i < head_dim; i++) {
            v_acc[i] += weight * __bfloat162float(v_vec[i]);
        }
    }

    __shared__ float shared_reduce[NUM_WARPS * MAX_HD];
    float2 sr = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);

    __nv_bfloat16* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, sr.x, sr.y, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// Batched paged KV cache write (prefill).
// ===========================================================================

struct PagedCopyKvBatchParams {
    unsigned int batch_size;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int block_size;
};

extern "C" __global__ void copy_to_paged_kv_cache_batch(
    const PagedCopyKvBatchParams params,
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ pool,
    const unsigned int* __restrict__ block_table,
    const unsigned int* __restrict__ positions
) {
    const unsigned int kv_dim = params.num_kv_heads * params.head_dim;
    const unsigned int total = params.batch_size * kv_dim;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    unsigned int batch = gid / kv_dim;
    unsigned int d     = gid % kv_dim;
    unsigned int pos   = positions[batch];

    unsigned int logical_block  = pos / params.block_size;
    unsigned int offset_in_block = pos % params.block_size;
    unsigned int physical_block = block_table[logical_block];

    unsigned int pool_idx = (physical_block * params.block_size + offset_in_block) * kv_dim + d;
    pool[pool_idx] = src[batch * kv_dim + d];
}

// ===========================================================================
// Causal prefill attention — fused single-pass softmax + V accumulation.
// One block per (query_position, head).
// ===========================================================================

struct PrefillAttentionParams {
    unsigned int chunk_size;
    unsigned int start_pos;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int head_dim;
    unsigned int window_size;
    float attn_scale;
};

extern "C" __global__ void prefill_attention(
    const PrefillAttentionParams params,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int chunk_size = params.chunk_size;
    const unsigned int head_dim = params.head_dim;
    const unsigned int num_heads = params.num_heads;
    const unsigned int num_kv_heads = params.num_kv_heads;
    const unsigned int heads_per_kv = num_heads / num_kv_heads;
    const unsigned int tg_id = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    unsigned int qi      = tg_id / num_heads;
    unsigned int head_id = tg_id % num_heads;

    if (qi >= chunk_size) return;

    const unsigned int kv_head = head_id / heads_per_kv;
    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrtf((float)head_dim);

    const unsigned int q_stride = num_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* q_ptr = q + qi * q_stride + head_id * head_dim;

    const unsigned int attend_len = qi + 1;
    const unsigned int attend_start = (params.window_size > 0 && attend_len > params.window_size)
                                      ? (attend_len - params.window_size) : 0;

    __shared__ float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = __bfloat162float(q_ptr[tid]);
    }
    __syncthreads();

    float v_acc[MAX_HD] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    for (unsigned int pos = attend_start + tid; pos < attend_len; pos += tg_size) {
        const __nv_bfloat16* k_vec = k + pos * kv_stride + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        if (score > local_max) {
            float correction = expf(local_max - score);
            for (unsigned int i = 0; i < head_dim; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += expf(score - local_max);
        }

        float weight = expf(score - local_max);
        const __nv_bfloat16* v_vec = v + pos * kv_stride + kv_head * head_dim;
        for (unsigned int i = 0; i < head_dim; i++) {
            v_acc[i] += weight * __bfloat162float(v_vec[i]);
        }
    }

    __shared__ float shared_reduce[NUM_WARPS * MAX_HD];
    float2 sr = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);

    __nv_bfloat16* out_ptr = output + qi * q_stride + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, sr.x, sr.y, tid, tg_size, head_dim, shared_reduce, out_ptr);
}
