// ===========================================================================
// Grouped-Query Attention (GQA) kernels — decode-path (flat + paged + fused).
//
// LEARNING OVERVIEW
//
// What these kernels do:
//   Compute the core attention operation for one token during decode:
//     output = softmax(Q · K^T / √d) · V
//   where Q is the current token's query, and K/V come from the KV cache.
//
// Fused single-pass algorithm (replaces the original two-pass approach):
//   The old design had two passes:
//     Pass 1: all 256 threads scan positions to find softmax constants (max, sum_exp)
//     Pass 2: recompute Q·K scores a second time, accumulate weighted V vectors
//   This computed every dot product TWICE.
//
//   The new fused design does everything in a SINGLE pass:
//     - All 256 threads cooperate over KV positions (strided by threadgroup size)
//     - Each thread maintains head_dim float accumulators in registers for the
//       weighted V sum, plus running (max, sum_exp) for online softmax
//     - When a new max is found, existing V accumulators are rescaled by
//       exp(old_max - new_max) — this is the key insight from "online softmax"
//     - After the position loop, cross-thread reduction combines partial results
//
// Why this is faster (5 wins):
//   1. Q·K dot products computed once, not twice — eliminates the entire Pass 2
//      recomputation.  For seq_len=1000 with head_dim=64, that's 64,000 fewer
//      FMA operations per head.
//   2. All 256 threads participate in BOTH softmax and V accumulation.  The old
//      Pass 2 used only head_dim=64 threads (wasting 75% of the threadgroup).
//   3. Q vector loaded into threadgroup shared memory once, then read from fast
//      on-chip memory instead of device memory for every dot product.
//   4. Vectorized bfloat4/float4 loads: 16 vector loads per dot product instead
//      of 64 scalar loads.  4x fewer memory transactions, better bandwidth
//      utilisation on the memory bus.
//   5. The fused paged_attention_fused kernel eliminates 2 kernel launch
//      overheads per layer per token by combining KV cache write + attention
//      into a single dispatch.
//
// Register budget:
//   Each thread holds 16 × float4 = 64 floats = 256 bytes for V accumulators.
//   With 256 threads/threadgroup, that's 64KB total — within Apple Silicon's
//   register file.  The Metal compiler may spill to stack for some threads,
//   but M-series GPUs have fast L1-backed stack access.
//
// Shared memory budget (MAX_HEAD_DIM=256):
//   q_shared[256] = 1024 bytes (Q vector, persistent during position loop)
//   shared_reduce[8 × 256] = 8192 bytes (reused for max/sum reduction then V reduction)
//   Total: 9216 bytes — well within the 32KB threadgroup memory limit.
//   For head_dim=64: effective usage is only ~2304 bytes (256 + 2048).
//
// Grouped-Query Attention (GQA):
//   Llama 3.2 uses 32 query heads but only 8 KV heads — every 4 query heads
//   share one KV head (4:1 ratio).
//   Mapping: kv_head_index = query_head_index / (num_heads / num_kv_heads)
//
// Dispatch model:
//   One threadgroup of 256 threads per query head.  Total threads =
//   num_heads * 256 = 32 * 256 = 8192.
//
// Related files:
//   Trait contract:  gpu/ops/attention.rs
//   Metal impl:     metal/kernels/attention.rs
//   Pipeline setup: metal/backend.rs
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Maximum head dimension supported by the attention kernels.
//
// Used to size threadgroup shared memory and per-thread register arrays at
// compile time (Metal requires constant-size threadgroup allocations).
// Supports all current architectures:
//   64  — Llama 3.2 1B/3B
//   96  — Gemma 3 4B
//   128 — Qwen 2.5, Mistral, Llama 8B, Phi-4
//   256 — Gemma 3 27B
//
// Shared memory budget at MAX_HEAD_DIM=256:
//   q_shared[256]          = 1024 bytes
//   shared_reduce[8 × 256] = 8192 bytes
//   Total: 9216 bytes — well within the 32KB threadgroup limit.
//
// Register budget at MAX_HEAD_DIM=256:
//   v_acc[64] × float4 = 256 floats × 4 bytes = 1024 bytes per thread.
//   With 256 threads: 256KB total.  Apple Silicon has ~96KB per SIMD group
//   (×8 = 768KB), so the compiler may spill a fraction to stack — acceptable
//   since M-series GPUs have fast L1-backed stack access.
// ---------------------------------------------------------------------------
// MAX_HEAD_DIM is injected at compile time via string replacement in backend.rs.
// Two variants are compiled: 128 (Llama, Qwen, Mistral, Phi) and 256 (Gemma).
// See backend.rs for the compilation logic.
#ifndef MAX_HEAD_DIM
#define MAX_HEAD_DIM 128
#endif
constant constexpr uint MAX_HD = MAX_HEAD_DIM;
constant constexpr uint MAX_HD_VEC4 = MAX_HD / 4;
constant constexpr uint NUM_SIMD_GROUPS = 8;  // = 256 threads / 32 lanes

// ---------------------------------------------------------------------------
// Helper: compute vectorised Q·K dot product using float4/bfloat4 loads.
//
// Reads Q from threadgroup shared memory (already float) and K from device
// memory (bfloat16).  Processes 4 elements per iteration — 16 iterations
// for head_dim=64 instead of 64 scalar loads.
//
// Requires head_dim to be a multiple of 4 (true for all supported models:
// 64, 96, 128 are all divisible by 4).
// ---------------------------------------------------------------------------
inline float dot_q_k(
    threadgroup const float* q_shared,
    device const bfloat* k_vec,
    uint head_dim
) {
    threadgroup const float4* q4 = (threadgroup const float4*)q_shared;
    device const bfloat4* k4 = (device const bfloat4*)k_vec;
    float4 acc4 = float4(0.0f);
    for (uint i = 0; i < head_dim / 4; i++) {
        acc4 += q4[i] * float4(k4[i]);
    }
    return acc4.x + acc4.y + acc4.z + acc4.w;
}

// ---------------------------------------------------------------------------
// Helper: cross-thread reduction of (max, sum_exp) for online softmax.
//
// Each thread has a local (max, sum_exp) pair from its subset of positions.
// This function combines them across all 256 threads in the threadgroup:
//   1. SIMD-level reduction (32 threads via hardware simd_max/simd_sum)
//   2. Cross-SIMD reduction (8 groups via shared memory)
//
// Returns (global_max, total_sum) broadcast to all threads.
// Uses shared_reduce[0..63] — caller must ensure this region is available.
// ---------------------------------------------------------------------------
inline float2 reduce_softmax(
    float local_max,
    float local_sum_exp,
    uint tid,
    uint tg_size,
    threadgroup float* shared_reduce
) {
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

    // --- Reduce max across SIMD groups ---
    float smax = simd_max(local_max);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = smax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_groups) ? shared_reduce[simd_lane_id] : -INFINITY;
        val = simd_max(val);
        if (simd_lane_id == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = shared_reduce[0];

    // --- Adjust and reduce sum_exp ---
    local_sum_exp *= exp(local_max - global_max);

    float ssum = simd_sum(local_sum_exp);
    if (simd_lane_id == 0) shared_reduce[32 + simd_group_id] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_groups) ? shared_reduce[32 + simd_lane_id] : 0.0f;
        val = simd_sum(val);
        if (simd_lane_id == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sum = shared_reduce[0];

    return float2(global_max, total_sum);
}

// ---------------------------------------------------------------------------
// Helper: cross-thread reduction of V accumulators and output write.
//
// After the fused position loop, each thread holds partial V accumulators
// (v_acc[16] as float4, = 64 floats) from its subset of positions.  We need
// to sum these across all 256 threads per output dimension.
//
// Algorithm:
//   1. Each thread rescales its accumulators to the global max and normalises
//      by total_sum (converting from unnormalised weighted sums to final
//      softmax-weighted sums).
//   2. SIMD-level reduction: simd_sum() within each of 8 SIMD groups of 32.
//   3. Lane 0 of each group writes 64 partial sums to shared memory.
//   4. After barrier, SIMD group 0 (lanes 0..63) sums the 8 partials per
//      output dimension and writes the final bf16 result.
//
// Uses shared_reduce[0..511] = [8 SIMD groups × 64 dims].
// ---------------------------------------------------------------------------
inline void reduce_v_and_write(
    float4 v_acc[16],
    float local_max,
    float global_max,
    float total_sum,
    uint tid,
    uint tg_size,
    uint head_dim,
    threadgroup float* shared_reduce,
    device bfloat* out_head
) {
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;
    uint num_groups = (tg_size + 31) / 32;  // = 8 for tg_size=256

    // Rescale this thread's accumulators: adjust from local_max to global_max,
    // then divide by total_sum to get final softmax-weighted values.
    float rescale = exp(local_max - global_max) / total_sum;
    for (uint i = 0; i < head_dim / 4; i++) {
        v_acc[i] *= rescale;
    }

    // SIMD-level reduction: sum accumulators across 32 lanes in each group.
    // After this, lane 0 of each group has the group's partial sum for each dim.
    for (uint i = 0; i < head_dim / 4; i++) {
        v_acc[i].x = simd_sum(v_acc[i].x);
        v_acc[i].y = simd_sum(v_acc[i].y);
        v_acc[i].z = simd_sum(v_acc[i].z);
        v_acc[i].w = simd_sum(v_acc[i].w);
    }

    // Lane 0 of each SIMD group writes its 64 partial sums to shared memory.
    // Layout: shared_reduce[group * head_dim + d] for d in 0..head_dim-1.
    if (simd_lane_id == 0) {
        for (uint i = 0; i < head_dim / 4; i++) {
            uint base = simd_group_id * head_dim + i * 4;
            shared_reduce[base + 0] = v_acc[i].x;
            shared_reduce[base + 1] = v_acc[i].y;
            shared_reduce[base + 2] = v_acc[i].z;
            shared_reduce[base + 3] = v_acc[i].w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD group 0, lanes 0..head_dim-1: sum the 8 partial sums per dimension.
    // This is the final reduction — each lane produces one output element.
    if (tid < head_dim) {
        float sum = 0.0f;
        for (uint g = 0; g < num_groups; g++) {
            sum += shared_reduce[g * head_dim + tid];
        }
        out_head[tid] = bfloat(sum);
    }
}

// ===========================================================================
// Host → GPU parameter block.  Must match Rust `AttentionParams`.
// ===========================================================================
struct AttentionParams {
    uint seq_len;       // Number of tokens in KV cache (including current).
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8, shared via GQA).
    uint head_dim;      // Dimension per head (64).
    uint window_size;   // Sliding window size (0 = attend to full context).
    float attn_scale;   // Custom attention scale (0 = use default 1/√head_dim).
};

// ===========================================================================
// Flat KV cache attention — fused single-pass softmax + V accumulation.
//
// This kernel is the flat-cache variant (currently unused in production, kept
// for reference and testing).  The paged variant below is the active path.
// ===========================================================================

kernel void attention(
    constant AttentionParams& params [[buffer(0)]],
    device const bfloat* q           [[buffer(1)]],  // [num_heads, head_dim]
    device const bfloat* k_cache     [[buffer(2)]],  // [max_seq_len, num_kv_heads, head_dim]
    device const bfloat* v_cache     [[buffer(3)]],  // [max_seq_len, num_kv_heads, head_dim]
    device bfloat* output            [[buffer(4)]],  // [num_heads, head_dim]
    uint head_id                     [[threadgroup_position_in_grid]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_dim = params.head_dim;
    const uint num_kv_heads = params.num_kv_heads;
    const uint heads_per_kv = params.num_heads / num_kv_heads;
    const uint kv_head = head_id / heads_per_kv;
    const uint kv_stride = num_kv_heads * head_dim;

    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrt(float(head_dim));
    const uint start = (params.window_size > 0 && seq_len > params.window_size)
                       ? (seq_len - params.window_size) : 0;

    device const bfloat* q_ptr = q + head_id * head_dim;

    // --- Load Q into threadgroup shared memory ---
    // All 256 threads will read Q repeatedly during the position loop.
    // Loading once into shared memory avoids redundant device memory reads.
    threadgroup float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Per-thread V accumulators and online softmax state ---
    // Each thread processes a subset of positions (strided by tg_size) and
    // maintains the FULL head_dim output vector in registers.  This is the
    // key difference from the old two-pass approach: we accumulate V weights
    // as we go, so we never need to recompute Q·K scores.
    float4 v_acc[MAX_HD_VEC4] = {};  // head_dim floats as float4s, zero-initialised
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    const uint hd4 = head_dim / 4;  // Number of float4 iterations for this model

    // --- Fused position loop: softmax + V accumulation in one pass ---
    for (uint pos = start + tid; pos < seq_len; pos += tg_size) {
        // Vectorised Q·K dot product from shared memory.
        device const bfloat* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        // Online softmax update with V accumulator rescaling.
        // When a new max is found, rescale existing accumulators and use
        // weight=1.0 (since exp(score - new_max) = exp(0) = 1.0).
        // Otherwise compute the exponential weight normally.
        float weight;
        if (score > local_max) {
            float correction = exp(local_max - score);
            for (uint i = 0; i < hd4; i++) {
                v_acc[i] *= correction;
            }
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
            weight = 1.0f;
        } else {
            weight = exp(score - local_max);
            local_sum_exp += weight;
        }
        device const bfloat* v_vec = v_cache + pos * kv_stride + kv_head * head_dim;
        device const bfloat4* v4 = (device const bfloat4*)v_vec;
        for (uint i = 0; i < hd4; i++) {
            v_acc[i] += weight * float4(v4[i]);
        }
    }

    // --- Cross-thread softmax reduction ---
    threadgroup float shared_reduce[NUM_SIMD_GROUPS * MAX_HD];
    float2 softmax_result = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);
    float global_max = softmax_result.x;
    float total_sum = softmax_result.y;

    // --- Reduce V accumulators across threads and write output ---
    device bfloat* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, global_max, total_sum, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// KV cache write kernel (flat cache).
//
// Copies a new K or V vector into the flat KV cache at position `pos`.
// This is trivially parallel — each thread copies one element.
//
// Cache layout: [max_seq_len, num_kv_heads, head_dim]
// We write: cache[pos * (num_kv_heads * head_dim) + gid] = src[gid]
// ===========================================================================

struct CopyKvParams {
    uint pos;           // Cache position to write to (0-indexed).
    uint num_kv_heads;  // Number of KV heads.
    uint head_dim;      // Dimension per head.
};

kernel void copy_to_kv_cache(
    constant CopyKvParams& params [[buffer(0)]],
    device const bfloat* src      [[buffer(1)]],
    device bfloat* cache          [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    const uint kv_size = params.num_kv_heads * params.head_dim;
    if (gid >= kv_size) return;
    cache[params.pos * kv_size + gid] = src[gid];
}

// ===========================================================================
// Paged KV cache kernels.
//
// LEARNING OVERVIEW
//
// These kernels work with a block-paged KV cache instead of a flat one.
// The pool is a single large buffer: [num_blocks * BLOCK_SIZE, kv_dim].
// Each sequence has a block table that maps logical block indices to
// physical block indices in the pool.
//
// Address translation:
//   logical_block  = pos / BLOCK_SIZE
//   offset_in_block = pos % BLOCK_SIZE
//   physical_block = block_table[logical_block]
//   pool_index = (physical_block * BLOCK_SIZE + offset_in_block) * kv_dim + d
//
// This indirection adds ~2 extra instructions per position lookup (integer
// divide + table read) but enables dynamic memory allocation and sharing.
// ===========================================================================

struct PagedCopyKvParams {
    uint pos;           // Logical position in the sequence.
    uint num_kv_heads;
    uint head_dim;
    uint block_size;    // = 16
};

kernel void copy_to_paged_kv_cache(
    constant PagedCopyKvParams& params [[buffer(0)]],
    device const bfloat* src           [[buffer(1)]],  // [kv_dim]
    device bfloat* pool                [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device const uint* block_table     [[buffer(3)]],  // [max_blocks_per_seq]
    uint gid                           [[thread_position_in_grid]]
) {
    const uint kv_dim = params.num_kv_heads * params.head_dim;
    if (gid >= kv_dim) return;

    // Address translation: logical pos -> physical location in pool.
    uint logical_block = params.pos / params.block_size;
    uint offset_in_block = params.pos % params.block_size;
    uint physical_block = block_table[logical_block];

    uint pool_idx = (physical_block * params.block_size + offset_in_block) * kv_dim + gid;
    pool[pool_idx] = src[gid];
}

// ===========================================================================
// Paged attention — fused single-pass softmax + V accumulation.
//
// Same fused algorithm as the flat attention kernel above, but K/V reads
// go through block table indirection.  The block table maps logical blocks
// (contiguous per sequence) to physical blocks (scattered across the pool).
//
// Block table lookup caching:
//   Instead of using macros (PAGED_K_VEC/PAGED_V_VEC) that recompute
//   the block table lookup for every access, we cache the current
//   physical_base and only re-read block_table[] when crossing a block
//   boundary (i.e., when logical_block changes).  This saves one table
//   read per position since K and V share the same physical block address.
//   With tg_size=256 and block_size=16, each thread jumps 256 positions
//   between iterations (16 blocks), so the cache rarely hits on the same
//   block — but we still save the duplicate lookup for K and V.
//
// Dispatch model: identical to flat attention — one threadgroup of 256
// threads per query head.
// ===========================================================================

struct PagedAttentionParams {
    uint seq_len;       // Number of tokens in this sequence's KV cache.
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8).
    uint head_dim;      // Dimension per head (64).
    uint block_size;    // = 16
    uint window_size;   // Sliding window (0 = full context).
    float attn_scale;   // Custom scale (0 = default 1/√head_dim).
    uint has_sinks;     // 0 = no sinks, 1 = sinks buffer present (attention sinks).
};

kernel void paged_attention(
    constant PagedAttentionParams& params [[buffer(0)]],
    device const bfloat* q               [[buffer(1)]],  // [num_heads, head_dim]
    device const bfloat* k_pool          [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device const bfloat* v_pool          [[buffer(3)]],  // [num_blocks * block_size, kv_dim]
    device const uint* block_table       [[buffer(4)]],  // [max_blocks_per_seq]
    device bfloat* output                [[buffer(5)]],  // [num_heads, head_dim]
    device const bfloat* sinks           [[buffer(6)]],  // [num_heads] — attention sink logits (ignored when has_sinks=0)
    uint head_id                         [[threadgroup_position_in_grid]],
    uint tid                             [[thread_position_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_dim = params.head_dim;
    const uint num_kv_heads = params.num_kv_heads;
    const uint heads_per_kv = params.num_heads / num_kv_heads;
    const uint kv_head = head_id / heads_per_kv;
    const uint kv_dim = num_kv_heads * head_dim;
    const uint block_size = params.block_size;

    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrt(float(head_dim));
    const uint start = (params.window_size > 0 && seq_len > params.window_size)
                       ? (seq_len - params.window_size) : 0;

    device const bfloat* q_ptr = q + head_id * head_dim;

    // --- Load Q into threadgroup shared memory ---
    threadgroup float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Per-thread V accumulators and online softmax state ---
    float4 v_acc[MAX_HD_VEC4] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    const uint hd4 = head_dim / 4;

    // --- Fused position loop with cached block table lookups ---
    // Track the current logical block to avoid redundant block_table[] reads.
    // When the logical block changes, we re-read the physical block from the
    // table.  Since K and V share the same block address, this halves lookups
    // compared to the old PAGED_K_VEC/PAGED_V_VEC macro approach.
    uint prev_logical_block = ~0u;  // Sentinel: forces lookup on first iteration.
    uint physical_base = 0;

    for (uint pos = start + tid; pos < seq_len; pos += tg_size) {
        // Block table lookup with caching.
        uint logical_block = pos / block_size;
        uint offset_in_block = pos % block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        uint pool_row = physical_base + offset_in_block;

        // Vectorised Q·K dot product.
        device const bfloat* k_vec = k_pool + pool_row * kv_dim + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        // Online softmax update with V accumulator rescaling.
        if (score > local_max) {
            float correction = exp(local_max - score);
            for (uint i = 0; i < hd4; i++) {
                v_acc[i] *= correction;
            }
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }

        // Accumulate weighted V vector.
        float weight = exp(score - local_max);
        device const bfloat* v_vec = v_pool + pool_row * kv_dim + kv_head * head_dim;
        device const bfloat4* v4 = (device const bfloat4*)v_vec;
        for (uint i = 0; i < hd4; i++) {
            v_acc[i] += weight * float4(v4[i]);
        }
    }

    // --- Attention sinks: add sink score to softmax denominator ---
    // Sinks are per-head scalar logits that participate in softmax but have no
    // associated V vector.  They absorb probability mass, gating how much the
    // actual KV content contributes.  Only thread 0 processes the sink to avoid
    // double-counting across the 256 cooperating threads.
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float correction = exp(local_max - sink_score);
            for (uint i = 0; i < hd4; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax reduction ---
    threadgroup float shared_reduce[NUM_SIMD_GROUPS * MAX_HD];
    float2 softmax_result = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);
    float global_max = softmax_result.x;
    float total_sum = softmax_result.y;

    // --- Reduce V accumulators across threads and write output ---
    device bfloat* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, global_max, total_sum, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// Fused paged KV cache write + paged attention.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Combines three operations that were previously separate kernel dispatches
//   into a single dispatch:
//     1. Write K vector to paged cache at position `pos`
//     2. Write V vector to paged cache at position `pos`
//     3. Compute paged attention over all positions 0..pos (inclusive)
//
// Why fusing is beneficial:
//   - Eliminates 2 kernel launch overheads per layer per token.  Metal kernel
//     launches have ~5-10µs overhead each.  With 32 layers, that's 64 saved
//     launches = ~320-640µs per token.
//   - The block table is loaded once (for the KV write) instead of three times
//     (K write, V write, attention each loading it independently).
//   - After writing K/V, the data is already in the GPU's L2 cache, so the
//     attention phase's read of the just-written position is essentially free.
//
// Dispatch model:
//   Same as paged_attention — one threadgroup of 256 per query head.
//   The KV write phase uses threads 0..kv_dim-1 (256 threads, kv_dim=512 for
//   Llama 3.2).  Since kv_dim > 256 for most models, threads loop to cover
//   all elements.  A device_and_threadgroup barrier ensures writes are visible
//   before the attention phase reads them.
//
// Params:
//   `pos` is the write position (0-indexed).  seq_len = pos + 1 is computed
//   internally.  This avoids the off-by-one confusion of the separate API
//   where copy_to_paged_kv_cache(pos=N) is followed by paged_attention(seq_len=N+1).
// ===========================================================================

struct PagedAttentionFusedParams {
    uint pos;           // Current token position (write position for K/V).
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8).
    uint head_dim;      // Dimension per head (64).
    uint block_size;    // = 16
    uint window_size;   // Sliding window (0 = full context).
    float attn_scale;   // Custom scale (0 = default 1/√head_dim).
    uint has_sinks;     // 0 = no sinks, 1 = sinks buffer present (attention sinks).
};

kernel void paged_attention_fused(
    constant PagedAttentionFusedParams& params [[buffer(0)]],
    device const bfloat* q               [[buffer(1)]],  // [num_heads, head_dim]
    device const bfloat* k_in            [[buffer(2)]],  // [num_kv_heads, head_dim] — current token K
    device const bfloat* v_in            [[buffer(3)]],  // [num_kv_heads, head_dim] — current token V
    device bfloat* k_pool                [[buffer(4)]],  // [num_blocks * block_size, kv_dim]
    device bfloat* v_pool                [[buffer(5)]],  // [num_blocks * block_size, kv_dim]
    device const uint* block_table       [[buffer(6)]],  // [max_blocks_per_seq]
    device bfloat* output                [[buffer(7)]],  // [num_heads, head_dim]
    device const bfloat* sinks           [[buffer(8)]],  // [num_heads] — attention sink logits (ignored when has_sinks=0)
    uint head_id                         [[threadgroup_position_in_grid]],
    uint tid                             [[thread_position_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]]
) {
    const uint pos = params.pos;
    const uint seq_len = pos + 1;   // After writing, we attend to pos+1 tokens.
    const uint head_dim = params.head_dim;
    const uint num_kv_heads = params.num_kv_heads;
    const uint heads_per_kv = params.num_heads / num_kv_heads;
    const uint kv_head = head_id / heads_per_kv;
    const uint kv_dim = num_kv_heads * head_dim;
    const uint block_size = params.block_size;

    // -------------------------------------------------------------------
    // Phase 1: Write K and V into the paged cache at position `pos`.
    //
    // Address translation: logical pos → physical pool location.
    // All threads with tid < kv_dim participate (loop if kv_dim > tg_size).
    // Only head_id==0's threadgroup needs to write (all heads share KV),
    // but since we need a device barrier anyway and the write is tiny
    // (512 elements), we let every threadgroup write — they all produce
    // the same result (idempotent) and it avoids cross-threadgroup sync.
    // -------------------------------------------------------------------
    {
        uint logical_block = pos / block_size;
        uint offset_in_block = pos % block_size;
        uint physical_block = block_table[logical_block];
        uint pool_base = (physical_block * block_size + offset_in_block) * kv_dim;

        for (uint d = tid; d < kv_dim; d += tg_size) {
            k_pool[pool_base + d] = k_in[d];
            v_pool[pool_base + d] = v_in[d];
        }
    }

    // Barrier: ensure KV writes are visible to all threads before attention reads.
    // mem_device ensures device memory writes are visible; mem_threadgroup for sync.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    // -------------------------------------------------------------------
    // Phase 2: Fused single-pass paged attention (identical to paged_attention).
    // -------------------------------------------------------------------
    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrt(float(head_dim));
    const uint start = (params.window_size > 0 && seq_len > params.window_size)
                       ? (seq_len - params.window_size) : 0;

    device const bfloat* q_ptr = q + head_id * head_dim;

    // Load Q into shared memory.
    threadgroup float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread V accumulators and online softmax state.
    float4 v_acc[MAX_HD_VEC4] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    const uint hd4 = head_dim / 4;

    // Fused position loop with cached block table lookups.
    uint prev_logical_block = ~0u;
    uint physical_base = 0;

    for (uint pos_i = start + tid; pos_i < seq_len; pos_i += tg_size) {
        uint logical_block = pos_i / block_size;
        uint offset_in_block = pos_i % block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        uint pool_row = physical_base + offset_in_block;

        device const bfloat* k_vec = k_pool + pool_row * kv_dim + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        // Online softmax update with V accumulator rescaling.
        // When a new max is found, rescale existing accumulators and use
        // weight=1.0 (since exp(score - new_max) = exp(0) = 1.0).
        // Otherwise compute the exponential weight normally.
        float weight;
        if (score > local_max) {
            float correction = exp(local_max - score);
            for (uint i = 0; i < hd4; i++) {
                v_acc[i] *= correction;
            }
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
            weight = 1.0f;
        } else {
            weight = exp(score - local_max);
            local_sum_exp += weight;
        }

        device const bfloat* v_vec = v_pool + pool_row * kv_dim + kv_head * head_dim;
        device const bfloat4* v4 = (device const bfloat4*)v_vec;
        for (uint i = 0; i < hd4; i++) {
            v_acc[i] += weight * float4(v4[i]);
        }
    }

    // Attention sinks (see paged_attention kernel for detailed comments).
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float correction = exp(local_max - sink_score);
            for (uint i = 0; i < hd4; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // Cross-thread softmax reduction.
    threadgroup float shared_reduce[NUM_SIMD_GROUPS * MAX_HD];
    float2 softmax_result = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);
    float global_max = softmax_result.x;
    float total_sum = softmax_result.y;

    // Reduce V accumulators and write output.
    device bfloat* out_head = output + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, global_max, total_sum, tid, tg_size, head_dim, shared_reduce, out_head);
}

// ===========================================================================
// Batched paged KV cache write.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Writes N K/V vectors at different positions into a paged pool.
//   During prefill, the model computes K and V for ALL prompt tokens at once
//   (via GEMM), producing [batch_size, kv_dim] tensors.  This kernel writes
//   each token's K/V into the paged cache at its correct position.
//
// Why we need this:
//   The GEMM-computed K/V are dense tensors (contiguous in memory), but the
//   paged cache scatters positions across physical blocks.  Each token's
//   position maps to a different physical block via the block table.
//   This kernel does that scatter write: for each (batch, d) element, it
//   looks up the correct physical block and writes to the right offset.
//
//   After this kernel runs, future decode steps can read these K/V values
//   from the paged cache via the standard paged_attention kernel.
//
// Dispatch model:
//   Grid: batch_size * kv_dim total threads.
//   One thread per (batch, dim) element.
// ===========================================================================

struct PagedCopyKvBatchParams {
    uint batch_size;
    uint num_kv_heads;
    uint head_dim;
    uint block_size;    // = 16
};

kernel void copy_to_paged_kv_cache_batch(
    constant PagedCopyKvBatchParams& params [[buffer(0)]],
    device const bfloat* src               [[buffer(1)]],  // [batch_size, kv_dim]
    device bfloat* pool                    [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device const uint* block_table         [[buffer(3)]],  // [max_blocks_per_seq]
    device const uint* positions           [[buffer(4)]],  // [batch_size]
    uint gid                               [[thread_position_in_grid]]
) {
    const uint kv_dim = params.num_kv_heads * params.head_dim;
    const uint total = params.batch_size * kv_dim;
    if (gid >= total) return;

    uint batch = gid / kv_dim;
    uint d     = gid % kv_dim;
    uint pos   = positions[batch];

    uint logical_block  = pos / params.block_size;
    uint offset_in_block = pos % params.block_size;
    uint physical_block = block_table[logical_block];

    uint pool_idx = (physical_block * params.block_size + offset_in_block) * kv_dim + d;
    pool[pool_idx] = src[batch * kv_dim + d];
}

// ===========================================================================
// Causal prefill attention kernel — fused single-pass softmax + V accumulation.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Computes causal self-attention for a chunk of tokens during prefill.
//   Token at query position i can attend to K/V positions 0..=i within the
//   chunk (plus any prior tokens via start_pos offset).
//
//   Q: [chunk_size, num_heads * head_dim]  (dense, from GEMM projections)
//   K: [chunk_size, num_kv_heads * head_dim]
//   V: [chunk_size, num_kv_heads * head_dim]
//   out: [chunk_size, num_heads * head_dim]
//
// Why this is different from decode attention:
//   Decode attention: 1 query token → attend to seq_len cached KV tokens.
//   Prefill attention: chunk_size query tokens, each attending to a DIFFERENT
//   number of key tokens due to the causal mask (no peeking at future tokens).
//
//   Crucially, Q/K/V are DENSE tensors from GEMM projections — not from the
//   paged cache.  This is intentional: during prefill, all tokens are
//   processed simultaneously, so their K/V exist as contiguous [chunk, dim]
//   matrices.  The paged cache is written to separately (by the
//   copy_to_paged_kv_cache_batch kernel) for use by FUTURE decode steps.
//
// Causal masking:
//   In language modelling, token i must not attend to tokens i+1, i+2, ...
//   (it would be "seeing the answer").  For query at chunk index qi, the
//   attend length is (qi + 1): it sees tokens 0, 1, ..., qi.  Token 0 sees
//   only itself; the last token sees the entire prompt.  This is the
//   lower-triangular attention mask.
//
// Fused single-pass algorithm (same as decode kernels):
//   Uses the same online softmax + V accumulation approach as the decode
//   kernels.  All 256 threads cooperate over K positions, each maintaining
//   per-thread V accumulators in registers.  This replaces the old two-pass
//   approach that computed every Q·K dot product twice.
//
// Dispatch model:
//   One threadgroup of 256 per (query_position, head).
//   Grid: chunk_size * num_heads * 256.
//   Each threadgroup computes attention for one query at one head.
// ===========================================================================

struct PrefillAttentionParams {
    uint chunk_size;    // Number of tokens in this prefill chunk.
    uint start_pos;     // Starting position for this chunk (0 for fresh prefill).
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8).
    uint head_dim;      // Dimension per head (64).
    uint window_size;   // Sliding window (0 = full causal attention).
    float attn_scale;   // Custom scale (0 = default 1/√head_dim).
    uint has_sinks;     // 0 = no sinks, 1 = sinks buffer present (attention sinks).
    uint causal;        // 1 = causal mask (LLM text), 0 = bidirectional (vision).
};

kernel void prefill_attention(
    constant PrefillAttentionParams& params [[buffer(0)]],
    device const bfloat* q       [[buffer(1)]],  // [chunk_size, num_heads * head_dim]
    device const bfloat* k       [[buffer(2)]],  // [chunk_size, num_kv_heads * head_dim]
    device const bfloat* v       [[buffer(3)]],  // [chunk_size, num_kv_heads * head_dim]
    device bfloat* output        [[buffer(4)]],  // [chunk_size, num_heads * head_dim]
    device const bfloat* sinks   [[buffer(5)]],  // [num_heads] — attention sink logits (ignored when has_sinks=0)
    uint tg_id                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    const uint chunk_size = params.chunk_size;
    const uint head_dim = params.head_dim;
    const uint num_heads = params.num_heads;
    const uint num_kv_heads = params.num_kv_heads;
    const uint heads_per_kv = num_heads / num_kv_heads;

    // Decompose threadgroup ID into (query_index, head).
    uint qi      = tg_id / num_heads;  // Which query token in the chunk.
    uint head_id = tg_id % num_heads;  // Which query head.

    if (qi >= chunk_size) return;

    const uint kv_head = head_id / heads_per_kv;
    const float scale = (params.attn_scale > 0.0f) ? params.attn_scale : rsqrt(float(head_dim));

    // Strides for the dense Q/K/V tensors.
    const uint q_stride = num_heads * head_dim;      // row stride in Q
    const uint kv_stride = num_kv_heads * head_dim;  // row stride in K/V

    // Pointer to this query token's head vector.
    device const bfloat* q_ptr = q + qi * q_stride + head_id * head_dim;

    // Attention mask: causal (LLM text) or bidirectional (vision encoder).
    //
    // Causal: each query attends to chunk positions 0..=qi.
    // Bidirectional: each query attends to all chunk positions 0..chunk_size-1.
    //
    // With sliding window (causal only): limit to at most window_size positions.
    const uint attend_len = params.causal ? (qi + 1) : chunk_size;
    const uint attend_start = (params.causal && params.window_size > 0 && attend_len > params.window_size)
                              ? (attend_len - params.window_size) : 0;

    // --- Load Q into threadgroup shared memory ---
    threadgroup float q_shared[MAX_HD];
    if (tid < head_dim) {
        q_shared[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Per-thread V accumulators and online softmax state ---
    float4 v_acc[MAX_HD_VEC4] = {};
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    const uint hd4 = head_dim / 4;

    // --- Fused position loop: softmax + V accumulation in one pass ---
    for (uint pos = attend_start + tid; pos < attend_len; pos += tg_size) {
        // Vectorised Q·K dot product from shared memory.
        device const bfloat* k_vec = k + pos * kv_stride + kv_head * head_dim;
        float score = dot_q_k(q_shared, k_vec, head_dim) * scale;

        // Online softmax update with V accumulator rescaling.
        if (score > local_max) {
            float correction = exp(local_max - score);
            for (uint i = 0; i < hd4; i++) {
                v_acc[i] *= correction;
            }
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }

        // Accumulate weighted V vector (vectorised bfloat4 loads).
        float weight = exp(score - local_max);
        device const bfloat* v_vec = v + pos * kv_stride + kv_head * head_dim;
        device const bfloat4* v4 = (device const bfloat4*)v_vec;
        for (uint i = 0; i < hd4; i++) {
            v_acc[i] += weight * float4(v4[i]);
        }
    }

    // Attention sinks (see paged_attention kernel for detailed comments).
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float correction = exp(local_max - sink_score);
            for (uint i = 0; i < hd4; i++) v_acc[i] *= correction;
            local_sum_exp = local_sum_exp * correction + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax reduction ---
    threadgroup float shared_reduce[NUM_SIMD_GROUPS * MAX_HD];
    float2 softmax_result = reduce_softmax(local_max, local_sum_exp, tid, tg_size, shared_reduce);
    float global_max = softmax_result.x;
    float total_sum = softmax_result.y;

    // --- Reduce V accumulators across threads and write output ---
    device bfloat* out_ptr = output + qi * q_stride + head_id * head_dim;
    reduce_v_and_write(v_acc, local_max, global_max, total_sum, tid, tg_size, head_dim, shared_reduce, out_ptr);
}
