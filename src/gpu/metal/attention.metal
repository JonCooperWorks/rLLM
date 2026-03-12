// ===========================================================================
// Grouped-Query Attention (GQA) kernel with flat KV cache.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Computes the core attention operation for one token:
//     output = softmax(Q · K^T / √d) · V
//   where Q is the current token's query, and K/V come from the KV cache
//   (all previously processed tokens plus the current one).
//
// Grouped-Query Attention (GQA):
//   In standard multi-head attention, each query head has its own K and V
//   heads.  GQA shares KV heads across multiple query heads to reduce
//   KV cache memory.  Llama 3.2 uses 32 query heads but only 8 KV heads
//   — every 4 query heads share one KV head (4:1 ratio).
//
//   Mapping: kv_head_index = query_head_index / (num_heads / num_kv_heads)
//
// KV cache layout:
//   Flat array: [max_seq_len, num_kv_heads, head_dim]
//   Position `pos`, KV head `h`, dimension `d` is at:
//     cache[pos * num_kv_heads * head_dim + h * head_dim + d]
//
// Online softmax (no score array):
//   A naive implementation would store all attention scores in threadgroup
//   memory, but that limits context length (32KB ÷ 4 bytes = 8192 max
//   positions, and shared memory for reductions pushes it over the 32KB
//   limit).
//
//   Instead, we use the ONLINE SOFTMAX algorithm:
//     - Pass 1: scan all positions, tracking running (max, sum_exp)
//     - Pass 2: recompute scores and accumulate weighted values
//
//   This uses O(head_dim) memory per thread instead of O(seq_len), at
//   the cost of computing Q·K dot products twice.  For Phase 1 with
//   short sequences this is fine; a fused flash-attention kernel would
//   avoid the recomputation.
//
// Dispatch model:
//   One threadgroup of 256 threads per query head.  Total threads =
//   num_heads * 256 = 32 * 256 = 8192.  Within each threadgroup:
//     - Pass 1: all 256 threads cooperatively scan positions to find
//       the softmax normalization constants (max and sum_exp)
//     - Pass 2: threads divide the head_dim output dimensions among
//       themselves and accumulate weighted value sums
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `AttentionParams`.
struct AttentionParams {
    uint seq_len;       // Number of tokens in KV cache (including current).
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8, shared via GQA).
    uint head_dim;      // Dimension per head (64).
};

// Threadgroup shared memory layout:
//   shared[0..31]  — for max reduction across SIMD groups
//   shared[32..63] — for sum reduction across SIMD groups

kernel void attention(
    constant AttentionParams& params [[buffer(0)]],
    // buffer(1): Q vector [num_heads, head_dim] — current token's queries.
    device const bfloat* q           [[buffer(1)]],
    // buffer(2): K cache [max_seq_len, num_kv_heads, head_dim].
    device const bfloat* k_cache     [[buffer(2)]],
    // buffer(3): V cache [max_seq_len, num_kv_heads, head_dim].
    device const bfloat* v_cache     [[buffer(3)]],
    // buffer(4): output [num_heads, head_dim].
    device bfloat* output            [[buffer(4)]],
    // Which threadgroup (= which query head) this thread belongs to.
    uint head_id                     [[threadgroup_position_in_grid]],
    // Thread index within the threadgroup (0..255).
    uint tid                         [[thread_position_in_threadgroup]],
    // Threadgroup size (256).
    uint tg_size                     [[threads_per_threadgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_dim = params.head_dim;
    const uint num_kv_heads = params.num_kv_heads;
    // GQA mapping: 4 query heads share 1 KV head.
    const uint heads_per_kv = params.num_heads / num_kv_heads;
    const uint kv_head = head_id / heads_per_kv;
    // Stride between consecutive positions in the KV cache.
    const uint kv_stride = num_kv_heads * head_dim;

    // Attention scale factor: 1/√d.  `rsqrt` = 1/sqrt (single instruction).
    const float scale = rsqrt(float(head_dim));

    // Pointer to this head's query vector.
    device const bfloat* q_ptr = q + head_id * head_dim;

    // -----------------------------------------------------------------------
    // Pass 1: Online softmax — find global max and sum of exp(score - max).
    //
    // Each thread scans ALL cache positions (strided by tg_size), computing
    // Q·K dot products on the fly.  The online softmax trick maintains a
    // running (max, sum_exp) pair that gets corrected when a new max is found:
    //
    //   if new_score > current_max:
    //       sum_exp = sum_exp * exp(old_max - new_max) + 1
    //       max = new_score
    //   else:
    //       sum_exp += exp(new_score - max)
    //
    // This avoids storing the full score vector in memory.
    // -----------------------------------------------------------------------

    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    for (uint pos = tid; pos < seq_len; pos += tg_size) {
        // Compute Q · K[pos] (dot product with one cached key vector).
        device const bfloat* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_vec[d]);
        }
        float score = dot * scale;

        // Online softmax update.
        if (score > local_max) {
            // New max found — rescale previous sum to the new max.
            local_sum_exp = local_sum_exp * exp(local_max - score) + exp(0.0f);
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }
    }

    // --- Cross-thread reduction: combine per-thread (max, sum_exp) ---

    threadgroup float shared[64];
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

    // Reduce max across SIMD groups via shared memory.
    float smax = simd_max(local_max);
    if (simd_lane_id == 0) shared[simd_group_id] = smax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_groups) ? shared[simd_lane_id] : -INFINITY;
        val = simd_max(val);
        if (simd_lane_id == 0) shared[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = shared[0];

    // Adjust each thread's local_sum_exp to the global max, then reduce.
    local_sum_exp *= exp(local_max - global_max);

    float ssum = simd_sum(local_sum_exp);
    if (simd_lane_id == 0) shared[32 + simd_group_id] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        uint num_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_groups) ? shared[32 + simd_lane_id] : 0.0f;
        val = simd_sum(val);
        if (simd_lane_id == 0) shared[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sum = shared[0];
    float inv_sum = 1.0f / total_sum;

    // -----------------------------------------------------------------------
    // Pass 2: Weighted sum of value vectors.
    //
    // Each thread handles a subset of the head_dim output dimensions.
    // For each assigned dimension d, iterate over ALL positions, recompute
    // the Q·K score, convert to softmax weight, and accumulate weight * V[d].
    //
    // Learning note: this recomputes Q·K scores from scratch.  This is the
    // trade-off of the online softmax approach — we compute each dot product
    // twice instead of storing scores in shared memory.  For head_dim=64
    // and short sequences, this is fast.  Flash Attention avoids this by
    // fusing both passes into a single tiled loop.
    // -----------------------------------------------------------------------

    device bfloat* out_head = output + head_id * head_dim;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint pos = 0; pos < seq_len; pos++) {
            // Recompute attention score for this position.
            device const bfloat* k_vec = k_cache + pos * kv_stride + kv_head * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += float(q_ptr[dd]) * float(k_vec[dd]);
            }
            float score = dot * scale;
            float weight = exp(score - global_max) * inv_sum;

            // Accumulate: output[d] += weight * V[pos][d]
            device const bfloat* v_vec = v_cache + pos * kv_stride + kv_head * head_dim;
            acc += weight * float(v_vec[d]);
        }
        out_head[d] = bfloat(acc);
    }
}

// ===========================================================================
// KV cache write kernel.
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
