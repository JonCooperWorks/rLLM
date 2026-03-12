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
// Paged attention kernel.
//
// Same two-pass online softmax as the flat attention kernel above, but
// K/V reads go through block table indirection.  The block table maps
// logical blocks (contiguous per sequence) to physical blocks (scattered
// across the shared pool).
//
// Dispatch model: identical to flat attention — one threadgroup of 256
// threads per query head.  The only difference is how K/V addresses
// are computed in the inner loops.
// ===========================================================================

struct PagedAttentionParams {
    uint seq_len;       // Number of tokens in this sequence's KV cache.
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8).
    uint head_dim;      // Dimension per head (64).
    uint block_size;    // = 16
};

kernel void paged_attention(
    constant PagedAttentionParams& params [[buffer(0)]],
    device const bfloat* q               [[buffer(1)]],  // [num_heads, head_dim]
    device const bfloat* k_pool          [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device const bfloat* v_pool          [[buffer(3)]],  // [num_blocks * block_size, kv_dim]
    device const uint* block_table       [[buffer(4)]],  // [max_blocks_per_seq]
    device bfloat* output                [[buffer(5)]],  // [num_heads, head_dim]
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

    const float scale = rsqrt(float(head_dim));
    device const bfloat* q_ptr = q + head_id * head_dim;

    // -------------------------------------------------------------------
    // Helper: translate logical position to physical address in the pool.
    // This replaces the simple `pos * kv_stride` from the flat kernel.
    // -------------------------------------------------------------------
    #define PAGED_K_VEC(pos) \
        (k_pool + (block_table[(pos) / block_size] * block_size + (pos) % block_size) * kv_dim + kv_head * head_dim)
    #define PAGED_V_VEC(pos) \
        (v_pool + (block_table[(pos) / block_size] * block_size + (pos) % block_size) * kv_dim + kv_head * head_dim)

    // -------------------------------------------------------------------
    // Pass 1: Online softmax — find global max and sum of exp(score - max).
    // Identical algorithm to flat attention, just different K addressing.
    // -------------------------------------------------------------------

    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    for (uint pos = tid; pos < seq_len; pos += tg_size) {
        device const bfloat* k_vec = PAGED_K_VEC(pos);
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_vec[d]);
        }
        float score = dot * scale;

        if (score > local_max) {
            local_sum_exp = local_sum_exp * exp(local_max - score) + exp(0.0f);
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }
    }

    // Cross-thread reduction (identical to flat kernel).
    threadgroup float shared[64];
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

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

    // -------------------------------------------------------------------
    // Pass 2: Weighted sum of value vectors (with paged addressing).
    // -------------------------------------------------------------------

    device bfloat* out_head = output + head_id * head_dim;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint pos = 0; pos < seq_len; pos++) {
            device const bfloat* k_vec = PAGED_K_VEC(pos);
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += float(q_ptr[dd]) * float(k_vec[dd]);
            }
            float score = dot * scale;
            float weight = exp(score - global_max) * inv_sum;

            device const bfloat* v_vec = PAGED_V_VEC(pos);
            acc += weight * float(v_vec[d]);
        }
        out_head[d] = bfloat(acc);
    }

    #undef PAGED_K_VEC
    #undef PAGED_V_VEC
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
// Causal prefill attention kernel.
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
// Dispatch model:
//   One threadgroup of 256 per (query_position, head).
//   Grid: chunk_size * num_heads * 256.
//   Each threadgroup computes attention for one query at one head.
//
// Same two-pass online softmax algorithm as the decode kernels:
//   Pass 1: scan K positions to find (max, sum_exp) for softmax normalisation
//   Pass 2: recompute scores and accumulate weighted V vectors
// ===========================================================================

struct PrefillAttentionParams {
    uint chunk_size;    // Number of tokens in this prefill chunk.
    uint start_pos;     // Starting position for this chunk (0 for fresh prefill).
    uint num_heads;     // Query heads (32).
    uint num_kv_heads;  // KV heads (8).
    uint head_dim;      // Dimension per head (64).
};

kernel void prefill_attention(
    constant PrefillAttentionParams& params [[buffer(0)]],
    device const bfloat* q       [[buffer(1)]],  // [chunk_size, num_heads * head_dim]
    device const bfloat* k       [[buffer(2)]],  // [chunk_size, num_kv_heads * head_dim]
    device const bfloat* v       [[buffer(3)]],  // [chunk_size, num_kv_heads * head_dim]
    device bfloat* output        [[buffer(4)]],  // [chunk_size, num_heads * head_dim]
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
    const float scale = rsqrt(float(head_dim));

    // Strides for the dense Q/K/V tensors.
    const uint q_stride = num_heads * head_dim;      // row stride in Q
    const uint kv_stride = num_kv_heads * head_dim;  // row stride in K/V

    // Pointer to this query token's head vector.
    device const bfloat* q_ptr = q + qi * q_stride + head_id * head_dim;

    // Causal mask: this query attends to chunk positions 0..=qi.
    // (If start_pos > 0, the chunk's tokens are at absolute positions
    //  start_pos..start_pos+chunk_size, but K/V within this chunk are
    //  indexed 0..chunk_size-1.  The causal constraint is on chunk indices.)
    const uint attend_len = qi + 1;

    // -------------------------------------------------------------------
    // Pass 1: Online softmax over K positions 0..attend_len-1.
    // -------------------------------------------------------------------
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    for (uint pos = tid; pos < attend_len; pos += tg_size) {
        device const bfloat* k_vec = k + pos * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_vec[d]);
        }
        float score = dot * scale;

        if (score > local_max) {
            local_sum_exp = local_sum_exp * exp(local_max - score) + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }
    }

    // Cross-thread reduction for max.
    threadgroup float shared[64];
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

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

    // Adjust and reduce sum_exp.
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

    // -------------------------------------------------------------------
    // Pass 2: Weighted sum of V vectors.
    // -------------------------------------------------------------------
    device bfloat* out_ptr = output + qi * q_stride + head_id * head_dim;

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint pos = 0; pos < attend_len; pos++) {
            device const bfloat* k_vec = k + pos * kv_stride + kv_head * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += float(q_ptr[dd]) * float(k_vec[dd]);
            }
            float score = dot * scale;
            float weight = exp(score - global_max) * inv_sum;

            device const bfloat* v_vec = v + pos * kv_stride + kv_head * head_dim;
            acc += weight * float(v_vec[d]);
        }
        out_ptr[d] = bfloat(acc);
    }
}
