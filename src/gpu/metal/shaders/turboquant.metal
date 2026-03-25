// ===========================================================================
// TurboQuant Metal Kernels — KV cache vector quantization for attention.
//
// Implements the TurboQuant algorithm (Zandieh et al., arXiv:2504.19874):
//   1. turbo_quantize_paged: Rotate + scalar quantize + write to paged pool
//   2. turbo_rotate_q: Pre-rotate query vector for efficient inner products
//   3. turbo_paged_attention: Attention with inline dequantization
//
// Storage format per KV head per position:
//   [2 bytes bf16 norm] [ceil(head_dim × bits / 8) bytes packed codes]
//
// Pool layout: positions are addressed via block table indirection, same as
// the BF16 paged cache.  The difference is bytes-per-position is smaller.
//   pool_offset = (physical_block * BLOCK_SIZE + offset_in_block) * bytes_per_pos
//                 + kv_head * bytes_per_head_pos
//
// Related files:
//   gpu/ops/turboquant.rs             — GpuTurboQuant trait
//   gpu/metal/kernels/turboquant.rs   — Rust dispatch code
//   model/turboquant.rs               — Algorithm and codebook constants
//
// Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion
//         Rate", Zandieh, Daliri, Hadian, Mirrokni. arXiv:2504.19874, 2025.
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Param structs — must match #[repr(C)] Rust structs byte-for-byte.
// ---------------------------------------------------------------------------

struct TurboQuantizeParams {
    uint pos;
    uint num_kv_heads;
    uint head_dim;
    uint bits;
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
};

struct TurboQuantizeBatchParams {
    uint batch_size;
    uint num_kv_heads;
    uint head_dim;
    uint bits;
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
};

struct TurboRotateQParams {
    uint num_heads;
    uint head_dim;
};

struct TurboPagedAttentionParams {
    uint seq_len;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint bits;
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
    uint window_size;
    float attn_scale;
    uint has_sinks;
};

// ---------------------------------------------------------------------------
// Helper: extract a b-bit code from a packed byte array.
//
// For 4-bit: 2 codes per byte (nibble extraction).
// For 2-bit: 4 codes per byte.
// For 3-bit: codes cross byte boundaries for 1/3 of positions.
// ---------------------------------------------------------------------------
inline uint extract_code(device const uchar* packed, uint idx, uint bits) {
    uint bit_offset = idx * bits;
    uint byte_idx = bit_offset / 8;
    uint bit_within_byte = bit_offset % 8;

    // Read up to 2 bytes to handle codes crossing a byte boundary.
    uint val = packed[byte_idx];
    if (bit_within_byte + bits > 8) {
        val |= (uint(packed[byte_idx + 1]) << 8);
    }

    return (val >> bit_within_byte) & ((1u << bits) - 1u);
}

// ---------------------------------------------------------------------------
// Helper: pack a b-bit code into a byte array at position idx.
// ---------------------------------------------------------------------------
inline void pack_code(device uchar* packed, uint idx, uint code, uint bits) {
    uint bit_offset = idx * bits;
    uint byte_idx = bit_offset / 8;
    uint bit_within_byte = bit_offset % 8;

    // Clear and set bits within first byte.
    uint mask = ((1u << bits) - 1u) << bit_within_byte;
    packed[byte_idx] = (packed[byte_idx] & ~uchar(mask & 0xFF)) | uchar((code << bit_within_byte) & 0xFF);

    // Handle overflow into next byte.
    if (bit_within_byte + bits > 8) {
        uint overflow_bits = bit_within_byte + bits - 8;
        uint overflow_mask = (1u << overflow_bits) - 1u;
        uint overflow_val = code >> (bits - overflow_bits);
        packed[byte_idx + 1] = (packed[byte_idx + 1] & ~uchar(overflow_mask)) | uchar(overflow_val);
    }
}

// ---------------------------------------------------------------------------
// turbo_quantize_paged — rotate + quantize one K/V vector into paged pool.
//
// Dispatch: one threadgroup per KV head.
// Threads: head_dim threads per threadgroup (up to 256).
//
// Algorithm per head:
//   1. Load bf16 head vector, compute L2 norm (parallel reduction)
//   2. Matrix-vector multiply: y = Pi × (x / norm)
//   3. Find nearest centroid for each y_j
//   4. Pack codes + bf16 norm into quantized pool at paged location
// ---------------------------------------------------------------------------

kernel void turbo_quantize_paged(
    constant TurboQuantizeParams& params [[buffer(0)]],
    device const bfloat* src            [[buffer(1)]],  // [num_kv_heads * head_dim] bf16
    device uchar* pool                  [[buffer(2)]],  // quantized paged pool
    device const uint* block_table      [[buffer(3)]],  // [MAX_BLOCKS_PER_SEQ]
    constant float* pi                  [[buffer(4)]],  // [head_dim, head_dim] f32
    constant float* centroids           [[buffer(5)]],  // [num_centroids] f32
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint kv_head = tg_id;
    if (kv_head >= params.num_kv_heads) return;

    uint hd = params.head_dim;

    // Pointer to this head's input vector.
    device const bfloat* x = src + kv_head * hd;

    // --- Step 1: Load input into shared memory and compute L2 norm. ---
    threadgroup float shared_x[256]; // max head_dim
    threadgroup float shared_norm[1];

    float val = (tid < hd) ? float(x[tid]) : 0.0f;
    shared_x[tid] = val;

    // Parallel sum of squares for L2 norm.
    float sq = val * val;
    sq = simd_sum(sq);
    if (simd_is_first()) {
        // Atomic add across SIMD groups.
        // Use shared memory for cross-SIMD reduction.
        if (tid == 0) shared_norm[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple reduction: lane 0 of each SIMD group atomically adds.
    if (tid % 32 == 0) {
        // Use atomic add for cross-SIMD accumulation.
        atomic_fetch_add_explicit(
            (threadgroup atomic_uint*)shared_norm,
            as_type<uint>(sq),
            memory_order_relaxed
        );
    }
    // NOTE: atomic float add on threadgroup memory is not standard Metal.
    // Use a simpler reduction approach instead.

    // Actually, let's use a cleaner reduction pattern.
    // Rewrite norm computation using shared memory array.
    threadgroup float norm_partials[8]; // up to 8 SIMD groups (256/32)
    uint simd_group = tid / 32;
    uint simd_lane = tid % 32;

    // Step 1a: SIMD-level sum of squares.
    // Already have sq = simd_sum(val^2) from above in lane 0.
    if (simd_lane == 0) {
        norm_partials[simd_group] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: Final reduction in first SIMD group.
    float norm_sq = 0.0f;
    if (tid == 0) {
        uint num_groups = (tg_size + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            norm_sq += norm_partials[i];
        }
        shared_norm[0] = sqrt(max(norm_sq, 1e-12f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = 1.0f / shared_norm[0];

    // --- Step 2: Rotate: y_j = sum_i Pi[j,i] * (x[i] / norm) ---
    // Each thread computes one output coordinate y[tid].
    float y = 0.0f;
    if (tid < hd) {
        for (uint i = 0; i < hd; i++) {
            y += pi[tid * hd + i] * (shared_x[i] * inv_norm);
        }
    }

    // --- Step 3: Find nearest centroid for y[tid]. ---
    uint best_idx = 0;
    if (tid < hd) {
        float best_dist = INFINITY;
        for (uint c = 0; c < params.num_centroids; c++) {
            float d = abs(y - centroids[c]);
            if (d < best_dist) {
                best_dist = d;
                best_idx = c;
            }
        }
    }

    // --- Step 4: Write to paged pool. ---
    // Compute pool address via block table indirection.
    uint logical_block = params.pos / params.block_size;
    uint offset_in_block = params.pos % params.block_size;
    uint physical_block = block_table[logical_block];
    uint pool_pos = physical_block * params.block_size + offset_in_block;

    // Each position stores: [head0_data][head1_data]...
    // where head_data = [2 bytes bf16 norm][packed_codes]
    uint num_kv_heads = params.num_kv_heads;
    uint bytes_per_pos = num_kv_heads * params.bytes_per_head_pos;
    device uchar* pos_base = pool + pool_pos * bytes_per_pos;
    device uchar* head_base = pos_base + kv_head * params.bytes_per_head_pos;

    // Thread 0 writes the bf16 norm.
    if (tid == 0) {
        bfloat norm_bf16 = bfloat(shared_norm[0]);
        device bfloat* norm_ptr = (device bfloat*)head_base;
        *norm_ptr = norm_bf16;
    }

    // Each thread packs its code.
    if (tid < hd) {
        device uchar* codes = head_base + 2; // skip 2-byte norm
        pack_code(codes, tid, best_idx, params.bits);
    }
}

// ---------------------------------------------------------------------------
// turbo_quantize_paged_batch — batched version for prefill.
//
// Dispatch: batch_size × num_kv_heads threadgroups.
// Each threadgroup processes one (batch_idx, kv_head) pair.
// ---------------------------------------------------------------------------

kernel void turbo_quantize_paged_batch(
    constant TurboQuantizeBatchParams& params [[buffer(0)]],
    device const bfloat* src            [[buffer(1)]],  // [batch_size, num_kv_heads * head_dim]
    device uchar* pool                  [[buffer(2)]],
    device const uint* block_table      [[buffer(3)]],
    device const uint* positions        [[buffer(4)]],  // [batch_size]
    constant float* pi                  [[buffer(5)]],
    constant float* centroids           [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint batch_idx = tg_id / params.num_kv_heads;
    uint kv_head = tg_id % params.num_kv_heads;
    if (batch_idx >= params.batch_size) return;

    uint hd = params.head_dim;
    uint kv_dim = params.num_kv_heads * hd;

    // Pointer to this (batch, head) input vector.
    device const bfloat* x = src + batch_idx * kv_dim + kv_head * hd;

    // Load + compute norm (same as single-vector version).
    threadgroup float shared_x[256];
    threadgroup float shared_norm[1];
    threadgroup float norm_partials[8];

    float val = (tid < hd) ? float(x[tid]) : 0.0f;
    shared_x[tid] = val;

    float sq = val * val;
    sq = simd_sum(sq);
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;
    if (simd_lane == 0) norm_partials[simd_group] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float norm_sq = 0.0f;
        uint num_groups = (tg_size + 31) / 32;
        for (uint i = 0; i < num_groups; i++) norm_sq += norm_partials[i];
        shared_norm[0] = sqrt(max(norm_sq, 1e-12f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = 1.0f / shared_norm[0];

    // Rotate.
    float y = 0.0f;
    if (tid < hd) {
        for (uint i = 0; i < hd; i++) {
            y += pi[tid * hd + i] * (shared_x[i] * inv_norm);
        }
    }

    // Nearest centroid.
    uint best_idx = 0;
    if (tid < hd) {
        float best_dist = INFINITY;
        for (uint c = 0; c < params.num_centroids; c++) {
            float d = abs(y - centroids[c]);
            if (d < best_dist) { best_dist = d; best_idx = c; }
        }
    }

    // Write to paged pool.
    uint pos = positions[batch_idx];
    uint logical_block = pos / params.block_size;
    uint offset_in_block = pos % params.block_size;
    uint physical_block = block_table[logical_block];
    uint pool_pos = physical_block * params.block_size + offset_in_block;

    uint bytes_per_pos = params.num_kv_heads * params.bytes_per_head_pos;
    device uchar* pos_base = pool + pool_pos * bytes_per_pos;
    device uchar* head_base = pos_base + kv_head * params.bytes_per_head_pos;

    if (tid == 0) {
        device bfloat* norm_ptr = (device bfloat*)head_base;
        *norm_ptr = bfloat(shared_norm[0]);
    }
    if (tid < hd) {
        pack_code(head_base + 2, tid, best_idx, params.bits);
    }
}

// ---------------------------------------------------------------------------
// turbo_rotate_q — pre-rotate query for quantized attention.
//
// Dispatch: num_heads threadgroups, head_dim threads each.
// q_rot[head][j] = sum_i Pi[j,i] * q[head][i]
// Output is f32 for dot-product precision.
// ---------------------------------------------------------------------------

kernel void turbo_rotate_q(
    constant TurboRotateQParams& params [[buffer(0)]],
    device const bfloat* q              [[buffer(1)]],  // [num_heads * head_dim] bf16
    device float* q_rot                 [[buffer(2)]],  // [num_heads * head_dim] f32
    constant float* pi                  [[buffer(3)]],  // [head_dim, head_dim] f32
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head = tg_id;
    if (head >= params.num_heads) return;

    uint hd = params.head_dim;
    device const bfloat* q_head = q + head * hd;
    device float* out_head = q_rot + head * hd;

    // Load q_head into shared memory.
    threadgroup float shared_q[256];
    if (tid < hd) shared_q[tid] = float(q_head[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute rotated output.
    if (tid < hd) {
        float sum = 0.0f;
        for (uint i = 0; i < hd; i++) {
            sum += pi[tid * hd + i] * shared_q[i];
        }
        out_head[tid] = sum;
    }
}

// ---------------------------------------------------------------------------
// turbo_paged_attention — paged attention with inline TurboQuant dequantization.
//
// Mirrors the structure of the BF16 paged_attention kernel:
//   - 256 threads per query head, strided position loop
//   - Online softmax (fused single-pass)
//   - Block table caching
//
// Differences from BF16 version:
//   - Q is f32 (pre-rotated), not bf16
//   - K/V are read as packed codes + bf16 norm, dequantized via centroid lookup
//   - V accumulation happens in rotated space
//   - After the position loop, Pi^T is applied to the accumulated V (once)
//
// Dispatch: num_heads threadgroups, 256 threads each.
// ---------------------------------------------------------------------------

kernel void turbo_paged_attention(
    constant TurboPagedAttentionParams& params [[buffer(0)]],
    device const float* q_rot           [[buffer(1)]],   // [num_heads, head_dim] f32
    device const uchar* k_pool          [[buffer(2)]],   // quantized K pool
    device const uchar* v_pool          [[buffer(3)]],   // quantized V pool
    device const uint* block_table      [[buffer(4)]],
    constant float* pi_t                [[buffer(5)]],   // [head_dim, head_dim] f32
    constant float* centroids           [[buffer(6)]],   // [num_centroids] f32
    device bfloat* output               [[buffer(7)]],   // [num_heads, head_dim] bf16
    device const bfloat* sinks          [[buffer(8)]],   // [num_heads] or dummy
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint head_id = tg_id;
    if (head_id >= params.num_heads) return;

    uint hd = params.head_dim;
    uint seq_len = params.seq_len;
    uint num_kv_heads = params.num_kv_heads;
    uint bits = params.bits;
    uint bytes_per_head_pos = params.bytes_per_head_pos;
    uint block_size = params.block_size;
    uint num_centroids = params.num_centroids;

    // GQA: map query head to KV head.
    uint heads_per_kv = params.num_heads / num_kv_heads;
    uint kv_head = head_id / heads_per_kv;

    // Load centroids into shared memory (tiny: 4-16 floats).
    threadgroup float shared_centroids[16];
    if (tid < num_centroids) {
        shared_centroids[tid] = centroids[tid];
    }

    // Load q_rot into shared memory.
    threadgroup float q_shared[256];
    if (tid < hd) {
        q_shared[tid] = q_rot[head_id * hd + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread V accumulator in rotated space + online softmax state.
    // Each thread handles a subset of output dimensions (strided by tg_size).
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    // V accumulator: each of 256 threads keeps partial sums for strided dims.
    // Thread t accumulates dims {t, t+256, t+512, ...} of the V vector.
    // We use a small register array per thread for the V accumulation.
    // For head_dim=128, each thread handles ceil(128/256) = 1 dim (or 0 for threads ≥ 128).
    float v_acc[1] = {0.0f}; // Simplified: each thread handles at most 1 V dim for hd≤256

    // Compute bytes_per_pos for the full position (all kv heads).
    uint bytes_per_pos = num_kv_heads * bytes_per_head_pos;

    // Block table caching.
    uint prev_logical_block = ~0u;
    uint physical_base = 0;

    // Determine attention window.
    uint start = 0;
    if (params.window_size > 0 && seq_len > params.window_size) {
        start = seq_len - params.window_size;
    }

    // --- Main position loop (strided across 256 threads). ---
    for (uint pos = start + tid; pos < seq_len; pos += tg_size) {
        // Block table lookup with caching.
        uint logical_block = pos / block_size;
        if (logical_block != prev_logical_block) {
            prev_logical_block = logical_block;
            physical_base = block_table[logical_block] * block_size;
        }
        uint offset_in_block = pos % block_size;
        uint pool_pos = physical_base + offset_in_block;

        // Address of this position's K data for our KV head.
        device const uchar* k_pos_base = k_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;

        // Read K norm (bf16 → f32).
        float k_norm = float(*(device const bfloat*)k_pos_base);
        device const uchar* k_codes = k_pos_base + 2;

        // Compute Q_rot · dequant(K) = sum_j q_rot[j] * (centroid[code_j] * k_norm)
        float score = 0.0f;
        for (uint j = 0; j < hd; j++) {
            uint code = extract_code(k_codes, j, bits);
            float k_val = shared_centroids[code] * k_norm;
            score += q_shared[j] * k_val;
        }

        // Apply attention scale.
        float scale = (params.attn_scale != 0.0f) ? params.attn_scale : rsqrt(float(hd));
        score *= scale;

        // Online softmax update.
        if (score > local_max) {
            float rescale = exp(local_max - score);
            // Rescale existing V accumulator.
            for (uint d = 0; d < 1; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }

        float weight = exp(score - local_max);

        // Accumulate weighted V in rotated space.
        // Each thread handles dimension tid (if tid < hd).
        device const uchar* v_pos_base = v_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;
        float v_norm = float(*(device const bfloat*)v_pos_base);
        device const uchar* v_codes = v_pos_base + 2;

        if (tid < hd) {
            uint v_code = extract_code(v_codes, tid, bits);
            float v_val = shared_centroids[v_code] * v_norm;
            v_acc[0] += weight * v_val;
        }
    }

    // Handle attention sinks.
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float rescale = exp(local_max - sink_score);
            for (uint d = 0; d < 1; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax reduction. ---
    // Reduce (local_max, local_sum_exp) across all 256 threads.
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];

    uint simd_group = tid / 32;
    uint simd_lane = tid % 32;

    // SIMD-level reduction of max.
    float group_max = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_group] = group_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD max reduction.
    float global_max;
    if (tid == 0) {
        global_max = shared_max[0];
        uint num_groups = (tg_size + 31) / 32;
        for (uint i = 1; i < num_groups; i++) {
            global_max = max(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    // Rescale each thread's state to global max.
    float rescale = exp(local_max - global_max);
    local_sum_exp *= rescale;
    for (uint d = 0; d < 1; d++) v_acc[d] *= rescale;

    // SIMD-level sum of exp.
    float group_sum = simd_sum(local_sum_exp);
    if (simd_lane == 0) shared_sum[simd_group] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        uint num_groups = (tg_size + 31) / 32;
        for (uint i = 0; i < num_groups; i++) total_sum += shared_sum[i];
        shared_sum[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum = shared_sum[0];

    // Normalise V accumulator.
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    if (tid < hd) {
        v_acc[0] *= inv_sum;
    }

    // --- Apply Pi^T to get from rotated space back to original. ---
    // v_out[i] = sum_j Pi_T[i,j] * v_acc_rotated[j]
    // Each thread has v_acc[0] for dimension tid (in rotated space).
    // We need to reduce across threads: v_out[i] = sum_j Pi_T[i,j] * thread_j_v_acc.

    // Store v_acc into shared memory for the mat-vec.
    threadgroup float shared_v_rot[256];
    if (tid < hd) {
        shared_v_rot[tid] = v_acc[0];
    } else {
        shared_v_rot[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one output dimension of Pi^T × v_rot.
    float v_out = 0.0f;
    if (tid < hd) {
        for (uint j = 0; j < hd; j++) {
            v_out += pi_t[tid * hd + j] * shared_v_rot[j];
        }
    }

    // Write bf16 output.
    if (tid < hd) {
        output[head_id * hd + tid] = bfloat(v_out);
    }
}
