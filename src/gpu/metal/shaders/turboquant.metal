// ===========================================================================
// TurboQuant Metal Kernels — KV cache vector quantization for attention.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the GPU kernels for TurboQuant (Zandieh et al., arXiv:2504.19874),
//   an online vector quantization algorithm that compresses the KV cache by ~4x
//   at 4-bit with quality-neutral results.  Three kernels work together:
//
//   1. turbo_quantize_paged: Rotate + scalar quantize K/V + write to paged pool
//   2. turbo_rotate_q: Pre-rotate query vector for efficient inner products
//   3. turbo_paged_attention: Decode attention with inline dequantization
//
// How quantization works:
//   For each K or V head vector x ∈ R^d:
//     - Compute L2 norm ||x||, store as bf16 (2 bytes)
//     - Normalise and rotate: y = Pi × (x / ||x||)
//     - Each coordinate y_j is approximately N(0, 1/√d) by concentration
//       of measure on the unit sphere — this is the key mathematical insight
//     - Quantise each y_j to the nearest Max-Lloyd centroid (4-16 entries)
//     - Pack codes into ceil(d × bits / 8) bytes
//
// How attention uses quantized data:
//   - Pre-rotate Q once: q_rot = Pi × q  (avoids per-position rotation)
//   - For each cached position: score = q_rot · (centroid[code_j] × norm)
//     Since centroid[code_j] ≈ (Pi × k/||k||)_j, this ≈ q · k (orthogonality)
//   - V accumulated in rotated space, Pi^T applied once at end
//
// Critical implementation detail — code packing:
//   For sub-byte codes (2-4 bits), multiple codes share a byte.  Concurrent
//   writes from SIMD lanes to the same byte cause data races on Apple Silicon
//   (the GPU picks one lane's write, zeroing the other's bits).  We avoid this
//   by collecting codes in threadgroup shared memory and having each thread
//   pack one complete byte.  See pack_codes_shared() below.
//
// Storage format per KV head per position:
//   [2 bytes bf16 norm] [ceil(head_dim × bits / 8) bytes packed codes]
//
// Pool layout: same paged block structure as BF16, addressed via block table
// indirection.  The only difference is bytes-per-position is smaller:
//   pool_offset = (physical_block * BLOCK_SIZE + offset_in_block) * bytes_per_pos
//                 + kv_head * bytes_per_head_pos
//
// Related files:
//   gpu/ops/turboquant.rs             — GpuTurboQuant trait (kernel interface)
//   gpu/metal/kernels/turboquant.rs   — Rust dispatch code (#[repr(C)] params)
//   model/turboquant.rs               — Algorithm, codebook constants, TurboContext
//   model/primitives.rs               — paged_kv_and_attention_maybe_quantized()
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
    uint bits;           // stage-1 bits (centroid codebook)
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
    uint is_plus;        // 1 = QJL residual (TurboQuant+)
};

struct TurboQuantizeBatchParams {
    uint batch_size;
    uint num_kv_heads;
    uint head_dim;
    uint bits;           // stage-1 bits
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
    uint is_plus;
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
    uint bits;           // stage-1 bits
    uint bytes_per_head_pos;
    uint block_size;
    uint num_centroids;
    uint window_size;
    float attn_scale;
    uint has_sinks;
    uint is_plus;
};

// V-only variant: K is BF16, V is TurboQuant.
struct TurboPagedAttentionVOnlyParams {
    uint seq_len;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint bits;              // V stage-1 quantization bits
    uint kv_dim;            // num_kv_heads * head_dim (for BF16 K addressing)
    uint v_bytes_per_head_pos;  // bytes per V head per position (quantized)
    uint block_size;
    uint num_centroids;
    uint window_size;
    float attn_scale;
    uint has_sinks;
    uint is_plus;
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
// Helper: pack codes from threadgroup memory into a packed byte array.
//
// The original pack_code() had a data race: for 4-bit codes, two threads
// share the same byte (e.g. thread 0 writes nibble 0 and thread 1 writes
// nibble 1 of byte 0).  Both do read-modify-write on the same device byte
// without synchronization.  Apple Silicon resolves the conflict by picking
// one lane's write, so every odd-indexed code gets zeroed out.
//
// Fix: collect all codes in threadgroup shared memory, then have each thread
// pack one complete byte from the relevant codes.  No concurrent writes to
// the same device address.  Caller must barrier before calling this.
// ---------------------------------------------------------------------------
inline void pack_codes_shared(
    threadgroup const uint* codes,
    device uchar* packed,
    uint tid,
    uint hd,
    uint bits
) {
    uint total_bytes = (hd * bits + 7) / 8;
    if (tid >= total_bytes) return;

    uchar val = 0;
    uint byte_bit_start = tid * 8;
    for (uint b = 0; b < 8; b++) {
        uint global_bit = byte_bit_start + b;
        uint code_idx = global_bit / bits;
        uint bit_in_code = global_bit % bits;
        if (code_idx < hd) {
            uint bit_val = (codes[code_idx] >> bit_in_code) & 1u;
            val |= uchar(bit_val << b);
        }
    }
    packed[tid] = val;
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

    // Parallel sum of squares for L2 norm via SIMD + shared memory reduction.
    float sq = val * val;
    sq = simd_sum(sq);

    threadgroup float norm_partials[8]; // up to 8 SIMD groups (256/32)
    uint simd_group = tid / 32;
    uint simd_lane = tid % 32;

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

    // Collect codes in shared memory for race-free packing.
    threadgroup uint shared_codes[256];
    shared_codes[tid] = (tid < hd) ? best_idx : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.is_plus) {
        // TurboQuant+ QJL residual: compute residual, gamma, and sign bits.
        // Layout: [2 bf16 norm] [2 bf16 gamma] [stage1 codes] [sign bits]
        uint stage1_code_bytes = (hd * params.bits + 7) / 8;

        // Compute residual and sign for this coordinate.
        float residual = 0.0f;
        uint sign_bit = 0;
        if (tid < hd) {
            residual = y - centroids[best_idx];
            sign_bit = (residual >= 0.0f) ? 1u : 0u;
        }

        // Compute gamma = ||residual||₂ via parallel reduction.
        float rsq = residual * residual;
        rsq = simd_sum(rsq);
        threadgroup float gamma_partials[8];
        if (simd_lane == 0) gamma_partials[simd_group] = rsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float gamma_sq = 0.0f;
            uint num_groups = (tg_size + 31) / 32;
            for (uint i = 0; i < num_groups; i++) gamma_sq += gamma_partials[i];
            // Write gamma as bf16 at offset 2 (after norm).
            device bfloat* gamma_ptr = (device bfloat*)(head_base + 2);
            *gamma_ptr = bfloat(sqrt(max(gamma_sq, 0.0f)));
        }

        // Pack stage-1 codes at offset 4 (after norm + gamma).
        pack_codes_shared(shared_codes, head_base + 4, tid, hd, params.bits);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pack sign bits at offset 4 + stage1_code_bytes.
        // Reuse shared_codes for sign bits (1-bit per coordinate).
        shared_codes[tid] = (tid < hd) ? sign_bit : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        pack_codes_shared(shared_codes, head_base + 4 + stage1_code_bytes, tid, hd, 1);
    } else {
        // Base TurboQuant: pack codes at offset 2 (after norm).
        pack_codes_shared(shared_codes, head_base + 2, tid, hd, params.bits);
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

    // Collect codes in shared memory for race-free packing.
    threadgroup uint shared_codes[256];
    shared_codes[tid] = (tid < hd) ? best_idx : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.is_plus) {
        // TurboQuant+ QJL residual.
        uint stage1_code_bytes = (hd * params.bits + 7) / 8;

        float residual = 0.0f;
        uint sign_bit = 0;
        if (tid < hd) {
            residual = y - centroids[best_idx];
            sign_bit = (residual >= 0.0f) ? 1u : 0u;
        }

        float rsq = residual * residual;
        rsq = simd_sum(rsq);
        threadgroup float gamma_partials[8];
        if (simd_lane == 0) gamma_partials[simd_group] = rsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float gamma_sq = 0.0f;
            uint num_groups = (tg_size + 31) / 32;
            for (uint i = 0; i < num_groups; i++) gamma_sq += gamma_partials[i];
            device bfloat* gamma_ptr = (device bfloat*)(head_base + 2);
            *gamma_ptr = bfloat(sqrt(max(gamma_sq, 0.0f)));
        }

        pack_codes_shared(shared_codes, head_base + 4, tid, hd, params.bits);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_codes[tid] = (tid < hd) ? sign_bit : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        pack_codes_shared(shared_codes, head_base + 4 + stage1_code_bytes, tid, hd, 1);
    } else {
        pack_codes_shared(shared_codes, head_base + 2, tid, hd, params.bits);
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
//   - Q is f32 (pre-rotated by Pi), not bf16
//   - K/V are read as packed codes + bf16 norm, dequantized via centroid lookup
//   - V accumulation happens in rotated space (all dims per thread, like BF16)
//   - After the position loop, Pi^T inverse rotation is applied to get back
//     to original space, fused with the cross-thread V reduction
//
// V accumulation design:
//   Each thread accumulates ALL head_dim V dimensions for the positions it
//   processes (strided by 256).  This matches the BF16 kernel's approach
//   where v_acc holds the full V vector.  The alternative — each thread
//   accumulating only its own dimension — is incorrect: V dimension d would
//   only get contributions from positions assigned to thread d, missing all
//   other positions.
//
// Register budget: v_acc[256] = 1KB per thread.  Apple Silicon has ~3KB
// registers per thread, so this fits.  For head_dim < 256, excess entries
// are zero-initialised and never touched.
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
    //
    // CRITICAL: each thread must accumulate ALL V dimensions for the positions
    // it processes, not just one dimension.  The BF16 attention kernel does
    // this with float4 v_acc[hd/4].  We do the same with scalar floats since
    // dequantization is per-coordinate anyway.
    //
    // Register budget: 256 floats × 4 bytes = 1024 bytes per thread.
    // Apple Silicon has ~3KB registers per thread, so this fits with room
    // for the other variables (score, weight, softmax state, etc.).
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;
    float v_acc[256] = {};  // max head_dim; zero-initialised

    // Compute bytes_per_pos for the full position (all kv heads).
    uint bytes_per_pos = num_kv_heads * bytes_per_head_pos;

    // Attention scale and QJL constants (computed once outside the loop).
    float scale = (params.attn_scale != 0.0f) ? params.attn_scale : rsqrt(float(hd));
    float qjl_scale = 1.2533141f / sqrt(float(hd)); // sqrt(pi/2) / sqrt(hd)
    uint code_bytes = (hd * bits + 7) / 8;           // stage-1 code bytes per head

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
        // Layout: [2B norm] [2B gamma] [stage1 codes] [sign bits]
        device const uchar* k_pos_base = k_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;

        // Read K norm + gamma, compute QJL correction scalar.
        float k_norm = float(*(device const bfloat*)k_pos_base);
        float k_correction = float(*(device const bfloat*)(k_pos_base + 2)) * qjl_scale * k_norm;
        device const uchar* k_codes = k_pos_base + 4;
        device const uchar* k_signs = k_pos_base + 4 + code_bytes;

        // Score = Q_rot · dequant(K)
        // dequant[j] = centroid[code_j] * k_norm + k_correction * sign_j
        // Batch-read sign bits as uint32 (32 signs per read) for efficiency.
        float score = 0.0f;
        for (uint j = 0; j < hd; j += 32) {
            uint sign_word = *(device const uint*)(k_signs + (j >> 3));
            uint end = min(j + 32, hd);
            for (uint d = j; d < end; d++) {
                uint code = extract_code(k_codes, d, bits);
                float sign_val = ((sign_word >> (d - j)) & 1u) ? 1.0f : -1.0f;
                score += q_shared[d] * (shared_centroids[code] * k_norm + k_correction * sign_val);
            }
        }

        score *= scale;

        // Online softmax update — rescale ALL V dimensions on new max.
        if (score > local_max) {
            float rescale = exp(local_max - score);
            for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }

        float weight = exp(score - local_max);

        // Accumulate weighted V in rotated space — ALL dimensions.
        // Sparse V dequantization: skip V dequant when weight is negligible.
        if (weight > 1e-6f) {
            device const uchar* v_pos_base = v_pool + pool_pos * bytes_per_pos + kv_head * bytes_per_head_pos;
            float v_norm = float(*(device const bfloat*)v_pos_base);
            float v_correction = float(*(device const bfloat*)(v_pos_base + 2)) * qjl_scale;
            device const uchar* v_codes = v_pos_base + 4;
            device const uchar* v_signs = v_pos_base + 4 + code_bytes;

            float wv_norm = weight * v_norm;
            float wv_correction = weight * v_correction;
            for (uint j = 0; j < hd; j += 32) {
                uint sign_word = *(device const uint*)(v_signs + (j >> 3));
                uint end = min(j + 32, hd);
                for (uint d = j; d < end; d++) {
                    uint v_code = extract_code(v_codes, d, bits);
                    float sign_val = ((sign_word >> (d - j)) & 1u) ? 1.0f : -1.0f;
                    v_acc[d] += shared_centroids[v_code] * wv_norm + wv_correction * sign_val;
                }
            }
        }
    }

    // Handle attention sinks.
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float rescale = exp(local_max - sink_score);
            for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax + V reduction. ---
    // Each thread has partial softmax state and partial V sums from its
    // subset of positions.  We need to combine them into a single global
    // softmax-normalised V output.
    //
    // Strategy: reduce (max, sum_exp) across threads, rescale each thread's
    // V accumulators, then reduce V across threads via shared memory.
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];

    uint simd_group = tid / 32;
    uint simd_lane = tid % 32;

    // SIMD-level max reduction.
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
    for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;

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

    // Normalise V accumulators.
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (uint d = 0; d < hd; d++) v_acc[d] *= inv_sum;

    // --- Cross-thread V reduction + Pi^T inverse rotation. ---
    //
    // Mirrors the BF16 kernel's reduce_v_and_write pattern:
    //   1. SIMD-level reduction: simd_sum() across 32 lanes per SIMD group
    //   2. Lane 0 of each group writes head_dim partial sums to shared memory
    //   3. Final reduction: threads 0..head_dim-1 sum across 8 SIMD groups
    //   4. Apply Pi^T rotation to get from rotated space to original
    //
    // Shared memory: 8 SIMD groups × 256 dims = 2048 floats = 8KB.
    threadgroup float shared_reduce[8 * 256];  // [num_simd_groups × max_head_dim]
    uint num_groups = (tg_size + 31) / 32;

    // Step 1: SIMD-level reduction of v_acc across 32 lanes.
    for (uint d = 0; d < hd; d++) {
        v_acc[d] = simd_sum(v_acc[d]);
    }

    // Step 2: Lane 0 of each SIMD group writes its partial sums.
    if (simd_lane == 0) {
        for (uint d = 0; d < hd; d++) {
            shared_reduce[simd_group * hd + d] = v_acc[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3+4: Threads 0..head_dim-1 do final reduction + Pi^T rotation.
    // For output dimension tid: sum across SIMD groups to get the full
    // rotated V vector, then multiply by Pi^T row tid.
    if (tid < hd) {
        // First, reduce the rotated V from shared memory into a local copy.
        // Then apply Pi^T[tid, :] · v_rot[:].
        float v_out = 0.0f;
        for (uint j = 0; j < hd; j++) {
            float v_rot_j = 0.0f;
            for (uint g = 0; g < num_groups; g++) {
                v_rot_j += shared_reduce[g * hd + j];
            }
            v_out += pi_t[tid * hd + j] * v_rot_j;
        }
        output[head_id * hd + tid] = bfloat(v_out);
    }
}

// ---------------------------------------------------------------------------
// turbo_paged_attention_v_only — asymmetric attention: BF16 K + TurboQuant V.
//
// For models with QKV bias (Qwen2, GPT-OSS), K quantization produces
// correlated errors that softmax amplifies.  V is tolerant because errors
// average out in weighted sums.  This kernel scores Q against BF16 K
// (standard dot product, no rotation) and accumulates turbo-quantized V
// (centroid dequant in rotated space, Pi^T inverse rotation at the end).
//
// K scoring follows the BF16 paged_attention pattern (attention.metal).
// V accumulation follows the turbo_paged_attention pattern above.
//
// Dispatch: num_heads threadgroups, 256 threads each.
// ---------------------------------------------------------------------------

kernel void turbo_paged_attention_v_only(
    constant TurboPagedAttentionVOnlyParams& params [[buffer(0)]],
    device const bfloat* q               [[buffer(1)]],   // [num_heads, head_dim] bf16 (NOT rotated)
    device const bfloat* k_pool          [[buffer(2)]],   // BF16 paged K pool
    device const uchar* v_pool           [[buffer(3)]],   // quantized paged V pool
    device const uint* block_table       [[buffer(4)]],
    constant float* pi_t                 [[buffer(5)]],   // [head_dim, head_dim] f32
    constant float* centroids            [[buffer(6)]],   // [num_centroids] f32
    device bfloat* output                [[buffer(7)]],   // [num_heads, head_dim] bf16
    device const bfloat* sinks           [[buffer(8)]],   // [num_heads] or dummy
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
    uint kv_dim = params.kv_dim;
    uint v_bytes_per_head_pos = params.v_bytes_per_head_pos;
    uint block_size = params.block_size;
    uint num_centroids = params.num_centroids;

    // GQA: map query head to KV head.
    uint heads_per_kv = params.num_heads / num_kv_heads;
    uint kv_head = head_id / heads_per_kv;

    // Load centroids into shared memory (for V dequant).
    threadgroup float shared_centroids[16];
    if (tid < num_centroids) {
        shared_centroids[tid] = centroids[tid];
    }

    // Load Q into shared memory (bf16, NOT rotated — K is BF16).
    threadgroup float q_shared[256];
    if (tid < hd) {
        q_shared[tid] = float(q[head_id * hd + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread V accumulator in rotated space + online softmax state.
    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;
    float v_acc[256] = {};  // max head_dim; zero-initialised

    // V pool addressing: bytes per position across all V KV heads.
    uint v_bytes_per_pos = num_kv_heads * v_bytes_per_head_pos;

    // Attention scale and QJL constants.
    float scale = (params.attn_scale != 0.0f) ? params.attn_scale : rsqrt(float(hd));
    float qjl_scale = 1.2533141f / sqrt(float(hd));
    uint v_code_bytes = (hd * bits + 7) / 8;

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

        // --- K scoring: standard BF16 dot product (same as paged_attention). ---
        device const bfloat* k_vec = k_pool + pool_pos * kv_dim + kv_head * hd;
        float score = 0.0f;
        for (uint j = 0; j < hd; j++) {
            score += q_shared[j] * float(k_vec[j]);
        }
        score *= scale;

        // Online softmax update — rescale ALL V dimensions on new max.
        if (score > local_max) {
            float rescale = exp(local_max - score);
            for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = score;
        } else {
            local_sum_exp += exp(score - local_max);
        }

        float weight = exp(score - local_max);

        // --- V accumulation: turbo-quantized dequant in rotated space. ---
        // Sparse V dequantization: skip when weight is negligible.
        if (weight > 1e-6f) {
            device const uchar* v_pos_base = v_pool + pool_pos * v_bytes_per_pos + kv_head * v_bytes_per_head_pos;
            float v_norm = float(*(device const bfloat*)v_pos_base);
            float v_correction = float(*(device const bfloat*)(v_pos_base + 2)) * qjl_scale;
            device const uchar* v_codes = v_pos_base + 4;
            device const uchar* v_signs = v_pos_base + 4 + v_code_bytes;

            float wv_norm = weight * v_norm;
            float wv_correction = weight * v_correction;
            for (uint j = 0; j < hd; j += 32) {
                uint sign_word = *(device const uint*)(v_signs + (j >> 3));
                uint end = min(j + 32, hd);
                for (uint d = j; d < end; d++) {
                    uint v_code = extract_code(v_codes, d, bits);
                    float sign_val = ((sign_word >> (d - j)) & 1u) ? 1.0f : -1.0f;
                    v_acc[d] += shared_centroids[v_code] * wv_norm + wv_correction * sign_val;
                }
            }
        }
    }

    // Handle attention sinks.
    if (params.has_sinks && tid == 0) {
        float sink_score = float(sinks[head_id]);
        if (sink_score > local_max) {
            float rescale = exp(local_max - sink_score);
            for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;
            local_sum_exp = local_sum_exp * rescale + 1.0f;
            local_max = sink_score;
        } else {
            local_sum_exp += exp(sink_score - local_max);
        }
    }

    // --- Cross-thread softmax + V reduction (same as turbo_paged_attention). ---
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];

    uint simd_group = tid / 32;
    uint simd_lane = tid % 32;

    float group_max = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_group] = group_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    float rescale = exp(local_max - global_max);
    local_sum_exp *= rescale;
    for (uint d = 0; d < hd; d++) v_acc[d] *= rescale;

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

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (uint d = 0; d < hd; d++) v_acc[d] *= inv_sum;

    // --- Cross-thread V reduction + Pi^T inverse rotation. ---
    // V was accumulated in rotated space (turbo centroids), so we need Pi^T
    // to get back to original space.
    threadgroup float shared_reduce[8 * 256];
    uint num_groups = (tg_size + 31) / 32;

    for (uint d = 0; d < hd; d++) {
        v_acc[d] = simd_sum(v_acc[d]);
    }

    if (simd_lane == 0) {
        for (uint d = 0; d < hd; d++) {
            shared_reduce[simd_group * hd + d] = v_acc[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < hd) {
        float v_out = 0.0f;
        for (uint j = 0; j < hd; j++) {
            float v_rot_j = 0.0f;
            for (uint g = 0; g < num_groups; g++) {
                v_rot_j += shared_reduce[g * hd + j];
            }
            v_out += pi_t[tid * hd + j] * v_rot_j;
        }
        output[head_id * hd + tid] = bfloat(v_out);
    }
}
