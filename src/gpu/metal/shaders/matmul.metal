// ===========================================================================
// Matrix-vector multiply kernels — multi-row SIMD with shared memory caching.
//
// LEARNING OVERVIEW
//
// What these kernels do:
//   Compute y = W · x, where W is a weight matrix and x is a vector.
//   This is the fundamental operation for all linear projections in the
//   transformer: Q/K/V projections, output projection, gate/up/down FFN
//   projections, and the final LM head.
//
// Multi-row SIMD (the key optimisation, inspired by llama.cpp):
//   Each SIMD group of 32 threads computes TWO output rows instead of one.
//   The input vector x is loaded once per iteration and reused for both rows,
//   giving the GPU two independent accumulator chains to pipeline:
//
//     for each x chunk:
//       x_val = x[j]              // loaded once
//       acc0 += w_row0[j] * x_val // row 0 dot product
//       acc1 += w_row1[j] * x_val // row 1 dot product (x reused, free ILP)
//
//   With 2 rows per SIMD, 8 SIMD groups per threadgroup = 16 rows per
//   threadgroup (up from 8).  The two accumulations are independent so the
//   GPU can pipeline loads and FMAs without stalls.
//
// Shared memory x caching:
//   All 256 threads cooperatively load the input vector x into threadgroup
//   shared memory (bf16 → f32 during load).  This means x is read from
//   device memory ONCE per threadgroup instead of once per SIMD group —
//   an 8× reduction in x memory traffic.  Combined with multi-row, each
//   threadgroup processes 16 rows with a single x load.
//
//   For a 4096-dim model, this eliminates ~56 KB of redundant x reads per
//   threadgroup (7 redundant SIMD groups × 8 KB per x vector).
//
// Dispatch model:
//   grid_size = ceil(M / 2) × 32 total threads.
//   threadgroup_size = 256 = 8 SIMD groups = 16 output rows per threadgroup.
//
// Precision:
//   Inputs (W and x) are bfloat16.  Dot-product accumulators are float32.
//   Final results are narrowed back to bfloat16 on store.
//
// Related files:
//   Trait contract:  gpu/ops/matmul.rs
//   Metal dispatch:  metal/kernels/matmul.rs
//   Fused variant:   metal/shaders/moe.metal (fused_gate_up_swiglu)
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `MatvecParams`.
struct MatvecParams {
    uint M;  // Output dimension (number of rows in W, length of y).
    uint K;  // Input dimension (number of columns in W, length of x).
};

// Number of output rows each SIMD group computes simultaneously.
// 2 rows per SIMD = 16 rows per 256-thread threadgroup (bf16).
// Q4 uses 1 row per SIMD to avoid register pressure from dequant.
// Each thread maintains ROWS_PER_SIMD independent float accumulators.
constant constexpr uint ROWS_PER_SIMD = 2;
constant constexpr uint ROWS_PER_SIMD_Q4 = 2;
constant constexpr uint ROWS_PER_SIMD_Q8 = 2;
constant constexpr uint ROWS_PER_SIMD_TQ3 = 2;

// Shared memory tile size for input vector caching.  4096 floats = 16 KB,
// well within Apple Silicon's 32 KB threadgroup memory limit.  Covers
// K ≤ 4096 in a single tile (all standard hidden sizes).  Larger K
// is handled by iterating over multiple tiles.
#define X_TILE 4096

kernel void matvec_bf16(
    constant MatvecParams& params [[buffer(0)]],
    device const bfloat* W        [[buffer(1)]],  // [M, K] row-major
    device const bfloat* x        [[buffer(2)]],  // [K]
    device bfloat* y              [[buffer(3)]],  // [M]
    uint gid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    // 32 threads (one SIMD group) cooperate on ROWS_PER_SIMD output rows.
    //   simd_idx = which pair of rows this SIMD group handles
    //   lane     = which thread within the SIMD group (0..31)
    //   row0     = first output row,  row1 = second output row
    uint simd_idx = gid / 32;
    uint lane = gid % 32;

    uint row0 = simd_idx * ROWS_PER_SIMD;
    uint row1 = row0 + 1;
    bool has_row1 = (row1 < M);

    device const bfloat* w_row0 = W + row0 * K;
    device const bfloat* w_row1 = W + row1 * K;

    // -----------------------------------------------------------------------
    // Shared memory cache for input vector x.
    //
    // All 256 threads cooperatively load x from device memory (bf16) into
    // shared memory (f32).  The bf16→f32 conversion happens during the load,
    // so the inner loop reads f32 from fast shared memory.
    //
    // Without caching: 8 SIMD groups each read x independently → 8× redundant.
    // With caching: one cooperative load, 8× less device memory traffic for x.
    //
    // For K > X_TILE, we tile: load a chunk, process, barrier, next chunk.
    // -----------------------------------------------------------------------
    threadgroup float x_shared[X_TILE];

    float acc0 = 0.0f, acc1 = 0.0f;

    for (uint tile = 0; tile < K; tile += X_TILE) {
        uint tile_len = min((uint)X_TILE, K - tile);

        // Cooperative load: 256 threads load tile_len bf16→f32 values.
        // ALL threads participate (even out-of-bounds rows) to ensure the
        // barrier is reached by every thread in the threadgroup.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------
        // Multi-row dot product with 4x unrolling.
        //
        // Each lane processes elements at indices: lane*4, lane*4+128, ...
        // For each element, the x value is loaded ONCE from shared memory
        // and used for both row0 and row1 — the core of multi-row SIMD.
        //
        // The two accumulations (acc0, acc1) are independent, so the GPU
        // can pipeline them: while acc0's FMA is in flight, the GPU issues
        // acc1's FMA on the same data.  This hides FMA latency.
        // ---------------------------------------------------------------
        if (row0 < M) {
            for (uint j = lane * 4; j < tile_len; j += 32 * 4) {
                float x0 = x_shared[j];
                float x1 = x_shared[j + 1];
                float x2 = x_shared[j + 2];
                float x3 = x_shared[j + 3];

                uint wj = tile + j;
                acc0 += float(w_row0[wj])     * x0;
                acc0 += float(w_row0[wj + 1]) * x1;
                acc0 += float(w_row0[wj + 2]) * x2;
                acc0 += float(w_row0[wj + 3]) * x3;

                if (has_row1) {
                    acc1 += float(w_row1[wj])     * x0;
                    acc1 += float(w_row1[wj + 1]) * x1;
                    acc1 += float(w_row1[wj + 2]) * x2;
                    acc1 += float(w_row1[wj + 3]) * x3;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // SIMD reduction: sum across all 32 lanes.
    if (row0 < M) {
        acc0 = simd_sum(acc0);
        if (lane == 0) y[row0] = bfloat(acc0);
    }
    if (has_row1) {
        acc1 = simd_sum(acc1);
        if (lane == 0) y[row1] = bfloat(acc1);
    }
}

// ===========================================================================
// Matrix-vector multiply kernel (Q4) — multi-row SIMD with fused dequant,
// packed uint loads, split accumulators, and shared memory x caching.
//
// LEARNING OVERVIEW
//
// Four GGML-inspired optimisations over the original kernel:
//
// 1. Fused dequant-multiply-accumulate:
//    Instead of precomputing 8 sx = scale*x intermediates (heavy register
//    pressure), compute neg8s = -8*scale once per block, then per weight:
//      w = fma(nibble, scale, neg8s)   // (nibble - 8) * scale
//      acc = fma(w, x_val, acc)        // accumulate
//    Same 2 FMAs per weight, but only 1 x_val register live at a time
//    instead of 8 sx registers.
//
// 2. Multi-row SIMD (2 rows per SIMD group):
//    Matches the bf16 kernel — each SIMD group computes 2 output rows,
//    sharing x values from shared memory.  Enabled by the register savings
//    from fused dequant.  Doubles useful compute per x-vector load.
//
// 3. Packed uint loads:
//    Read 4 nibble bytes as a single uint instead of 4 separate uchars.
//    Extract with shifts: (packed >> k*4) & 0xF for weight k.
//    4 wide loads per block instead of 16 narrow loads.
//
// 4. Split accumulators:
//    Two independent accumulators per row (acc_a for weights 0-15, acc_b
//    for weights 16-31).  Breaks the serial dependency chain, giving the
//    GPU two independent FMA streams to pipeline.
//
// Block layout (18 bytes per block of 32 weights):
//   bytes  0-1:  bf16 scale factor
//   bytes  2-17: 16 bytes of packed nibbles (2 weights per byte)
//
// Packing: byte[i] = (q[2i] & 0xF) | (q[2i+1] << 4)
// Dequant: weight = (nibble - 8) * scale
//
// Dispatch: grid_size = ceil(M / 2) × 32, threadgroup_size = 256.
// ===========================================================================

kernel void matvec_q4(
    constant MatvecParams& params [[buffer(0)]],
    device const uchar* W_q4     [[buffer(1)]],  // [M * blocks_per_row * 18 bytes]
    device const bfloat* x       [[buffer(2)]],  // [K]
    device bfloat* y             [[buffer(3)]],  // [M]
    uint gid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint simd_idx = gid / 32;
    uint lane = gid % 32;

    // Multi-row: each SIMD group computes 2 output rows.
    uint row0 = simd_idx * ROWS_PER_SIMD_Q4;
    uint row1 = row0 + 1;
    bool has_row1 = (row1 < M);

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 18;  // 2 (bf16 scale) + 16 (packed nibbles)

    device const uchar* row0_data = W_q4 + row0 * blocks_per_row * bytes_per_block;
    device const uchar* row1_data = W_q4 + row1 * blocks_per_row * bytes_per_block;

    // Shared memory cache for input vector x.
    threadgroup float x_shared[X_TILE];

    // Split accumulators: acc_a for weights 0-15, acc_b for 16-31.
    float acc0_a = 0.0f, acc0_b = 0.0f;
    float acc1_a = 0.0f, acc1_b = 0.0f;

    for (uint tile = 0; tile < K; tile += X_TILE) {
        uint tile_len = min((uint)X_TILE, K - tile);

        // Cooperative load: 256 threads load tile_len bf16→f32 values.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < M) {
            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                // Row 0 block header.
                device const uchar* bp0 = row0_data + block_idx * bytes_per_block;
                float scale0 = float(*((device const bfloat*)bp0));
                float neg8s0 = -8.0f * scale0;
                device const uchar* d0 = bp0 + 2;

                // Row 1 block header (conditional).
                float scale1, neg8s1;
                device const uchar* d1;
                if (has_row1) {
                    device const uchar* bp1 = row1_data + block_idx * bytes_per_block;
                    scale1 = float(*((device const bfloat*)bp1));
                    neg8s1 = -8.0f * scale1;
                    d1 = bp1 + 2;
                }

                uint xb = (block_idx * 32) - tile;

                // --- Chunk 0: weights 0-7, packed uint load → acc_a ---
                {
                    uint pk = uint(d0[0]) | (uint(d0[1]) << 8) | (uint(d0[2]) << 16) | (uint(d0[3]) << 24);
                    float x0 = x_shared[xb],     x1 = x_shared[xb + 1];
                    float x2 = x_shared[xb + 2], x3 = x_shared[xb + 3];
                    float x4 = x_shared[xb + 4], x5 = x_shared[xb + 5];
                    float x6 = x_shared[xb + 6], x7 = x_shared[xb + 7];

                    acc0_a = fma(fma(float(pk & 0xF),         scale0, neg8s0), x0, acc0_a);
                    acc0_a = fma(fma(float((pk >> 4) & 0xF),  scale0, neg8s0), x1, acc0_a);
                    acc0_a = fma(fma(float((pk >> 8) & 0xF),  scale0, neg8s0), x2, acc0_a);
                    acc0_a = fma(fma(float((pk >> 12) & 0xF), scale0, neg8s0), x3, acc0_a);
                    acc0_a = fma(fma(float((pk >> 16) & 0xF), scale0, neg8s0), x4, acc0_a);
                    acc0_a = fma(fma(float((pk >> 20) & 0xF), scale0, neg8s0), x5, acc0_a);
                    acc0_a = fma(fma(float((pk >> 24) & 0xF), scale0, neg8s0), x6, acc0_a);
                    acc0_a = fma(fma(float((pk >> 28)),        scale0, neg8s0), x7, acc0_a);

                    if (has_row1) {
                        uint p1 = uint(d1[0]) | (uint(d1[1]) << 8) | (uint(d1[2]) << 16) | (uint(d1[3]) << 24);
                        acc1_a = fma(fma(float(p1 & 0xF),         scale1, neg8s1), x0, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 4) & 0xF),  scale1, neg8s1), x1, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 8) & 0xF),  scale1, neg8s1), x2, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 12) & 0xF), scale1, neg8s1), x3, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 16) & 0xF), scale1, neg8s1), x4, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 20) & 0xF), scale1, neg8s1), x5, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 24) & 0xF), scale1, neg8s1), x6, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 28)),        scale1, neg8s1), x7, acc1_a);
                    }
                }

                // --- Chunk 1: weights 8-15 → acc_a ---
                {
                    uint pk = uint(d0[4]) | (uint(d0[5]) << 8) | (uint(d0[6]) << 16) | (uint(d0[7]) << 24);
                    float x0 = x_shared[xb + 8],  x1 = x_shared[xb + 9];
                    float x2 = x_shared[xb + 10], x3 = x_shared[xb + 11];
                    float x4 = x_shared[xb + 12], x5 = x_shared[xb + 13];
                    float x6 = x_shared[xb + 14], x7 = x_shared[xb + 15];

                    acc0_a = fma(fma(float(pk & 0xF),         scale0, neg8s0), x0, acc0_a);
                    acc0_a = fma(fma(float((pk >> 4) & 0xF),  scale0, neg8s0), x1, acc0_a);
                    acc0_a = fma(fma(float((pk >> 8) & 0xF),  scale0, neg8s0), x2, acc0_a);
                    acc0_a = fma(fma(float((pk >> 12) & 0xF), scale0, neg8s0), x3, acc0_a);
                    acc0_a = fma(fma(float((pk >> 16) & 0xF), scale0, neg8s0), x4, acc0_a);
                    acc0_a = fma(fma(float((pk >> 20) & 0xF), scale0, neg8s0), x5, acc0_a);
                    acc0_a = fma(fma(float((pk >> 24) & 0xF), scale0, neg8s0), x6, acc0_a);
                    acc0_a = fma(fma(float((pk >> 28)),        scale0, neg8s0), x7, acc0_a);

                    if (has_row1) {
                        uint p1 = uint(d1[4]) | (uint(d1[5]) << 8) | (uint(d1[6]) << 16) | (uint(d1[7]) << 24);
                        acc1_a = fma(fma(float(p1 & 0xF),         scale1, neg8s1), x0, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 4) & 0xF),  scale1, neg8s1), x1, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 8) & 0xF),  scale1, neg8s1), x2, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 12) & 0xF), scale1, neg8s1), x3, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 16) & 0xF), scale1, neg8s1), x4, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 20) & 0xF), scale1, neg8s1), x5, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 24) & 0xF), scale1, neg8s1), x6, acc1_a);
                        acc1_a = fma(fma(float((p1 >> 28)),        scale1, neg8s1), x7, acc1_a);
                    }
                }

                // --- Chunk 2: weights 16-23 → acc_b ---
                {
                    uint pk = uint(d0[8]) | (uint(d0[9]) << 8) | (uint(d0[10]) << 16) | (uint(d0[11]) << 24);
                    float x0 = x_shared[xb + 16], x1 = x_shared[xb + 17];
                    float x2 = x_shared[xb + 18], x3 = x_shared[xb + 19];
                    float x4 = x_shared[xb + 20], x5 = x_shared[xb + 21];
                    float x6 = x_shared[xb + 22], x7 = x_shared[xb + 23];

                    acc0_b = fma(fma(float(pk & 0xF),         scale0, neg8s0), x0, acc0_b);
                    acc0_b = fma(fma(float((pk >> 4) & 0xF),  scale0, neg8s0), x1, acc0_b);
                    acc0_b = fma(fma(float((pk >> 8) & 0xF),  scale0, neg8s0), x2, acc0_b);
                    acc0_b = fma(fma(float((pk >> 12) & 0xF), scale0, neg8s0), x3, acc0_b);
                    acc0_b = fma(fma(float((pk >> 16) & 0xF), scale0, neg8s0), x4, acc0_b);
                    acc0_b = fma(fma(float((pk >> 20) & 0xF), scale0, neg8s0), x5, acc0_b);
                    acc0_b = fma(fma(float((pk >> 24) & 0xF), scale0, neg8s0), x6, acc0_b);
                    acc0_b = fma(fma(float((pk >> 28)),        scale0, neg8s0), x7, acc0_b);

                    if (has_row1) {
                        uint p1 = uint(d1[8]) | (uint(d1[9]) << 8) | (uint(d1[10]) << 16) | (uint(d1[11]) << 24);
                        acc1_b = fma(fma(float(p1 & 0xF),         scale1, neg8s1), x0, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 4) & 0xF),  scale1, neg8s1), x1, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 8) & 0xF),  scale1, neg8s1), x2, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 12) & 0xF), scale1, neg8s1), x3, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 16) & 0xF), scale1, neg8s1), x4, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 20) & 0xF), scale1, neg8s1), x5, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 24) & 0xF), scale1, neg8s1), x6, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 28)),        scale1, neg8s1), x7, acc1_b);
                    }
                }

                // --- Chunk 3: weights 24-31 → acc_b ---
                {
                    uint pk = uint(d0[12]) | (uint(d0[13]) << 8) | (uint(d0[14]) << 16) | (uint(d0[15]) << 24);
                    float x0 = x_shared[xb + 24], x1 = x_shared[xb + 25];
                    float x2 = x_shared[xb + 26], x3 = x_shared[xb + 27];
                    float x4 = x_shared[xb + 28], x5 = x_shared[xb + 29];
                    float x6 = x_shared[xb + 30], x7 = x_shared[xb + 31];

                    acc0_b = fma(fma(float(pk & 0xF),         scale0, neg8s0), x0, acc0_b);
                    acc0_b = fma(fma(float((pk >> 4) & 0xF),  scale0, neg8s0), x1, acc0_b);
                    acc0_b = fma(fma(float((pk >> 8) & 0xF),  scale0, neg8s0), x2, acc0_b);
                    acc0_b = fma(fma(float((pk >> 12) & 0xF), scale0, neg8s0), x3, acc0_b);
                    acc0_b = fma(fma(float((pk >> 16) & 0xF), scale0, neg8s0), x4, acc0_b);
                    acc0_b = fma(fma(float((pk >> 20) & 0xF), scale0, neg8s0), x5, acc0_b);
                    acc0_b = fma(fma(float((pk >> 24) & 0xF), scale0, neg8s0), x6, acc0_b);
                    acc0_b = fma(fma(float((pk >> 28)),        scale0, neg8s0), x7, acc0_b);

                    if (has_row1) {
                        uint p1 = uint(d1[12]) | (uint(d1[13]) << 8) | (uint(d1[14]) << 16) | (uint(d1[15]) << 24);
                        acc1_b = fma(fma(float(p1 & 0xF),         scale1, neg8s1), x0, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 4) & 0xF),  scale1, neg8s1), x1, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 8) & 0xF),  scale1, neg8s1), x2, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 12) & 0xF), scale1, neg8s1), x3, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 16) & 0xF), scale1, neg8s1), x4, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 20) & 0xF), scale1, neg8s1), x5, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 24) & 0xF), scale1, neg8s1), x6, acc1_b);
                        acc1_b = fma(fma(float((p1 >> 28)),        scale1, neg8s1), x7, acc1_b);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Merge split accumulators, then SIMD reduce.
    if (row0 < M) {
        float acc0 = acc0_a + acc0_b;
        acc0 = simd_sum(acc0);
        if (lane == 0) y[row0] = bfloat(acc0);
    }
    if (has_row1) {
        float acc1 = acc1_a + acc1_b;
        acc1 = simd_sum(acc1);
        if (lane == 0) y[row1] = bfloat(acc1);
    }
}

// ===========================================================================
// Batched matrix multiply (GEMM) — bf16 weights.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Computes Y = X @ W^T, where X is [batch_size, K] and W is [M, K].
//   Output Y is [batch_size, M].  This is the batched version of matvec —
//   instead of multiplying one input vector by the weight matrix, we multiply
//   a whole batch of vectors at once.
//
// Why this is the most important optimisation for prefill:
//
//   The fundamental problem with token-by-token prefill is arithmetic
//   intensity — the ratio of compute (FLOPs) to memory traffic (bytes).
//
//   SINGLE MATVEC (one token):
//     FLOPs:  2 × M × K           (one multiply-add per weight element)
//     Memory: M × K × 2 bytes     (load entire weight matrix) + K × 2 (input)
//     Intensity: ≈ 2 FLOPs/byte   → MEMORY BOUND
//     The GPU loads ~8 MB of weights but only does ~8M multiply-adds.
//     At 546 GB/s bandwidth, the GPU finishes in ~15μs — but the ALUs are
//     idle 99% of the time, waiting for data.
//
//   BATCHED GEMM (B tokens):
//     FLOPs:  2 × B × M × K       (B times more compute)
//     Memory: M × K × 2 bytes     (load weight matrix ONCE) + B × K × 2
//     Intensity: ≈ 2B FLOPs/byte  → COMPUTE BOUND for B ≥ ~50
//     For B=100: the same weight matrix is loaded once but used for 100
//     dot products.  The GPU's ALUs are now the bottleneck, not memory.
//     This is where the 3-10x speedup comes from.
//
// Dispatch model:
//   Same SIMD-cooperative pattern as matvec: 32 threads per output element.
//   Grid: batch_size * M * 32 threads.  Thread (gid) computes:
//     batch = (gid / 32) / M
//     row   = (gid / 32) % M
//     lane  = gid % 32
//   Output: y[batch * M + row] = dot(W[row], X[batch])
//
// Note: this kernel reuses the same SIMD-cooperative structure as matvec.
// A more advanced implementation would use tiling (loading blocks of W and
// X into threadgroup memory) for better cache behaviour, but the simple
// approach already achieves the key insight: weight reuse across batch.
// ===========================================================================

struct GemmParams {
    uint batch_size;
    uint M;  // Output dimension (rows of W).
    uint K;  // Input dimension (cols of W).
};

kernel void gemm_bf16(
    constant GemmParams& params  [[buffer(0)]],
    device const bfloat* W       [[buffer(1)]],  // [M, K]
    device const bfloat* X       [[buffer(2)]],  // [batch_size, K]
    device bfloat* Y             [[buffer(3)]],  // [batch_size, M]
    uint gid                     [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint elem = gid / 32;  // Which (batch, row) pair.
    uint lane = gid % 32;

    uint batch = elem / M;
    uint row   = elem % M;

    if (batch >= params.batch_size) return;

    device const bfloat* w_row = W + row * K;
    device const bfloat* x_vec = X + batch * K;

    float acc = 0.0f;
    for (uint j = lane * 4; j < K; j += 32 * 4) {
        acc += float(w_row[j])     * float(x_vec[j]);
        acc += float(w_row[j + 1]) * float(x_vec[j + 1]);
        acc += float(w_row[j + 2]) * float(x_vec[j + 2]);
        acc += float(w_row[j + 3]) * float(x_vec[j + 3]);
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = bfloat(acc);
    }
}

// ===========================================================================
// Batched matrix multiply (GEMM) — Q4 weights.
//
// Same optimisations as matvec_q4 except multi-row (each SIMD group still
// handles one (batch, row) element since GEMM parallelises across batch):
//   1. Fused dequant: neg8s = -8*scale once per block.
//   3. Packed uint loads: 4 uint loads per block instead of 16 uchar.
//   4. Split accumulators: acc_a (first 16 weights), acc_b (last 16).
//
// Q4 GEMM compounds TWO bandwidth savings:
//   1. Batching: weight matrix loaded once for B tokens (B× fewer loads)
//   2. Quantisation: 18 bytes per 32 weights vs 64 bytes (3.6× smaller)
//   Combined: for B=100, memory traffic drops ~360× compared to 100 bf16 matvecs.
// ===========================================================================

kernel void gemm_q4(
    constant GemmParams& params  [[buffer(0)]],
    device const uchar* W_q4     [[buffer(1)]],  // [M * blocks_per_row * 18 bytes]
    device const bfloat* X       [[buffer(2)]],  // [batch_size, K]
    device bfloat* Y             [[buffer(3)]],  // [batch_size, M]
    uint gid                     [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint elem = gid / 32;
    uint lane = gid % 32;

    uint batch = elem / M;
    uint row   = elem % M;

    if (batch >= params.batch_size) return;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 18;

    device const uchar* row_data = W_q4 + row * blocks_per_row * bytes_per_block;
    device const bfloat* x_vec = X + batch * K;

    float acc_a = 0.0f, acc_b = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* block_ptr = row_data + block_idx * bytes_per_block;
        float scale = float(*((device const bfloat*)block_ptr));
        float neg8s = -8.0f * scale;
        device const uchar* data = block_ptr + 2;
        uint xb = block_idx * 32;

        // Chunk 0: weights 0-7 → acc_a.
        {
            uint pk = uint(data[0]) | (uint(data[1]) << 8) | (uint(data[2]) << 16) | (uint(data[3]) << 24);
            float x0 = float(x_vec[xb]),     x1 = float(x_vec[xb + 1]);
            float x2 = float(x_vec[xb + 2]), x3 = float(x_vec[xb + 3]);
            float x4 = float(x_vec[xb + 4]), x5 = float(x_vec[xb + 5]);
            float x6 = float(x_vec[xb + 6]), x7 = float(x_vec[xb + 7]);

            acc_a = fma(fma(float(pk & 0xF),         scale, neg8s), x0, acc_a);
            acc_a = fma(fma(float((pk >> 4) & 0xF),  scale, neg8s), x1, acc_a);
            acc_a = fma(fma(float((pk >> 8) & 0xF),  scale, neg8s), x2, acc_a);
            acc_a = fma(fma(float((pk >> 12) & 0xF), scale, neg8s), x3, acc_a);
            acc_a = fma(fma(float((pk >> 16) & 0xF), scale, neg8s), x4, acc_a);
            acc_a = fma(fma(float((pk >> 20) & 0xF), scale, neg8s), x5, acc_a);
            acc_a = fma(fma(float((pk >> 24) & 0xF), scale, neg8s), x6, acc_a);
            acc_a = fma(fma(float((pk >> 28)),        scale, neg8s), x7, acc_a);
        }

        // Chunk 1: weights 8-15 → acc_a.
        {
            uint pk = uint(data[4]) | (uint(data[5]) << 8) | (uint(data[6]) << 16) | (uint(data[7]) << 24);
            float x0 = float(x_vec[xb + 8]),  x1 = float(x_vec[xb + 9]);
            float x2 = float(x_vec[xb + 10]), x3 = float(x_vec[xb + 11]);
            float x4 = float(x_vec[xb + 12]), x5 = float(x_vec[xb + 13]);
            float x6 = float(x_vec[xb + 14]), x7 = float(x_vec[xb + 15]);

            acc_a = fma(fma(float(pk & 0xF),         scale, neg8s), x0, acc_a);
            acc_a = fma(fma(float((pk >> 4) & 0xF),  scale, neg8s), x1, acc_a);
            acc_a = fma(fma(float((pk >> 8) & 0xF),  scale, neg8s), x2, acc_a);
            acc_a = fma(fma(float((pk >> 12) & 0xF), scale, neg8s), x3, acc_a);
            acc_a = fma(fma(float((pk >> 16) & 0xF), scale, neg8s), x4, acc_a);
            acc_a = fma(fma(float((pk >> 20) & 0xF), scale, neg8s), x5, acc_a);
            acc_a = fma(fma(float((pk >> 24) & 0xF), scale, neg8s), x6, acc_a);
            acc_a = fma(fma(float((pk >> 28)),        scale, neg8s), x7, acc_a);
        }

        // Chunk 2: weights 16-23 → acc_b.
        {
            uint pk = uint(data[8]) | (uint(data[9]) << 8) | (uint(data[10]) << 16) | (uint(data[11]) << 24);
            float x0 = float(x_vec[xb + 16]), x1 = float(x_vec[xb + 17]);
            float x2 = float(x_vec[xb + 18]), x3 = float(x_vec[xb + 19]);
            float x4 = float(x_vec[xb + 20]), x5 = float(x_vec[xb + 21]);
            float x6 = float(x_vec[xb + 22]), x7 = float(x_vec[xb + 23]);

            acc_b = fma(fma(float(pk & 0xF),         scale, neg8s), x0, acc_b);
            acc_b = fma(fma(float((pk >> 4) & 0xF),  scale, neg8s), x1, acc_b);
            acc_b = fma(fma(float((pk >> 8) & 0xF),  scale, neg8s), x2, acc_b);
            acc_b = fma(fma(float((pk >> 12) & 0xF), scale, neg8s), x3, acc_b);
            acc_b = fma(fma(float((pk >> 16) & 0xF), scale, neg8s), x4, acc_b);
            acc_b = fma(fma(float((pk >> 20) & 0xF), scale, neg8s), x5, acc_b);
            acc_b = fma(fma(float((pk >> 24) & 0xF), scale, neg8s), x6, acc_b);
            acc_b = fma(fma(float((pk >> 28)),        scale, neg8s), x7, acc_b);
        }

        // Chunk 3: weights 24-31 → acc_b.
        {
            uint pk = uint(data[12]) | (uint(data[13]) << 8) | (uint(data[14]) << 16) | (uint(data[15]) << 24);
            float x0 = float(x_vec[xb + 24]), x1 = float(x_vec[xb + 25]);
            float x2 = float(x_vec[xb + 26]), x3 = float(x_vec[xb + 27]);
            float x4 = float(x_vec[xb + 28]), x5 = float(x_vec[xb + 29]);
            float x6 = float(x_vec[xb + 30]), x7 = float(x_vec[xb + 31]);

            acc_b = fma(fma(float(pk & 0xF),         scale, neg8s), x0, acc_b);
            acc_b = fma(fma(float((pk >> 4) & 0xF),  scale, neg8s), x1, acc_b);
            acc_b = fma(fma(float((pk >> 8) & 0xF),  scale, neg8s), x2, acc_b);
            acc_b = fma(fma(float((pk >> 12) & 0xF), scale, neg8s), x3, acc_b);
            acc_b = fma(fma(float((pk >> 16) & 0xF), scale, neg8s), x4, acc_b);
            acc_b = fma(fma(float((pk >> 20) & 0xF), scale, neg8s), x5, acc_b);
            acc_b = fma(fma(float((pk >> 24) & 0xF), scale, neg8s), x6, acc_b);
            acc_b = fma(fma(float((pk >> 28)),        scale, neg8s), x7, acc_b);
        }
    }

    float acc = acc_a + acc_b;
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = bfloat(acc);
    }
}

// ===========================================================================
// Matrix-vector multiply — Q8 (8-bit) weights with shared memory x caching.
//
// Same multi-row SIMD structure as matvec_q4 but simpler dequantisation:
// each weight is a signed int8 byte, so dequant is just float(byte) * scale
// with no nibble extraction or offset.
//
// Block layout (34 bytes per block of 32 weights):
//   bytes 0-1:   bf16 scale factor
//   bytes 2-33:  32 signed int8 values
//
// Dequant: weight = float((device const char*)data)[i] * scale
//
// Dispatch: grid_size = ceil(M / 2) × 32, threadgroup_size = 256.
// ===========================================================================

kernel void matvec_q8(
    constant MatvecParams& params [[buffer(0)]],
    device const uchar* W_q8     [[buffer(1)]],  // [M * blocks_per_row * 34 bytes]
    device const bfloat* x       [[buffer(2)]],  // [K]
    device bfloat* y             [[buffer(3)]],  // [M]
    uint gid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint simd_idx = gid / 32;
    uint lane = gid % 32;

    // Multi-row: each SIMD group computes 2 output rows.
    uint row0 = simd_idx * ROWS_PER_SIMD_Q8;
    uint row1 = row0 + 1;
    bool has_row1 = (row1 < M);

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 34;  // 2 (bf16 scale) + 32 (int8 values)

    device const uchar* row0_data = W_q8 + row0 * blocks_per_row * bytes_per_block;
    device const uchar* row1_data = W_q8 + row1 * blocks_per_row * bytes_per_block;

    // Shared memory cache for input vector x.
    threadgroup float x_shared[X_TILE];

    // Split accumulators: acc_a for weights 0-15, acc_b for 16-31.
    float acc0_a = 0.0f, acc0_b = 0.0f;
    float acc1_a = 0.0f, acc1_b = 0.0f;

    for (uint tile = 0; tile < K; tile += X_TILE) {
        uint tile_len = min((uint)X_TILE, K - tile);

        // Cooperative load: 256 threads load tile_len bf16→f32 values.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < M) {
            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                // Row 0 block header.
                device const uchar* bp0 = row0_data + block_idx * bytes_per_block;
                float scale0 = float(*((device const bfloat*)bp0));
                device const char* d0 = (device const char*)(bp0 + 2);

                // Row 1 block header (conditional).
                float scale1;
                device const char* d1;
                if (has_row1) {
                    device const uchar* bp1 = row1_data + block_idx * bytes_per_block;
                    scale1 = float(*((device const bfloat*)bp1));
                    d1 = (device const char*)(bp1 + 2);
                }

                uint xb = (block_idx * 32) - tile;

                // --- Chunk 0: weights 0-3 → acc_a ---
                {
                    float x0 = x_shared[xb],     x1 = x_shared[xb + 1];
                    float x2 = x_shared[xb + 2], x3 = x_shared[xb + 3];
                    acc0_a = fma(float(d0[0]) * scale0, x0, acc0_a);
                    acc0_a = fma(float(d0[1]) * scale0, x1, acc0_a);
                    acc0_a = fma(float(d0[2]) * scale0, x2, acc0_a);
                    acc0_a = fma(float(d0[3]) * scale0, x3, acc0_a);
                    if (has_row1) {
                        acc1_a = fma(float(d1[0]) * scale1, x0, acc1_a);
                        acc1_a = fma(float(d1[1]) * scale1, x1, acc1_a);
                        acc1_a = fma(float(d1[2]) * scale1, x2, acc1_a);
                        acc1_a = fma(float(d1[3]) * scale1, x3, acc1_a);
                    }
                }

                // --- Chunk 1: weights 4-7 → acc_a ---
                {
                    float x0 = x_shared[xb + 4], x1 = x_shared[xb + 5];
                    float x2 = x_shared[xb + 6], x3 = x_shared[xb + 7];
                    acc0_a = fma(float(d0[4]) * scale0, x0, acc0_a);
                    acc0_a = fma(float(d0[5]) * scale0, x1, acc0_a);
                    acc0_a = fma(float(d0[6]) * scale0, x2, acc0_a);
                    acc0_a = fma(float(d0[7]) * scale0, x3, acc0_a);
                    if (has_row1) {
                        acc1_a = fma(float(d1[4]) * scale1, x0, acc1_a);
                        acc1_a = fma(float(d1[5]) * scale1, x1, acc1_a);
                        acc1_a = fma(float(d1[6]) * scale1, x2, acc1_a);
                        acc1_a = fma(float(d1[7]) * scale1, x3, acc1_a);
                    }
                }

                // --- Chunk 2: weights 8-11 → acc_a ---
                {
                    float x0 = x_shared[xb + 8],  x1 = x_shared[xb + 9];
                    float x2 = x_shared[xb + 10], x3 = x_shared[xb + 11];
                    acc0_a = fma(float(d0[8])  * scale0, x0, acc0_a);
                    acc0_a = fma(float(d0[9])  * scale0, x1, acc0_a);
                    acc0_a = fma(float(d0[10]) * scale0, x2, acc0_a);
                    acc0_a = fma(float(d0[11]) * scale0, x3, acc0_a);
                    if (has_row1) {
                        acc1_a = fma(float(d1[8])  * scale1, x0, acc1_a);
                        acc1_a = fma(float(d1[9])  * scale1, x1, acc1_a);
                        acc1_a = fma(float(d1[10]) * scale1, x2, acc1_a);
                        acc1_a = fma(float(d1[11]) * scale1, x3, acc1_a);
                    }
                }

                // --- Chunk 3: weights 12-15 → acc_a ---
                {
                    float x0 = x_shared[xb + 12], x1 = x_shared[xb + 13];
                    float x2 = x_shared[xb + 14], x3 = x_shared[xb + 15];
                    acc0_a = fma(float(d0[12]) * scale0, x0, acc0_a);
                    acc0_a = fma(float(d0[13]) * scale0, x1, acc0_a);
                    acc0_a = fma(float(d0[14]) * scale0, x2, acc0_a);
                    acc0_a = fma(float(d0[15]) * scale0, x3, acc0_a);
                    if (has_row1) {
                        acc1_a = fma(float(d1[12]) * scale1, x0, acc1_a);
                        acc1_a = fma(float(d1[13]) * scale1, x1, acc1_a);
                        acc1_a = fma(float(d1[14]) * scale1, x2, acc1_a);
                        acc1_a = fma(float(d1[15]) * scale1, x3, acc1_a);
                    }
                }

                // --- Chunk 4: weights 16-19 → acc_b ---
                {
                    float x0 = x_shared[xb + 16], x1 = x_shared[xb + 17];
                    float x2 = x_shared[xb + 18], x3 = x_shared[xb + 19];
                    acc0_b = fma(float(d0[16]) * scale0, x0, acc0_b);
                    acc0_b = fma(float(d0[17]) * scale0, x1, acc0_b);
                    acc0_b = fma(float(d0[18]) * scale0, x2, acc0_b);
                    acc0_b = fma(float(d0[19]) * scale0, x3, acc0_b);
                    if (has_row1) {
                        acc1_b = fma(float(d1[16]) * scale1, x0, acc1_b);
                        acc1_b = fma(float(d1[17]) * scale1, x1, acc1_b);
                        acc1_b = fma(float(d1[18]) * scale1, x2, acc1_b);
                        acc1_b = fma(float(d1[19]) * scale1, x3, acc1_b);
                    }
                }

                // --- Chunk 5: weights 20-23 → acc_b ---
                {
                    float x0 = x_shared[xb + 20], x1 = x_shared[xb + 21];
                    float x2 = x_shared[xb + 22], x3 = x_shared[xb + 23];
                    acc0_b = fma(float(d0[20]) * scale0, x0, acc0_b);
                    acc0_b = fma(float(d0[21]) * scale0, x1, acc0_b);
                    acc0_b = fma(float(d0[22]) * scale0, x2, acc0_b);
                    acc0_b = fma(float(d0[23]) * scale0, x3, acc0_b);
                    if (has_row1) {
                        acc1_b = fma(float(d1[20]) * scale1, x0, acc1_b);
                        acc1_b = fma(float(d1[21]) * scale1, x1, acc1_b);
                        acc1_b = fma(float(d1[22]) * scale1, x2, acc1_b);
                        acc1_b = fma(float(d1[23]) * scale1, x3, acc1_b);
                    }
                }

                // --- Chunk 6: weights 24-27 → acc_b ---
                {
                    float x0 = x_shared[xb + 24], x1 = x_shared[xb + 25];
                    float x2 = x_shared[xb + 26], x3 = x_shared[xb + 27];
                    acc0_b = fma(float(d0[24]) * scale0, x0, acc0_b);
                    acc0_b = fma(float(d0[25]) * scale0, x1, acc0_b);
                    acc0_b = fma(float(d0[26]) * scale0, x2, acc0_b);
                    acc0_b = fma(float(d0[27]) * scale0, x3, acc0_b);
                    if (has_row1) {
                        acc1_b = fma(float(d1[24]) * scale1, x0, acc1_b);
                        acc1_b = fma(float(d1[25]) * scale1, x1, acc1_b);
                        acc1_b = fma(float(d1[26]) * scale1, x2, acc1_b);
                        acc1_b = fma(float(d1[27]) * scale1, x3, acc1_b);
                    }
                }

                // --- Chunk 7: weights 28-31 → acc_b ---
                {
                    float x0 = x_shared[xb + 28], x1 = x_shared[xb + 29];
                    float x2 = x_shared[xb + 30], x3 = x_shared[xb + 31];
                    acc0_b = fma(float(d0[28]) * scale0, x0, acc0_b);
                    acc0_b = fma(float(d0[29]) * scale0, x1, acc0_b);
                    acc0_b = fma(float(d0[30]) * scale0, x2, acc0_b);
                    acc0_b = fma(float(d0[31]) * scale0, x3, acc0_b);
                    if (has_row1) {
                        acc1_b = fma(float(d1[28]) * scale1, x0, acc1_b);
                        acc1_b = fma(float(d1[29]) * scale1, x1, acc1_b);
                        acc1_b = fma(float(d1[30]) * scale1, x2, acc1_b);
                        acc1_b = fma(float(d1[31]) * scale1, x3, acc1_b);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Merge split accumulators, then SIMD reduce.
    if (row0 < M) {
        float acc0 = acc0_a + acc0_b;
        acc0 = simd_sum(acc0);
        if (lane == 0) y[row0] = bfloat(acc0);
    }
    if (has_row1) {
        float acc1 = acc1_a + acc1_b;
        acc1 = simd_sum(acc1);
        if (lane == 0) y[row1] = bfloat(acc1);
    }
}

// ===========================================================================
// Batched matrix multiply (GEMM) — Q8 weights.
//
// Same structure as gemm_q4 but with simpler 8-bit dequantisation.
// Each weight is a signed int8 byte: dequant = float(byte) * scale.
//
// Block layout (34 bytes per block of 32 weights):
//   bytes 0-1:   bf16 scale factor
//   bytes 2-33:  32 signed int8 values
// ===========================================================================

kernel void gemm_q8(
    constant GemmParams& params  [[buffer(0)]],
    device const uchar* W_q8     [[buffer(1)]],  // [M * blocks_per_row * 34 bytes]
    device const bfloat* X       [[buffer(2)]],  // [batch_size, K]
    device bfloat* Y             [[buffer(3)]],  // [batch_size, M]
    uint gid                     [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint elem = gid / 32;
    uint lane = gid % 32;

    uint batch = elem / M;
    uint row   = elem % M;

    if (batch >= params.batch_size) return;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 34;

    device const uchar* row_data = W_q8 + row * blocks_per_row * bytes_per_block;
    device const bfloat* x_vec = X + batch * K;

    float acc_a = 0.0f, acc_b = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* block_ptr = row_data + block_idx * bytes_per_block;
        float scale = float(*((device const bfloat*)block_ptr));
        device const char* data = (device const char*)(block_ptr + 2);
        uint xb = block_idx * 32;

        // Chunk 0: weights 0-3 → acc_a.
        {
            float x0 = float(x_vec[xb]),     x1 = float(x_vec[xb + 1]);
            float x2 = float(x_vec[xb + 2]), x3 = float(x_vec[xb + 3]);
            acc_a = fma(float(data[0]) * scale, x0, acc_a);
            acc_a = fma(float(data[1]) * scale, x1, acc_a);
            acc_a = fma(float(data[2]) * scale, x2, acc_a);
            acc_a = fma(float(data[3]) * scale, x3, acc_a);
        }

        // Chunk 1: weights 4-7 → acc_a.
        {
            float x0 = float(x_vec[xb + 4]), x1 = float(x_vec[xb + 5]);
            float x2 = float(x_vec[xb + 6]), x3 = float(x_vec[xb + 7]);
            acc_a = fma(float(data[4]) * scale, x0, acc_a);
            acc_a = fma(float(data[5]) * scale, x1, acc_a);
            acc_a = fma(float(data[6]) * scale, x2, acc_a);
            acc_a = fma(float(data[7]) * scale, x3, acc_a);
        }

        // Chunk 2: weights 8-11 → acc_a.
        {
            float x0 = float(x_vec[xb + 8]),  x1 = float(x_vec[xb + 9]);
            float x2 = float(x_vec[xb + 10]), x3 = float(x_vec[xb + 11]);
            acc_a = fma(float(data[8])  * scale, x0, acc_a);
            acc_a = fma(float(data[9])  * scale, x1, acc_a);
            acc_a = fma(float(data[10]) * scale, x2, acc_a);
            acc_a = fma(float(data[11]) * scale, x3, acc_a);
        }

        // Chunk 3: weights 12-15 → acc_a.
        {
            float x0 = float(x_vec[xb + 12]), x1 = float(x_vec[xb + 13]);
            float x2 = float(x_vec[xb + 14]), x3 = float(x_vec[xb + 15]);
            acc_a = fma(float(data[12]) * scale, x0, acc_a);
            acc_a = fma(float(data[13]) * scale, x1, acc_a);
            acc_a = fma(float(data[14]) * scale, x2, acc_a);
            acc_a = fma(float(data[15]) * scale, x3, acc_a);
        }

        // Chunk 4: weights 16-19 → acc_b.
        {
            float x0 = float(x_vec[xb + 16]), x1 = float(x_vec[xb + 17]);
            float x2 = float(x_vec[xb + 18]), x3 = float(x_vec[xb + 19]);
            acc_b = fma(float(data[16]) * scale, x0, acc_b);
            acc_b = fma(float(data[17]) * scale, x1, acc_b);
            acc_b = fma(float(data[18]) * scale, x2, acc_b);
            acc_b = fma(float(data[19]) * scale, x3, acc_b);
        }

        // Chunk 5: weights 20-23 → acc_b.
        {
            float x0 = float(x_vec[xb + 20]), x1 = float(x_vec[xb + 21]);
            float x2 = float(x_vec[xb + 22]), x3 = float(x_vec[xb + 23]);
            acc_b = fma(float(data[20]) * scale, x0, acc_b);
            acc_b = fma(float(data[21]) * scale, x1, acc_b);
            acc_b = fma(float(data[22]) * scale, x2, acc_b);
            acc_b = fma(float(data[23]) * scale, x3, acc_b);
        }

        // Chunk 6: weights 24-27 → acc_b.
        {
            float x0 = float(x_vec[xb + 24]), x1 = float(x_vec[xb + 25]);
            float x2 = float(x_vec[xb + 26]), x3 = float(x_vec[xb + 27]);
            acc_b = fma(float(data[24]) * scale, x0, acc_b);
            acc_b = fma(float(data[25]) * scale, x1, acc_b);
            acc_b = fma(float(data[26]) * scale, x2, acc_b);
            acc_b = fma(float(data[27]) * scale, x3, acc_b);
        }

        // Chunk 7: weights 28-31 → acc_b.
        {
            float x0 = float(x_vec[xb + 28]), x1 = float(x_vec[xb + 29]);
            float x2 = float(x_vec[xb + 30]), x3 = float(x_vec[xb + 31]);
            acc_b = fma(float(data[28]) * scale, x0, acc_b);
            acc_b = fma(float(data[29]) * scale, x1, acc_b);
            acc_b = fma(float(data[30]) * scale, x2, acc_b);
            acc_b = fma(float(data[31]) * scale, x3, acc_b);
        }
    }

    float acc = acc_a + acc_b;
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = bfloat(acc);
    }
}

// ===========================================================================
// TQ3 support — TurboQuant 3-bit weight quantization with Walsh-Hadamard
// rotation.
//
// LEARNING OVERVIEW
//
// TQ3 applies the TurboQuant algorithm (Zandieh et al., arXiv:2504.19874) to
// weight quantization.  Each block of 32 weights is:
//   1. Walsh-Hadamard transformed (orthonormal rotation)
//   2. Quantized to 8 Max-Lloyd optimal centroids (3-bit codes)
//   3. Packed with dual bf16 scales (scale_lo for [0:15], scale_hi for [16:32])
//
// Block layout (16 bytes per block of 32 weights):
//   [0:2]   bf16 scale_lo  (scale for WHT coefficients 0-15)
//   [2:4]   bf16 scale_hi  (scale for WHT coefficients 16-31)
//   [4:16]  12 bytes packed 3-bit centroid codes (32 × 3 = 96 bits)
//
// Centroids (Max-Lloyd optimal for N(0,1), 3-bit = 8 levels):
//   [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]
//
// Inference requires WHT-rotating the activation vector before dot product:
//   <w, x> = <WHT(w), WHT(x)>  (WHT is orthonormal)
//
// The WHT is applied per 32-element block in shared memory using the
// butterfly algorithm (5 stages for n=32, O(n log n)).
//
// Dispatch: grid_size = ceil(M / 2) × 32, threadgroup_size = 256.
// ===========================================================================

// Constant centroid table — 8 Max-Lloyd centroids for N(0,1).
constant constexpr float TQ3_CENTROIDS[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

// 1/sqrt(32) normalisation factor for WHT.
constant constexpr float WHT_NORM = 0.176776695f;

// Extract a 3-bit code from a packed byte array at the given index.
// Codes can span byte boundaries (e.g., code at bit offset 6 spans bytes 0-1).
inline uint extract_3bit(device const uchar* packed, uint idx) {
    uint bit_offset = idx * 3;
    uint byte_idx = bit_offset / 8;
    uint bit_within = bit_offset % 8;
    uint val = packed[byte_idx];
    if (bit_within + 3 > 8) {
        val |= (uint(packed[byte_idx + 1]) << 8);
    }
    return (val >> bit_within) & 0x7;
}

// Bulk-extract all 32 × 3-bit codes from 12 packed bytes into a float array
// of centroid values, pre-scaled by scale_lo (first 16) and scale_hi (last 16).
// This loads the 12 bytes as 3 uint32s and extracts codes without per-element
// byte addressing or branching — much faster than 32 individual extract_3bit calls.
inline void decode_tq3_block(
    device const uchar* codes,
    float scale_lo,
    float scale_hi,
    thread float* w  // output: 32 dequantized weights
) {
    // Load 12 bytes as 3 × uint32 (96 bits total = 32 × 3-bit codes).
    uint w0 = uint(codes[0])  | (uint(codes[1]) << 8)  | (uint(codes[2]) << 16)  | (uint(codes[3]) << 24);
    uint w1 = uint(codes[4])  | (uint(codes[5]) << 8)  | (uint(codes[6]) << 16)  | (uint(codes[7]) << 24);
    uint w2 = uint(codes[8])  | (uint(codes[9]) << 8)  | (uint(codes[10]) << 16) | (uint(codes[11]) << 24);

    // Extract codes 0-9 from w0 (bits 0-29, 30 bits = 10 codes × 3 bits).
    w[0]  = TQ3_CENTROIDS[(w0)       & 0x7] * scale_lo;
    w[1]  = TQ3_CENTROIDS[(w0 >> 3)  & 0x7] * scale_lo;
    w[2]  = TQ3_CENTROIDS[(w0 >> 6)  & 0x7] * scale_lo;
    w[3]  = TQ3_CENTROIDS[(w0 >> 9)  & 0x7] * scale_lo;
    w[4]  = TQ3_CENTROIDS[(w0 >> 12) & 0x7] * scale_lo;
    w[5]  = TQ3_CENTROIDS[(w0 >> 15) & 0x7] * scale_lo;
    w[6]  = TQ3_CENTROIDS[(w0 >> 18) & 0x7] * scale_lo;
    w[7]  = TQ3_CENTROIDS[(w0 >> 21) & 0x7] * scale_lo;
    w[8]  = TQ3_CENTROIDS[(w0 >> 24) & 0x7] * scale_lo;
    w[9]  = TQ3_CENTROIDS[(w0 >> 27) & 0x7] * scale_lo;

    // Code 10 spans w0[30:31] and w1[0:0] (2 bits from w0, 1 bit from w1).
    w[10] = TQ3_CENTROIDS[((w0 >> 30) | (w1 << 2)) & 0x7] * scale_lo;

    // Extract codes 11-20 from w1 (starting at bit 1).
    w[11] = TQ3_CENTROIDS[(w1 >> 1)  & 0x7] * scale_lo;
    w[12] = TQ3_CENTROIDS[(w1 >> 4)  & 0x7] * scale_lo;
    w[13] = TQ3_CENTROIDS[(w1 >> 7)  & 0x7] * scale_lo;
    w[14] = TQ3_CENTROIDS[(w1 >> 10) & 0x7] * scale_lo;
    w[15] = TQ3_CENTROIDS[(w1 >> 13) & 0x7] * scale_lo;

    // Codes 16-31 use scale_hi.
    w[16] = TQ3_CENTROIDS[(w1 >> 16) & 0x7] * scale_hi;
    w[17] = TQ3_CENTROIDS[(w1 >> 19) & 0x7] * scale_hi;
    w[18] = TQ3_CENTROIDS[(w1 >> 22) & 0x7] * scale_hi;
    w[19] = TQ3_CENTROIDS[(w1 >> 25) & 0x7] * scale_hi;
    w[20] = TQ3_CENTROIDS[(w1 >> 28) & 0x7] * scale_hi;

    // Code 21 spans w1[31:31] and w2[0:1] (1 bit from w1, 2 bits from w2).
    w[21] = TQ3_CENTROIDS[((w1 >> 31) | (w2 << 1)) & 0x7] * scale_hi;

    // Extract codes 22-31 from w2 (starting at bit 2).
    w[22] = TQ3_CENTROIDS[(w2 >> 2)  & 0x7] * scale_hi;
    w[23] = TQ3_CENTROIDS[(w2 >> 5)  & 0x7] * scale_hi;
    w[24] = TQ3_CENTROIDS[(w2 >> 8)  & 0x7] * scale_hi;
    w[25] = TQ3_CENTROIDS[(w2 >> 11) & 0x7] * scale_hi;
    w[26] = TQ3_CENTROIDS[(w2 >> 14) & 0x7] * scale_hi;
    w[27] = TQ3_CENTROIDS[(w2 >> 17) & 0x7] * scale_hi;
    w[28] = TQ3_CENTROIDS[(w2 >> 20) & 0x7] * scale_hi;
    w[29] = TQ3_CENTROIDS[(w2 >> 23) & 0x7] * scale_hi;
    w[30] = TQ3_CENTROIDS[(w2 >> 26) & 0x7] * scale_hi;
    w[31] = TQ3_CENTROIDS[(w2 >> 29) & 0x7] * scale_hi;
}

// Apply in-place 32-point Walsh-Hadamard Transform to a shared memory block.
// Uses 5 butterfly stages (log2(32) = 5).  Only threads 0-15 do the butterflies,
// but all threads must hit the barriers.
inline void wht_32_shared(
    threadgroup float* block,
    uint base,
    uint lid
) {
    // Stage 1: step=1, 16 butterflies
    if (lid < 16) {
        uint i = base + (lid / 1) * 2 + (lid % 1);
        float a = block[i], b = block[i + 1];
        block[i] = a + b; block[i + 1] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: step=2
    if (lid < 16) {
        uint i = base + (lid / 2) * 4 + (lid % 2);
        float a = block[i], b = block[i + 2];
        block[i] = a + b; block[i + 2] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3: step=4
    if (lid < 16) {
        uint i = base + (lid / 4) * 8 + (lid % 4);
        float a = block[i], b = block[i + 4];
        block[i] = a + b; block[i + 4] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 4: step=8
    if (lid < 16) {
        uint i = base + (lid / 8) * 16 + (lid % 8);
        float a = block[i], b = block[i + 8];
        block[i] = a + b; block[i + 8] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: step=16
    if (lid < 16) {
        uint i = base + lid;
        float a = block[i], b = block[i + 16];
        block[i] = a + b; block[i + 16] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalise: multiply by 1/sqrt(32).
    for (uint i = lid; i < 32; i += 256) {
        block[base + i] *= WHT_NORM;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ===========================================================================
// WHT pre-rotation kernel — transforms x in-place before TQ3 matvec.
//
// Dispatched once before matvec_tq3.  Runs O(K) work total (K/32 blocks ×
// 5 butterfly stages each).  The matvec itself is O(M×K), so this is negligible.
// After this kernel, x contains WHT(x) in float32 and matvec_tq3 reads it
// directly — no WHT overhead in the hot matmul loop.
//
// Dispatch: grid = K, threadgroup = 256.
// ===========================================================================

struct WhtParams {
    uint K;  // Total number of elements.
};

kernel void wht_rotate_x(
    constant WhtParams& params   [[buffer(0)]],
    device float* x_wht          [[buffer(1)]],  // [K] float32 output
    device const bfloat* x_bf16  [[buffer(2)]],  // [K] bfloat16 source
    uint gid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]]
) {
    // Each threadgroup processes 8 blocks of 32 elements (256 threads / 32 = 8).
    uint block_base = (gid / 256) * 8;  // first block for this threadgroup
    uint my_block   = block_base + (lid / 32);
    uint my_elem    = lid % 32;

    uint idx = my_block * 32 + my_elem;
    if (idx >= params.K) return;

    // Load bf16 → f32.
    threadgroup float shared[256];  // 8 blocks × 32 elements
    shared[lid] = float(x_bf16[idx]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5 butterfly stages within each 32-element block.
    uint block_offset = (lid / 32) * 32;  // start of this block in shared
    for (uint step = 1; step < 32; step <<= 1) {
        uint partner = my_elem ^ step;
        float my_val   = shared[block_offset + my_elem];
        float pair_val = shared[block_offset + partner];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (my_elem < partner) {
            shared[block_offset + my_elem]  = my_val + pair_val;
            shared[block_offset + partner]  = my_val - pair_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise and write to output.
    x_wht[idx] = shared[lid] * WHT_NORM;
}

// ===========================================================================
// Matrix-vector multiply kernel (TQ3) — pre-rotated x, bulk centroid decode.
//
// IMPORTANT: Assumes x has already been WHT-rotated by wht_rotate_x into
// float32.  The matvec structure is identical to Q4: shared memory x caching,
// multi-row SIMD (2 rows per group), lanes strided across blocks.
//
// The WHT pre-rotation is dispatched as a separate O(K) kernel before this
// O(M×K) kernel — the cost is amortized across all M rows.
//
// Dispatch: grid = ceil(M / 2) × 32, threadgroup_size = 256.
// ===========================================================================

kernel void matvec_tq3(
    constant MatvecParams& params [[buffer(0)]],
    device const uchar* W_tq3     [[buffer(1)]],  // [M * (K/32) * 16 bytes]
    device const float* x_wht     [[buffer(2)]],  // [K] float32, pre-rotated
    device bfloat* y              [[buffer(3)]],  // [M]
    uint gid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint simd_idx = gid / 32;
    uint lane = gid % 32;

    uint row0 = simd_idx * ROWS_PER_SIMD_TQ3;
    uint row1 = row0 + 1;
    bool has_row1 = (row1 < M);

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 16;

    device const uchar* row0_data = W_tq3 + row0 * blocks_per_row * bytes_per_block;
    device const uchar* row1_data = W_tq3 + row1 * blocks_per_row * bytes_per_block;

    // Shared memory cache for pre-rotated x (same tiling as Q4).
    threadgroup float x_shared[X_TILE];

    float acc0_a = 0.0f, acc0_b = 0.0f;
    float acc1_a = 0.0f, acc1_b = 0.0f;

    for (uint tile = 0; tile < K; tile += X_TILE) {
        uint tile_len = min((uint)X_TILE, K - tile);

        // Cooperative load: x_wht is already float32.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = x_wht[tile + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < M) {
            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                device const uchar* bp0 = row0_data + block_idx * bytes_per_block;
                float scale0_lo = float(*((device const bfloat*)bp0));
                float scale0_hi = float(*((device const bfloat*)(bp0 + 2)));

                float w0[32];
                decode_tq3_block(bp0 + 4, scale0_lo, scale0_hi, w0);

                float w1[32];
                if (has_row1) {
                    device const uchar* bp1 = row1_data + block_idx * bytes_per_block;
                    float scale1_lo = float(*((device const bfloat*)bp1));
                    float scale1_hi = float(*((device const bfloat*)(bp1 + 2)));
                    decode_tq3_block(bp1 + 4, scale1_lo, scale1_hi, w1);
                }

                uint xb = (block_idx * 32) - tile;

                for (uint i = 0; i < 16; i++) {
                    float x_val = x_shared[xb + i];
                    acc0_a = fma(w0[i], x_val, acc0_a);
                    if (has_row1) acc1_a = fma(w1[i], x_val, acc1_a);
                }
                for (uint i = 16; i < 32; i++) {
                    float x_val = x_shared[xb + i];
                    acc0_b = fma(w0[i], x_val, acc0_b);
                    if (has_row1) acc1_b = fma(w1[i], x_val, acc1_b);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < M) {
        float acc0 = acc0_a + acc0_b;
        acc0 = simd_sum(acc0);
        if (lane == 0) y[row0] = bfloat(acc0);
    }
    if (has_row1) {
        float acc1 = acc1_a + acc1_b;
        acc1 = simd_sum(acc1);
        if (lane == 0) y[row1] = bfloat(acc1);
    }
}

// ===========================================================================
// Batch WHT kernel — transforms all rows of X in-place before gemm_tq3.
//
// X is [batch_size, K] bf16.  Output is [batch_size, K] float32.
// Each threadgroup handles 8 blocks (256 threads / 32 per block).
// Dispatch: grid = batch_size * K, threadgroup = 256.
// ===========================================================================

kernel void wht_rotate_x_batch(
    constant GemmParams& params  [[buffer(0)]],
    device float* x_wht          [[buffer(1)]],  // [batch_size, K] float32 output
    device const bfloat* X       [[buffer(2)]],  // [batch_size, K] bf16 source
    uint gid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]]
) {
    const uint K = params.K;
    const uint total = params.batch_size * K;

    uint block_base = (gid / 256) * 8;
    uint my_block   = block_base + (lid / 32);
    uint my_elem    = lid % 32;

    uint idx = my_block * 32 + my_elem;
    if (idx >= total) return;

    threadgroup float shared[256];
    shared[lid] = float(X[idx]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_offset = (lid / 32) * 32;
    for (uint step = 1; step < 32; step <<= 1) {
        uint partner = my_elem ^ step;
        float my_val   = shared[block_offset + my_elem];
        float pair_val = shared[block_offset + partner];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (my_elem < partner) {
            shared[block_offset + my_elem]  = my_val + pair_val;
            shared[block_offset + partner]  = my_val - pair_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    x_wht[idx] = shared[lid] * WHT_NORM;
}

// ===========================================================================
// Batched matrix multiply (GEMM) — TQ3 weights, pre-rotated float32 X.
//
// Assumes X has been WHT-rotated by wht_rotate_x_batch.  Same structure as
// gemm_q4: lanes strided across blocks, bulk centroid decode.
// ===========================================================================

kernel void gemm_tq3(
    constant GemmParams& params  [[buffer(0)]],
    device const uchar* W_tq3    [[buffer(1)]],  // [M * (K/32) * 16 bytes]
    device const float* X_wht    [[buffer(2)]],  // [batch_size, K] float32 pre-rotated
    device bfloat* Y             [[buffer(3)]],  // [batch_size, M]
    uint gid                     [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint elem = gid / 32;
    uint lane = gid % 32;

    uint batch = elem / M;
    uint row   = elem % M;

    if (batch >= params.batch_size) return;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 16;

    device const uchar* row_data = W_tq3 + row * blocks_per_row * bytes_per_block;
    device const float* x_vec = X_wht + batch * K;

    float acc_a = 0.0f, acc_b = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* bp = row_data + block_idx * bytes_per_block;
        float scale_lo = float(*((device const bfloat*)bp));
        float scale_hi = float(*((device const bfloat*)(bp + 2)));

        float w[32];
        decode_tq3_block(bp + 4, scale_lo, scale_hi, w);

        uint xb = block_idx * 32;
        for (uint i = 0; i < 16; i++) {
            acc_a = fma(w[i], x_vec[xb + i], acc_a);
        }
        for (uint i = 16; i < 32; i++) {
            acc_b = fma(w[i], x_vec[xb + i], acc_b);
        }
    }

    float acc = acc_a + acc_b;
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = bfloat(acc);
    }
}
