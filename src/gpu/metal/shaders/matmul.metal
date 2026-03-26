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
constant constexpr uint ROWS_PER_SIMD_Q4 = 1;

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
// Matrix-vector multiply kernel (Q4) — multi-row SIMD with shared memory
// input caching and inline dequantisation.
//
// LEARNING OVERVIEW
//
// Same multi-row SIMD + shared memory caching pattern as matvec_bf16, but
// weights are stored in block-wise 4-bit quantisation.
//
// Multi-row benefit for Q4:
//   Q4 dequantisation is compute-heavy (nibble extract + scale multiply per
//   weight).  With 2 rows per SIMD, each thread reads weight blocks from
//   TWO rows but shares x values from shared memory.  The x values are
//   already loaded and the scale*x products are reusable across rows.
//   This doubles the useful compute per memory cycle, improving the
//   compute-to-bandwidth ratio.
//
// Block layout (18 bytes per block of 32 weights):
//   bytes  0-1:  bf16 scale factor
//   bytes  2-17: 16 bytes of packed nibbles (2 weights per byte)
//
// Packing: byte[i] = (q[2i] & 0xF) | (q[2i+1] << 4)
// Dequant: weight = (nibble - 8) * scale
//   (symmetric quantisation with offset encoding: q ∈ [0,15] → signed [-8,7])
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

    uint row = simd_idx;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 18;  // 2 (bf16 scale) + 16 (packed nibbles)

    device const uchar* row_data = W_q4 + row * blocks_per_row * bytes_per_block;

    // Shared memory cache for input vector x.
    threadgroup float x_shared[X_TILE];

    float acc = 0.0f;

    for (uint tile = 0; tile < K; tile += X_TILE) {
        uint tile_len = min((uint)X_TILE, K - tile);

        // Cooperative load: 256 threads load tile_len bf16→f32 values.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M) {
            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                device const uchar* block_ptr = row_data + block_idx * bytes_per_block;
                float scale = float(*((device const half*)block_ptr));
                device const uchar* data = block_ptr + 2;

                uint x_local = (block_idx * 32) - tile;

                // FMA-optimised dequant: precompute scale*x, then
                // fma(nibble, sx, -8*sx) = (nibble - 8) * scale * x.
                for (uint i = 0; i < 16; i += 4) {
                    uchar b0 = data[i], b1 = data[i+1], b2 = data[i+2], b3 = data[i+3];

                    float sx0 = scale * x_shared[x_local + i * 2];
                    float sx1 = scale * x_shared[x_local + i * 2 + 1];
                    float sx2 = scale * x_shared[x_local + i * 2 + 2];
                    float sx3 = scale * x_shared[x_local + i * 2 + 3];
                    float sx4 = scale * x_shared[x_local + i * 2 + 4];
                    float sx5 = scale * x_shared[x_local + i * 2 + 5];
                    float sx6 = scale * x_shared[x_local + i * 2 + 6];
                    float sx7 = scale * x_shared[x_local + i * 2 + 7];

                    acc = fma(float(b0 & 0xF), sx0, fma(-8.0f, sx0, acc));
                    acc = fma(float(b0 >> 4),  sx1, fma(-8.0f, sx1, acc));
                    acc = fma(float(b1 & 0xF), sx2, fma(-8.0f, sx2, acc));
                    acc = fma(float(b1 >> 4),  sx3, fma(-8.0f, sx3, acc));
                    acc = fma(float(b2 & 0xF), sx4, fma(-8.0f, sx4, acc));
                    acc = fma(float(b2 >> 4),  sx5, fma(-8.0f, sx5, acc));
                    acc = fma(float(b3 & 0xF), sx6, fma(-8.0f, sx6, acc));
                    acc = fma(float(b3 >> 4),  sx7, fma(-8.0f, sx7, acc));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (lane == 0 && row < M) {
        y[row] = bfloat(acc);
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
// Same as gemm_bf16 but with inline dequantisation of 4-bit packed weights.
// Each SIMD group handles one (batch, row) output element.
//
// Q4 GEMM compounds TWO bandwidth savings:
//   1. Batching: weight matrix loaded once for B tokens (B× fewer loads)
//   2. Quantisation: 18 bytes per 32 weights vs 64 bytes (3.6× smaller)
//   Combined: for B=100, memory traffic drops ~360× compared to 100 bf16 matvecs.
//
// Block layout: the Q4 GEMM kernel uses the same 18-byte block format as
// matvec_q4 (2-byte bf16 scale + 16 bytes packed nibbles).
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

    float acc = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* block_ptr = row_data + block_idx * bytes_per_block;
        float scale = float(*((device const half*)block_ptr));
        device const uchar* data = block_ptr + 2;
        uint x_base = block_idx * 32;

        // FMA-optimised dequant (same as matvec_q4 — see comment there).
        for (uint i = 0; i < 16; i += 4) {
            uchar b0 = data[i];
            uchar b1 = data[i + 1];
            uchar b2 = data[i + 2];
            uchar b3 = data[i + 3];

            float sx0 = scale * float(x_vec[x_base + i * 2]);
            float sx1 = scale * float(x_vec[x_base + i * 2 + 1]);
            float sx2 = scale * float(x_vec[x_base + i * 2 + 2]);
            float sx3 = scale * float(x_vec[x_base + i * 2 + 3]);
            float sx4 = scale * float(x_vec[x_base + i * 2 + 4]);
            float sx5 = scale * float(x_vec[x_base + i * 2 + 5]);
            float sx6 = scale * float(x_vec[x_base + i * 2 + 6]);
            float sx7 = scale * float(x_vec[x_base + i * 2 + 7]);

            acc = fma(float(b0 & 0xF), sx0, fma(-8.0f, sx0, acc));
            acc = fma(float(b0 >> 4),  sx1, fma(-8.0f, sx1, acc));
            acc = fma(float(b1 & 0xF), sx2, fma(-8.0f, sx2, acc));
            acc = fma(float(b1 >> 4),  sx3, fma(-8.0f, sx3, acc));
            acc = fma(float(b2 & 0xF), sx4, fma(-8.0f, sx4, acc));
            acc = fma(float(b2 >> 4),  sx5, fma(-8.0f, sx5, acc));
            acc = fma(float(b3 & 0xF), sx6, fma(-8.0f, sx6, acc));
            acc = fma(float(b3 >> 4),  sx7, fma(-8.0f, sx7, acc));
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = bfloat(acc);
    }
}
