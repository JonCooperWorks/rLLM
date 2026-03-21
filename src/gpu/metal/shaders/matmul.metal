// ===========================================================================
// Matrix-vector multiply kernel (bf16) — SIMD-cooperative version.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Computes y = W · x, where W is a weight matrix and x is a vector.
//   This is the fundamental operation for all linear projections in the
//   transformer: Q/K/V projections, output projection, gate/up/down FFN
//   projections, and the final LM head.
//
// SIMD-cooperative approach:
//   Instead of one thread computing an entire dot product (K=2048 multiply-
//   adds), 32 threads (one SIMD group) cooperate on each output row.
//   Each thread handles K/32 elements in a strided pattern, then `simd_sum`
//   reduces across all 32 lanes in a single hardware instruction.
//
//   For K=2048: each thread does 64 multiply-adds (vs 2048 in the naive kernel).
//   For K=8192: each thread does 256 multiply-adds (vs 8192 in the naive kernel).
//
// Why this is faster:
//   1. 32x more parallelism per output row — GPU cores stay busy
//   2. Strided access across SIMD lanes → coalesced memory reads
//   3. `simd_sum` is a single-cycle hardware operation (no shared memory)
//   4. 4x loop unrolling gives the compiler room for ILP (instruction-level
//      parallelism) — it can interleave loads with multiplies
//
// Dispatch model:
//   grid_size = M × 32 total threads (32 threads per output row).
//   threadgroup_size = 256 = 8 SIMD groups = 8 output rows per threadgroup.
//
// Precision:
//   Inputs (W and x) are bfloat16.  The dot-product accumulator is float32.
//   The final result is narrowed back to bfloat16 on store.
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `MatvecParams`.
struct MatvecParams {
    uint M;  // Output dimension (number of rows in W, length of y).
    uint K;  // Input dimension (number of columns in W, length of x).
};

kernel void matvec_bf16(
    constant MatvecParams& params [[buffer(0)]],
    // buffer(1): weight matrix W [M, K] in bfloat16, row-major.
    device const bfloat* W        [[buffer(1)]],
    // buffer(2): input vector x [K] in bfloat16.
    device const bfloat* x        [[buffer(2)]],
    // buffer(3): output vector y [M] in bfloat16.
    device bfloat* y              [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    // 32 threads (one SIMD group) cooperate on one output row.
    //   row  = which output element (0..M-1)
    //   lane = which thread within the SIMD group (0..31)
    uint row = gid / 32;
    uint lane = gid % 32;

    if (row >= M) return;

    // Pointer to this row's weights — K consecutive bfloat16 values.
    device const bfloat* w_row = W + row * K;

    // -----------------------------------------------------------------------
    // Strided dot product with 4x unrolling.
    //
    // Each lane processes elements at indices: lane*4, lane*4+128, lane*4+256, ...
    // The stride of 128 (= 32 lanes × 4 unroll) ensures all 32 lanes read
    // consecutive 4-element blocks, maximising memory coalescing.
    //
    // For K=2048: each lane does 2048/(32*4) = 16 iterations of 4 multiply-adds.
    // For K=8192: each lane does 8192/(32*4) = 64 iterations of 4 multiply-adds.
    //
    // Learning note: unrolling by 4 helps because the GPU can issue the next
    // memory load while the previous multiply-add is still in the pipeline.
    // Without unrolling, the loop has a load→compute→branch dependency chain
    // that serialises operations.
    // -----------------------------------------------------------------------
    float acc = 0.0f;
    for (uint j = lane * 4; j < K; j += 32 * 4) {
        acc += float(w_row[j])     * float(x[j]);
        acc += float(w_row[j + 1]) * float(x[j + 1]);
        acc += float(w_row[j + 2]) * float(x[j + 2]);
        acc += float(w_row[j + 3]) * float(x[j + 3]);
    }

    // -----------------------------------------------------------------------
    // SIMD reduction: sum across all 32 lanes.
    //
    // `simd_sum` is a hardware intrinsic that sums a value across all lanes
    // of the SIMD group in a single cycle.  No shared memory, no barriers.
    // After this call, all 32 lanes hold the same total dot product.
    //
    // Only lane 0 writes the result — the other 31 lanes discard their copy.
    // -----------------------------------------------------------------------
    acc = simd_sum(acc);

    if (lane == 0) {
        y[row] = bfloat(acc);
    }
}

// ===========================================================================
// Matrix-vector multiply kernel (Q4) — SIMD-cooperative with inline dequant.
//
// LEARNING OVERVIEW
//
// Same SIMD-cooperative structure as matvec_bf16, but the weight matrix is
// stored in block-wise 4-bit quantisation.  This halves memory bandwidth
// compared to bf16 (20 bytes per 32 weights vs 64 bytes), giving ~2x speedup
// for memory-bound matmuls.
//
// Block layout (20 bytes per block of 32 weights):
//   bytes  0-3:  f32 scale factor
//   bytes  4-19: 16 bytes of packed nibbles (2 weights per byte)
//
// Packing: byte[i] = (q[2i] & 0xF) | (q[2i+1] << 4)
// Dequant: weight = (nibble - 8) * scale
//   (symmetric quantisation with offset encoding: q ∈ [0,15] → signed [-8,7])
//
// Dispatch: same as bf16 — grid_size = M × 32, threadgroup_size = 256.
// ===========================================================================

kernel void matvec_q4(
    constant MatvecParams& params [[buffer(0)]],
    // buffer(1): packed Q4 weight data [M * blocks_per_row * 20 bytes].
    device const uchar* W_q4     [[buffer(1)]],
    // buffer(2): input vector x [K] in bfloat16.
    device const bfloat* x       [[buffer(2)]],
    // buffer(3): output vector y [M] in bfloat16.
    device bfloat* y             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint row = gid / 32;
    uint lane = gid % 32;
    if (row >= M) return;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 20;  // 4 (scale) + 16 (packed nibbles)

    // Pointer to this row's Q4 data.
    device const uchar* row_data = W_q4 + row * blocks_per_row * bytes_per_block;

    // -----------------------------------------------------------------------
    // Each lane processes blocks in a strided pattern (lane, lane+32, ...).
    // Within each block, dequantise all 32 weights and dot with x.
    //
    // For K=2048: blocks_per_row=64, each lane handles 2 blocks = 64 weights.
    // For K=8192: blocks_per_row=256, each lane handles 8 blocks = 256 weights.
    //
    // Learning note: Q4 is memory-bandwidth bound.  Reading 20 bytes per 32
    // weights (vs 64 bytes for bf16) is 3.2x less data, but dequantisation
    // adds compute.  Net effect: ~1.5-2x faster because we're memory-bound.
    // -----------------------------------------------------------------------
    float acc = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* block_ptr = row_data + block_idx * bytes_per_block;

        // Read scale (first 4 bytes, always 4-byte aligned since 20%4==0).
        float scale = *((device const float*)block_ptr);
        device const uchar* data = block_ptr + 4;

        // Base index in x vector for this block's 32 weights.
        uint x_base = block_idx * 32;

        // Unpack 16 bytes → 32 nibbles → 32 dequantised weights.
        // 4x unrolled: process 4 bytes (8 weights) per iteration.
        //
        // FMA optimisation (from flash-moe): precompute scale*x per element,
        // then use fma(nibble, sx, -8*sx) instead of (nibble-8)*scale*x.
        // This lets the GPU issue a single fused multiply-add per nibble
        // instead of a subtract + two multiplies.  The -8*sx bias term is
        // hoisted out of the nibble extraction path.
        for (uint i = 0; i < 16; i += 4) {
            uchar b0 = data[i];
            uchar b1 = data[i + 1];
            uchar b2 = data[i + 2];
            uchar b3 = data[i + 3];

            // Precompute scale*x and bias (-8*scale*x) for each element.
            float sx0 = scale * float(x[x_base + i * 2]);
            float sx1 = scale * float(x[x_base + i * 2 + 1]);
            float sx2 = scale * float(x[x_base + i * 2 + 2]);
            float sx3 = scale * float(x[x_base + i * 2 + 3]);
            float sx4 = scale * float(x[x_base + i * 2 + 4]);
            float sx5 = scale * float(x[x_base + i * 2 + 5]);
            float sx6 = scale * float(x[x_base + i * 2 + 6]);
            float sx7 = scale * float(x[x_base + i * 2 + 7]);

            // fma(nibble, scale*x, -8*scale*x) = (nibble - 8) * scale * x
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
//   Example numbers (Llama 3.2 1B, Q projection, K=2048, M=2048):
//     Matvec: 8M FLOPs, 8 MB loaded → intensity 1.0 → 15μs at 546 GB/s
//     GEMM B=100: 800M FLOPs, 8 MB loaded → intensity 100 → same 15μs memory
//       but now fully utilising the GPU's 14 TFLOPS compute → 57μs compute
//     Net: 100 tokens in 57μs vs 100 × 15μs = 1500μs → 26x faster!
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
//   2. Quantisation: 20 bytes per 32 weights vs 64 bytes (3.2× smaller)
//   Combined: for B=100, memory traffic drops ~320× compared to 100 bf16 matvecs.
// ===========================================================================

kernel void gemm_q4(
    constant GemmParams& params  [[buffer(0)]],
    device const uchar* W_q4     [[buffer(1)]],  // [M * blocks_per_row * 20 bytes]
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
    const uint bytes_per_block = 20;

    device const uchar* row_data = W_q4 + row * blocks_per_row * bytes_per_block;
    device const bfloat* x_vec = X + batch * K;

    float acc = 0.0f;

    for (uint block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        device const uchar* block_ptr = row_data + block_idx * bytes_per_block;
        float scale = *((device const float*)block_ptr);
        device const uchar* data = block_ptr + 4;
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
