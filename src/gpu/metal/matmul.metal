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
        for (uint i = 0; i < 16; i += 4) {
            uchar b0 = data[i];
            uchar b1 = data[i + 1];
            uchar b2 = data[i + 2];
            uchar b3 = data[i + 3];

            // Each byte packs two 4-bit values: low nibble = even, high = odd.
            acc += float(int(b0 & 0xF) - 8) * scale * float(x[x_base + i * 2]);
            acc += float(int(b0 >> 4)  - 8) * scale * float(x[x_base + i * 2 + 1]);
            acc += float(int(b1 & 0xF) - 8) * scale * float(x[x_base + i * 2 + 2]);
            acc += float(int(b1 >> 4)  - 8) * scale * float(x[x_base + i * 2 + 3]);
            acc += float(int(b2 & 0xF) - 8) * scale * float(x[x_base + i * 2 + 4]);
            acc += float(int(b2 >> 4)  - 8) * scale * float(x[x_base + i * 2 + 5]);
            acc += float(int(b3 & 0xF) - 8) * scale * float(x[x_base + i * 2 + 6]);
            acc += float(int(b3 >> 4)  - 8) * scale * float(x[x_base + i * 2 + 7]);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = bfloat(acc);
    }
}
