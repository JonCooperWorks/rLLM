// ===========================================================================
// Matrix-vector multiply and batched GEMM CUDA kernels (bf16 + Q4).
//
// LEARNING OVERVIEW
//
// Port of the Metal matmul.metal kernels to CUDA for NVIDIA GPUs.
//
// These kernels compute y = W · x (matvec) and Y = X @ W^T (GEMM).
// The SIMD-cooperative pattern from Metal maps naturally to CUDA warps:
// 32 threads (one warp) cooperate on each output row via `__shfl_xor_sync`
// reduction instead of Metal's `simd_sum`.
//
// Four kernel variants:
//   matvec_bf16 — single-token decode, bf16 weights
//   matvec_q4   — single-token decode, Q4 packed weights
//   gemm_bf16   — batched prefill, bf16 weights
//   gemm_q4     — batched prefill, Q4 packed weights
//
// CUDA vs Metal differences:
//   - Warp size is 32 on both platforms — the cooperative pattern is identical.
//   - CUDA `__shfl_xor_sync` replaces Metal `simd_sum`.
//   - CUDA `__nv_bfloat16` replaces Metal `bfloat`.
//   - H100 has 3.35 TB/s HBM3 bandwidth (vs M4 Max 546 GB/s) — these kernels
//     are memory-bound, so the H100 sees ~6x higher throughput.
//
// Related files:
//   Metal shader:  metal/shaders/matmul.metal
//   CUDA bridge:   cuda/kernels/matmul.rs
//   Trait contract: gpu/ops/matmul.rs
// ===========================================================================

#include <cuda_bf16.h>

__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

struct MatvecParams {
    unsigned int M;
    unsigned int K;
};

// ---------------------------------------------------------------------------
// matvec_bf16 — SIMD-cooperative matrix-vector multiply with bf16 weights.
// 32 threads per output row, 4x loop unrolling for ILP.
// ---------------------------------------------------------------------------
extern "C" __global__ void matvec_bf16(
    const MatvecParams params,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const __nv_bfloat16* w_row = W + row * K;

    float acc = 0.0f;
    for (unsigned int j = lane * 4; j < K; j += 32 * 4) {
        acc += __bfloat162float(w_row[j])     * __bfloat162float(x[j]);
        acc += __bfloat162float(w_row[j + 1]) * __bfloat162float(x[j + 1]);
        acc += __bfloat162float(w_row[j + 2]) * __bfloat162float(x[j + 2]);
        acc += __bfloat162float(w_row[j + 3]) * __bfloat162float(x[j + 3]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        y[row] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// matvec_q4 — SIMD-cooperative matvec with inline Q4 dequantisation.
//
// Q4 block layout (20 bytes per 32 weights):
//   bytes  0-3:  f32 scale
//   bytes  4-19: 16 packed nibble bytes (2 weights per byte)
//   Dequant: weight = (nibble - 8) * scale
//
// Two access patterns selected by blocks_per_row:
//
//   PER-LANE (blocks_per_row <= 64, i.e. K <= 2048):
//     Each lane owns separate blocks (strided by 32).  The inner loop unpacks
//     all 32 weights from each block with 4x unrolling — 8 multiply-adds per
//     iteration with excellent L1 locality.  No shuffle overhead.  Better for
//     small matrices (MoE experts, attention projections at hidden_size=2048)
//     where the working set fits in cache and launch count is high.
//
//   COOPERATIVE (blocks_per_row > 64, i.e. K > 2048):
//     All 32 lanes cooperate on the SAME block.  Lane l handles weight l:
//     reads nibble byte l/2, extracts low/high nibble, reads x[block*32 + l].
//     Coalesced x reads and broadcast scale via __shfl_sync.  4x unrolled
//     over blocks.  Better for large K where memory coalescing dominates.
// ---------------------------------------------------------------------------
extern "C" __global__ void matvec_q4(
    const MatvecParams params,
    const unsigned char* __restrict__ W_q4,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned int bytes_per_block = 20;
    const unsigned char* row_data = W_q4 + row * blocks_per_row * bytes_per_block;

    float acc = 0.0f;

    if (blocks_per_row <= 64) {
        // Per-lane pattern: each lane processes its own blocks.
        for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
            const unsigned char* block_ptr = row_data + block_idx * bytes_per_block;
            float scale = *((const float*)block_ptr);
            const unsigned char* data = block_ptr + 4;
            unsigned int x_base = block_idx * 32;

            for (unsigned int i = 0; i < 16; i += 4) {
                unsigned char b0 = data[i];
                unsigned char b1 = data[i + 1];
                unsigned char b2 = data[i + 2];
                unsigned char b3 = data[i + 3];

                acc += (float)((int)(b0 & 0xF) - 8) * scale * __bfloat162float(x[x_base + i * 2]);
                acc += (float)((int)(b0 >> 4)  - 8) * scale * __bfloat162float(x[x_base + i * 2 + 1]);
                acc += (float)((int)(b1 & 0xF) - 8) * scale * __bfloat162float(x[x_base + i * 2 + 2]);
                acc += (float)((int)(b1 >> 4)  - 8) * scale * __bfloat162float(x[x_base + i * 2 + 3]);
                acc += (float)((int)(b2 & 0xF) - 8) * scale * __bfloat162float(x[x_base + i * 2 + 4]);
                acc += (float)((int)(b2 >> 4)  - 8) * scale * __bfloat162float(x[x_base + i * 2 + 5]);
                acc += (float)((int)(b3 & 0xF) - 8) * scale * __bfloat162float(x[x_base + i * 2 + 6]);
                acc += (float)((int)(b3 >> 4)  - 8) * scale * __bfloat162float(x[x_base + i * 2 + 7]);
            }
        }
    } else {
        // Cooperative pattern: all 32 lanes process the same block together.
        const unsigned int byte_in_block = lane / 2;
        const unsigned int nibble_hi = lane & 1;

        unsigned int base = 0;
        for (; base + 3 < blocks_per_row; base += 4) {
            float my_scale = 0.0f;
            if (lane < 4) {
                my_scale = *((const float*)(row_data + (base + lane) * bytes_per_block));
            }

            {
                float scale = __shfl_sync(0xffffffff, my_scale, 0);
                unsigned char packed = row_data[(base + 0) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 0) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 1);
                unsigned char packed = row_data[(base + 1) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 1) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 2);
                unsigned char packed = row_data[(base + 2) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 2) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 3);
                unsigned char packed = row_data[(base + 3) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 3) * 32 + lane]);
            }
        }

        for (; base < blocks_per_row; base++) {
            float scale_val = 0.0f;
            if (lane == 0) {
                scale_val = *((const float*)(row_data + base * bytes_per_block));
            }
            float scale = __shfl_sync(0xffffffff, scale_val, 0);
            unsigned char packed = row_data[base * bytes_per_block + 4 + byte_in_block];
            int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
            acc += (float)(nib - 8) * scale * __bfloat162float(x[base * 32 + lane]);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        y[row] = __float2bfloat16(acc);
    }
}

// ===========================================================================
// Batched GEMM — bf16 weights.
// Y = X @ W^T, where X is [batch_size, K] and W is [M, K].
// Same warp-cooperative pattern as matvec applied across batch × M pairs.
// ===========================================================================

struct GemmParams {
    unsigned int batch_size;
    unsigned int M;
    unsigned int K;
};

extern "C" __global__ void gemm_bf16(
    const GemmParams params,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int elem = gid / 32;
    unsigned int lane = gid % 32;
    unsigned int batch = elem / M;
    unsigned int row   = elem % M;

    if (batch >= params.batch_size) return;

    const __nv_bfloat16* w_row = W + row * K;
    const __nv_bfloat16* x_vec = X + batch * K;

    float acc = 0.0f;
    for (unsigned int j = lane * 4; j < K; j += 32 * 4) {
        acc += __bfloat162float(w_row[j])     * __bfloat162float(x_vec[j]);
        acc += __bfloat162float(w_row[j + 1]) * __bfloat162float(x_vec[j + 1]);
        acc += __bfloat162float(w_row[j + 2]) * __bfloat162float(x_vec[j + 2]);
        acc += __bfloat162float(w_row[j + 3]) * __bfloat162float(x_vec[j + 3]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = __float2bfloat16(acc);
    }
}

// ===========================================================================
// Batched GEMM — Q4 weights.
// Same as gemm_bf16 but with inline Q4 dequantisation.
// ===========================================================================

// Same dual access pattern as matvec_q4 — see comments there.
extern "C" __global__ void gemm_q4(
    const GemmParams params,
    const unsigned char* __restrict__ W_q4,
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int elem = gid / 32;
    unsigned int lane = gid % 32;
    unsigned int batch = elem / M;
    unsigned int row   = elem % M;

    if (batch >= params.batch_size) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned int bytes_per_block = 20;
    const unsigned char* row_data = W_q4 + row * blocks_per_row * bytes_per_block;
    const __nv_bfloat16* x_vec = X + batch * K;

    float acc = 0.0f;

    if (blocks_per_row <= 64) {
        for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
            const unsigned char* block_ptr = row_data + block_idx * bytes_per_block;
            float scale = *((const float*)block_ptr);
            const unsigned char* data = block_ptr + 4;
            unsigned int x_base = block_idx * 32;

            for (unsigned int i = 0; i < 16; i += 4) {
                unsigned char b0 = data[i];
                unsigned char b1 = data[i + 1];
                unsigned char b2 = data[i + 2];
                unsigned char b3 = data[i + 3];

                acc += (float)((int)(b0 & 0xF) - 8) * scale * __bfloat162float(x_vec[x_base + i * 2]);
                acc += (float)((int)(b0 >> 4)  - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 1]);
                acc += (float)((int)(b1 & 0xF) - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 2]);
                acc += (float)((int)(b1 >> 4)  - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 3]);
                acc += (float)((int)(b2 & 0xF) - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 4]);
                acc += (float)((int)(b2 >> 4)  - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 5]);
                acc += (float)((int)(b3 & 0xF) - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 6]);
                acc += (float)((int)(b3 >> 4)  - 8) * scale * __bfloat162float(x_vec[x_base + i * 2 + 7]);
            }
        }
    } else {
        const unsigned int byte_in_block = lane / 2;
        const unsigned int nibble_hi = lane & 1;

        unsigned int base = 0;
        for (; base + 3 < blocks_per_row; base += 4) {
            float my_scale = 0.0f;
            if (lane < 4) {
                my_scale = *((const float*)(row_data + (base + lane) * bytes_per_block));
            }

            {
                float scale = __shfl_sync(0xffffffff, my_scale, 0);
                unsigned char packed = row_data[(base + 0) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 0) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 1);
                unsigned char packed = row_data[(base + 1) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 1) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 2);
                unsigned char packed = row_data[(base + 2) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 2) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 3);
                unsigned char packed = row_data[(base + 3) * bytes_per_block + 4 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 3) * 32 + lane]);
            }
        }

        for (; base < blocks_per_row; base++) {
            float scale_val = 0.0f;
            if (lane == 0) {
                scale_val = *((const float*)(row_data + base * bytes_per_block));
            }
            float scale = __shfl_sync(0xffffffff, scale_val, 0);
            unsigned char packed = row_data[base * bytes_per_block + 4 + byte_in_block];
            int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
            acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[base * 32 + lane]);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = __float2bfloat16(acc);
    }
}
