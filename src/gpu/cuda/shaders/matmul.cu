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
// Q4 block layout (18 bytes per 32 weights):
//   bytes  0-1:  bf16 scale
//   bytes  2-17: 16 packed nibble bytes (2 weights per byte)
//   Dequant: weight = (nibble - 8) * scale
//   bf16 scale (vs f32) saves 10% I/O per block — critical for NVMe-bound
//   expert streaming where bandwidth is the bottleneck.
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
    const unsigned int bytes_per_block = 18;
    const unsigned char* row_data = W_q4 + row * blocks_per_row * bytes_per_block;

    float acc = 0.0f;

    if (blocks_per_row <= 64) {
        // Per-lane pattern: each lane processes its own blocks.
        for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
            const unsigned char* block_ptr = row_data + block_idx * bytes_per_block;
            float scale = __bfloat162float(*((const __nv_bfloat16*)block_ptr));
            const unsigned char* data = block_ptr + 2;
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
                my_scale = __bfloat162float(*((const __nv_bfloat16*)(row_data + (base + lane) * bytes_per_block)));
            }

            {
                float scale = __shfl_sync(0xffffffff, my_scale, 0);
                unsigned char packed = row_data[(base + 0) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 0) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 1);
                unsigned char packed = row_data[(base + 1) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 1) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 2);
                unsigned char packed = row_data[(base + 2) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 2) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 3);
                unsigned char packed = row_data[(base + 3) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x[(base + 3) * 32 + lane]);
            }
        }

        for (; base < blocks_per_row; base++) {
            float scale_val = 0.0f;
            if (lane == 0) {
                scale_val = __bfloat162float(*((const __nv_bfloat16*)(row_data + base * bytes_per_block)));
            }
            float scale = __shfl_sync(0xffffffff, scale_val, 0);
            unsigned char packed = row_data[base * bytes_per_block + 2 + byte_in_block];
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
    const unsigned int bytes_per_block = 18;
    const unsigned char* row_data = W_q4 + row * blocks_per_row * bytes_per_block;
    const __nv_bfloat16* x_vec = X + batch * K;

    float acc = 0.0f;

    if (blocks_per_row <= 64) {
        for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
            const unsigned char* block_ptr = row_data + block_idx * bytes_per_block;
            float scale = __bfloat162float(*((const __nv_bfloat16*)block_ptr));
            const unsigned char* data = block_ptr + 2;
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
                my_scale = __bfloat162float(*((const __nv_bfloat16*)(row_data + (base + lane) * bytes_per_block)));
            }

            {
                float scale = __shfl_sync(0xffffffff, my_scale, 0);
                unsigned char packed = row_data[(base + 0) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 0) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 1);
                unsigned char packed = row_data[(base + 1) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 1) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 2);
                unsigned char packed = row_data[(base + 2) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 2) * 32 + lane]);
            }
            {
                float scale = __shfl_sync(0xffffffff, my_scale, 3);
                unsigned char packed = row_data[(base + 3) * bytes_per_block + 2 + byte_in_block];
                int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
                acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[(base + 3) * 32 + lane]);
            }
        }

        for (; base < blocks_per_row; base++) {
            float scale_val = 0.0f;
            if (lane == 0) {
                scale_val = __bfloat162float(*((const __nv_bfloat16*)(row_data + base * bytes_per_block)));
            }
            float scale = __shfl_sync(0xffffffff, scale_val, 0);
            unsigned char packed = row_data[base * bytes_per_block + 2 + byte_in_block];
            int nib = nibble_hi ? (packed >> 4) : (packed & 0xF);
            acc += (float)(nib - 8) * scale * __bfloat162float(x_vec[base * 32 + lane]);
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = __float2bfloat16(acc);
    }
}

// ===========================================================================
// Q8 kernels — 8-bit symmetric quantization.
//
// Q8 block layout (34 bytes per 32 weights):
//   bytes  0-1:  bf16 scale
//   bytes  2-33: 32 signed int8 values
//   Dequant: weight = float(q) * scale
//
// Simpler than Q4 — one byte per weight, no nibble extraction.
// ===========================================================================

static constexpr unsigned int Q8_BYTES_PER_BLOCK = 34;

// ---------------------------------------------------------------------------
// matvec_q8 — SIMD-cooperative matvec with inline Q8 dequantisation.
// ---------------------------------------------------------------------------
extern "C" __global__ void matvec_q8(
    const MatvecParams params,
    const unsigned char* __restrict__ W_q8,
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
    const unsigned char* row_data = W_q8 + row * blocks_per_row * Q8_BYTES_PER_BLOCK;

    float acc = 0.0f;

    // Cooperative: all 32 lanes process the same block, each handles weight[lane].
    for (unsigned int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
        const unsigned char* block_ptr = row_data + block_idx * Q8_BYTES_PER_BLOCK;
        float scale = __bfloat162float(*((const __nv_bfloat16*)block_ptr));
        signed char q_val = (signed char)block_ptr[2 + lane];
        acc += (float)q_val * scale * __bfloat162float(x[block_idx * 32 + lane]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        y[row] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// gemm_q8 — batched GEMM with inline Q8 dequantisation.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemm_q8(
    const GemmParams params,
    const unsigned char* __restrict__ W_q8,
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
    const unsigned char* row_data = W_q8 + row * blocks_per_row * Q8_BYTES_PER_BLOCK;
    const __nv_bfloat16* x_vec = X + batch * K;

    float acc = 0.0f;

    for (unsigned int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
        const unsigned char* block_ptr = row_data + block_idx * Q8_BYTES_PER_BLOCK;
        float scale = __bfloat162float(*((const __nv_bfloat16*)block_ptr));
        signed char q_val = (signed char)block_ptr[2 + lane];
        acc += (float)q_val * scale * __bfloat162float(x_vec[block_idx * 32 + lane]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = __float2bfloat16(acc);
    }
}

// ===========================================================================
// FP8 E4M3 kernels — IEEE 8-bit float, 1 byte per weight, no block structure.
//
// Used on NVIDIA SM 89+ (Ada/Hopper) where native FP8 hardware is available.
// Unlike Q4/Q8, there is no per-block scale — each byte is an independent
// FP8 E4M3 value that can be directly converted to float.
//
// FP8 E4M3: 1 sign + 4 exponent (bias 7) + 3 mantissa.  Range ±448.
// No infinity; NaN = 0x7F.
//
// Dequantisation is a simple format conversion, not scale*quantized_int.
// ===========================================================================

// ---------------------------------------------------------------------------
// fp8_e4m3_to_float — convert a single FP8 E4M3 byte to float.
//
// Manual bit manipulation for portability across CUDA versions.
// On SM 89+ this could use __nv_cvt_fp8_to_halfraw but the manual path
// avoids version-gating and compiles everywhere.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char bits) {
    unsigned int sign = (bits >> 7) & 1;
    unsigned int exp  = (bits >> 3) & 0xF;
    unsigned int man  = bits & 0x7;

    // NaN: E=1111, M=111
    if (exp == 0xF && man == 0x7) return __uint_as_float(0x7FC00000); // quiet NaN

    float val;
    if (exp == 0) {
        // Subnormal: 0.man * 2^(-6)
        val = ((float)man / 8.0f) * 0.015625f; // 2^-6 = 0.015625
    } else {
        // Normal: (1 + man/8) * 2^(exp - 7)
        // Build f32 directly: rebias exponent from 7 to 127.
        unsigned int f32_exp = (unsigned int)(exp - 7 + 127);
        unsigned int f32_bits = (sign << 31) | (f32_exp << 23) | (man << 20);
        return __uint_as_float(f32_bits);
    }
    return sign ? -val : val;
}

// ---------------------------------------------------------------------------
// matvec_fp8 — SIMD-cooperative matvec with inline FP8 dequantisation.
//
// Same warp-cooperative pattern as matvec_q8 but with linear addressing
// (no block structure).  Each lane processes one weight per iteration,
// 32 lanes cover 32 consecutive weights.
// ---------------------------------------------------------------------------
extern "C" __global__ void matvec_fp8(
    const MatvecParams params,
    const unsigned char* __restrict__ W_fp8,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned char* row_data = W_fp8 + row * K;

    float acc = 0.0f;

    // 32 lanes cooperatively process 32 weights at a time.
    const unsigned int iters = K / 32;
    for (unsigned int i = 0; i < iters; i++) {
        unsigned int col = i * 32 + lane;
        float w = fp8_e4m3_to_float(row_data[col]);
        acc += w * __bfloat162float(x[col]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        y[row] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// gemm_fp8 — batched GEMM with inline FP8 dequantisation.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemm_fp8(
    const GemmParams params,
    const unsigned char* __restrict__ W_fp8,
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

    const unsigned char* row_data = W_fp8 + row * K;
    const __nv_bfloat16* x_vec = X + batch * K;

    float acc = 0.0f;

    const unsigned int iters = K / 32;
    for (unsigned int i = 0; i < iters; i++) {
        unsigned int col = i * 32 + lane;
        float w = fp8_e4m3_to_float(row_data[col]);
        acc += w * __bfloat162float(x_vec[col]);
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        Y[batch * M + row] = __float2bfloat16(acc);
    }
}
