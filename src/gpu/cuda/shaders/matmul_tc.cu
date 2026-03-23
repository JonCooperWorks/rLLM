// ===========================================================================
// Tensor-core GEMM kernels using WMMA (Warp Matrix Multiply-Accumulate).
//
// LEARNING OVERVIEW
//
// These kernels replace the scalar warp-cooperative GEMM in matmul.cu with
// NVIDIA tensor core operations for sm_80+ GPUs (A100, H100, RTX 40xx).
// The scalar kernels in matmul.cu remain as fallback for older hardware.
//
// WHY TENSOR CORES?
//   The scalar GEMM uses 32 threads per output row doing FMA (fused multiply-
//   add) — one multiply-add per thread per cycle.  Tensor cores perform a
//   16×16×16 matrix multiply-accumulate in a SINGLE instruction across one
//   warp (32 threads).  That's 16×16×16 = 4096 FMAs per warp per instruction,
//   vs ~32 scalar FMAs.  For compute-bound workloads (prefill with batch ≥ 4),
//   this is an 8-16× throughput improvement.
//
// WHY TILES?
//   Tensor cores operate on small 16×16 fragments, but global memory is slow.
//   We load a large tile (128×32 of weights, 32×128 of inputs) into fast shared
//   memory once, then each warp reads its 16×16 fragments from shared memory
//   many times.  This amortises the global memory cost across many WMMA ops.
//   128×32 × 2 bytes = 8 KB per tile — small enough for shared memory.
//
// WHY ACCUMULATE IN F32?
//   bf16 has only ~3 decimal digits of precision.  Accumulating thousands of
//   multiply-adds in bf16 would lose significant precision.  WMMA accumulates
//   in f32 (7 digits) throughout the K-loop, then converts to bf16 only when
//   writing the final result.  This matches cuBLAS behaviour.
//
// WHY DOUBLE BUFFERING?
//   While warps compute WMMA on tile k, the threads also load tile k+1 into
//   a second shared memory buffer.  This overlaps global memory latency with
//   compute, keeping the tensor cores busy.  Two buffers × 16 KB = 32 KB total.
//
// TILE GEOMETRY
//   128×128 output tile, K tiled in steps of 32.
//   256 threads = 8 warps.  Warp layout within the output tile:
//
//     Rows 0-31:    warp0 [cols 0-63]   warp1 [cols 64-127]
//     Rows 32-63:   warp2 [cols 0-63]   warp3 [cols 64-127]
//     Rows 64-95:   warp4 [cols 0-63]   warp5 [cols 64-127]
//     Rows 96-127:  warp6 [cols 0-63]   warp7 [cols 64-127]
//
//   Each warp owns a 2×4 grid of 16×16 WMMA fragments (32 rows × 64 cols).
//
// GEMM DIMENSIONS (Y = X @ W^T):
//   X is [batch_size, K], W is [M, K], Y is [batch_size, M].
//   Grid tiles over (M, batch_size) — blockIdx.x tiles M, blockIdx.y tiles batch.
//
// Related files:
//   Scalar fallback: cuda/shaders/matmul.cu
//   Rust dispatch:   cuda/kernels/matmul.rs
//   Trait contract:  gpu/ops/matmul.rs
// ===========================================================================

#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// Tile dimensions.
// 128×128 output tile fits well on both A100 (164 KB smem/SM) and
// H100 (228 KB smem/SM) — we only need 32 KB.
static constexpr int TILE_M = 128;  // rows of output tile (batch dimension)
static constexpr int TILE_N = 128;  // cols of output tile (M/weight-row dimension)
static constexpr int TILE_K = 32;   // K-dimension tile width

// WMMA fragment dimensions (fixed by hardware).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Warp layout: 4 rows × 2 cols of warp regions within the output tile.
// Each warp owns a 32×64 region = 2×4 grid of 16×16 fragments.
static constexpr int WARPS_M = 4;   // warp rows
static constexpr int WARPS_N = 2;   // warp cols
static constexpr int WARP_TILES_M = 2;  // 16×16 fragments per warp in M
static constexpr int WARP_TILES_N = 4;  // 16×16 fragments per warp in N

// Reuse the same params struct layout as the scalar GEMM.
struct GemmParams {
    unsigned int batch_size;
    unsigned int M;
    unsigned int K;
};

// Q4 block layout (18 bytes per 32 weights):
//   bytes  0-1:  bf16 scale
//   bytes  2-17: 16 packed nibble bytes (2 weights per byte)
//   Dequant: weight = (nibble - 8) * scale
static constexpr int Q4_BLOCK_SIZE = 32;
static constexpr int Q4_BYTES_PER_BLOCK = 18;

// ---------------------------------------------------------------------------
// Shared memory tile loaders.
//
// 256 threads cooperatively load tiles from global memory into shared memory.
// Each thread handles multiple elements to cover the full tile.
// Out-of-bounds loads are zero-filled for correct boundary handling.
// ---------------------------------------------------------------------------

// Load a bf16 weight tile: A_shared[TILE_N][TILE_K] from W[M, K].
// W is row-major: W[row, col] = W[row * K + col].
// A_shared stores TILE_N rows (weight rows) × TILE_K columns (K-slice).
__device__ __forceinline__ void load_weight_tile_bf16(
    __nv_bfloat16 A_shared[TILE_N][TILE_K],
    const __nv_bfloat16* __restrict__ W,
    unsigned int tile_n_start,  // first weight row in this tile
    unsigned int k_start,       // first K column in this tile
    unsigned int M,             // total weight rows
    unsigned int K,             // total K dimension
    unsigned int tid            // thread index within block
) {
    // 128 rows × 32 cols = 4096 elements. 256 threads → 16 elements/thread.
    const unsigned int total_elems = TILE_N * TILE_K;
    for (unsigned int idx = tid; idx < total_elems; idx += 256) {
        unsigned int row = idx / TILE_K;
        unsigned int col = idx % TILE_K;
        unsigned int global_row = tile_n_start + row;
        unsigned int global_col = k_start + col;

        __nv_bfloat16 val;
        if (global_row < M && global_col < K) {
            val = W[global_row * K + global_col];
        } else {
            val = __float2bfloat16(0.0f);
        }
        A_shared[row][col] = val;
    }
}

// Load a bf16 input tile: B_shared[TILE_K][TILE_M] from X[batch_size, K].
// X is row-major: X[batch, col] = X[batch * K + col].
// B_shared stores TILE_K rows (K-slice) × TILE_M columns (batch-slice).
// Note: B is stored transposed relative to X for coalesced WMMA fragment loads.
__device__ __forceinline__ void load_input_tile(
    __nv_bfloat16 B_shared[TILE_K][TILE_M],
    const __nv_bfloat16* __restrict__ X,
    unsigned int tile_m_start,  // first batch index in this tile
    unsigned int k_start,       // first K column in this tile
    unsigned int batch_size,
    unsigned int K,
    unsigned int tid
) {
    // 32 rows × 128 cols = 4096 elements. 256 threads → 16 elements/thread.
    const unsigned int total_elems = TILE_K * TILE_M;
    for (unsigned int idx = tid; idx < total_elems; idx += 256) {
        unsigned int k_row = idx / TILE_M;
        unsigned int batch_col = idx % TILE_M;
        unsigned int global_k = k_start + k_row;
        unsigned int global_batch = tile_m_start + batch_col;

        __nv_bfloat16 val;
        if (global_batch < batch_size && global_k < K) {
            val = X[global_batch * K + global_k];
        } else {
            val = __float2bfloat16(0.0f);
        }
        B_shared[k_row][batch_col] = val;
    }
}

// Load Q4 weight tile: dequantize Q4 blocks into A_shared[TILE_N][TILE_K] as bf16.
//
// Q4 blocks are 18 bytes each (32 weights).  For a TILE_N × TILE_K tile,
// each row has TILE_K/32 = 1 Q4 block.  Total: 128 blocks × 18 bytes = 2304 bytes.
// 256 threads dequant 128×32 = 4096 weights → 16 weights per thread.
__device__ __forceinline__ void load_weight_tile_q4(
    __nv_bfloat16 A_shared[TILE_N][TILE_K],
    const unsigned char* __restrict__ W_q4,
    unsigned int tile_n_start,
    unsigned int k_start,
    unsigned int M,
    unsigned int K,
    unsigned int tid
) {
    const unsigned int blocks_per_row = K / Q4_BLOCK_SIZE;
    const unsigned int k_block = k_start / Q4_BLOCK_SIZE;  // which Q4 block along K

    // Each thread handles 16 output elements (4096 total / 256 threads).
    const unsigned int total_elems = TILE_N * TILE_K;
    for (unsigned int idx = tid; idx < total_elems; idx += 256) {
        unsigned int row = idx / TILE_K;
        unsigned int col = idx % TILE_K;
        unsigned int global_row = tile_n_start + row;

        __nv_bfloat16 val;
        if (global_row < M && (k_start + col) < K) {
            // Locate the Q4 block for this weight row and K position.
            const unsigned char* row_data = W_q4 + global_row * blocks_per_row * Q4_BYTES_PER_BLOCK;
            const unsigned char* block_ptr = row_data + k_block * Q4_BYTES_PER_BLOCK;

            // Read scale from first 2 bytes of the block.
            float scale = __bfloat162float(*((const __nv_bfloat16*)block_ptr));

            // col is the position within the 32-weight block (since TILE_K=32).
            unsigned int nibble_idx = col;
            unsigned int byte_idx = nibble_idx / 2;
            unsigned int is_high = nibble_idx & 1;
            unsigned char packed = block_ptr[2 + byte_idx];
            int nibble = is_high ? (packed >> 4) : (packed & 0xF);

            val = __float2bfloat16((float)(nibble - 8) * scale);
        } else {
            val = __float2bfloat16(0.0f);
        }
        A_shared[row][col] = val;
    }
}

// ---------------------------------------------------------------------------
// Tensor-core GEMM implementation (templated on weight format).
//
// Y = X @ W^T
//   X: [batch_size, K]  (bf16)
//   W: [M, K]           (bf16 or Q4)
//   Y: [batch_size, M]  (bf16)
//
// Grid: (ceil(M/128), ceil(batch_size/128))
// Block: 256 threads (8 warps)
// Shared memory: 32 KB (double-buffered A and B tiles)
// ---------------------------------------------------------------------------
template<bool IS_Q4>
__global__ void gemm_tc_impl(
    const GemmParams params,
    const void* __restrict__ W,       // bf16* or uint8* depending on IS_Q4
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y
) {
    const unsigned int batch_size = params.batch_size;
    const unsigned int M = params.M;
    const unsigned int K = params.K;

    // Which 128×128 output tile does this block own?
    const unsigned int tile_n_start = blockIdx.x * TILE_N;  // weight-row (M) dimension
    const unsigned int tile_m_start = blockIdx.y * TILE_M;  // batch dimension

    // Early exit if this tile is entirely out of bounds.
    if (tile_n_start >= M && tile_m_start >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int warp_row = warp_id / WARPS_N;  // 0..3
    const unsigned int warp_col = warp_id % WARPS_N;  // 0..1

    // -----------------------------------------------------------------------
    // Accumulator fragments: each warp owns a 2×4 grid of 16×16 f32 fragments.
    // Initialised to zero — we accumulate across K tiles.
    // -----------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];
    for (int i = 0; i < WARP_TILES_M; i++)
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // -----------------------------------------------------------------------
    // Double-buffered shared memory.
    //
    // A_shared: weight tile [TILE_N][TILE_K] — rows are weight rows (M dim),
    //   cols are K-slice.  WMMA loads row_major fragments from this.
    //
    // B_shared: input tile [TILE_K][TILE_M] — rows are K-slice, cols are batch
    //   indices.  WMMA loads col_major fragments from this (each column is one
    //   batch element's K-slice, stored contiguously down columns).
    //
    // Total: 2 × (128×32 + 32×128) × 2 bytes = 32 KB.
    // -----------------------------------------------------------------------
    __shared__ __nv_bfloat16 A_shared[2][TILE_N][TILE_K];
    __shared__ __nv_bfloat16 B_shared[2][TILE_K][TILE_M];

    const unsigned int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // -----------------------------------------------------------------------
    // K-loop with double buffering.
    //
    // Iteration 0: load tile 0 into buffer 0, sync, compute on buffer 0.
    // Iteration k: load tile k into buffer (k%2), sync, compute on buffer (k%2).
    //
    // (A simpler approach than true async double-buffering, but still effective
    // because the load and compute phases use different hardware units — the
    // memory subsystem pipelines loads while tensor cores compute.)
    // -----------------------------------------------------------------------
    for (unsigned int kt = 0; kt < num_k_tiles; kt++) {
        int buf = kt % 2;
        unsigned int k_start = kt * TILE_K;

        // Load weight tile (A) into shared memory.
        if constexpr (IS_Q4) {
            load_weight_tile_q4(
                A_shared[buf],
                (const unsigned char*)W,
                tile_n_start, k_start, M, K, tid
            );
        } else {
            load_weight_tile_bf16(
                A_shared[buf],
                (const __nv_bfloat16*)W,
                tile_n_start, k_start, M, K, tid
            );
        }

        // Load input tile (B) into shared memory.
        load_input_tile(
            B_shared[buf],
            X,
            tile_m_start, k_start, batch_size, K, tid
        );

        // All threads must finish loading before warps start reading fragments.
        __syncthreads();

        // -------------------------------------------------------------------
        // WMMA compute: each warp processes its 2×4 fragment grid.
        //
        // The K tile is 32 wide, but WMMA operates on K=16 slices, so we do
        // two WMMA passes per K-tile (k_inner = 0 and k_inner = 16).
        // -------------------------------------------------------------------
        for (int k_inner = 0; k_inner < TILE_K; k_inner += WMMA_K) {
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                // A fragment: 16 rows of weight starting at this warp's row offset.
                // A_shared layout: [TILE_N][TILE_K], row-major.
                // Warp's row offset: warp_row * 32 + wi * 16.
                unsigned int a_row = warp_row * (TILE_N / WARPS_M) + wi * WMMA_M;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __nv_bfloat16, wmma::row_major> a_frag;

                wmma::load_matrix_sync(
                    a_frag,
                    &A_shared[buf][a_row][k_inner],
                    TILE_K  // leading dimension (stride between rows)
                );

                for (int wj = 0; wj < WARP_TILES_N; wj++) {
                    // B fragment: 16 columns of input at this warp's column offset.
                    // B_shared layout: [TILE_K][TILE_M] — K is rows, batch is cols.
                    // row_major: B_frag(k, n) = B_shared[k_inner + k][b_col + n]
                    //   = X[tile_m + b_col + n, k_start + k_inner + k]
                    // So k maps to K dimension, n maps to batch dimension. Correct.
                    unsigned int b_col = warp_col * (TILE_M / WARPS_N) + wj * WMMA_N;

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(
                        b_frag,
                        &B_shared[buf][k_inner][b_col],
                        TILE_M  // leading dimension (stride between rows)
                    );

                    // Tensor core 16×16×16 multiply-accumulate.
                    // c_frag += a_frag × b_frag (all in one warp instruction).
                    wmma::mma_sync(c_frag[wi][wj], a_frag, b_frag, c_frag[wi][wj]);
                }
            }
        }

        // Ensure all warps are done reading shared memory before next load.
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Epilogue: store accumulated f32 fragments to global memory as bf16.
    //
    // Each warp's 2×4 fragment grid maps to a 32×64 region of the output tile.
    // We stage each fragment through a per-warp region of shared memory (reusing
    // A_shared which is no longer needed after the K-loop), convert f32→bf16,
    // and write to global Y with bounds checking.
    //
    // Dimension mapping:
    //   c_frag[wi][wj] is a 16×16 matrix where:
    //     rows  = M dimension (weight rows), offset by warp_row * 32 + wi * 16
    //     cols  = batch dimension,           offset by warp_col * 64 + wj * 16
    //   Y layout: Y[batch, m] = Y[batch * M + m]  (row-major)
    // -----------------------------------------------------------------------

    // Reuse A_shared as per-warp staging buffers.  Each warp needs 16×16 floats
    // = 1 KB.  8 warps × 1 KB = 8 KB, easily fits in the 16 KB A_shared[0] buffer.
    float* warp_staging = (float*)&A_shared[0][0][0] + warp_id * WMMA_M * WMMA_N;

    const unsigned int lane = tid % 32;

    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            // Store this 16×16 f32 fragment to this warp's staging region.
            // store_matrix_sync is warp-cooperative (no cross-warp sync needed).
            wmma::store_matrix_sync(
                warp_staging,
                c_frag[wi][wj],
                WMMA_N,           // leading dimension
                wmma::mem_row_major
            );

            // Global output position for this fragment.
            // Rows of the fragment = M dimension (from A/weight side).
            // Cols of the fragment = batch dimension (from B/input side).
            unsigned int frag_m_start = tile_n_start + warp_row * (TILE_N / WARPS_M) + wi * WMMA_M;
            unsigned int frag_batch_start = tile_m_start + warp_col * (TILE_M / WARPS_N) + wj * WMMA_N;

            // 32 threads store 16×16 = 256 elements → 8 elements per thread.
            for (unsigned int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
                unsigned int r = idx / WMMA_N;  // M dimension offset
                unsigned int c = idx % WMMA_N;  // batch dimension offset
                unsigned int global_m = frag_m_start + r;
                unsigned int global_batch = frag_batch_start + c;

                if (global_batch < batch_size && global_m < M) {
                    Y[global_batch * M + global_m] = __float2bfloat16(warp_staging[idx]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points — extern "C" wrappers for NVRTC (no C++ name mangling).
//
// These have the same argument layout as the scalar gemm_bf16/gemm_q4 in
// matmul.cu, so the Rust dispatch can use identical launch_builder args.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bf16_tc(
    const GemmParams params,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y
) {
    gemm_tc_impl<false>(params, (const void*)W, X, Y);
}

extern "C" __global__ void gemm_q4_tc(
    const GemmParams params,
    const unsigned char* __restrict__ W_q4,
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y
) {
    gemm_tc_impl<true>(params, (const void*)W_q4, X, Y);
}
