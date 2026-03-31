// ===========================================================================
// Fused MoE (Mixture of Experts) kernels.
//
// LEARNING OVERVIEW
//
// These kernels fuse multiple operations in the MoE expert FFN pipeline.
// Without fusion, each expert requires four kernel dispatches:
//   1. gate_buf = W_gate @ input       (matvec)
//   2. up_buf   = W_up   @ input       (matvec)
//   3. act_buf  = silu(gate_buf) * up_buf  (silu_mul)
//   4. out_buf  = W_down @ act_buf     (matvec)
//
// The fused gate+up+SwiGLU kernel combines steps 1-3 into a single dispatch
// that reads the input vector once instead of twice (halving bandwidth for
// the input, which is shared between gate and up projections).
//
// The combine+residual kernel replaces the loop of scale_add calls plus
// the final residual add with a single element-wise pass.
//
// Inspired by flash-moe (github.com/danveloper/flash-moe).
//
// Rust trait:  gpu/ops/moe.rs (GpuMoe)
// Rust impl:   gpu/metal/kernels/moe.rs
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with bf16 weights.
//
// Each SIMD group (32 threads) handles one output row: it computes two dot
// products simultaneously (gate_proj @ x and up_proj @ x), then lane 0
// applies SiLU to the gate result and multiplies by the up result.
//
// Why this is faster than separate kernels:
//   1. Input bandwidth: reads x[K] once instead of twice (gate and up share it)
//   2. Dispatch overhead: 1 kernel launch instead of 3 (gate + up + silu_mul)
//   3. Register reuse: the loaded x values are used for both dot products
//      without a round-trip through global memory
//
// The two accumulators (acc_gate, acc_up) double register pressure per SIMD
// group, but MoE expert matrices are small (e.g. M=768, K=2048) so occupancy
// is not the bottleneck.
//
// Dispatch: grid = M * 32, threadgroup = 256 (same as matvec_bf16).
// ---------------------------------------------------------------------------

struct FusedGateUpParams {
    uint M;  // Output dimension (moe_intermediate_size).
    uint K;  // Input dimension (hidden_size).
};

kernel void fused_gate_up_swiglu_bf16(
    constant FusedGateUpParams& params [[buffer(0)]],
    device const bfloat* W_gate        [[buffer(1)]],  // [M, K]
    device const bfloat* W_up          [[buffer(2)]],  // [M, K]
    device const bfloat* x             [[buffer(3)]],  // [K]
    device bfloat* output              [[buffer(4)]],  // [M]
    uint gid                           [[thread_position_in_grid]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint row = gid / 32;
    uint lane = gid % 32;

    if (row >= M) return;

    device const bfloat* gate_row = W_gate + row * K;
    device const bfloat* up_row   = W_up   + row * K;

    // Two accumulators: one for gate projection, one for up projection.
    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    // Strided dot product with 4x unrolling (same pattern as matvec_bf16).
    for (uint j = lane * 4; j < K; j += 32 * 4) {
        float x0 = float(x[j]);
        float x1 = float(x[j + 1]);
        float x2 = float(x[j + 2]);
        float x3 = float(x[j + 3]);

        acc_gate += float(gate_row[j])     * x0;
        acc_gate += float(gate_row[j + 1]) * x1;
        acc_gate += float(gate_row[j + 2]) * x2;
        acc_gate += float(gate_row[j + 3]) * x3;

        acc_up += float(up_row[j])     * x0;
        acc_up += float(up_row[j + 1]) * x1;
        acc_up += float(up_row[j + 2]) * x2;
        acc_up += float(up_row[j + 3]) * x3;
    }

    // SIMD reduction for both accumulators.
    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    // Lane 0 applies SwiGLU: silu(gate) * up.
    if (lane == 0) {
        float silu = acc_gate / (1.0f + exp(-acc_gate));
        output[row] = bfloat(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with Q4 weights.
//
// Same structure as the bf16 variant but both weight matrices are block-wise
// 4-bit quantized.  Uses the FMA-optimized dequant pattern: precompute
// scale*x per element, then fma(nibble, sx, -8*sx).
//
// Dispatch: grid = M * 32, threadgroup = 256 (same as matvec_q4).
// ---------------------------------------------------------------------------

kernel void fused_gate_up_swiglu_q4(
    constant FusedGateUpParams& params [[buffer(0)]],
    device const uchar* W_gate_q4      [[buffer(1)]],  // [M * blocks_per_row * 18]
    device const uchar* W_up_q4        [[buffer(2)]],  // [M * blocks_per_row * 18]
    device const bfloat* x             [[buffer(3)]],   // [K]
    device bfloat* output              [[buffer(4)]],   // [M]
    uint gid                           [[thread_position_in_grid]],
    uint lid                           [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint row = gid / 32;
    uint lane = gid % 32;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 18;

    // Shared memory cache for x — same tiled approach as matvec_q4.
    // Gate and up projections share x, so caching gives a double benefit:
    // x loaded once from device memory, used for both dot products.
    threadgroup float x_shared[4096];

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint tile = 0; tile < K; tile += 4096) {
        uint tile_len = min((uint)4096, K - tile);

        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M) {
            device const uchar* gate_row = W_gate_q4 + row * blocks_per_row * bytes_per_block;
            device const uchar* up_row   = W_up_q4   + row * blocks_per_row * bytes_per_block;

            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                device const uchar* g_ptr = gate_row + block_idx * bytes_per_block;
                float g_scale = float(*((device const bfloat*)g_ptr));
                device const uchar* g_data = g_ptr + 2;

                device const uchar* u_ptr = up_row + block_idx * bytes_per_block;
                float u_scale = float(*((device const bfloat*)u_ptr));
                device const uchar* u_data = u_ptr + 2;

                uint x_local = (block_idx * 32) - tile;

                // FMA-optimised dequant for both gate and up, sharing cached x.
                for (uint i = 0; i < 16; i += 4) {
                    uchar gb0 = g_data[i], gb1 = g_data[i+1], gb2 = g_data[i+2], gb3 = g_data[i+3];
                    uchar ub0 = u_data[i], ub1 = u_data[i+1], ub2 = u_data[i+2], ub3 = u_data[i+3];

                    float x0 = x_shared[x_local + i*2];
                    float x1 = x_shared[x_local + i*2 + 1];
                    float x2 = x_shared[x_local + i*2 + 2];
                    float x3 = x_shared[x_local + i*2 + 3];
                    float x4 = x_shared[x_local + i*2 + 4];
                    float x5 = x_shared[x_local + i*2 + 5];
                    float x6 = x_shared[x_local + i*2 + 6];
                    float x7 = x_shared[x_local + i*2 + 7];

                    float gsx0 = g_scale * x0, gsx1 = g_scale * x1;
                    float gsx2 = g_scale * x2, gsx3 = g_scale * x3;
                    float gsx4 = g_scale * x4, gsx5 = g_scale * x5;
                    float gsx6 = g_scale * x6, gsx7 = g_scale * x7;

                    acc_gate = fma(float(gb0 & 0xF), gsx0, fma(-8.0f, gsx0, acc_gate));
                    acc_gate = fma(float(gb0 >> 4),  gsx1, fma(-8.0f, gsx1, acc_gate));
                    acc_gate = fma(float(gb1 & 0xF), gsx2, fma(-8.0f, gsx2, acc_gate));
                    acc_gate = fma(float(gb1 >> 4),  gsx3, fma(-8.0f, gsx3, acc_gate));
                    acc_gate = fma(float(gb2 & 0xF), gsx4, fma(-8.0f, gsx4, acc_gate));
                    acc_gate = fma(float(gb2 >> 4),  gsx5, fma(-8.0f, gsx5, acc_gate));
                    acc_gate = fma(float(gb3 & 0xF), gsx6, fma(-8.0f, gsx6, acc_gate));
                    acc_gate = fma(float(gb3 >> 4),  gsx7, fma(-8.0f, gsx7, acc_gate));

                    float usx0 = u_scale * x0, usx1 = u_scale * x1;
                    float usx2 = u_scale * x2, usx3 = u_scale * x3;
                    float usx4 = u_scale * x4, usx5 = u_scale * x5;
                    float usx6 = u_scale * x6, usx7 = u_scale * x7;

                    acc_up = fma(float(ub0 & 0xF), usx0, fma(-8.0f, usx0, acc_up));
                    acc_up = fma(float(ub0 >> 4),  usx1, fma(-8.0f, usx1, acc_up));
                    acc_up = fma(float(ub1 & 0xF), usx2, fma(-8.0f, usx2, acc_up));
                    acc_up = fma(float(ub1 >> 4),  usx3, fma(-8.0f, usx3, acc_up));
                    acc_up = fma(float(ub2 & 0xF), usx4, fma(-8.0f, usx4, acc_up));
                    acc_up = fma(float(ub2 >> 4),  usx5, fma(-8.0f, usx5, acc_up));
                    acc_up = fma(float(ub3 & 0xF), usx6, fma(-8.0f, usx6, acc_up));
                    acc_up = fma(float(ub3 >> 4),  usx7, fma(-8.0f, usx7, acc_up));
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + exp(-acc_gate));
        output[row] = bfloat(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with Q8 weights.
//
// Same structure as the Q4 variant but with simpler 8-bit dequantisation:
// each weight is a signed int8 byte, dequant = float(byte) * scale.
//
// Block layout (34 bytes per block of 32 weights):
//   bytes 0-1:   bf16 scale factor
//   bytes 2-33:  32 signed int8 values
//
// Dispatch: grid = M * 32, threadgroup = 256.
// ---------------------------------------------------------------------------

kernel void fused_gate_up_swiglu_q8(
    constant FusedGateUpParams& params [[buffer(0)]],
    device const uchar* W_gate_q8      [[buffer(1)]],  // [M * blocks_per_row * 34]
    device const uchar* W_up_q8        [[buffer(2)]],  // [M * blocks_per_row * 34]
    device const bfloat* x             [[buffer(3)]],   // [K]
    device bfloat* output              [[buffer(4)]],   // [M]
    uint gid                           [[thread_position_in_grid]],
    uint lid                           [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint row = gid / 32;
    uint lane = gid % 32;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 34;

    threadgroup float x_shared[4096];

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint tile = 0; tile < K; tile += 4096) {
        uint tile_len = min((uint)4096, K - tile);

        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M) {
            device const uchar* gate_row = W_gate_q8 + row * blocks_per_row * bytes_per_block;
            device const uchar* up_row   = W_up_q8   + row * blocks_per_row * bytes_per_block;

            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                device const uchar* g_ptr = gate_row + block_idx * bytes_per_block;
                float g_scale = float(*((device const bfloat*)g_ptr));
                device const char* g_data = (device const char*)(g_ptr + 2);

                device const uchar* u_ptr = up_row + block_idx * bytes_per_block;
                float u_scale = float(*((device const bfloat*)u_ptr));
                device const char* u_data = (device const char*)(u_ptr + 2);

                uint x_local = (block_idx * 32) - tile;

                // Process 32 signed int8 weights in chunks of 4.
                for (uint i = 0; i < 32; i += 4) {
                    float x0 = x_shared[x_local + i];
                    float x1 = x_shared[x_local + i + 1];
                    float x2 = x_shared[x_local + i + 2];
                    float x3 = x_shared[x_local + i + 3];

                    acc_gate = fma(float(g_data[i])     * g_scale, x0, acc_gate);
                    acc_gate = fma(float(g_data[i + 1]) * g_scale, x1, acc_gate);
                    acc_gate = fma(float(g_data[i + 2]) * g_scale, x2, acc_gate);
                    acc_gate = fma(float(g_data[i + 3]) * g_scale, x3, acc_gate);

                    acc_up = fma(float(u_data[i])     * u_scale, x0, acc_up);
                    acc_up = fma(float(u_data[i + 1]) * u_scale, x1, acc_up);
                    acc_up = fma(float(u_data[i + 2]) * u_scale, x2, acc_up);
                    acc_up = fma(float(u_data[i + 3]) * u_scale, x3, acc_up);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + exp(-acc_gate));
        output[row] = bfloat(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with TQ3 weights.
//
// Same structure as the Q4 variant but with:
//   1. WHT rotation of x in shared memory (same as matvec_tq3)
//   2. Centroid lookup dequant (3-bit → 8 centroids)
//   3. Dual scales (scale_lo for [0:15], scale_hi for [16:31])
//
// Block layout (16 bytes per block of 32 weights):
//   [0:2]   bf16 scale_lo
//   [2:4]   bf16 scale_hi
//   [4:16]  12 bytes packed 3-bit codes
//
// Dispatch: grid = M * 32, threadgroup = 256.
// ---------------------------------------------------------------------------

// TQ3 centroids and helpers are defined in matmul.metal (same compilation unit
// via include_str! in backend.rs).  We re-declare them here since moe.metal is
// a separate shader source.
constant constexpr float MOE_TQ3_CENTROIDS[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

constant constexpr float MOE_WHT_NORM = 0.176776695f;  // 1/sqrt(32)

inline uint moe_extract_3bit(device const uchar* packed, uint idx) {
    uint bit_offset = idx * 3;
    uint byte_idx = bit_offset / 8;
    uint bit_within = bit_offset % 8;
    uint val = packed[byte_idx];
    if (bit_within + 3 > 8) {
        val |= (uint(packed[byte_idx + 1]) << 8);
    }
    return (val >> bit_within) & 0x7;
}

// Bulk-decode 32 × 3-bit codes into dequantized weights (same as matmul.metal).
inline void moe_decode_tq3_block(
    device const uchar* codes, float scale_lo, float scale_hi, thread float* w
) {
    uint w0 = uint(codes[0]) | (uint(codes[1]) << 8) | (uint(codes[2]) << 16) | (uint(codes[3]) << 24);
    uint w1 = uint(codes[4]) | (uint(codes[5]) << 8) | (uint(codes[6]) << 16) | (uint(codes[7]) << 24);
    uint w2 = uint(codes[8]) | (uint(codes[9]) << 8) | (uint(codes[10]) << 16) | (uint(codes[11]) << 24);
    w[0]  = MOE_TQ3_CENTROIDS[(w0)       & 0x7] * scale_lo;
    w[1]  = MOE_TQ3_CENTROIDS[(w0 >> 3)  & 0x7] * scale_lo;
    w[2]  = MOE_TQ3_CENTROIDS[(w0 >> 6)  & 0x7] * scale_lo;
    w[3]  = MOE_TQ3_CENTROIDS[(w0 >> 9)  & 0x7] * scale_lo;
    w[4]  = MOE_TQ3_CENTROIDS[(w0 >> 12) & 0x7] * scale_lo;
    w[5]  = MOE_TQ3_CENTROIDS[(w0 >> 15) & 0x7] * scale_lo;
    w[6]  = MOE_TQ3_CENTROIDS[(w0 >> 18) & 0x7] * scale_lo;
    w[7]  = MOE_TQ3_CENTROIDS[(w0 >> 21) & 0x7] * scale_lo;
    w[8]  = MOE_TQ3_CENTROIDS[(w0 >> 24) & 0x7] * scale_lo;
    w[9]  = MOE_TQ3_CENTROIDS[(w0 >> 27) & 0x7] * scale_lo;
    w[10] = MOE_TQ3_CENTROIDS[((w0 >> 30) | (w1 << 2)) & 0x7] * scale_lo;
    w[11] = MOE_TQ3_CENTROIDS[(w1 >> 1)  & 0x7] * scale_lo;
    w[12] = MOE_TQ3_CENTROIDS[(w1 >> 4)  & 0x7] * scale_lo;
    w[13] = MOE_TQ3_CENTROIDS[(w1 >> 7)  & 0x7] * scale_lo;
    w[14] = MOE_TQ3_CENTROIDS[(w1 >> 10) & 0x7] * scale_lo;
    w[15] = MOE_TQ3_CENTROIDS[(w1 >> 13) & 0x7] * scale_lo;
    w[16] = MOE_TQ3_CENTROIDS[(w1 >> 16) & 0x7] * scale_hi;
    w[17] = MOE_TQ3_CENTROIDS[(w1 >> 19) & 0x7] * scale_hi;
    w[18] = MOE_TQ3_CENTROIDS[(w1 >> 22) & 0x7] * scale_hi;
    w[19] = MOE_TQ3_CENTROIDS[(w1 >> 25) & 0x7] * scale_hi;
    w[20] = MOE_TQ3_CENTROIDS[(w1 >> 28) & 0x7] * scale_hi;
    w[21] = MOE_TQ3_CENTROIDS[((w1 >> 31) | (w2 << 1)) & 0x7] * scale_hi;
    w[22] = MOE_TQ3_CENTROIDS[(w2 >> 2)  & 0x7] * scale_hi;
    w[23] = MOE_TQ3_CENTROIDS[(w2 >> 5)  & 0x7] * scale_hi;
    w[24] = MOE_TQ3_CENTROIDS[(w2 >> 8)  & 0x7] * scale_hi;
    w[25] = MOE_TQ3_CENTROIDS[(w2 >> 11) & 0x7] * scale_hi;
    w[26] = MOE_TQ3_CENTROIDS[(w2 >> 14) & 0x7] * scale_hi;
    w[27] = MOE_TQ3_CENTROIDS[(w2 >> 17) & 0x7] * scale_hi;
    w[28] = MOE_TQ3_CENTROIDS[(w2 >> 20) & 0x7] * scale_hi;
    w[29] = MOE_TQ3_CENTROIDS[(w2 >> 23) & 0x7] * scale_hi;
    w[30] = MOE_TQ3_CENTROIDS[(w2 >> 26) & 0x7] * scale_hi;
    w[31] = MOE_TQ3_CENTROIDS[(w2 >> 29) & 0x7] * scale_hi;
}

kernel void fused_gate_up_swiglu_tq3(
    constant FusedGateUpParams& params [[buffer(0)]],
    device const uchar* W_gate_tq3     [[buffer(1)]],  // [M * (K/32) * 16]
    device const uchar* W_up_tq3       [[buffer(2)]],  // [M * (K/32) * 16]
    device const bfloat* x             [[buffer(3)]],   // [K]
    device bfloat* output              [[buffer(4)]],   // [M]
    uint gid                           [[thread_position_in_grid]],
    uint lid                           [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint K = params.K;

    uint row = gid / 32;
    uint lane = gid % 32;

    const uint blocks_per_row = K / 32;
    const uint bytes_per_block = 16;

    // Shared memory: load x, then WHT-rotate all blocks in parallel.
    threadgroup float x_shared[4096];

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint tile = 0; tile < K; tile += 4096) {
        uint tile_len = min((uint)4096, K - tile);

        // Cooperative load.
        for (uint i = lid; i < tile_len; i += 256) {
            x_shared[i] = float(x[tile + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WHT all blocks in parallel: 256 threads / 32 per block = 8 blocks per pass.
        uint num_blocks = tile_len / 32;
        for (uint batch = 0; batch < num_blocks; batch += 8) {
            uint my_block = batch + (lid / 32);
            uint my_elem  = lid % 32;

            if (my_block < num_blocks) {
                uint base = my_block * 32;
                for (uint step = 1; step < 32; step <<= 1) {
                    uint pair = my_elem ^ step;
                    float my_val   = x_shared[base + my_elem];
                    float pair_val = x_shared[base + pair];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (my_elem < pair) {
                        x_shared[base + my_elem] = my_val + pair_val;
                        x_shared[base + pair]    = my_val - pair_val;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                x_shared[base + my_elem] *= MOE_WHT_NORM;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < M) {
            device const uchar* gate_row = W_gate_tq3 + row * blocks_per_row * bytes_per_block;
            device const uchar* up_row   = W_up_tq3   + row * blocks_per_row * bytes_per_block;

            uint block_start = tile / 32;
            uint block_end   = (tile + tile_len) / 32;

            for (uint block_idx = block_start + lane; block_idx < block_end; block_idx += 32) {
                device const uchar* g_ptr = gate_row + block_idx * bytes_per_block;
                float g_scale_lo = float(*((device const bfloat*)g_ptr));
                float g_scale_hi = float(*((device const bfloat*)(g_ptr + 2)));

                device const uchar* u_ptr = up_row + block_idx * bytes_per_block;
                float u_scale_lo = float(*((device const bfloat*)u_ptr));
                float u_scale_hi = float(*((device const bfloat*)(u_ptr + 2)));

                float gw[32], uw[32];
                moe_decode_tq3_block(g_ptr + 4, g_scale_lo, g_scale_hi, gw);
                moe_decode_tq3_block(u_ptr + 4, u_scale_lo, u_scale_hi, uw);

                uint xb = (block_idx * 32) - tile;

                for (uint i = 0; i < 32; i++) {
                    float x_val = x_shared[xb + i];
                    acc_gate = fma(gw[i], x_val, acc_gate);
                    acc_up   = fma(uw[i], x_val, acc_up);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + exp(-acc_gate));
        output[row] = bfloat(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused MoE combine + residual add.
//
// Combines k expert outputs with routing weights and adds the residual in
// a single pass:
//   output[j] = residual[j] + sum_i(weights[i] * expert_outs[i * hidden + j])
//
// Replaces k separate scale_add dispatches + 1 add dispatch.
//
// The routing weights are passed in a constant buffer (max 32 experts,
// easily fits in a Metal constant argument buffer).
//
// Dispatch: grid = hidden_size, threadgroup = 256.
// ---------------------------------------------------------------------------

struct MoeCombineParams {
    uint hidden_size;
    uint k;
    float weights[32];  // Max 32 experts.
};

kernel void moe_combine_residual(
    constant MoeCombineParams& params  [[buffer(0)]],
    device const bfloat* residual      [[buffer(1)]],  // [hidden_size]
    device const bfloat* expert_outs   [[buffer(2)]],  // [k, hidden_size]
    device bfloat* output              [[buffer(3)]],  // [hidden_size]
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_size) return;

    float sum = float(residual[gid]);
    for (uint i = 0; i < params.k; i++) {
        sum += params.weights[i] * float(expert_outs[i * params.hidden_size + gid]);
    }
    output[gid] = bfloat(sum);
}
