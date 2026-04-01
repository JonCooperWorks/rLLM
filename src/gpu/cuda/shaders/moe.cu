// ===========================================================================
// Fused MoE (Mixture of Experts) CUDA kernels.
//
// LEARNING OVERVIEW
//
// Port of the Metal moe.metal kernels to CUDA for NVIDIA GPUs.
// These kernels are the GPU-side compute for the expert streaming pipeline.
//
// EXPERT STREAMING CONTEXT
//
// In a large MoE model (e.g. Qwen3.5 397B — 512 experts/layer, ~214 GB Q4),
// expert weights live on NVMe.  On each token, the router selects K experts
// (e.g. K=4), their weights are pread()'d from disk in parallel, then copied
// to GPU via pinned-memory async DMA on a dedicated CUDA transfer stream.
// An LRU expert cache (64 slots, ~432 MB Q4) means cache hits skip both
// NVMe reads and PCIe transfers entirely.
//
// Once the expert weights land in GPU memory, THESE kernels run the actual
// FFN computation.  The fused design is critical for streamed inference:
// each expert is only resident briefly, so minimising kernel launches per
// expert directly reduces wall-clock time.
//
// Streaming approach inspired by flash-moe (github.com/danveloper/flash-moe).
//
// TWO FUSED KERNELS
//
//   fused_gate_up_swiglu — combines gate matmul + up matmul + SwiGLU
//     activation into a single kernel.  Reads input vector x once instead
//     of twice (shared between gate and up projections).  Without fusion
//     this would be 3 separate dispatches (gate matvec, up matvec, SiLU+mul).
//     Two variants: bf16 weights and Q4 (4-bit quantized) weights.
//
//   moe_combine_residual — combines k weighted expert outputs + residual
//     add into a single element-wise pass.  Without fusion this would be
//     k scale_add dispatches + 1 add dispatch.
//
// CUDA vs Metal differences:
//   - Warp size is 32 on both platforms — pattern is identical.
//   - CUDA `__shfl_xor_sync` / warp_sum replaces Metal `simd_sum`.
//   - CUDA `__nv_bfloat16` replaces Metal `bfloat`.
//   - CUDA shared memory (`__shared__`) replaces Metal `threadgroup`.
//   - On CUDA (discrete GPU), expert weights arrive via PCIe after async DMA.
//     On Metal (unified memory), weights are directly accessible after memcpy.
//
// Related files:
//   Metal shader:     metal/shaders/moe.metal
//   CUDA bridge:      cuda/kernels/moe.rs
//   Trait contract:   gpu/ops/moe.rs
//   Expert streamer:  model/expert_stream.rs  (pread + LRU cache + DMA)
//   Streaming dispatch: model/primitives.rs   (moe_expert_dispatch_streamed)
// ===========================================================================

#include <cuda_bf16.h>

__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with bf16 weights.
//
// Each warp (32 threads) handles one output row: computes two dot products
// simultaneously (gate_proj @ x and up_proj @ x), then lane 0 applies SiLU
// to the gate result and multiplies by the up result.
//
// Why fuse?  A single MoE expert FFN is gate_proj, up_proj, SiLU, mul, then
// down_proj.  Without fusion that's 3 kernel launches per expert × K experts
// per token × L layers.  For 397B Q4 with K=4 and 96 layers, fusion saves
// ~768 kernel launches per token — significant when expert weights are
// streaming from NVMe and each expert is only GPU-resident briefly.
//
// The bf16 variant is used when experts are stored in bfloat16 (larger models
// where Q4 pre-quantization hasn't been applied).  Q4 variant below is 3.5x
// less I/O per expert — critical for NVMe-bound streaming.
//
// Dispatch: grid = ceil(M*32 / 256), block = 256.
// ---------------------------------------------------------------------------

struct FusedGateUpParams {
    unsigned int M;  // Output dimension (moe_intermediate_size).
    unsigned int K;  // Input dimension (hidden_size).
};

extern "C" __global__ void fused_gate_up_swiglu_bf16(
    const FusedGateUpParams params,
    const __nv_bfloat16* __restrict__ W_gate,  // [M, K]
    const __nv_bfloat16* __restrict__ W_up,    // [M, K]
    const __nv_bfloat16* __restrict__ x,       // [K]
    __nv_bfloat16* __restrict__ output         // [M]
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const __nv_bfloat16* gate_row = W_gate + row * K;
    const __nv_bfloat16* up_row   = W_up   + row * K;

    // Two accumulators: one for gate, one for up projection.
    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    // Strided dot product with 4x unrolling (same pattern as matvec_bf16).
    for (unsigned int j = lane * 4; j < K; j += 32 * 4) {
        float x0 = __bfloat162float(x[j]);
        float x1 = __bfloat162float(x[j + 1]);
        float x2 = __bfloat162float(x[j + 2]);
        float x3 = __bfloat162float(x[j + 3]);

        acc_gate += __bfloat162float(gate_row[j])     * x0;
        acc_gate += __bfloat162float(gate_row[j + 1]) * x1;
        acc_gate += __bfloat162float(gate_row[j + 2]) * x2;
        acc_gate += __bfloat162float(gate_row[j + 3]) * x3;

        acc_up += __bfloat162float(up_row[j])     * x0;
        acc_up += __bfloat162float(up_row[j + 1]) * x1;
        acc_up += __bfloat162float(up_row[j + 2]) * x2;
        acc_up += __bfloat162float(up_row[j + 3]) * x3;
    }

    // Warp reduction for both accumulators.
    acc_gate = warp_sum(acc_gate);
    acc_up   = warp_sum(acc_up);

    // Lane 0 applies SwiGLU: silu(gate) * up.
    if (lane == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        output[row] = __float2bfloat16(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with Q4 weights.
//
// Q4 block layout: 18 bytes per 32 weights (2-byte bf16 scale + 16 packed
// nibbles).  Dequant: weight = (nibble - 8) * scale.  The bf16 scale (vs
// f32) saves 10% I/O per block — critical when NVMe bandwidth is the
// bottleneck during expert streaming.
//
// MoE experts have small K (e.g. 2048), so blocks_per_row <= 64.
// Uses the per-lane Q4 pattern from matvec_q4: each lane processes
// its own blocks independently with no shared memory overhead.
// Both gate and up accumulators share the same x reads for double
// bandwidth savings — each byte of x is loaded once, used for both
// the gate and up dot products.
//
// Performance impact: Q4 streaming gives 3.5x less NVMe I/O per expert
// load vs bf16.  On RTX 4090 this translates to 4.0 tok/s (Q4) vs
// 1.2 tok/s (bf16) on Qwen3.5-122B — the kernel compute is trivial
// compared to the PCIe transfer time.
//
// Dispatch: grid = ceil(M*32 / 256), block = 256.
// ---------------------------------------------------------------------------

extern "C" __global__ void fused_gate_up_swiglu_q4(
    const FusedGateUpParams params,
    const unsigned char* __restrict__ W_gate_q4,  // [M * blocks_per_row * 18]
    const unsigned char* __restrict__ W_up_q4,    // [M * blocks_per_row * 18]
    const __nv_bfloat16* __restrict__ x,          // [K]
    __nv_bfloat16* __restrict__ output            // [M]
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned int bytes_per_block = 18;
    const unsigned char* gate_row = W_gate_q4 + row * blocks_per_row * bytes_per_block;
    const unsigned char* up_row   = W_up_q4   + row * blocks_per_row * bytes_per_block;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    // Per-lane pattern: each lane processes its own blocks.
    // Both gate and up share x reads within each block.
    for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        const unsigned char* g_ptr = gate_row + block_idx * bytes_per_block;
        float g_scale = __bfloat162float(*((const __nv_bfloat16*)g_ptr));
        const unsigned char* g_data = g_ptr + 2;

        const unsigned char* u_ptr = up_row + block_idx * bytes_per_block;
        float u_scale = __bfloat162float(*((const __nv_bfloat16*)u_ptr));
        const unsigned char* u_data = u_ptr + 2;

        unsigned int x_base = block_idx * 32;

        for (unsigned int i = 0; i < 16; i += 4) {
            unsigned char gb0 = g_data[i], gb1 = g_data[i+1], gb2 = g_data[i+2], gb3 = g_data[i+3];
            unsigned char ub0 = u_data[i], ub1 = u_data[i+1], ub2 = u_data[i+2], ub3 = u_data[i+3];

            float x0 = __bfloat162float(x[x_base + i*2]);
            float x1 = __bfloat162float(x[x_base + i*2 + 1]);
            float x2 = __bfloat162float(x[x_base + i*2 + 2]);
            float x3 = __bfloat162float(x[x_base + i*2 + 3]);
            float x4 = __bfloat162float(x[x_base + i*2 + 4]);
            float x5 = __bfloat162float(x[x_base + i*2 + 5]);
            float x6 = __bfloat162float(x[x_base + i*2 + 6]);
            float x7 = __bfloat162float(x[x_base + i*2 + 7]);

            acc_gate += (float)((int)(gb0 & 0xF) - 8) * g_scale * x0;
            acc_gate += (float)((int)(gb0 >> 4)  - 8) * g_scale * x1;
            acc_gate += (float)((int)(gb1 & 0xF) - 8) * g_scale * x2;
            acc_gate += (float)((int)(gb1 >> 4)  - 8) * g_scale * x3;
            acc_gate += (float)((int)(gb2 & 0xF) - 8) * g_scale * x4;
            acc_gate += (float)((int)(gb2 >> 4)  - 8) * g_scale * x5;
            acc_gate += (float)((int)(gb3 & 0xF) - 8) * g_scale * x6;
            acc_gate += (float)((int)(gb3 >> 4)  - 8) * g_scale * x7;

            acc_up += (float)((int)(ub0 & 0xF) - 8) * u_scale * x0;
            acc_up += (float)((int)(ub0 >> 4)  - 8) * u_scale * x1;
            acc_up += (float)((int)(ub1 & 0xF) - 8) * u_scale * x2;
            acc_up += (float)((int)(ub1 >> 4)  - 8) * u_scale * x3;
            acc_up += (float)((int)(ub2 & 0xF) - 8) * u_scale * x4;
            acc_up += (float)((int)(ub2 >> 4)  - 8) * u_scale * x5;
            acc_up += (float)((int)(ub3 & 0xF) - 8) * u_scale * x6;
            acc_up += (float)((int)(ub3 >> 4)  - 8) * u_scale * x7;
        }
    }

    acc_gate = warp_sum(acc_gate);
    acc_up   = warp_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        output[row] = __float2bfloat16(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused MoE combine + residual add.
//
// After all K selected experts have run their FFN (gate+up+SwiGLU → down),
// this kernel weighted-sums their outputs and adds the residual connection:
//   output[gid] = residual[gid] + sum_i(weights[i] * expert_outs[i * hidden + gid])
//
// Replaces k separate scale_add dispatches + 1 add dispatch.
// Routing weights (from the softmax over top-k router logits) are passed
// in a constant struct (max 32 experts) — fits in constant memory/registers.
//
// This is the final kernel in the MoE block before the residual stream
// continues to the next layer.  Pure element-wise, no reductions needed —
// one thread per hidden dimension element.
//
// Dispatch: grid = ceil(hidden_size / 256), block = 256.
// ---------------------------------------------------------------------------

struct MoeCombineParams {
    unsigned int hidden_size;
    unsigned int k;
    float weights[32];  // Max 32 experts.
};

extern "C" __global__ void moe_combine_residual(
    const MoeCombineParams params,
    const __nv_bfloat16* __restrict__ residual,      // [hidden_size]
    const __nv_bfloat16* __restrict__ expert_outs,   // [k, hidden_size]
    __nv_bfloat16* __restrict__ output               // [hidden_size]
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.hidden_size) return;

    float sum = __bfloat162float(residual[gid]);
    for (unsigned int i = 0; i < params.k; i++) {
        sum += params.weights[i] * __bfloat162float(expert_outs[i * params.hidden_size + gid]);
    }
    output[gid] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with Q8 weights.
//
// Q8 block layout: 34 bytes per 32 weights (2-byte bf16 scale + 32 int8 values).
// Same warp-cooperative pattern as bf16 but with inline Q8 dequantisation.
// ---------------------------------------------------------------------------

static constexpr unsigned int Q8_BYTES_PER_BLOCK = 34;

extern "C" __global__ void fused_gate_up_swiglu_q8(
    const FusedGateUpParams params,
    const unsigned char* __restrict__ W_gate_q8,
    const unsigned char* __restrict__ W_up_q8,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned char* gate_row = W_gate_q8 + row * blocks_per_row * Q8_BYTES_PER_BLOCK;
    const unsigned char* up_row   = W_up_q8   + row * blocks_per_row * Q8_BYTES_PER_BLOCK;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (unsigned int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
        const unsigned char* g_ptr = gate_row + block_idx * Q8_BYTES_PER_BLOCK;
        float g_scale = __bfloat162float(*((const __nv_bfloat16*)g_ptr));
        signed char g_val = (signed char)g_ptr[2 + lane];

        const unsigned char* u_ptr = up_row + block_idx * Q8_BYTES_PER_BLOCK;
        float u_scale = __bfloat162float(*((const __nv_bfloat16*)u_ptr));
        signed char u_val = (signed char)u_ptr[2 + lane];

        float x_val = __bfloat162float(x[block_idx * 32 + lane]);

        acc_gate += (float)g_val * g_scale * x_val;
        acc_up   += (float)u_val * u_scale * x_val;
    }

    acc_gate = warp_sum(acc_gate);
    acc_up   = warp_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        output[row] = __float2bfloat16(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU with FP8 E4M3 weights.
//
// FP8 has no block structure — 1 byte per weight, linear addressing.
// Same warp-cooperative pattern as Q8 but with FP8 dequantisation.
// The fp8 conversion function is duplicated here because NVRTC compiles
// each .cu file independently.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fp8_e4m3_to_float_moe(unsigned char bits) {
    unsigned int sign = (bits >> 7) & 1;
    unsigned int exp  = (bits >> 3) & 0xF;
    unsigned int man  = bits & 0x7;
    if (exp == 0xF && man == 0x7) return __uint_as_float(0x7FC00000);
    float val;
    if (exp == 0) {
        val = ((float)man / 8.0f) * 0.015625f;
    } else {
        unsigned int f32_exp = (unsigned int)(exp - 7 + 127);
        unsigned int f32_bits = (sign << 31) | (f32_exp << 23) | (man << 20);
        return __uint_as_float(f32_bits);
    }
    return sign ? -val : val;
}

extern "C" __global__ void fused_gate_up_swiglu_fp8(
    const FusedGateUpParams params,
    const unsigned char* __restrict__ W_gate_fp8,
    const unsigned char* __restrict__ W_up_fp8,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned char* gate_row = W_gate_fp8 + row * K;
    const unsigned char* up_row   = W_up_fp8   + row * K;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    const unsigned int iters = K / 32;
    for (unsigned int i = 0; i < iters; i++) {
        unsigned int col = i * 32 + lane;
        float g_val = fp8_e4m3_to_float_moe(gate_row[col]);
        float u_val = fp8_e4m3_to_float_moe(up_row[col]);
        float x_val = __bfloat162float(x[col]);

        acc_gate += g_val * x_val;
        acc_up   += u_val * x_val;
    }

    acc_gate = warp_sum(acc_gate);
    acc_up   = warp_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        output[row] = __float2bfloat16(silu * acc_up);
    }
}

// ---------------------------------------------------------------------------
// Fused gate+up+SwiGLU for NVFP4 E2M1 expert weights.
//
// Same block layout as Q4 (18 bytes per 32 weights) but nibbles decoded
// via FP4 E2M1 lookup table instead of (nibble - 8) integer subtraction.
// ---------------------------------------------------------------------------

__device__ __constant__ float nvfp4_lut_moe[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void fused_gate_up_swiglu_nvfp4(
    const FusedGateUpParams params,
    const unsigned char* __restrict__ W_gate_nvfp4,
    const unsigned char* __restrict__ W_up_nvfp4,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int M = params.M;
    const unsigned int K = params.K;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = gid / 32;
    unsigned int lane = gid % 32;
    if (row >= M) return;

    const unsigned int blocks_per_row = K / 32;
    const unsigned int bytes_per_block = 18;
    const unsigned char* gate_row = W_gate_nvfp4 + row * blocks_per_row * bytes_per_block;
    const unsigned char* up_row   = W_up_nvfp4   + row * blocks_per_row * bytes_per_block;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (unsigned int block_idx = lane; block_idx < blocks_per_row; block_idx += 32) {
        const unsigned char* g_ptr = gate_row + block_idx * bytes_per_block;
        float g_scale = __bfloat162float(*((const __nv_bfloat16*)g_ptr));
        const unsigned char* g_data = g_ptr + 2;

        const unsigned char* u_ptr = up_row + block_idx * bytes_per_block;
        float u_scale = __bfloat162float(*((const __nv_bfloat16*)u_ptr));
        const unsigned char* u_data = u_ptr + 2;

        unsigned int x_base = block_idx * 32;

        for (unsigned int i = 0; i < 16; i += 4) {
            unsigned char gb0 = g_data[i], gb1 = g_data[i+1], gb2 = g_data[i+2], gb3 = g_data[i+3];
            unsigned char ub0 = u_data[i], ub1 = u_data[i+1], ub2 = u_data[i+2], ub3 = u_data[i+3];

            float x0 = __bfloat162float(x[x_base + i*2]);
            float x1 = __bfloat162float(x[x_base + i*2 + 1]);
            float x2 = __bfloat162float(x[x_base + i*2 + 2]);
            float x3 = __bfloat162float(x[x_base + i*2 + 3]);
            float x4 = __bfloat162float(x[x_base + i*2 + 4]);
            float x5 = __bfloat162float(x[x_base + i*2 + 5]);
            float x6 = __bfloat162float(x[x_base + i*2 + 6]);
            float x7 = __bfloat162float(x[x_base + i*2 + 7]);

            acc_gate += nvfp4_lut_moe[gb0 & 0xF] * g_scale * x0;
            acc_gate += nvfp4_lut_moe[gb0 >> 4]  * g_scale * x1;
            acc_gate += nvfp4_lut_moe[gb1 & 0xF] * g_scale * x2;
            acc_gate += nvfp4_lut_moe[gb1 >> 4]  * g_scale * x3;
            acc_gate += nvfp4_lut_moe[gb2 & 0xF] * g_scale * x4;
            acc_gate += nvfp4_lut_moe[gb2 >> 4]  * g_scale * x5;
            acc_gate += nvfp4_lut_moe[gb3 & 0xF] * g_scale * x6;
            acc_gate += nvfp4_lut_moe[gb3 >> 4]  * g_scale * x7;

            acc_up += nvfp4_lut_moe[ub0 & 0xF] * u_scale * x0;
            acc_up += nvfp4_lut_moe[ub0 >> 4]  * u_scale * x1;
            acc_up += nvfp4_lut_moe[ub1 & 0xF] * u_scale * x2;
            acc_up += nvfp4_lut_moe[ub1 >> 4]  * u_scale * x3;
            acc_up += nvfp4_lut_moe[ub2 & 0xF] * u_scale * x4;
            acc_up += nvfp4_lut_moe[ub2 >> 4]  * u_scale * x5;
            acc_up += nvfp4_lut_moe[ub3 & 0xF] * u_scale * x6;
            acc_up += nvfp4_lut_moe[ub3 >> 4]  * u_scale * x7;
        }
    }

    acc_gate = warp_sum(acc_gate);
    acc_up   = warp_sum(acc_up);

    if (lane == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        output[row] = __float2bfloat16(silu * acc_up);
    }
}
