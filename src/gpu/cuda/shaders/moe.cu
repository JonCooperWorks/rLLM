// ===========================================================================
// Fused MoE (Mixture of Experts) CUDA kernels.
//
// LEARNING OVERVIEW
//
// Port of the Metal moe.metal kernels to CUDA for NVIDIA GPUs.
//
// Two fused kernels that reduce dispatch overhead for MoE expert FFN:
//
//   fused_gate_up_swiglu — combines gate matmul + up matmul + SwiGLU
//     activation into a single kernel.  Reads input vector once instead
//     of twice (shared between gate and up projections).  Two variants:
//     bf16 weights and Q4 (4-bit quantized) weights.
//
//   moe_combine_residual — combines k weighted expert outputs + residual
//     add into a single element-wise pass.
//
// CUDA vs Metal differences:
//   - Warp size is 32 on both platforms — pattern is identical.
//   - CUDA `__shfl_xor_sync` / warp_sum replaces Metal `simd_sum`.
//   - CUDA `__nv_bfloat16` replaces Metal `bfloat`.
//   - CUDA shared memory (`__shared__`) replaces Metal `threadgroup`.
//
// Related files:
//   Metal shader:  metal/shaders/moe.metal
//   CUDA bridge:   cuda/kernels/moe.rs
//   Trait contract: gpu/ops/moe.rs
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
// MoE experts have small K (e.g. 2048), so blocks_per_row <= 64.
// Uses the per-lane Q4 pattern from matvec_q4: each lane processes
// its own blocks independently with no shared memory overhead.
// Both gate and up accumulators share the same x reads for double
// bandwidth savings.
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
// Combines k expert outputs with routing weights and adds the residual:
//   output[gid] = residual[gid] + sum_i(weights[i] * expert_outs[i * hidden + gid])
//
// Replaces k separate scale_add dispatches + 1 add dispatch.
// Routing weights are passed in a constant struct (max 32 experts).
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
