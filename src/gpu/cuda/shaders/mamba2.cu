// ===========================================================================
// Mamba-2 CUDA kernels — Selective State Space Model for Nemotron-H.
//
// Port of the Metal mamba2.metal kernels to CUDA for NVIDIA GPUs.
//
// Three kernels:
//   mamba2_conv1d_silu    — depthwise conv1d + bias + SiLU (one thread/channel)
//   mamba2_ssm_step       — SSM state update + output (one block/head)
//   mamba2_gated_rms_norm — gated grouped RMSNorm (one block/group)
//
// Related files:
//   Metal shader:     metal/shaders/mamba2.metal
//   CUDA bridge:      cuda/kernels/mamba2.rs
//   Trait contract:   gpu/ops/mamba2.rs
//   CPU reference:    cpu/mod.rs
// ===========================================================================

#include <cuda_bf16.h>

// -----------------------------------------------------------------------
// Warp-level sum reduction using butterfly shuffle.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ===========================================================================
// Depthwise Conv1D with bias and SiLU activation.
//
// One thread per channel.  Identical to DeltaNet's conv1d but with an
// additive bias BEFORE the SiLU activation.
//
// Dispatch: cfg_1d(dim, 256)
// ===========================================================================

struct MambaConv1dParams {
    unsigned int dim;
    unsigned int kernel_size;
    unsigned int input_offset;
};

extern "C" __global__ void mamba2_conv1d_silu(
    const MambaConv1dParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ history,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.dim) return;

    // Start with bias.
    float acc = __bfloat162float(bias[gid]);

    // Convolve over history buffer (kernel_size - 1 prior tokens).
    for (unsigned int j = 0; j < params.kernel_size - 1; j++) {
        acc += __bfloat162float(weight[gid * params.kernel_size + j]) *
               __bfloat162float(history[j * params.dim + gid]);
    }

    // Current token occupies the last weight tap.
    acc += __bfloat162float(weight[gid * params.kernel_size + params.kernel_size - 1]) *
           __bfloat162float(input[params.input_offset + gid]);

    // SiLU activation.
    out[gid] = __float2bfloat16(acc / (1.0f + expf(-acc)));
}

// ===========================================================================
// Mamba-2 SSM step: state update + output readout.
//
// One block (256 threads) per head.
//
// Algorithm per head h:
//   1. Thread 0 computes shared scalars: dt, dA, D
//   2. All threads update state rows and compute output
//   3. Write raw output (gated norm done separately)
//
// Dispatch: cfg_blocks(num_heads, 256)
// ===========================================================================

struct Mamba2SsmStepParams {
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int state_size;
    unsigned int n_groups;
    unsigned int b_offset;
    unsigned int c_offset;
    unsigned int dt_offset;
    float eps;
};

extern "C" __global__ void mamba2_ssm_step(
    const Mamba2SsmStepParams params,
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bc_buf,
    const __nv_bfloat16* __restrict__ dt_buf,
    const float* __restrict__ a_log,
    const float* __restrict__ d_skip,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out
) {
    const unsigned int h = blockIdx.x;
    if (h >= params.num_heads) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;
    const unsigned int hd = params.head_dim;
    const unsigned int ss = params.state_size;

    // Which B/C group does this head belong to?
    unsigned int g = h * params.n_groups / params.num_heads;

    // --- Step 1: Shared scalars (thread 0 computes, broadcast via shared) ---
    __shared__ float shared_dt_val;
    __shared__ float shared_dA;
    __shared__ float shared_D_val;

    if (tid == 0) {
        float dt_raw = __bfloat162float(dt_buf[params.dt_offset + h]) + dt_bias[h];
        float dt_val = logf(1.0f + expf(dt_raw));  // softplus

        float dA = expf(dt_val * (-expf(a_log[h])));

        shared_dt_val = dt_val;
        shared_dA = dA;
        shared_D_val = d_skip[h];
    }
    __syncthreads();

    float dt_val = shared_dt_val;
    float dA = shared_dA;
    float D_val = shared_D_val;

    // --- Step 2: State update + output computation ---
    __shared__ float y_shared[256];
    y_shared[tid] = 0.0f;

    for (unsigned int i = tid; i < hd; i += tg_size) {
        float x_val = __bfloat162float(x[h * hd + i]);

        unsigned int state_base = h * hd * ss + i * ss;

        float y_val = 0.0f;
        for (unsigned int s = 0; s < ss; s++) {
            unsigned int idx = state_base + s;
            float b_val = __bfloat162float(bc_buf[params.b_offset + g * ss + s]);

            // State update: state = dA * state + dt * x * B
            state[idx] = dA * state[idx] + dt_val * x_val * b_val;

            // Accumulate output: y += state * C
            float c_val = __bfloat162float(bc_buf[params.c_offset + g * ss + s]);
            y_val += state[idx] * c_val;
        }

        // Skip connection
        y_val += D_val * x_val;

        y_shared[i] = y_val;
    }
    __syncthreads();

    // --- Step 3: Write raw output ---
    for (unsigned int i = tid; i < hd; i += tg_size) {
        out[h * hd + i] = __float2bfloat16(y_shared[i]);
    }
}

// ===========================================================================
// Gated grouped RMSNorm for Mamba-2 output.
//
// Computes: out[i] = rmsnorm_grouped(y[i] * silu(z[i])) * weight[i]
//
// One block per group of group_size elements.
//
// Dispatch: cfg_blocks(d_inner / group_size, 256)
// ===========================================================================

struct MambaGatedNormParams {
    unsigned int d_inner;
    unsigned int group_size;
    unsigned int z_offset;
    float eps;
};

extern "C" __global__ void mamba2_gated_rms_norm(
    const MambaGatedNormParams params,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ z_buf,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out
) {
    unsigned int g = blockIdx.x;
    unsigned int gs = params.group_size;
    unsigned int base = g * gs;
    if (base >= params.d_inner) return;

    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    // Phase 1: Compute gated values and accumulate sum-of-squares.
    __shared__ float shared_vals[512];  // Max group_size.
    float local_ss = 0.0f;

    for (unsigned int i = tid; i < gs; i += tg_size) {
        float y_val = __bfloat162float(y[base + i]);
        float z_val = __bfloat162float(z_buf[params.z_offset + base + i]);
        float gate = z_val / (1.0f + expf(-z_val));  // silu(z)
        float gated = y_val * gate;
        shared_vals[i] = gated;
        local_ss += gated * gated;
    }

    // Phase 2: Reduce sum-of-squares across block.
    __shared__ float reduce_buf[256];
    reduce_buf[tid] = local_ss;
    __syncthreads();

    for (unsigned int stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(reduce_buf[0] / (float)gs + params.eps);

    // Phase 3: Normalize and write.
    for (unsigned int i = tid; i < gs; i += tg_size) {
        float normed = shared_vals[i] * rms;
        float w = __bfloat162float(weight[base + i]);
        out[base + i] = __float2bfloat16(normed * w);
    }
}
