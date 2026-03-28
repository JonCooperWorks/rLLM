// ===========================================================================
// Vision encoder utility CUDA kernels.
//
// Port of the Metal vision.metal kernels to CUDA for NVIDIA GPUs.
//
// Three kernels:
//   spatial_merge      — rearrange 2D grid tokens by merging spatial neighbours
//   spatial_merge_norm — fused merge + LayerNorm
//   scatter_vision_tokens — overwrite text embeddings at image placeholders
//
// Related files:
//   Metal shader:    metal/shaders/vision.metal
//   CUDA bridge:     cuda/kernels/vision.rs
//   Trait contract:  gpu/ops/vision.rs
// ===========================================================================

#include <cuda_bf16.h>

__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Spatial merge: rearrange [grid_h*grid_w, hidden] → [merged_tokens, hidden*ms*ms]
//
// Dispatch: cfg_1d(total_elements, 256)
// ---------------------------------------------------------------------------

struct SpatialMergeParams {
    unsigned int grid_h;
    unsigned int grid_w;
    unsigned int hidden_dim;
    unsigned int merge_size;
};

extern "C" __global__ void spatial_merge(
    const SpatialMergeParams params,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int ms = params.merge_size;
    const unsigned int hd = params.hidden_dim;
    const unsigned int out_w = params.grid_w / ms;
    const unsigned int merged_hd = hd * ms * ms;
    const unsigned int total_elements = (params.grid_h / ms) * out_w * merged_hd;

    if (gid >= total_elements) return;

    unsigned int out_token = gid / merged_hd;
    unsigned int elem      = gid % merged_hd;

    unsigned int out_row = out_token / out_w;
    unsigned int out_col = out_token % out_w;

    unsigned int sub_token = elem / hd;
    unsigned int sub_elem  = elem % hd;

    unsigned int dy = sub_token / ms;
    unsigned int dx = sub_token % ms;

    unsigned int src_row = out_row * ms + dy;
    unsigned int src_col = out_col * ms + dx;
    unsigned int src_idx = (src_row * params.grid_w + src_col) * hd + sub_elem;

    output[gid] = input[src_idx];
}

// ---------------------------------------------------------------------------
// Fused spatial merge + LayerNorm.
//
// One block of 256 threads per merged output token.
//
// Dispatch: cfg_blocks(num_merged_tokens, 256)
// ---------------------------------------------------------------------------

struct SpatialMergeNormParams {
    unsigned int grid_h;
    unsigned int grid_w;
    unsigned int hidden_dim;
    unsigned int merge_size;
    float eps;
};

extern "C" __global__ void spatial_merge_norm(
    const SpatialMergeNormParams params,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int row_id = blockIdx.x;
    const unsigned int ms = params.merge_size;
    const unsigned int hd = params.hidden_dim;
    const unsigned int out_w = params.grid_w / ms;
    const unsigned int merged_hd = hd * ms * ms;
    const unsigned int num_merged = (params.grid_h / ms) * out_w;

    if (row_id >= num_merged) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    unsigned int out_row = row_id / out_w;
    unsigned int out_col = row_id % out_w;

    __nv_bfloat16* row_out = output + row_id * merged_hd;

    // Phase 1: gather patches and compute sum (for mean).
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < merged_hd; i += tg_size) {
        unsigned int sub_token = i / hd;
        unsigned int sub_elem = i % hd;
        unsigned int dy = sub_token / ms;
        unsigned int dx = sub_token % ms;
        unsigned int src_row = out_row * ms + dy;
        unsigned int src_col = out_col * ms + dx;
        unsigned int src_idx = (src_row * params.grid_w + src_col) * hd + sub_elem;
        float val = __bfloat162float(input[src_idx]);
        row_out[i] = __float2bfloat16(val);  // Temporary write for phase 2.
        local_sum += val;
    }

    // Warp + cross-warp reduction for mean.
    local_sum = warp_sum(local_sum);
    __shared__ float shared[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float v = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        v = warp_sum(v);
        if (lane_id == 0) shared[0] = v;
    }
    __syncthreads();
    float mean = shared[0] / (float)merged_hd;

    // Phase 2: compute variance.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < merged_hd; i += tg_size) {
        float d = __bfloat162float(row_out[i]) - mean;
        sum_sq += d * d;
    }

    sum_sq = warp_sum(sum_sq);
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int num_warps = (tg_size + 31) / 32;
        float v = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        v = warp_sum(v);
        if (lane_id == 0) shared[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(shared[0] / (float)merged_hd + params.eps);

    // Phase 3: normalise, scale, bias.
    for (unsigned int i = tid; i < merged_hd; i += tg_size) {
        float val = (__bfloat162float(row_out[i]) - mean) * scale;
        row_out[i] = __float2bfloat16(val * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}

// ---------------------------------------------------------------------------
// Scatter vision tokens into text embeddings.
//
// Serial scan over token_ids, parallel copy of matching rows.
// Single block of 256 threads.
//
// Dispatch: cfg_blocks(1, 256)
// ---------------------------------------------------------------------------

struct ScatterVisionParams {
    unsigned int image_token_id;
    unsigned int seq_len;
    unsigned int hidden_dim;
};

extern "C" __global__ void scatter_vision_tokens(
    const ScatterVisionParams params,
    __nv_bfloat16* __restrict__ text_embeds,
    const __nv_bfloat16* __restrict__ vision_embeds,
    const unsigned int* __restrict__ token_ids
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tg_size = blockDim.x;
    const unsigned int hd = params.hidden_dim;
    unsigned int vision_idx = 0;

    for (unsigned int pos = 0; pos < params.seq_len; pos++) {
        if (token_ids[pos] == params.image_token_id) {
            __nv_bfloat16* dst = text_embeds + pos * hd;
            const __nv_bfloat16* src = vision_embeds + vision_idx * hd;
            for (unsigned int i = tid; i < hd; i += tg_size) {
                dst[i] = src[i];
            }
            __syncthreads();
            vision_idx++;
        }
    }
}
