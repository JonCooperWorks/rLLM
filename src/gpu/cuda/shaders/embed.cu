// ===========================================================================
// Embedding lookup CUDA kernels.
//
// LEARNING OVERVIEW
//
// Port of the Metal embed.metal kernels to CUDA for NVIDIA GPUs.
//
// Converts discrete token IDs into continuous vectors by looking up rows
// in the embedding table.  Trivially parallel — one thread per element.
//
// Related files:
//   Metal shader:  metal/shaders/embed.metal
//   CUDA bridge:   cuda/kernels/embed.rs
//   Trait contract: gpu/ops/embed.rs
// ===========================================================================

#include <cuda_bf16.h>

struct EmbedParams {
    unsigned int token_id;
    unsigned int hidden_dim;
};

extern "C" __global__ void embed_lookup(
    const EmbedParams params,
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.hidden_dim) return;
    output[gid] = table[params.token_id * params.hidden_dim + gid];
}

// ===========================================================================
// Batched embedding lookup — N tokens in one dispatch.
// ===========================================================================

struct EmbedBatchParams {
    unsigned int batch_size;
    unsigned int hidden_dim;
};

extern "C" __global__ void embed_lookup_batch(
    const EmbedBatchParams params,
    const __nv_bfloat16* __restrict__ table,
    const unsigned int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ output
) {
    const unsigned int hidden_dim = params.hidden_dim;
    const unsigned int total = params.batch_size * hidden_dim;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    unsigned int batch = gid / hidden_dim;
    unsigned int d     = gid % hidden_dim;
    unsigned int token_id = token_ids[batch];

    output[batch * hidden_dim + d] = table[token_id * hidden_dim + d];
}
