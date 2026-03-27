// ===========================================================================
// Mamba-2 CUDA kernels — Selective State Space Model for Nemotron-H.
//
// STUB: Not yet implemented.  See the Metal shader (metal/shaders/mamba2.metal)
// for the reference GPU implementation and cpu/mod.rs for pure Rust reference.
//
// Planned kernels:
//   mamba2_conv1d_silu — depthwise conv1d + bias + SiLU activation
//     One thread per channel.  Identical to DeltaNet's conv1d kernel except
//     an additive per-channel bias is applied before the SiLU activation.
//
//   mamba2_ssm_step — selective SSM state update + output + RMSNorm
//     One block per head.  Each head maintains a [head_dim, state_size]
//     recurrent state matrix in f32.  B and C vectors are shared across
//     groups of heads (like GQA sharing KV heads).
//
// Related files:
//   Trait contract:   gpu/ops/mamba2.rs
//   Metal shader:     metal/shaders/mamba2.metal
//   CUDA bridge:      cuda/kernels/mamba2.rs
//   CPU reference:    cpu/mod.rs
// ===========================================================================

#include <cuda_bf16.h>

// TODO: Implement mamba2_conv1d_silu kernel
// TODO: Implement mamba2_ssm_step kernel
