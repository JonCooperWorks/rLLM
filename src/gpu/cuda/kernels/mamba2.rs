// ---------------------------------------------------------------------------
// CUDA impl: GpuMamba2 — Mamba-2 Selective State Space Model kernels.
//
// Trait contract: gpu/ops/mamba2.rs
// CUDA shader:    cuda/shaders/mamba2.cu (stub)
//
// Nemotron-H uses Mamba-2 (SSD) layers alongside standard attention layers.
// Two kernels are needed:
//
//   mamba2_conv1d_silu — depthwise conv1d + bias + SiLU (like DeltaNet's
//     conv1d but with an additive bias before activation)
//   mamba2_ssm_step   — selective SSM state update: discretize with dt,
//     update [head_dim, state_size] recurrent state, output + RMSNorm
//
// Related files:
//   Metal shader:     metal/shaders/mamba2.metal
//   Metal bridge:     metal/kernels/mamba2.rs
//   CPU reference:    cpu/mod.rs (full reference implementation)
//   Trait contract:   gpu/ops/mamba2.rs
// ---------------------------------------------------------------------------

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuMamba2;

impl GpuMamba2 for CudaBackend {
    fn mamba2_conv1d_silu(
        &self,
        _input: &CudaTensor,
        _history: &CudaTensor,
        _weight: &CudaTensor,
        _bias: &CudaTensor,
        _out: &CudaTensor,
        _dim: u32,
        _kernel_size: u32,
    ) {
        unimplemented!("Mamba2 conv1d+silu CUDA kernel not yet implemented")
    }

    fn mamba2_ssm_step(
        &self,
        _state: &CudaTensor,
        _x: &CudaTensor,
        _b: &CudaTensor,
        _c: &CudaTensor,
        _dt: &CudaTensor,
        _a_log: &CudaTensor,
        _d_skip: &CudaTensor,
        _dt_bias: &CudaTensor,
        _norm_weight: &CudaTensor,
        _out: &CudaTensor,
        _num_heads: u32,
        _head_dim: u32,
        _state_size: u32,
        _n_groups: u32,
        _eps: f32,
    ) {
        unimplemented!("Mamba2 SSM step CUDA kernel not yet implemented")
    }
}
