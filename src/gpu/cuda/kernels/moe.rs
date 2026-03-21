// ---------------------------------------------------------------------------
// CUDA impl: GpuMoe — fused MoE kernels (stub).
//
// Trait contract: gpu/ops/moe.rs
//
// The fused kernels are currently Metal-only.  The CUDA backend falls back
// to calling the separate matmul + silu_mul + scale_add operations via the
// unfused path in `moe_expert_dispatch`.  These stubs exist to satisfy the
// trait bound; they should not be called directly.
// ---------------------------------------------------------------------------

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuMoe;

impl GpuMoe for CudaBackend {
    fn fused_gate_up_swiglu(
        &self,
        _w_gate: &CudaTensor,
        _w_up: &CudaTensor,
        _input: &CudaTensor,
        _output: &CudaTensor,
        _m: u32,
        _k: u32,
    ) {
        panic!(
            "CUDA fused_gate_up_swiglu not yet implemented; \
             use separate matmul + silu_mul calls via moe_expert_dispatch"
        );
    }

    fn moe_combine_residual(
        &self,
        _residual: &CudaTensor,
        _expert_outputs: &CudaTensor,
        _weights: &[f32],
        _output: &CudaTensor,
        _hidden_size: u32,
        _k: u32,
    ) {
        panic!(
            "CUDA moe_combine_residual not yet implemented; \
             use separate scale_add + add calls via moe_expert_dispatch"
        );
    }
}
