// ---------------------------------------------------------------------------
// Metal impl: GpuMoe — fused MoE kernels.
//
// Trait contract: gpu/ops/moe.rs
// Metal shader:   metal/shaders/moe.metal
//
// Fused gate+up+SwiGLU halves the number of dispatches per expert by
// computing both dot products and applying the activation in a single kernel.
// The combine+residual kernel replaces k scale_add + 1 add with one pass.
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::TensorDtype;
use crate::gpu::ops::GpuMoe;

// Must match the Metal shader's FusedGateUpParams.
#[repr(C)]
#[derive(Clone, Copy)]
struct FusedGateUpParams {
    m: u32,
    k: u32,
}

// Must match the Metal shader's MoeCombineParams.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)] // matches Metal shader struct; moe_combine_residual trait method not yet called
struct MoeCombineParams {
    hidden_size: u32,
    k: u32,
    weights: [f32; 32],
}

impl GpuMoe for MetalBackend {
    fn fused_gate_up_swiglu(
        &self,
        w_gate: &MetalTensor,
        w_up: &MetalTensor,
        input: &MetalTensor,
        output: &MetalTensor,
        m: u32,
        k: u32,
    ) {
        let params = FusedGateUpParams { m, k };
        // Dispatch Q4 or bf16 variant based on weight dtype (same as matmul).
        let pipeline = match w_gate.dtype {
            TensorDtype::Q4 => &self.pipeline_fused_gate_up_swiglu_q4,
            TensorDtype::Q8 => &self.pipeline_fused_gate_up_swiglu_q8,
            TensorDtype::FP8 => panic!("FP8 tensors not supported on Metal — use Q8 block format"),
            _ => &self.pipeline_fused_gate_up_swiglu,
        };
        // Same dispatch geometry as matvec: 32 threads per output row.
        let grid = MTLSize::new(m as u64 * 32, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        self.dispatch_async(
            pipeline,
            &params,
            &[
                (&w_gate.buffer, 1),
                (&w_up.buffer, 2),
                (&input.buffer, 3),
                (&output.buffer, 4),
            ],
            grid,
            tg,
        );
    }

    fn moe_combine_residual(
        &self,
        residual: &MetalTensor,
        expert_outputs: &MetalTensor,
        weights: &[f32],
        output: &MetalTensor,
        hidden_size: u32,
        k: u32,
    ) {
        let mut weight_arr = [0.0f32; 32];
        for (i, &w) in weights.iter().enumerate().take(32) {
            weight_arr[i] = w;
        }
        let params = MoeCombineParams {
            hidden_size,
            k,
            weights: weight_arr,
        };
        self.dispatch_async(
            &self.pipeline_moe_combine_residual,
            &params,
            &[
                (&residual.buffer, 1),
                (&expert_outputs.buffer, 2),
                (&output.buffer, 3),
            ],
            MTLSize::new(hidden_size as u64, 1, 1),
            MTLSize::new(256.min(hidden_size as u64), 1, 1),
        );
    }
}
