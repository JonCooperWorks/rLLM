// ---------------------------------------------------------------------------
// CUDA impl: GpuMoe — fused MoE kernels for expert streaming.
//
// These kernels run the expert FFN compute after weights arrive on GPU via
// the expert streaming pipeline (pread from NVMe → pinned host memory →
// async DMA to device via dedicated CUDA transfer stream).
//
// Fused gate+up+SwiGLU halves the number of dispatches per expert by
// computing both dot products and applying the activation in a single kernel.
// The combine+residual kernel replaces k scale_add + 1 add with one pass.
//
// Trait contract:    gpu/ops/moe.rs
// CUDA shader:       cuda/shaders/moe.cu
// Expert streamer:   model/expert_stream.rs
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::TensorDtype;
use crate::gpu::ops::GpuMoe;

// Must match the CUDA shader's FusedGateUpParams.
#[repr(C)]
#[derive(Clone, Copy)]
struct FusedGateUpParams {
    m: u32,
    k: u32,
}
unsafe impl DeviceRepr for FusedGateUpParams {}

// Must match the CUDA shader's MoeCombineParams.
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeCombineParams {
    hidden_size: u32,
    k: u32,
    weights: [f32; 32],
}
unsafe impl DeviceRepr for MoeCombineParams {}

impl GpuMoe for CudaBackend {
    fn fused_gate_up_swiglu(
        &self,
        w_gate: &CudaTensor,
        w_up: &CudaTensor,
        input: &CudaTensor,
        output: &CudaTensor,
        m: u32,
        k: u32,
    ) {
        let params = FusedGateUpParams { m, k };
        // Dispatch Q4 or bf16 variant based on weight dtype (same as matmul).
        let func = match w_gate.dtype {
            TensorDtype::Q4 => &self.fn_fused_gate_up_swiglu_q4,
            TensorDtype::Q8 => &self.fn_fused_gate_up_swiglu_q8,
            TensorDtype::FP8 => &self.fn_fused_gate_up_swiglu_fp8,
            _ => &self.fn_fused_gate_up_swiglu_bf16,
        };
        // M rows × 32 threads per row = M*32 total threads.
        let cfg = CudaBackend::cfg_1d(m * 32, 256);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&params)
                .arg(&w_gate.buf)
                .arg(&w_up.buf)
                .arg(&input.buf)
                .arg(&output.buf)
                .launch(cfg)
        }
        .expect("fused_gate_up_swiglu launch failed");
    }

    fn moe_combine_residual(
        &self,
        residual: &CudaTensor,
        expert_outputs: &CudaTensor,
        weights: &[f32],
        output: &CudaTensor,
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
        let cfg = CudaBackend::cfg_1d(hidden_size, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_moe_combine_residual)
                .arg(&params)
                .arg(&residual.buf)
                .arg(&expert_outputs.buf)
                .arg(&output.buf)
                .launch(cfg)
        }
        .expect("moe_combine_residual launch failed");
    }
}
