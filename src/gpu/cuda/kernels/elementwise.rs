// ---------------------------------------------------------------------------
// CUDA impl: GpuElementwise — point-wise and element-wise kernels.
//
// Trait contract: gpu/ops/elementwise.rs
// CUDA shader:    cuda/shaders/elementwise.cu
//
// These are all bandwidth-bound one-thread-per-element ops: activations
// (SiLU, GELU, sigmoid), arithmetic (add, mul, scale_add, scalar_mul),
// bias-add, fill-zero, and MoE routing (top_k_softmax).
//
// Most share the same ElemParams { size } struct.  The few that need extra
// constants (scale, dim, k) get their own param struct.
//
// top_k_softmax is the exception to the "one thread per element" pattern —
// it runs a single thread that does sequential selection over a small
// expert array (typically 8–64 experts).
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuElementwise;

#[repr(C)]
#[derive(Clone, Copy)]
struct ElemParams {
    size: u32,
}
unsafe impl DeviceRepr for ElemParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleAddParams {
    size: u32,
    scale: f32,
}
unsafe impl DeviceRepr for ScaleAddParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScalarMulParams {
    size: u32,
    scalar: f32,
}
unsafe impl DeviceRepr for ScalarMulParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct BiasAddParams {
    total: u32,
    dim: u32,
}
unsafe impl DeviceRepr for BiasAddParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct SigmoidParams {
    size: u32,
}
unsafe impl DeviceRepr for SigmoidParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct TopKParams {
    num_experts: u32,
    k: u32,
}
unsafe impl DeviceRepr for TopKParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct SiluMulClampParams {
    size: u32,
    limit: f32,
}
unsafe impl DeviceRepr for SiluMulClampParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct GptOssGatedActParams {
    size: u32,
    alpha: f32,
    limit: f32,
}
unsafe impl DeviceRepr for GptOssGatedActParams {}

impl GpuElementwise for CudaBackend {
    fn silu_mul(&self, gate: &CudaTensor, up: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_silu_mul)
                .arg(&params)
                .arg(&gate.buf)
                .arg(&up.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("silu_mul launch failed");
    }

    fn gelu_mul(&self, gate: &CudaTensor, up: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_gelu_mul)
                .arg(&params)
                .arg(&gate.buf)
                .arg(&up.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("gelu_mul launch failed");
    }

    fn gelu(&self, _input: &CudaTensor, _out: &CudaTensor, _size: u32) {
        todo!("gelu not yet implemented for CUDA backend")
    }

    fn scalar_mul(&self, input: &CudaTensor, out: &CudaTensor, scalar: f32, size: u32) {
        let params = ScalarMulParams { size, scalar };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_scalar_mul)
                .arg(&params)
                .arg(&input.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("scalar_mul launch failed");
    }

    fn add(&self, a: &CudaTensor, b: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_add)
                .arg(&params)
                .arg(&a.buf)
                .arg(&b.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("add launch failed");
    }

    fn scale_add(&self, dst: &CudaTensor, src: &CudaTensor, scale: f32, size: u32) {
        let params = ScaleAddParams { size, scale };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_scale_add)
                .arg(&params)
                .arg(&dst.buf)
                .arg(&src.buf)
                .launch(cfg)
        }
        .expect("scale_add launch failed");
    }

    fn fill_zero(&self, dst: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_fill_zero)
                .arg(&params)
                .arg(&dst.buf)
                .launch(cfg)
        }
        .expect("fill_zero launch failed");
    }

    fn bias_add_batch(
        &self,
        input: &CudaTensor,
        bias: &CudaTensor,
        out: &CudaTensor,
        batch_size: u32,
        dim: u32,
    ) {
        let total = batch_size * dim;
        let params = BiasAddParams { total, dim };
        let block = 256.min(total);
        let cfg = CudaBackend::cfg_1d(total, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_bias_add)
                .arg(&params)
                .arg(&input.buf)
                .arg(&bias.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("bias_add launch failed");
    }

    fn sigmoid(&self, input: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = SigmoidParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_sigmoid)
                .arg(&params)
                .arg(&input.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("sigmoid launch failed");
    }

    fn sigmoid_bf16(&self, input: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = SigmoidParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_sigmoid_bf16)
                .arg(&params)
                .arg(&input.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("sigmoid_bf16 launch failed");
    }

    fn silu(&self, input: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_silu_elem)
                .arg(&params)
                .arg(&input.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("silu launch failed");
    }

    fn silu_mul_clamp(
        &self,
        gate: &CudaTensor,
        up: &CudaTensor,
        out: &CudaTensor,
        size: u32,
        limit: f32,
    ) {
        let params = SiluMulClampParams { size, limit };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_silu_mul_clamp)
                .arg(&params)
                .arg(&gate.buf)
                .arg(&up.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("silu_mul_clamp launch failed");
    }

    fn mul(&self, a: &CudaTensor, b: &CudaTensor, out: &CudaTensor, size: u32) {
        let params = ElemParams { size };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_mul_elem)
                .arg(&params)
                .arg(&a.buf)
                .arg(&b.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("mul launch failed");
    }

    fn gpt_oss_gated_act(
        &self,
        gate: &CudaTensor,
        up: &CudaTensor,
        out: &CudaTensor,
        size: u32,
        alpha: f32,
        limit: f32,
    ) {
        let params = GptOssGatedActParams { size, alpha, limit };
        let block = 256.min(size);
        let cfg = CudaBackend::cfg_1d(size, block);
        unsafe {
            self.stream
                .launch_builder(&self.fn_gpt_oss_gated_act)
                .arg(&params)
                .arg(&gate.buf)
                .arg(&up.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("gpt_oss_gated_act launch failed");
    }

    fn top_k_softmax(&self, logits: &CudaTensor, output: &CudaTensor, num_experts: u32, k: u32) {
        let params = TopKParams { num_experts, k };
        // Single thread — sequential top-k selection.
        let cfg = CudaBackend::cfg_blocks(1, 1);
        unsafe {
            self.stream
                .launch_builder(&self.fn_top_k_softmax)
                .arg(&params)
                .arg(&logits.buf)
                .arg(&output.buf)
                .launch(cfg)
        }
        .expect("top_k_softmax launch failed");
    }
}
