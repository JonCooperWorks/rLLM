// ---------------------------------------------------------------------------
// Metal impl: GpuElementwise — point-wise and element-wise kernels.
//
// Trait contract: gpu/ops/elementwise.rs
// Metal shader:   metal/shaders/elementwise.metal
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

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuElementwise;

#[repr(C)]
#[derive(Clone, Copy)]
struct ElemParams {
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleAddParams {
    size: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScalarMulParams {
    size: u32,
    scalar: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BiasAddParams {
    total: u32,
    dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SigmoidParams {
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TopKParams {
    num_experts: u32,
    k: u32,
}

impl GpuElementwise for MetalBackend {
    fn silu_mul(&self, gate: &MetalTensor, up: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_silu_mul,
            &params,
            &[(&gate.buffer, 1), (&up.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn gelu_mul(&self, gate: &MetalTensor, up: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_gelu_mul,
            &params,
            &[(&gate.buffer, 1), (&up.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn scalar_mul(&self, input: &MetalTensor, out: &MetalTensor, scalar: f32, size: u32) {
        let params = ScalarMulParams { size, scalar };
        self.dispatch_async(
            &self.pipeline_scalar_mul,
            &params,
            &[(&input.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn add(&self, a: &MetalTensor, b: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_add,
            &params,
            &[(&a.buffer, 1), (&b.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn scale_add(&self, dst: &MetalTensor, src: &MetalTensor, scale: f32, size: u32) {
        let params = ScaleAddParams { size, scale };
        self.dispatch_async(
            &self.pipeline_scale_add,
            &params,
            &[(&dst.buffer, 1), (&src.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn fill_zero(&self, dst: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_fill_zero,
            &params,
            &[(&dst.buffer, 1)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn bias_add_batch(
        &self,
        input: &MetalTensor,
        bias: &MetalTensor,
        out: &MetalTensor,
        batch_size: u32,
        dim: u32,
    ) {
        let total = batch_size * dim;
        let params = BiasAddParams { total, dim };
        self.dispatch_async(
            &self.pipeline_bias_add,
            &params,
            &[(&input.buffer, 1), (&bias.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(total as u64, 1, 1),
            MTLSize::new(256.min(total as u64), 1, 1),
        );
    }

    fn sigmoid(&self, input: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = SigmoidParams { size };
        self.dispatch_async(
            &self.pipeline_sigmoid,
            &params,
            &[(&input.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn sigmoid_bf16(&self, input: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = SigmoidParams { size };
        self.dispatch_async(
            &self.pipeline_sigmoid_bf16,
            &params,
            &[(&input.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn silu(&self, input: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_silu_elem,
            &params,
            &[(&input.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn mul(&self, a: &MetalTensor, b: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_mul_elem,
            &params,
            &[(&a.buffer, 1), (&b.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    fn top_k_softmax(
        &self,
        logits: &MetalTensor,
        output: &MetalTensor,
        num_experts: u32,
        k: u32,
    ) {
        let params = TopKParams { num_experts, k };
        self.dispatch_async(
            &self.pipeline_top_k_softmax,
            &params,
            &[(&logits.buffer, 1), (&output.buffer, 2)],
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );
    }
}
