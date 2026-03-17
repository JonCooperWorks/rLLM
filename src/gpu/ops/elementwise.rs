// ---------------------------------------------------------------------------
// GpuElementwise — point-wise and element-wise kernels.
//
// Grab-bag of ops that are one thread per element: activations (SiLU, GELU,
// sigmoid), arithmetic (add, mul, scale_add, scalar_mul, fill_zero),
// bias-add, and MoE routing (top_k_softmax).
//
// Most are trivially bandwidth-bound — the kernel cost is dominated by
// reading/writing the tensor, not by the arithmetic.
//
// Metal shaders: shaders/elementwise.metal, shaders/deltanet.metal
// Metal impl:    gpu/metal/kernels/elementwise.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuElementwise: GpuCore {
    /// SwiGLU: out[i] = silu(gate[i]) * up[i].
    fn silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// GeGLU: out[i] = gelu(gate[i]) * up[i].
    fn gelu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Scalar multiply: out[i] = scalar * input[i].
    fn scalar_mul(&self, input: &Self::Tensor, out: &Self::Tensor, scalar: f32, size: u32);

    /// Element-wise add: out[i] = a[i] + b[i].
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Scaled accumulate: dst[i] += scale * src[i].
    fn scale_add(&self, dst: &Self::Tensor, src: &Self::Tensor, scale: f32, size: u32);

    /// Fill tensor with zeros.
    fn fill_zero(&self, dst: &Self::Tensor, size: u32);

    /// Broadcast bias-add: out[i] = input[i] + bias[i % dim].
    fn bias_add_batch(
        &self,
        input: &Self::Tensor,
        bias: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        dim: u32,
    );

    /// Element-wise sigmoid (bf16→f32).
    fn sigmoid(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise sigmoid (bf16→bf16).
    fn sigmoid_bf16(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise SiLU on bf16 tensors.
    fn silu(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise multiply (bf16).
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Clamped SwiGLU: out[i] = clamp(silu(gate[i]) * up[i], -limit, limit).
    /// Used by GPT-OSS which applies swiglu_limit=7.0 to stabilize MoE training.
    fn silu_mul_clamp(
        &self,
        gate: &Self::Tensor,
        up: &Self::Tensor,
        out: &Self::Tensor,
        size: u32,
        limit: f32,
    );

    /// GPU-side top-k selection with softmax for MoE expert routing.
    fn top_k_softmax(
        &self,
        logits: &Self::Tensor,
        output: &Self::Tensor,
        num_experts: u32,
        k: u32,
    );
}
