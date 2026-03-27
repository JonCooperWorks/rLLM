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
// Metal shaders: shaders/elementwise.metal
// Metal impl:    gpu/metal/kernels/elementwise.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuElementwise: GpuCore {
    /// SwiGLU: out[i] = silu(gate[i]) * up[i].
    fn silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// GeGLU: out[i] = gelu(gate[i]) * up[i].
    fn gelu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Plain GELU: out[i] = gelu(input[i]).
    /// Used by vision encoder FFN (SigLIP ViT) which uses GELU without gating.
    fn gelu(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

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

    /// GPT-OSS gated activation: (clamp(up,-lim,lim) + 1) * clamp(gate,max=lim) * sigmoid(gate*alpha)
    ///
    /// NOT standard SwiGLU — uses scaled sigmoid, gate-only upper clamp, and (up + 1) offset.
    fn gpt_oss_gated_act(
        &self,
        gate: &Self::Tensor,
        up: &Self::Tensor,
        out: &Self::Tensor,
        size: u32,
        alpha: f32,
        limit: f32,
    );

    /// GPU-side top-k selection with softmax for MoE expert routing.
    fn top_k_softmax(&self, logits: &Self::Tensor, output: &Self::Tensor, num_experts: u32, k: u32);

    /// ReLU-squared activation: out[i] = max(0, in[i])².
    ///
    /// Learning note: ReLU-squared (also written ReLU²) is an activation used
    /// by Nemotron-H's MoE experts instead of SwiGLU.  Unlike SwiGLU which
    /// needs TWO projections (gate + up) and a multiply, relu² needs only ONE
    /// projection (up) making each expert simpler: up_proj → relu² → down_proj.
    ///
    /// The squaring amplifies larger activations relative to smaller ones,
    /// providing a softer "gating" effect compared to plain ReLU.  This was
    /// found to match SwiGLU quality with fewer parameters per expert.
    fn relu_squared(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// GPU-side top-k selection with sigmoid routing for Nemotron-H MoE.
    ///
    /// Learning note: standard MoE routing (Mixtral, Qwen) uses softmax over
    /// all experts to produce routing weights.  Nemotron-H instead uses
    /// DeepSeek-V3 style routing:
    ///
    ///   1. Compute sigmoid scores: score[i] = sigmoid(logits[i])
    ///   2. Add correction bias for SELECTION only: adj[i] = score[i] + bias[i]
    ///   3. Select top-k by adjusted score
    ///   4. Routing weights come from ORIGINAL sigmoid scores (without bias)
    ///   5. Optionally normalize weights to sum to 1
    ///   6. Multiply by scaling factor
    ///
    /// The correction bias steers which experts get selected (load balancing)
    /// without distorting the actual routing weights — a clever decoupling.
    ///
    /// Output format: [2*k] f32 — interleaved (expert_index, weight) pairs,
    /// same layout as `top_k_softmax` for downstream compatibility.
    fn top_k_sigmoid(
        &self,
        logits: &Self::Tensor,          // [num_experts] bf16 — raw router output
        correction_bias: &Self::Tensor, // [num_experts] f32 — load-balancing bias
        output: &Self::Tensor,          // [2*k] f32 — (index, weight) pairs
        num_experts: u32,
        k: u32,
        scaling_factor: f32,
        norm_topk_prob: bool,
    );
}
