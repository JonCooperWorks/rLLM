// ---------------------------------------------------------------------------
// GpuNorm — normalisation kernels.
//
// RMSNorm is used before every attention and FFN block in the transformer.
// The three variants cover: single-token decode, batched prefill, and the
// weight-free version used in DeltaNet output paths.
//
// Metal shaders: shaders/rms_norm.metal, shaders/deltanet.metal
// Metal impl:    gpu/metal/kernels/norm.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuNorm: GpuCore {
    /// RMSNorm: out = weight * (input / sqrt(mean(input²) + eps))
    fn rms_norm(&self, input: &Self::Tensor, weight: &Self::Tensor, eps: f32, out: &Self::Tensor);

    /// Batched RMSNorm: normalise each row of [batch_size, hidden_dim] independently.
    fn rms_norm_batch(
        &self,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &Self::Tensor,
        batch_size: u32,
    );

    /// RMSNorm without learned weights: out = input / sqrt(mean(input^2) + eps).
    fn rms_norm_no_weight(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32, eps: f32);
}
