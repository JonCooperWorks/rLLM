// ---------------------------------------------------------------------------
// GpuNorm — normalisation kernels.
//
// RMSNorm is used before every attention and FFN block in the transformer.
// Two variants: single-token decode and batched prefill.
//
// Metal shaders: shaders/rms_norm.metal
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
}
