// ---------------------------------------------------------------------------
// GpuNorm — normalisation kernels.
//
// RMSNorm is used before every attention and FFN block in the transformer.
// LayerNorm is used by vision encoders (SigLIP ViT) which need both
// mean-centering and learned bias.
//
// LEARNING NOTE: Why two normalisation variants?
//
// Modern LLMs (Llama, Qwen, etc.) all use RMSNorm because it's cheaper
// than LayerNorm — it skips the mean subtraction and bias addition, saving
// two passes over the data.  This works fine for text transformers trained
// from scratch with RMSNorm.
//
// But SigLIP vision encoders (used by both Qwen 3.5 VL and Gemma 3) were
// pre-trained with full LayerNorm (mean-centering + learned bias).  We
// can't swap to RMSNorm without retraining the vision encoder, so we need
// both variants.  The `layer_norm_batch` method exists specifically for the
// vision encoding pipeline.
//
// Metal shaders: shaders/rms_norm.metal (layer_norm_batch kernel is here too)
// Metal impl:    gpu/metal/kernels/norm.rs
// Vision encoder forward pass: model/vision.rs
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

    /// Batched LayerNorm: out = weight * ((input - mean) / sqrt(var + eps)) + bias.
    ///
    /// Full LayerNorm with mean-centering and learned bias, used by vision
    /// encoders (SigLIP ViT).  Each row of [batch_size, hidden_dim] is
    /// normalised independently.
    ///
    /// LEARNING NOTE: Unlike RMSNorm (which only divides by the root-mean-square),
    /// LayerNorm first subtracts the mean, making it invariant to both scale AND
    /// shift of the input distribution.  The learned bias parameter re-introduces
    /// a shift after normalisation.  Vision encoders rely on this because image
    /// patch embeddings have non-zero means that carry spatial information.
    fn layer_norm_batch(
        &self,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        eps: f32,
        out: &Self::Tensor,
        batch_size: u32,
    );
}
