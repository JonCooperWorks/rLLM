// ---------------------------------------------------------------------------
// GpuVision — vision encoder utility kernels.
//
// Operations specific to the vision encoding pipeline that don't fit into
// the existing kernel families:
//
//   spatial_merge        — Qwen 3.5 VLM: rearrange 2D grid tokens into
//                          merged tokens by concatenating merge_size×merge_size
//                          spatial neighbours.
//   scatter_vision_tokens — overwrite placeholder embedding rows with vision
//                          encoder output at image_token positions.
//
// LEARNING NOTE: These two kernels bridge the gap between the vision encoder
// (which outputs a 2D grid of patch tokens) and the text transformer (which
// expects a 1D sequence of embeddings).
//
//   spatial_merge performs a 2D→1D rearrangement: it takes the vision encoder's
//   output grid (grid_h × grid_w tokens, each of `hidden_dim` dimensions) and
//   groups every merge_size×merge_size block of neighbouring tokens into a
//   single wider token by concatenation.  This reduces the token count by ms²
//   while increasing hidden_dim by ms² — a form of learned spatial downsampling
//   that cuts the sequence length the text transformer must attend over.
//
//   scatter_vision_tokens is the final stitching step: the tokenizer emits
//   placeholder `<image>` token IDs where vision tokens belong.  This kernel
//   does a serial scan over the token ID buffer to find those placeholders,
//   then copies vision embeddings into the corresponding rows in parallel.
//   The serial scan is necessary because vision tokens must be assigned in
//   order (the N-th placeholder gets the N-th vision token), but each row
//   copy is parallelised across threads for throughput.
//
// Metal shaders: shaders/vision.metal
// Metal impl:    gpu/metal/kernels/vision.rs
// Vision encoder forward pass: model/vision.rs
// Vision weight loading: model/loader/ (load_vision_weights)
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuVision: GpuCore {
    /// Spatial merge: rearrange tokens from a 2D grid so that every
    /// merge_size×merge_size block of adjacent tokens is concatenated
    /// into a single token.
    ///
    /// Input:  [grid_h * grid_w, hidden_dim]
    /// Output: [(grid_h/ms) * (grid_w/ms), hidden_dim * ms * ms]
    ///
    /// The rearrangement follows row-major order within each merge block.
    fn spatial_merge(
        &self,
        input: &Self::Tensor,
        output: &Self::Tensor,
        grid_h: u32,
        grid_w: u32,
        hidden_dim: u32,
        merge_size: u32,
    );

    /// Fused spatial merge + LayerNorm: rearrange 2D grid tokens AND normalise
    /// in one kernel dispatch, avoiding an intermediate global memory write.
    ///
    /// Each output token is computed by:
    ///   1. Gather merge_size×merge_size patches from the input grid
    ///   2. Concatenate into one wide vector
    ///   3. Apply LayerNorm (mean-center, scale by weight, add bias)
    ///   4. Write the normalised merged token to output
    fn spatial_merge_norm(
        &self,
        input: &Self::Tensor,
        output: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        grid_h: u32,
        grid_w: u32,
        hidden_dim: u32,
        merge_size: u32,
        eps: f32,
    );

    /// Scatter vision embeddings into a text embedding buffer.
    ///
    /// For each position `i` in `token_ids` where `token_ids[i] == image_token_id`,
    /// copy the next vision embedding row into `text_embeds[i]`.  Vision rows are
    /// consumed sequentially: the first match gets vision row 0, second gets row 1, etc.
    ///
    /// text_embeds:  [seq_len, hidden_dim] — modified in-place
    /// vision_embeds: [num_vision_tokens, hidden_dim]
    /// token_ids:    [seq_len] — u32 token IDs
    fn scatter_vision_tokens(
        &self,
        text_embeds: &Self::Tensor,
        vision_embeds: &Self::Tensor,
        token_ids: &Self::Tensor,
        image_token_id: u32,
        seq_len: u32,
        hidden_dim: u32,
    );
}
