// ---------------------------------------------------------------------------
// GpuEmbed — embedding lookup kernels.
//
// Converts discrete token IDs into continuous vector representations by
// copying rows from the embedding table.  Single-token version for decode,
// batched version for prefill.
//
// Metal shaders: shaders/embed.metal
// Metal impl:    gpu/metal/kernels/embed.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuEmbed: GpuCore {
    /// Embedding lookup: copy row `token_id` from the embedding table to `out`.
    fn embed_lookup(&self, table: &Self::Tensor, token_id: u32, out: &Self::Tensor, hidden_dim: u32);

    /// Batched embedding lookup: N token IDs → [batch_size, hidden_dim].
    fn embed_lookup_batch(
        &self,
        table: &Self::Tensor,
        token_ids: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        hidden_dim: u32,
    );
}
