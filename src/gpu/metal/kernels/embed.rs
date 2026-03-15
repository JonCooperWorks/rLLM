// ---------------------------------------------------------------------------
// Metal impl: GpuEmbed — embedding lookup kernels.
//
// Trait contract: gpu/ops/embed.rs
// Metal shader:   metal/shaders/embed.metal
//
// Copies rows from the embedding weight table to the output buffer.
// One thread per element (hidden_dim threads per token).  The batched
// variant handles prefill by processing batch_size tokens in one dispatch.
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuEmbed;

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    token_id: u32,
    hidden_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedBatchParams {
    batch_size: u32,
    hidden_dim: u32,
}

impl GpuEmbed for MetalBackend {
    fn embed_lookup(&self, table: &MetalTensor, token_id: u32, out: &MetalTensor, hidden_dim: u32) {
        let params = EmbedParams {
            token_id,
            hidden_dim,
        };
        self.dispatch_async(
            &self.pipeline_embed_lookup,
            &params,
            &[(&table.buffer, 1), (&out.buffer, 2)],
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(256.min(hidden_dim as u64), 1, 1),
        );
    }

    fn embed_lookup_batch(
        &self,
        table: &MetalTensor,
        token_ids: &MetalTensor,
        out: &MetalTensor,
        batch_size: u32,
        hidden_dim: u32,
    ) {
        let params = EmbedBatchParams {
            batch_size,
            hidden_dim,
        };
        let total = batch_size as u64 * hidden_dim as u64;
        self.dispatch_async(
            &self.pipeline_embed_lookup_batch,
            &params,
            &[(&table.buffer, 1), (&token_ids.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(total, 1, 1),
            MTLSize::new(256.min(total), 1, 1),
        );
    }
}
