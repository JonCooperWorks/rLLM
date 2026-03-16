// ---------------------------------------------------------------------------
// CUDA impl: GpuEmbed — embedding lookup kernels.
//
// Trait contract: gpu/ops/embed.rs
// CUDA shader:    cuda/shaders/embed.cu
//
// Copies rows from the embedding weight table to the output buffer.
// One thread per element (hidden_dim threads per token).  The batched
// variant handles prefill by processing batch_size tokens in one dispatch.
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuEmbed;

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    token_id: u32,
    hidden_dim: u32,
}
unsafe impl DeviceRepr for EmbedParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedBatchParams {
    batch_size: u32,
    hidden_dim: u32,
}
unsafe impl DeviceRepr for EmbedBatchParams {}

impl GpuEmbed for CudaBackend {
    fn embed_lookup(&self, table: &CudaTensor, token_id: u32, out: &CudaTensor, hidden_dim: u32) {
        let params = EmbedParams {
            token_id,
            hidden_dim,
        };
        let block = 256.min(hidden_dim);
        let cfg = CudaBackend::cfg_1d(hidden_dim, block);
        unsafe {
            self.stream.launch_builder(&self.fn_embed_lookup)
                .arg(&params)
                .arg(&table.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("embed_lookup launch failed");
    }

    fn embed_lookup_batch(
        &self,
        table: &CudaTensor,
        token_ids: &CudaTensor,
        out: &CudaTensor,
        batch_size: u32,
        hidden_dim: u32,
    ) {
        let params = EmbedBatchParams {
            batch_size,
            hidden_dim,
        };
        let total = batch_size * hidden_dim;
        let block = 256.min(total);
        let cfg = CudaBackend::cfg_1d(total, block);
        unsafe {
            self.stream.launch_builder(&self.fn_embed_lookup_batch)
                .arg(&params)
                .arg(&table.buf)
                .arg(&token_ids.buf)
                .arg(&out.buf)
                .launch(cfg)
        }.expect("embed_lookup_batch launch failed");
    }
}
