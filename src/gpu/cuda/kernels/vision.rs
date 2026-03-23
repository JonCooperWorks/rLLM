// ---------------------------------------------------------------------------
// CUDA stub: GpuVision — vision encoder utility kernels.
//
// Trait contract: gpu/ops/vision.rs
// ---------------------------------------------------------------------------

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuVision;

impl GpuVision for CudaBackend {
    fn spatial_merge(
        &self,
        _input: &CudaTensor,
        _output: &CudaTensor,
        _grid_h: u32,
        _grid_w: u32,
        _hidden_dim: u32,
        _merge_size: u32,
    ) {
        todo!("spatial_merge not yet implemented for CUDA backend")
    }

    fn scatter_vision_tokens(
        &self,
        _text_embeds: &CudaTensor,
        _vision_embeds: &CudaTensor,
        _token_ids: &CudaTensor,
        _image_token_id: u32,
        _seq_len: u32,
        _hidden_dim: u32,
    ) {
        todo!("scatter_vision_tokens not yet implemented for CUDA backend")
    }
}
