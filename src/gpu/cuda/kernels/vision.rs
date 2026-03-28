// ---------------------------------------------------------------------------
// CUDA impl: GpuVision — vision encoder utility kernels.
//
// Trait contract: gpu/ops/vision.rs
// CUDA shader:    cuda/shaders/vision.cu
//
// Three kernels:
//   spatial_merge          — rearrange 2D grid tokens
//   spatial_merge_norm     — fused merge + LayerNorm
//   scatter_vision_tokens  — overwrite text embeds at image placeholders
//
// Related files:
//   Metal shader:  metal/shaders/vision.metal
//   Metal bridge:  metal/kernels/vision.rs
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuVision;

#[repr(C)]
#[derive(Clone, Copy)]
struct SpatialMergeParams {
    grid_h: u32,
    grid_w: u32,
    hidden_dim: u32,
    merge_size: u32,
}
unsafe impl DeviceRepr for SpatialMergeParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct SpatialMergeNormParams {
    grid_h: u32,
    grid_w: u32,
    hidden_dim: u32,
    merge_size: u32,
    eps: f32,
}
unsafe impl DeviceRepr for SpatialMergeNormParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScatterVisionParams {
    image_token_id: u32,
    seq_len: u32,
    hidden_dim: u32,
}
unsafe impl DeviceRepr for ScatterVisionParams {}

impl GpuVision for CudaBackend {
    fn spatial_merge(
        &self,
        input: &CudaTensor,
        output: &CudaTensor,
        grid_h: u32,
        grid_w: u32,
        hidden_dim: u32,
        merge_size: u32,
    ) {
        let params = SpatialMergeParams {
            grid_h,
            grid_w,
            hidden_dim,
            merge_size,
        };
        let total = (grid_h / merge_size) * (grid_w / merge_size) * hidden_dim * merge_size * merge_size;
        let cfg = CudaBackend::cfg_1d(total, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_spatial_merge)
                .arg(&params)
                .arg(&input.buf)
                .arg(&output.buf)
                .launch(cfg)
        }
        .expect("spatial_merge launch failed");
    }

    fn spatial_merge_norm(
        &self,
        input: &CudaTensor,
        output: &CudaTensor,
        weight: &CudaTensor,
        bias: &CudaTensor,
        grid_h: u32,
        grid_w: u32,
        hidden_dim: u32,
        merge_size: u32,
        eps: f32,
    ) {
        let params = SpatialMergeNormParams {
            grid_h,
            grid_w,
            hidden_dim,
            merge_size,
            eps,
        };
        let num_merged = (grid_h / merge_size) * (grid_w / merge_size);
        let cfg = CudaBackend::cfg_blocks(num_merged, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_spatial_merge_norm)
                .arg(&params)
                .arg(&input.buf)
                .arg(&weight.buf)
                .arg(&bias.buf)
                .arg(&output.buf)
                .launch(cfg)
        }
        .expect("spatial_merge_norm launch failed");
    }

    fn scatter_vision_tokens(
        &self,
        text_embeds: &CudaTensor,
        vision_embeds: &CudaTensor,
        token_ids: &CudaTensor,
        image_token_id: u32,
        seq_len: u32,
        hidden_dim: u32,
    ) {
        let params = ScatterVisionParams {
            image_token_id,
            seq_len,
            hidden_dim,
        };
        let cfg = CudaBackend::cfg_blocks(1, 256);
        unsafe {
            self.stream
                .launch_builder(&self.fn_scatter_vision_tokens)
                .arg(&params)
                .arg(&text_embeds.buf)
                .arg(&vision_embeds.buf)
                .arg(&token_ids.buf)
                .launch(cfg)
        }
        .expect("scatter_vision_tokens launch failed");
    }
}
