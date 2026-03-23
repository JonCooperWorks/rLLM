// ---------------------------------------------------------------------------
// Metal impl: GpuVision — vision encoder utility kernels.
//
// Trait contract: gpu/ops/vision.rs
// Metal shader:   metal/shaders/vision.metal
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuVision;

#[repr(C)]
#[derive(Clone, Copy)]
struct SpatialMergeParams {
    grid_h: u32,
    grid_w: u32,
    hidden_dim: u32,
    merge_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScatterVisionParams {
    image_token_id: u32,
    seq_len: u32,
    hidden_dim: u32,
}

impl GpuVision for MetalBackend {
    fn spatial_merge(
        &self,
        input: &MetalTensor,
        output: &MetalTensor,
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
        let out_tokens = (grid_h / merge_size) * (grid_w / merge_size);
        let merged_hd = hidden_dim * merge_size * merge_size;
        let total_elements = out_tokens as u64 * merged_hd as u64;
        let threads_per_group: u64 = 256;
        let num_groups = (total_elements + threads_per_group - 1) / threads_per_group;
        self.dispatch_async(
            &self.pipeline_spatial_merge,
            &params,
            &[(&input.buffer, 1), (&output.buffer, 2)],
            MTLSize::new(num_groups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn scatter_vision_tokens(
        &self,
        text_embeds: &MetalTensor,
        vision_embeds: &MetalTensor,
        token_ids: &MetalTensor,
        image_token_id: u32,
        seq_len: u32,
        hidden_dim: u32,
    ) {
        let params = ScatterVisionParams {
            image_token_id,
            seq_len,
            hidden_dim,
        };
        // Single threadgroup of 256 threads — serial scan, parallel copy.
        self.dispatch_async(
            &self.pipeline_scatter_vision_tokens,
            &params,
            &[
                (&text_embeds.buffer, 1),
                (&vision_embeds.buffer, 2),
                (&token_ids.buffer, 3),
            ],
            MTLSize::new(256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
