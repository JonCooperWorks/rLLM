// ---------------------------------------------------------------------------
// Metal impl: GpuRope — Rotary Positional Embedding kernels.
//
// Trait contract: gpu/ops/rope.rs
// Metal shader:   metal/shaders/rope.metal
//
// Each thread rotates one (cos, sin) pair of elements.  Thread count =
// (num_heads + num_kv_heads) * (head_dim / 2) — one thread per pair across
// both Q and K vectors.
//
// Three variants:
//   - rope: single-token decode, fixed position
//   - rope_batch: prefill with per-token position array
//   - rope_partial: Qwen 3.5 GQA layers where only the first `rotary_dim`
//     elements are rotated (the rest pass through unchanged)
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuRope;

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeBatchParams {
    batch_size: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopePartialParams {
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeYarnParams {
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_pos: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeYarnBatchParams {
    batch_size: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_pos: u32,
}

impl GpuRope for MetalBackend {
    fn rope(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = RopeParams {
            pos,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let total_pairs = (num_heads + num_kv_heads) * (head_dim / 2);
        self.dispatch_async(
            &self.pipeline_rope,
            &params,
            &[(&q.buffer, 1), (&k.buffer, 2)],
            MTLSize::new(total_pairs as u64, 1, 1),
            MTLSize::new(256.min(total_pairs as u64), 1, 1),
        );
    }

    fn rope_batch(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        positions: &MetalTensor,
        rope_theta: f32,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = RopeBatchParams {
            batch_size,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let pairs_per_token = (num_heads + num_kv_heads) * (head_dim / 2);
        let total = batch_size as u64 * pairs_per_token as u64;
        self.dispatch_async(
            &self.pipeline_rope_batch,
            &params,
            &[(&q.buffer, 1), (&k.buffer, 2), (&positions.buffer, 3)],
            MTLSize::new(total, 1, 1),
            MTLSize::new(256.min(total), 1, 1),
        );
    }

    fn rope_partial(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    ) {
        let params = RopePartialParams {
            pos,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
        };
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads + num_kv_heads) * half_rotary;
        self.dispatch_async(
            &self.pipeline_rope_partial,
            &params,
            &[(&q.buffer, 1), (&k.buffer, 2)],
            MTLSize::new(total_pairs as u64, 1, 1),
            MTLSize::new(256.min(total_pairs as u64), 1, 1),
        );
    }

    fn rope_yarn(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_pos: u32,
    ) {
        let params = RopeYarnParams {
            pos,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
            factor,
            beta_fast,
            beta_slow,
            original_max_pos,
        };
        let total_pairs = (num_heads + num_kv_heads) * (head_dim / 2);
        self.dispatch_async(
            &self.pipeline_rope_yarn,
            &params,
            &[(&q.buffer, 1), (&k.buffer, 2)],
            MTLSize::new(total_pairs as u64, 1, 1),
            MTLSize::new(256.min(total_pairs as u64), 1, 1),
        );
    }

    fn rope_yarn_batch(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        positions: &MetalTensor,
        rope_theta: f32,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_pos: u32,
    ) {
        let params = RopeYarnBatchParams {
            batch_size,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
            factor,
            beta_fast,
            beta_slow,
            original_max_pos,
        };
        let pairs_per_token = (num_heads + num_kv_heads) * (head_dim / 2);
        let total = batch_size as u64 * pairs_per_token as u64;
        self.dispatch_async(
            &self.pipeline_rope_yarn_batch,
            &params,
            &[(&q.buffer, 1), (&k.buffer, 2), (&positions.buffer, 3)],
            MTLSize::new(total, 1, 1),
            MTLSize::new(256.min(total), 1, 1),
        );
    }
}
