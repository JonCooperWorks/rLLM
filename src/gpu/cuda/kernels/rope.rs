// ---------------------------------------------------------------------------
// CUDA impl: GpuRope — Rotary Positional Embedding kernels.
//
// Trait contract: gpu/ops/rope.rs
// CUDA shader:    cuda/shaders/rope.cu
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

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
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
unsafe impl DeviceRepr for RopeParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeBatchParams {
    batch_size: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}
unsafe impl DeviceRepr for RopeBatchParams {}

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
unsafe impl DeviceRepr for RopePartialParams {}

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
unsafe impl DeviceRepr for RopeYarnParams {}

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
unsafe impl DeviceRepr for RopeYarnBatchParams {}

impl GpuRope for CudaBackend {
    fn rope(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
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
        let block = 256.min(total_pairs);
        let cfg = CudaBackend::cfg_1d(total_pairs, block);
        unsafe {
            self.stream.launch_builder(&self.fn_rope)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .launch(cfg)
        }.expect("rope launch failed");
    }

    fn rope_batch(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        positions: &CudaTensor,
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
        let total = batch_size * pairs_per_token;
        let block = 256.min(total);
        let cfg = CudaBackend::cfg_1d(total, block);
        unsafe {
            self.stream.launch_builder(&self.fn_rope_batch)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&positions.buf)
                .launch(cfg)
        }.expect("rope_batch launch failed");
    }

    fn rope_partial(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
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
        let block = 256.min(total_pairs);
        let cfg = CudaBackend::cfg_1d(total_pairs, block);
        unsafe {
            self.stream.launch_builder(&self.fn_rope_partial)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .launch(cfg)
        }.expect("rope_partial launch failed");
    }

    fn rope_yarn(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
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
        let block = 256.min(total_pairs);
        let cfg = CudaBackend::cfg_1d(total_pairs, block);
        unsafe {
            self.stream.launch_builder(&self.fn_rope_yarn)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .launch(cfg)
        }.expect("rope_yarn launch failed");
    }

    fn rope_yarn_batch(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        positions: &CudaTensor,
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
        let total = batch_size * pairs_per_token;
        let block = 256.min(total);
        let cfg = CudaBackend::cfg_1d(total, block);
        unsafe {
            self.stream.launch_builder(&self.fn_rope_yarn_batch)
                .arg(&params)
                .arg(&q.buf)
                .arg(&k.buf)
                .arg(&positions.buf)
                .launch(cfg)
        }.expect("rope_yarn_batch launch failed");
    }
}
