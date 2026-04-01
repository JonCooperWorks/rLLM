// ---------------------------------------------------------------------------
// Metal dispatch for GpuTurboQuant — TurboQuant KV cache quantization kernels.
//
// Param structs are #[repr(C)] and must match the Metal shader structs in
// shaders/turboquant.metal byte-for-byte.
//
// Related files:
//   gpu/ops/turboquant.rs              — GpuTurboQuant trait
//   gpu/metal/shaders/turboquant.metal — Metal kernels
//   model/turboquant.rs                — Algorithm and codebook constants
// ---------------------------------------------------------------------------

use metal::MTLSize;

use crate::gpu::metal::MetalBackend;
use crate::gpu::metal::MetalTensor;
use crate::gpu::GpuTurboQuant;
use crate::model::kv_cache;

// ---------------------------------------------------------------------------
// Param structs — must match Metal shader structs byte-for-byte.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboQuantizeParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    is_plus: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboQuantizeBatchParams {
    batch_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    is_plus: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboRotateQParams {
    num_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboPagedAttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    window_size: u32,
    attn_scale: f32,
    has_sinks: u32,
    is_plus: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurboPagedAttentionVOnlyParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    bits: u32,
    kv_dim: u32,
    v_bytes_per_head_pos: u32,
    block_size: u32,
    num_centroids: u32,
    window_size: u32,
    attn_scale: f32,
    has_sinks: u32,
    is_plus: u32,
}

impl GpuTurboQuant for MetalBackend {
    fn turbo_quantize_to_paged(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        pi: &MetalTensor,
        centroids: &MetalTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        is_plus: bool,
    ) {
        let params = TurboQuantizeParams {
            pos,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: kv_cache::BLOCK_SIZE as u32,
            num_centroids: 1 << bits,
            is_plus: is_plus as u32,
        };

        // One threadgroup per KV head, head_dim threads per group.
        let threads_per_group = head_dim.max(32) as u64; // at least 32 for SIMD
        let num_threadgroups = num_kv_heads as u64;

        self.dispatch_async(
            &self.pipeline_turbo_quantize_paged,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
                (&pi.buffer, 4),
                (&centroids.buffer, 5),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn turbo_quantize_to_paged_batch(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        positions: &MetalTensor,
        pi: &MetalTensor,
        centroids: &MetalTensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        is_plus: bool,
    ) {
        let params = TurboQuantizeBatchParams {
            batch_size,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: kv_cache::BLOCK_SIZE as u32,
            num_centroids: 1 << bits,
            is_plus: is_plus as u32,
        };

        let threads_per_group = head_dim.max(32) as u64;
        let num_threadgroups = batch_size as u64 * num_kv_heads as u64;

        self.dispatch_async(
            &self.pipeline_turbo_quantize_paged_batch,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
                (&positions.buffer, 4),
                (&pi.buffer, 5),
                (&centroids.buffer, 6),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn turbo_rotate_q(
        &self,
        q: &MetalTensor,
        q_rot: &MetalTensor,
        pi: &MetalTensor,
        num_heads: u32,
        head_dim: u32,
    ) {
        let params = TurboRotateQParams {
            num_heads,
            head_dim,
        };

        let threads_per_group = head_dim.max(32) as u64;
        let num_threadgroups = num_heads as u64;

        self.dispatch_async(
            &self.pipeline_turbo_rotate_q,
            &params,
            &[
                (&q.buffer, 1),
                (&q_rot.buffer, 2),
                (&pi.buffer, 3),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn turbo_paged_attention(
        &self,
        q_rot: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_table: &MetalTensor,
        pi_t: &MetalTensor,
        centroids: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&MetalTensor>,
        is_plus: bool,
    ) {
        let params = TurboPagedAttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            bits,
            bytes_per_head_pos,
            block_size: kv_cache::BLOCK_SIZE as u32,
            num_centroids: 1 << bits,
            window_size,
            attn_scale,
            has_sinks: if sinks.is_some() { 1 } else { 0 },
            is_plus: is_plus as u32,
        };

        let sinks_buf = sinks.map(|s| &s.buffer).unwrap_or(&out.buffer);
        let threads_per_group: u64 = 256;
        let num_threadgroups = num_heads as u64;

        self.dispatch_async(
            &self.pipeline_turbo_paged_attention,
            &params,
            &[
                (&q_rot.buffer, 1),
                (&k_pool.buffer, 2),
                (&v_pool.buffer, 3),
                (&block_table.buffer, 4),
                (&pi_t.buffer, 5),
                (&centroids.buffer, 6),
                (&out.buffer, 7),
                (sinks_buf, 8),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    fn turbo_paged_attention_v_only(
        &self,
        q: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_table: &MetalTensor,
        pi_t: &MetalTensor,
        centroids: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        kv_dim: u32,
        v_bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&MetalTensor>,
        is_plus: bool,
    ) {
        let params = TurboPagedAttentionVOnlyParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            bits,
            kv_dim,
            v_bytes_per_head_pos,
            block_size: kv_cache::BLOCK_SIZE as u32,
            num_centroids: 1 << bits,
            window_size,
            attn_scale,
            has_sinks: if sinks.is_some() { 1 } else { 0 },
            is_plus: is_plus as u32,
        };

        let sinks_buf = sinks.map(|s| &s.buffer).unwrap_or(&out.buffer);
        let threads_per_group: u64 = 256;
        let num_threadgroups = num_heads as u64;

        self.dispatch_async(
            &self.pipeline_turbo_paged_attention_v_only,
            &params,
            &[
                (&q.buffer, 1),
                (&k_pool.buffer, 2),
                (&v_pool.buffer, 3),
                (&block_table.buffer, 4),
                (&pi_t.buffer, 5),
                (&centroids.buffer, 6),
                (&out.buffer, 7),
                (sinks_buf, 8),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }
}
