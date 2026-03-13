// ===========================================================================
// Metal GPU backend for LLM inference.
//
// LEARNING OVERVIEW — read this first if you're new to GPU programming.
//
// What this file does:
//   Implements the `GpuBackend` trait for Apple's Metal API.  All the
//   tensor operations needed by the Llama transformer (matmul, attention,
//   normalization, etc.) are dispatched as Metal compute kernels.
//
// Key concepts for understanding this backend:
//
//   1. KERNEL COMPILATION
//      Metal shader sources (.metal files) are embedded in the binary at
//      compile time via `include_str!()`.  At runtime, `new_library_with_source`
//      compiles them to GPU machine code.  This adds ~200ms to startup but
//      avoids shipping precompiled metallib bundles.  Each compiled function
//      becomes a `ComputePipelineState` — Metal's handle for a launchable
//      GPU kernel.
//
//   2. DISPATCH MODEL
//      Metal's `dispatch_threads(grid_size, threadgroup_size)` launches a
//      grid of threads.  The grid is divided into threadgroups of up to
//      `threadgroup_size` threads each.  Within a threadgroup, threads can
//      share data via `threadgroup` memory and synchronise with barriers.
//      Across threadgroups, threads are fully independent.
//
//      IMPORTANT: `grid_size` specifies the TOTAL number of threads, NOT
//      the number of threadgroups.  Metal automatically computes:
//        num_threadgroups = ceil(grid_size / threadgroup_size)
//      To get N threadgroups of M threads: grid_size = N * M.
//
//   3. MEMORY MODEL (UNIFIED MEMORY)
//      On Apple Silicon, CPU and GPU share the same physical memory.  All
//      buffers use `StorageModeShared`, meaning both CPU and GPU can read
//      and write the same memory — no explicit host↔device copies needed.
//      CPU writes are visible to the GPU after command buffer commit; GPU
//      writes are visible to the CPU after `wait_until_completed()`.
//
//   4. ASYNC DISPATCH
//      Kernel dispatches are committed to the GPU command queue WITHOUT
//      waiting for completion.  Metal's serial command queue guarantees
//      that committed command buffers execute in order.  The CPU only
//      blocks when it needs results (in `copy_to_host` → `flush`).
//
//      This eliminates ~144 unnecessary GPU→CPU sync points per token.
//      The `flush()` method submits a fence command buffer and waits for
//      it — since the queue is serial, this guarantees all prior work is
//      complete.
//
//   5. BUFFER BINDING
//      Each kernel declares its buffers with `[[buffer(N)]]` in the Metal
//      source.  The host (Rust) side must bind buffers to matching indices
//      via `encoder.set_buffer(N, ...)`.  The params struct always goes in
//      slot 0; tensor data goes in slots 1, 2, 3, etc.
//
//   6. PARAMS STRUCTS
//      Small constant data (dimensions, positions, epsilon) is passed to
//      kernels via a `#[repr(C)]` struct that is copied into a Metal buffer.
//      The Rust struct layout must match the Metal struct EXACTLY —
//      field order, sizes, and alignment.  `#[repr(C)]` guarantees this.
// ===========================================================================

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::anyhow;
use metal::{CompileOptions, MTLResourceOptions, MTLSize};

use super::{GpuBackend, TensorDtype};

// ---------------------------------------------------------------------------
// Embedded shader sources.
// ---------------------------------------------------------------------------

const METAL_SOURCE_RMS_NORM: &str = include_str!("rms_norm.metal");
const METAL_SOURCE_MATMUL: &str = include_str!("matmul.metal");
const METAL_SOURCE_ROPE: &str = include_str!("rope.metal");
const METAL_SOURCE_ATTENTION: &str = include_str!("attention.metal");
const METAL_SOURCE_ELEMENTWISE: &str = include_str!("elementwise.metal");
const METAL_SOURCE_EMBED: &str = include_str!("embed.metal");

// ---------------------------------------------------------------------------
// MetalTensor — the backend's opaque tensor type.
//
// Wraps a Metal buffer plus shape and dtype metadata.  The buffer is
// allocated with `StorageModeShared` (unified memory), so the CPU can
// read/write it directly via `buffer.contents()` and the GPU accesses
// the same physical memory.
// ---------------------------------------------------------------------------

pub(crate) struct MetalTensor {
    pub buffer: metal::Buffer,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

impl MetalTensor {
    pub fn byte_count(&self) -> usize {
        match self.dtype {
            TensorDtype::Q4 => {
                assert!(self.shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                super::q4_byte_count(self.shape[0], self.shape[1])
            }
            _ => self.shape.iter().product::<usize>() * self.dtype.byte_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// MetalBackend — holds the Metal device, command queue, and all compiled
// kernel pipeline states.
//
// The `has_pending_work` flag tracks whether any kernel dispatches have
// been committed but not yet waited on.  `flush()` waits for all pending
// GPU work to complete — it's called automatically by `copy_to_host()`.
// ---------------------------------------------------------------------------

pub(crate) struct MetalBackend {
    device: metal::Device,
    queue: metal::CommandQueue,
    name: String,

    // Compiled kernel pipelines — one per kernel entry point.
    pipeline_rms_norm: metal::ComputePipelineState,
    pipeline_matvec: metal::ComputePipelineState,
    pipeline_matvec_q4: metal::ComputePipelineState,
    pipeline_rope: metal::ComputePipelineState,
    #[allow(dead_code)]
    pipeline_attention: metal::ComputePipelineState,
    pipeline_silu_mul: metal::ComputePipelineState,
    pipeline_add: metal::ComputePipelineState,
    pipeline_bias_add: metal::ComputePipelineState,
    pipeline_embed_lookup: metal::ComputePipelineState,
    #[allow(dead_code)]
    pipeline_copy_kv: metal::ComputePipelineState,
    pipeline_paged_copy_kv: metal::ComputePipelineState,
    pipeline_paged_attention: metal::ComputePipelineState,

    // Phase 3: batched prefill pipelines.
    pipeline_gemm_bf16: metal::ComputePipelineState,
    pipeline_gemm_q4: metal::ComputePipelineState,
    pipeline_rms_norm_batch: metal::ComputePipelineState,
    pipeline_embed_lookup_batch: metal::ComputePipelineState,
    pipeline_rope_batch: metal::ComputePipelineState,
    pipeline_paged_copy_kv_batch: metal::ComputePipelineState,
    pipeline_prefill_attention: metal::ComputePipelineState,

    // Async dispatch tracking.
    // Set to true when any command buffer is committed without waiting.
    // Cleared by `flush()`.
    has_pending_work: AtomicBool,
}

impl MetalBackend {
    pub fn new() -> anyhow::Result<Self> {
        let device =
            metal::Device::system_default().ok_or_else(|| anyhow!("no Metal device found"))?;
        let queue = device.new_command_queue();
        let name = device.name().to_string();

        let compile_opts = CompileOptions::new();

        let pipeline_rms_norm =
            Self::make_pipeline(&device, METAL_SOURCE_RMS_NORM, "rms_norm", &compile_opts)?;
        let pipeline_matvec =
            Self::make_pipeline(&device, METAL_SOURCE_MATMUL, "matvec_bf16", &compile_opts)?;
        let pipeline_matvec_q4 =
            Self::make_pipeline(&device, METAL_SOURCE_MATMUL, "matvec_q4", &compile_opts)?;
        let pipeline_rope = Self::make_pipeline(
            &device,
            METAL_SOURCE_ROPE,
            "rotary_embedding",
            &compile_opts,
        )?;
        let pipeline_attention =
            Self::make_pipeline(&device, METAL_SOURCE_ATTENTION, "attention", &compile_opts)?;
        let pipeline_silu_mul =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "silu_mul", &compile_opts)?;
        let pipeline_add = Self::make_pipeline(
            &device,
            METAL_SOURCE_ELEMENTWISE,
            "add_tensors",
            &compile_opts,
        )?;
        let pipeline_bias_add =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "bias_add", &compile_opts)?;
        let pipeline_embed_lookup =
            Self::make_pipeline(&device, METAL_SOURCE_EMBED, "embed_lookup", &compile_opts)?;
        let pipeline_copy_kv = Self::make_pipeline(
            &device,
            METAL_SOURCE_ATTENTION,
            "copy_to_kv_cache",
            &compile_opts,
        )?;
        let pipeline_paged_copy_kv = Self::make_pipeline(
            &device,
            METAL_SOURCE_ATTENTION,
            "copy_to_paged_kv_cache",
            &compile_opts,
        )?;
        let pipeline_paged_attention = Self::make_pipeline(
            &device,
            METAL_SOURCE_ATTENTION,
            "paged_attention",
            &compile_opts,
        )?;

        // Phase 3: batched prefill pipelines.
        let pipeline_gemm_bf16 =
            Self::make_pipeline(&device, METAL_SOURCE_MATMUL, "gemm_bf16", &compile_opts)?;
        let pipeline_gemm_q4 =
            Self::make_pipeline(&device, METAL_SOURCE_MATMUL, "gemm_q4", &compile_opts)?;
        let pipeline_rms_norm_batch = Self::make_pipeline(
            &device,
            METAL_SOURCE_RMS_NORM,
            "rms_norm_batch",
            &compile_opts,
        )?;
        let pipeline_embed_lookup_batch = Self::make_pipeline(
            &device,
            METAL_SOURCE_EMBED,
            "embed_lookup_batch",
            &compile_opts,
        )?;
        let pipeline_rope_batch = Self::make_pipeline(
            &device,
            METAL_SOURCE_ROPE,
            "rotary_embedding_batch",
            &compile_opts,
        )?;
        let pipeline_paged_copy_kv_batch = Self::make_pipeline(
            &device,
            METAL_SOURCE_ATTENTION,
            "copy_to_paged_kv_cache_batch",
            &compile_opts,
        )?;
        let pipeline_prefill_attention = Self::make_pipeline(
            &device,
            METAL_SOURCE_ATTENTION,
            "prefill_attention",
            &compile_opts,
        )?;

        Ok(Self {
            device,
            queue,
            name,
            pipeline_rms_norm,
            pipeline_matvec,
            pipeline_matvec_q4,
            pipeline_rope,
            pipeline_attention,
            pipeline_silu_mul,
            pipeline_add,
            pipeline_bias_add,
            pipeline_embed_lookup,
            pipeline_copy_kv,
            pipeline_paged_copy_kv,
            pipeline_paged_attention,
            pipeline_gemm_bf16,
            pipeline_gemm_q4,
            pipeline_rms_norm_batch,
            pipeline_embed_lookup_batch,
            pipeline_rope_batch,
            pipeline_paged_copy_kv_batch,
            pipeline_prefill_attention,
            has_pending_work: AtomicBool::new(false),
        })
    }

    /// Compile a Metal source string, extract a named function, and create
    /// a compute pipeline state (ready-to-dispatch kernel handle).
    fn make_pipeline(
        device: &metal::Device,
        source: &str,
        function_name: &str,
        opts: &CompileOptions,
    ) -> anyhow::Result<metal::ComputePipelineState> {
        let library = device
            .new_library_with_source(source, opts)
            .map_err(|e| anyhow!("failed to compile Metal kernel source: {e}"))?;
        let function = library
            .get_function(function_name, None)
            .map_err(|e| anyhow!("function '{function_name}' not found: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("failed to create pipeline for '{function_name}': {e}"))
    }

    // =======================================================================
    // ASYNC DISPATCH
    //
    // Instead of the old synchronous pattern (encode → commit → wait per
    // kernel), we now commit without waiting.  Metal's serial command queue
    // guarantees execution order, so we only need to wait when the CPU
    // needs to read GPU results.
    //
    // `dispatch_async`:   encode + commit, NO wait.
    // `flush`:            wait for all pending GPU work.
    // `copy_to_host`:     calls flush automatically.
    //
    // Learning note: why not batch into a single command buffer?
    //   That would require storing the command buffer and encoder across
    //   method calls (complex lifetime management with Objective-C objects).
    //   The per-kernel-commit approach is simpler and almost as fast — the
    //   ~10μs overhead per command buffer creation is negligible compared
    //   to eliminating ~144 blocking wait cycles.
    //
    // Safety of param buffer lifetime:
    //   `new_command_buffer()` creates a command buffer with RETAINED
    //   references (the default).  After `commit()`, Metal retains all
    //   buffers set on the encoder.  So when our local `param_buf` drops,
    //   the retain count goes from 2→1; when the GPU finishes, Metal
    //   releases its reference (1→0) and the buffer is deallocated.
    // =======================================================================

    /// Encode and commit a kernel dispatch WITHOUT waiting for completion.
    fn dispatch_async<T: Copy>(
        &self,
        pipeline: &metal::ComputePipelineState,
        params: &T,
        buffers: &[(&metal::Buffer, u64)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        let param_bytes = unsafe {
            std::slice::from_raw_parts(params as *const T as *const u8, std::mem::size_of::<T>())
        };
        let param_buf = self.device.new_buffer_with_data(
            param_bytes.as_ptr() as *const _,
            param_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(0, Some(&param_buf), 0);

        for &(buf, idx) in buffers {
            encoder.set_buffer(idx, Some(buf), 0);
        }
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd.commit();
        // No wait! The GPU processes this while we encode the next kernel.

        self.has_pending_work.store(true, Ordering::Release);
    }

    /// Wait for all pending GPU work to complete.
    ///
    /// Submits a fence command buffer and waits for it.  Since Metal's
    /// serial command queue processes buffers in commit order, waiting
    /// for this buffer guarantees all prior buffers have completed.
    fn flush(&self) {
        if self.has_pending_work.swap(false, Ordering::AcqRel) {
            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}

// ---------------------------------------------------------------------------
// Host → GPU parameter blocks.
//
// These structs are copied byte-for-byte into Metal buffers and read
// directly by GPU kernels.  `#[repr(C)]` is REQUIRED to guarantee that
// the Rust compiler lays out fields in declaration order with C alignment
// rules — matching the Metal struct definitions in the .metal files.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormParams {
    hidden_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatvecParams {
    m: u32,
    k: u32,
}

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
#[allow(dead_code)]
struct AttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ElemParams {
    size: u32,
}

/// Params for broadcast bias-add kernel. Must match Metal `BiasAddParams`.
#[repr(C)]
#[derive(Clone, Copy)]
struct BiasAddParams {
    total: u32, // batch_size * dim
    dim: u32,   // bias vector length
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    token_id: u32,
    hidden_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct CopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedCopyKvParams {
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PagedAttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}

// Phase 3: batched prefill param structs.

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    batch_size: u32,
    m: u32,
    k: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormBatchParams {
    hidden_size: u32,
    eps: f32,
    batch_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedBatchParams {
    batch_size: u32,
    hidden_dim: u32,
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
struct PagedCopyKvBatchParams {
    batch_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PrefillAttentionParams {
    chunk_size: u32,
    start_pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

// ---------------------------------------------------------------------------
// GpuBackend trait implementation.
//
// All kernel dispatches use `dispatch_async` — commit without waiting.
// Only `copy_to_host` calls `flush()` to ensure GPU results are ready.
// ---------------------------------------------------------------------------

impl GpuBackend for MetalBackend {
    type Tensor = MetalTensor;

    fn device_name(&self) -> &str {
        &self.name
    }

    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> MetalTensor {
        let byte_count = match dtype {
            TensorDtype::Q4 => {
                assert!(shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                super::q4_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        let buffer = self
            .device
            .new_buffer(byte_count as u64, MTLResourceOptions::StorageModeShared);
        MetalTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> MetalTensor {
        let expected = match dtype {
            TensorDtype::Q4 => {
                assert!(shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                super::q4_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        assert_eq!(
            data.len(),
            expected,
            "upload_tensor: data length {} != expected {}",
            data.len(),
            expected
        );
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn copy_to_tensor(&self, tensor: &MetalTensor, src: &[u8]) {
        let byte_count = tensor.byte_count();
        assert!(
            src.len() <= byte_count,
            "copy_to_tensor: src too large ({} > {})",
            src.len(),
            byte_count
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                tensor.buffer.contents() as *mut u8,
                src.len(),
            );
        }
    }

    fn tensor_byte_count(&self, tensor: &MetalTensor) -> usize {
        tensor.byte_count()
    }

    /// Copy GPU tensor data to a host byte buffer.
    ///
    /// Calls `flush()` first to ensure all pending GPU writes are complete.
    /// On Apple Silicon unified memory, the actual "copy" is just a memcpy
    /// from the shared buffer — the data is already in the same physical memory.
    fn copy_to_host(&self, tensor: &MetalTensor, dst: &mut [u8]) {
        // Ensure all pending GPU work is done before reading.
        self.flush();

        let byte_count = tensor.byte_count();
        assert!(
            dst.len() >= byte_count,
            "copy_to_host: dst too small ({} < {})",
            dst.len(),
            byte_count
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor.buffer.contents() as *const u8,
                dst.as_mut_ptr(),
                byte_count,
            );
        }
    }

    /// RMSNorm: dispatched as a single threadgroup of 256 threads.
    fn rms_norm(&self, input: &MetalTensor, weight: &MetalTensor, eps: f32, out: &MetalTensor) {
        let hidden_size = input.shape[0] as u32;
        let params = RmsNormParams { hidden_size, eps };
        self.dispatch_async(
            &self.pipeline_rms_norm,
            &params,
            &[(&input.buffer, 1), (&weight.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Matrix-vector multiply: SIMD-cooperative, 32 threads per output row.
    /// Auto-detects Q4 vs BF16 weights and dispatches the right kernel.
    fn matmul(&self, weight: &MetalTensor, input: &MetalTensor, out: &MetalTensor, m: u32, k: u32) {
        let params = MatvecParams { m, k };
        let pipeline = match weight.dtype {
            TensorDtype::Q4 => &self.pipeline_matvec_q4,
            _ => &self.pipeline_matvec,
        };
        self.dispatch_async(
            pipeline,
            &params,
            &[(&weight.buffer, 1), (&input.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(m as u64 * 32, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// RoPE: one thread per (head, pair) across both Q and K tensors.
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

    /// Attention: one threadgroup of 256 threads per query head.
    fn attention(
        &self,
        q: &MetalTensor,
        k_cache: &MetalTensor,
        v_cache: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = AttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k_cache.buffer, 2),
                (&v_cache.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    /// SwiGLU activation: one thread per element.
    fn silu_mul(&self, gate: &MetalTensor, up: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_silu_mul,
            &params,
            &[(&gate.buffer, 1), (&up.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    /// Element-wise add: one thread per element.
    fn add(&self, a: &MetalTensor, b: &MetalTensor, out: &MetalTensor, size: u32) {
        let params = ElemParams { size };
        self.dispatch_async(
            &self.pipeline_add,
            &params,
            &[(&a.buffer, 1), (&b.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    /// Broadcast bias-add: out[i] = input[i] + bias[i % dim].
    /// Adds a [dim] bias to each row of [batch_size, dim].
    fn bias_add_batch(
        &self,
        input: &MetalTensor,
        bias: &MetalTensor,
        out: &MetalTensor,
        batch_size: u32,
        dim: u32,
    ) {
        let total = batch_size * dim;
        let params = BiasAddParams { total, dim };
        self.dispatch_async(
            &self.pipeline_bias_add,
            &params,
            &[(&input.buffer, 1), (&bias.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(total as u64, 1, 1),
            MTLSize::new(256.min(total as u64), 1, 1),
        );
    }

    /// Embedding lookup: copy one row from the embedding table.
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

    /// Copy a new K or V vector into the flat KV cache at position `pos`.
    fn copy_to_kv_cache(
        &self,
        src: &MetalTensor,
        cache: &MetalTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = CopyKvParams {
            pos,
            num_kv_heads,
            head_dim,
        };
        let size = num_kv_heads * head_dim;
        self.dispatch_async(
            &self.pipeline_copy_kv,
            &params,
            &[(&src.buffer, 1), (&cache.buffer, 2)],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    /// Copy a new K or V vector into a paged KV cache pool.
    fn copy_to_paged_kv_cache(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PagedCopyKvParams {
            pos,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
        };
        let size = num_kv_heads * head_dim;
        self.dispatch_async(
            &self.pipeline_paged_copy_kv,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
            ],
            MTLSize::new(size as u64, 1, 1),
            MTLSize::new(256.min(size as u64), 1, 1),
        );
    }

    /// Paged attention: one threadgroup of 256 threads per query head.
    fn paged_attention(
        &self,
        q: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_table: &MetalTensor,
        out: &MetalTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PagedAttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
        };
        let threads_per_group: u64 = 256;
        self.dispatch_async(
            &self.pipeline_paged_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k_pool.buffer, 2),
                (&v_pool.buffer, 3),
                (&block_table.buffer, 4),
                (&out.buffer, 5),
            ],
            MTLSize::new(num_heads as u64 * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    // --- Batched / prefill operations ---
    //
    // These methods replace the single-token operations above for the prefill
    // phase.  Instead of processing one token at a time (mat-vec), they process
    // the entire prompt (mat-mat / GEMM).
    //
    // The key difference in dispatch: where single-token operations have
    //   grid = M * 32   (one SIMD group per output row)
    // batched operations have
    //   grid = batch_size * M * 32   (one SIMD group per (batch, row) pair)
    //
    // This launches batch_size × more thread groups, fully saturating the GPU's
    // compute units.  The weight matrix is loaded once and reused across all
    // batch elements — the fundamental reason GEMM is faster than repeated matvec.

    /// Batched GEMM: out = input @ weight^T for batch_size input rows.
    /// Grid: batch_size * M * 32 threads (32 per output element).
    fn matmul_batch(
        &self,
        weight: &MetalTensor,
        input: &MetalTensor,
        out: &MetalTensor,
        batch_size: u32,
        m: u32,
        k: u32,
    ) {
        let params = GemmParams { batch_size, m, k };
        let pipeline = match weight.dtype {
            TensorDtype::Q4 => &self.pipeline_gemm_q4,
            _ => &self.pipeline_gemm_bf16,
        };
        self.dispatch_async(
            pipeline,
            &params,
            &[(&weight.buffer, 1), (&input.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(batch_size as u64 * m as u64 * 32, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Batched RMSNorm: one threadgroup of 256 per row.
    fn rms_norm_batch(
        &self,
        input: &MetalTensor,
        weight: &MetalTensor,
        eps: f32,
        out: &MetalTensor,
        batch_size: u32,
    ) {
        let hidden_size = weight.shape[0] as u32;
        let params = RmsNormBatchParams {
            hidden_size,
            eps,
            batch_size,
        };
        self.dispatch_async(
            &self.pipeline_rms_norm_batch,
            &params,
            &[(&input.buffer, 1), (&weight.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(batch_size as u64 * 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Batched embedding lookup: N token IDs → [batch_size, hidden_dim].
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

    /// Batched RoPE: per-token positions.
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

    /// Batched paged KV cache write: N vectors at different positions.
    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &MetalTensor,
        pool: &MetalTensor,
        block_table: &MetalTensor,
        positions: &MetalTensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PagedCopyKvBatchParams {
            batch_size,
            num_kv_heads,
            head_dim,
            block_size: crate::model::kv_cache::BLOCK_SIZE as u32,
        };
        let kv_dim = num_kv_heads * head_dim;
        let total = batch_size as u64 * kv_dim as u64;
        self.dispatch_async(
            &self.pipeline_paged_copy_kv_batch,
            &params,
            &[
                (&src.buffer, 1),
                (&pool.buffer, 2),
                (&block_table.buffer, 3),
                (&positions.buffer, 4),
            ],
            MTLSize::new(total, 1, 1),
            MTLSize::new(256.min(total), 1, 1),
        );
    }

    /// Causal prefill attention on dense Q/K/V tensors.
    fn prefill_attention(
        &self,
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        out: &MetalTensor,
        chunk_size: u32,
        start_pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let params = PrefillAttentionParams {
            chunk_size,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let threads_per_group: u64 = 256;
        let num_threadgroups = chunk_size as u64 * num_heads as u64;
        self.dispatch_async(
            &self.pipeline_prefill_attention,
            &params,
            &[
                (&q.buffer, 1),
                (&k.buffer, 2),
                (&v.buffer, 3),
                (&out.buffer, 4),
            ],
            MTLSize::new(num_threadgroups * threads_per_group, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }
}
