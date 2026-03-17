// ===========================================================================
// Metal GPU backend for LLM inference.
//
// LEARNING OVERVIEW — read this first if you're new to GPU programming.
//
// What this file does:
//   Implements the Metal device management, pipeline compilation, and
//   async dispatch infrastructure.  The actual kernel dispatch methods
//   (trait implementation) live in kernels.rs.
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

use anyhow::anyhow;
use metal::{CompileOptions, MTLResourceOptions, MTLSize};

// ---------------------------------------------------------------------------
// Embedded shader sources.
// ---------------------------------------------------------------------------

const METAL_SOURCE_RMS_NORM: &str = include_str!("shaders/rms_norm.metal");
const METAL_SOURCE_MATMUL: &str = include_str!("shaders/matmul.metal");
const METAL_SOURCE_ROPE: &str = include_str!("shaders/rope.metal");
const METAL_SOURCE_ATTENTION: &str = include_str!("shaders/attention.metal");
const METAL_SOURCE_ELEMENTWISE: &str = include_str!("shaders/elementwise.metal");
const METAL_SOURCE_EMBED: &str = include_str!("shaders/embed.metal");
const METAL_SOURCE_DELTANET: &str = include_str!("shaders/deltanet.metal");

// ---------------------------------------------------------------------------
// MetalBackend — holds the Metal device, command queue, and all compiled
// kernel pipeline states.
//
// Kernel dispatches are batched into a single Metal command buffer (stored
// in `current_cmd`).  `flush()` commits and waits for the entire batch —
// called automatically by `copy_to_host()`.
// ---------------------------------------------------------------------------

pub(crate) struct MetalBackend {
    pub(crate) device: metal::Device,
    pub(crate) queue: metal::CommandQueue,
    pub(crate) name: String,

    // Compiled kernel pipelines — one per kernel entry point.
    pub(crate) pipeline_rms_norm: metal::ComputePipelineState,
    pub(crate) pipeline_matvec: metal::ComputePipelineState,
    pub(crate) pipeline_matvec_q4: metal::ComputePipelineState,
    pub(crate) pipeline_rope: metal::ComputePipelineState,
    #[allow(dead_code)]
    pub(crate) pipeline_attention: metal::ComputePipelineState,
    #[allow(dead_code)]
    pub(crate) pipeline_attention_hd256: metal::ComputePipelineState,
    pub(crate) pipeline_silu_mul: metal::ComputePipelineState,
    pub(crate) pipeline_gelu_mul: metal::ComputePipelineState,
    pub(crate) pipeline_scalar_mul: metal::ComputePipelineState,
    pub(crate) pipeline_add: metal::ComputePipelineState,
    pub(crate) pipeline_bias_add: metal::ComputePipelineState,
    pub(crate) pipeline_scale_add: metal::ComputePipelineState,
    pub(crate) pipeline_fill_zero: metal::ComputePipelineState,
    pub(crate) pipeline_embed_lookup: metal::ComputePipelineState,
    #[allow(dead_code)]
    pub(crate) pipeline_copy_kv: metal::ComputePipelineState,
    pub(crate) pipeline_paged_copy_kv: metal::ComputePipelineState,
    pub(crate) pipeline_paged_attention: metal::ComputePipelineState,
    pub(crate) pipeline_paged_attention_hd256: metal::ComputePipelineState,
    pub(crate) pipeline_paged_attention_fused: metal::ComputePipelineState,
    pub(crate) pipeline_paged_attention_fused_hd256: metal::ComputePipelineState,

    // Phase 3: batched prefill pipelines.
    pub(crate) pipeline_gemm_bf16: metal::ComputePipelineState,
    pub(crate) pipeline_gemm_q4: metal::ComputePipelineState,
    pub(crate) pipeline_rms_norm_batch: metal::ComputePipelineState,
    pub(crate) pipeline_embed_lookup_batch: metal::ComputePipelineState,
    pub(crate) pipeline_rope_batch: metal::ComputePipelineState,
    pub(crate) pipeline_paged_copy_kv_batch: metal::ComputePipelineState,
    pub(crate) pipeline_prefill_attention: metal::ComputePipelineState,
    pub(crate) pipeline_prefill_attention_hd256: metal::ComputePipelineState,

    // MoE routing kernel.
    pub(crate) pipeline_top_k_softmax: metal::ComputePipelineState,

    // DeltaNet kernels (Qwen 3.5 hybrid attention).
    pub(crate) pipeline_conv1d_depthwise: metal::ComputePipelineState,
    pub(crate) pipeline_conv1d_shift: metal::ComputePipelineState,
    pub(crate) pipeline_l2_normalize: metal::ComputePipelineState,
    pub(crate) pipeline_sigmoid: metal::ComputePipelineState,
    pub(crate) pipeline_sigmoid_bf16: metal::ComputePipelineState,
    pub(crate) pipeline_decay_gate: metal::ComputePipelineState,
    pub(crate) pipeline_silu_elem: metal::ComputePipelineState,
    pub(crate) pipeline_mul_elem: metal::ComputePipelineState,
    pub(crate) pipeline_deltanet_step: metal::ComputePipelineState,
    pub(crate) pipeline_rope_partial: metal::ComputePipelineState,

    // GPT-OSS kernels.
    pub(crate) pipeline_silu_mul_clamp: metal::ComputePipelineState,
    pub(crate) pipeline_gpt_oss_gated_act: metal::ComputePipelineState,
    pub(crate) pipeline_rope_yarn: metal::ComputePipelineState,
    pub(crate) pipeline_rope_yarn_batch: metal::ComputePipelineState,

    // Batched command encoding state.
    //
    // Instead of creating and committing a command buffer per kernel dispatch,
    // we accumulate dispatches into a single command buffer.  This reduces Metal
    // overhead dramatically for MoE models (~2500 dispatches/token → 1 commit
    // per flush instead of 2500 individual commits).
    //
    // Metal allows creating sequential compute command encoders on one command
    // buffer.  Each dispatch creates an encoder, encodes the kernel, and ends
    // the encoder.  The command buffer is only committed when flush() is called.
    pub(crate) current_cmd: std::sync::Mutex<Option<metal::CommandBuffer>>,
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
        // Compile attention kernels with two MAX_HEAD_DIM specialisations:
        //   128 — covers Llama (64), Gemma 4B (96), Qwen/Mistral/Phi (128)
        //   256 — covers Gemma 27B (256)
        // The dispatch layer picks the right pipeline based on runtime head_dim.
        // This avoids register pressure from oversized v_acc arrays: head_dim=64
        // with MAX_HEAD_DIM=256 would waste 4x registers per thread.
        let attn_src_128 = format!("#define MAX_HEAD_DIM 128\n{METAL_SOURCE_ATTENTION}");
        let attn_src_256 = format!("#define MAX_HEAD_DIM 256\n{METAL_SOURCE_ATTENTION}");

        let pipeline_attention =
            Self::make_pipeline(&device, &attn_src_128, "attention", &compile_opts)?;
        let pipeline_attention_hd256 =
            Self::make_pipeline(&device, &attn_src_256, "attention", &compile_opts)?;
        let pipeline_silu_mul =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "silu_mul", &compile_opts)?;
        let pipeline_gelu_mul =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "gelu_mul", &compile_opts)?;
        let pipeline_scalar_mul =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "scalar_mul", &compile_opts)?;
        let pipeline_add = Self::make_pipeline(
            &device,
            METAL_SOURCE_ELEMENTWISE,
            "add_tensors",
            &compile_opts,
        )?;
        let pipeline_bias_add =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "bias_add", &compile_opts)?;
        let pipeline_scale_add =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "scale_add", &compile_opts)?;
        let pipeline_fill_zero =
            Self::make_pipeline(&device, METAL_SOURCE_ELEMENTWISE, "fill_zero", &compile_opts)?;
        let pipeline_embed_lookup =
            Self::make_pipeline(&device, METAL_SOURCE_EMBED, "embed_lookup", &compile_opts)?;
        let pipeline_copy_kv = Self::make_pipeline(
            &device,
            &attn_src_128,
            "copy_to_kv_cache",
            &compile_opts,
        )?;
        let pipeline_paged_copy_kv = Self::make_pipeline(
            &device,
            &attn_src_128,
            "copy_to_paged_kv_cache",
            &compile_opts,
        )?;
        let pipeline_paged_attention = Self::make_pipeline(
            &device,
            &attn_src_128,
            "paged_attention",
            &compile_opts,
        )?;
        let pipeline_paged_attention_hd256 = Self::make_pipeline(
            &device,
            &attn_src_256,
            "paged_attention",
            &compile_opts,
        )?;
        let pipeline_paged_attention_fused = Self::make_pipeline(
            &device,
            &attn_src_128,
            "paged_attention_fused",
            &compile_opts,
        )?;
        let pipeline_paged_attention_fused_hd256 = Self::make_pipeline(
            &device,
            &attn_src_256,
            "paged_attention_fused",
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
            &attn_src_128,
            "copy_to_paged_kv_cache_batch",
            &compile_opts,
        )?;
        let pipeline_prefill_attention = Self::make_pipeline(
            &device,
            &attn_src_128,
            "prefill_attention",
            &compile_opts,
        )?;
        let pipeline_prefill_attention_hd256 = Self::make_pipeline(
            &device,
            &attn_src_256,
            "prefill_attention",
            &compile_opts,
        )?;
        let pipeline_top_k_softmax = Self::make_pipeline(
            &device,
            METAL_SOURCE_ELEMENTWISE,
            "top_k_softmax",
            &compile_opts,
        )?;

        // DeltaNet kernels.
        let pipeline_conv1d_depthwise = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "conv1d_depthwise_single",
            &compile_opts,
        )?;
        let pipeline_conv1d_shift = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "conv1d_shift_history",
            &compile_opts,
        )?;
        let pipeline_l2_normalize = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "l2_normalize_heads",
            &compile_opts,
        )?;
        let pipeline_sigmoid =
            Self::make_pipeline(&device, METAL_SOURCE_DELTANET, "sigmoid_kernel", &compile_opts)?;
        let pipeline_sigmoid_bf16 =
            Self::make_pipeline(&device, METAL_SOURCE_DELTANET, "sigmoid_bf16", &compile_opts)?;
        let pipeline_decay_gate =
            Self::make_pipeline(&device, METAL_SOURCE_DELTANET, "deltanet_decay_gate", &compile_opts)?;
        let pipeline_silu_elem = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "silu_elementwise",
            &compile_opts,
        )?;
        let pipeline_mul_elem = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "mul_elementwise",
            &compile_opts,
        )?;
        let pipeline_deltanet_step = Self::make_pipeline(
            &device,
            METAL_SOURCE_DELTANET,
            "deltanet_step",
            &compile_opts,
        )?;
        let pipeline_rope_partial = Self::make_pipeline(
            &device,
            METAL_SOURCE_ROPE,
            "rotary_embedding_partial",
            &compile_opts,
        )?;

        // GPT-OSS kernels.
        let pipeline_silu_mul_clamp = Self::make_pipeline(
            &device,
            METAL_SOURCE_ELEMENTWISE,
            "silu_mul_clamp",
            &compile_opts,
        )?;
        let pipeline_gpt_oss_gated_act = Self::make_pipeline(
            &device,
            METAL_SOURCE_ELEMENTWISE,
            "gpt_oss_gated_act",
            &compile_opts,
        )?;
        let pipeline_rope_yarn = Self::make_pipeline(
            &device,
            METAL_SOURCE_ROPE,
            "rotary_embedding_yarn",
            &compile_opts,
        )?;
        let pipeline_rope_yarn_batch = Self::make_pipeline(
            &device,
            METAL_SOURCE_ROPE,
            "rotary_embedding_yarn_batch",
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
            pipeline_attention_hd256,
            pipeline_silu_mul,
            pipeline_gelu_mul,
            pipeline_scalar_mul,
            pipeline_add,
            pipeline_bias_add,
            pipeline_scale_add,
            pipeline_fill_zero,
            pipeline_embed_lookup,
            pipeline_copy_kv,
            pipeline_paged_copy_kv,
            pipeline_paged_attention,
            pipeline_paged_attention_hd256,
            pipeline_paged_attention_fused,
            pipeline_paged_attention_fused_hd256,
            pipeline_gemm_bf16,
            pipeline_gemm_q4,
            pipeline_rms_norm_batch,
            pipeline_embed_lookup_batch,
            pipeline_rope_batch,
            pipeline_paged_copy_kv_batch,
            pipeline_prefill_attention,
            pipeline_prefill_attention_hd256,
            pipeline_top_k_softmax,
            pipeline_conv1d_depthwise,
            pipeline_conv1d_shift,
            pipeline_l2_normalize,
            pipeline_sigmoid,
            pipeline_sigmoid_bf16,
            pipeline_decay_gate,
            pipeline_silu_elem,
            pipeline_mul_elem,
            pipeline_deltanet_step,
            pipeline_rope_partial,
            pipeline_silu_mul_clamp,
            pipeline_gpt_oss_gated_act,
            pipeline_rope_yarn,
            pipeline_rope_yarn_batch,
            current_cmd: std::sync::Mutex::new(None),
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
    // BATCHED ASYNC DISPATCH
    //
    // All kernel dispatches are accumulated into a single Metal command buffer.
    // Each dispatch creates a compute encoder, encodes the kernel, and ends
    // the encoder — but the command buffer is NOT committed until flush().
    //
    // Metal allows sequential compute encoders on one command buffer, and the
    // serial command queue guarantees execution order.  This batching reduces
    // Metal API overhead dramatically:
    //
    //   Before: 2500 command buffer creations + commits per MoE token
    //   After:  1 command buffer, committed only on flush (~48 flushes/token)
    //
    // `dispatch_async`:   encode into the current command buffer, NO commit.
    // `flush`:            commit + wait for all accumulated work.
    // `copy_to_host`:     calls flush automatically.
    //
    // Safety of param buffer lifetime:
    //   Metal command buffers with retained references (the default) keep all
    //   set_buffer references alive until completion.  When the local param_buf
    //   drops, its retain count goes from 2→1; when the GPU finishes and the
    //   command buffer is released, the count goes to 0 and the buffer is freed.
    // =======================================================================

    /// Get or create the current command buffer for batching.
    ///
    /// `new_command_buffer()` returns a borrowed `&CommandBufferRef` with
    /// autorelease semantics.  We retain it via `to_owned()` so it lives
    /// in the Mutex until flush() commits it.
    fn get_or_create_cmd(&self) -> std::sync::MutexGuard<'_, Option<metal::CommandBuffer>> {
        let mut guard = self.current_cmd.lock().unwrap();
        if guard.is_none() {
            let cmd_ref = self.queue.new_command_buffer();
            *guard = Some(cmd_ref.to_owned());
        }
        guard
    }

    /// Encode a kernel dispatch into the current batched command buffer.
    pub(crate) fn dispatch_async<T: Copy>(
        &self,
        pipeline: &metal::ComputePipelineState,
        params: &T,
        buffers: &[(&metal::Buffer, u64)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        let guard = self.get_or_create_cmd();
        let cmd = guard.as_ref().unwrap();

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
        // No commit — stays in the command buffer until flush().
    }

    /// Commit and wait for all pending GPU work in the current command buffer.
    pub(crate) fn flush(&self) {
        let cmd = self.current_cmd.lock().unwrap().take();
        if let Some(cmd) = cmd {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    /// Commit pending GPU work without waiting — allows GPU to start executing
    /// while the CPU continues encoding the next batch of dispatches.
    pub(crate) fn submit(&self) {
        let cmd = self.current_cmd.lock().unwrap().take();
        if let Some(cmd) = cmd {
            cmd.commit();
        }
    }
}
