// ===========================================================================
// CUDA GPU backend for LLM inference.
//
// LEARNING OVERVIEW — read this first if you're new to GPU programming.
//
// What this file does:
//   Implements the CUDA device management, NVRTC shader compilation, and
//   stream-based async dispatch infrastructure.  The actual kernel dispatch
//   methods (trait implementations) live in the kernels/ directory.
//
// Key concepts for understanding this backend:
//
//   1. KERNEL COMPILATION
//      CUDA shader sources (.cu files) are embedded in the binary at compile
//      time via `include_str!()`.  At runtime, NVRTC compiles them to PTX
//      (assembly), which is then loaded into a CudaModule.  Each compiled
//      function becomes a `CudaFunction` — CUDA's handle for a launchable
//      GPU kernel.
//
//   2. DISPATCH MODEL
//      CUDA's launch model uses `(grid_dim, block_dim)`:
//        - `block_dim` = number of threads per block (e.g. 256)
//        - `grid_dim`  = number of blocks = ceil(total_threads / block_dim)
//      Unlike Metal where you specify total threads, CUDA specifies blocks.
//
//   3. MEMORY MODEL (DISCRETE)
//      Unlike Apple Silicon's unified memory, NVIDIA GPUs have separate device
//      memory (HBM3 on H100).  All tensor data lives in device memory; host
//      access requires explicit copies (cuMemcpyHtoD / cuMemcpyDtoH).
//
//   4. ASYNC DISPATCH
//      Kernel launches on a CUDA stream are asynchronous — the CPU returns
//      immediately while the GPU executes.  Stream ordering guarantees that
//      kernels on the same stream execute in submission order.  The CPU only
//      blocks when it calls `stream.synchronize()` (our `flush()`).
//
//   5. PARAMETER PASSING
//      Kernel parameters are passed by value using cudarc's `DeviceRepr` trait.
//      The CUDA driver copies small structs (<4KB) to constant memory at launch
//      time.  This avoids device memory allocation for per-kernel params and is
//      the natural pattern for cudarc's launch_builder API.
//
//   6. PARAMS STRUCTS
//      Small constant data (dimensions, positions, epsilon) is passed to
//      kernels via a `#[repr(C)]` struct.  The Rust struct layout must match
//      the CUDA struct EXACTLY — field order, sizes, and alignment.
//      `#[repr(C)]` guarantees this.
//
// Related files:
//   Metal backend: metal/backend.rs (equivalent for macOS)
//   Trait defs:    gpu/ops/*.rs
//   CUDA kernels:  cuda/kernels/*.rs
//   CUDA shaders:  cuda/shaders/*.cu
// ===========================================================================

use std::sync::Arc;

use anyhow::anyhow;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};

// ---------------------------------------------------------------------------
// Embedded shader sources.
// ---------------------------------------------------------------------------

const CUDA_SOURCE_RMS_NORM: &str = include_str!("shaders/rms_norm.cu");
const CUDA_SOURCE_MATMUL: &str = include_str!("shaders/matmul.cu");
const CUDA_SOURCE_ROPE: &str = include_str!("shaders/rope.cu");
const CUDA_SOURCE_ATTENTION: &str = include_str!("shaders/attention.cu");
const CUDA_SOURCE_ELEMENTWISE: &str = include_str!("shaders/elementwise.cu");
const CUDA_SOURCE_EMBED: &str = include_str!("shaders/embed.cu");
const CUDA_SOURCE_DELTANET: &str = include_str!("shaders/deltanet.cu");

// ---------------------------------------------------------------------------
// CudaBackend — holds the CUDA device context, stream, and all compiled
// kernel function handles.
//
// Kernel dispatches are submitted to a single CUDA stream for serialised
// execution.  `flush()` calls `stream.synchronize()` to wait for all
// pending GPU work.  `submit()` is a no-op on CUDA since stream submissions
// are already committed to the GPU immediately.
// ---------------------------------------------------------------------------

pub(crate) struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) name: String,

    // Compiled kernel functions — one per kernel entry point.
    pub(crate) fn_rms_norm: CudaFunction,
    pub(crate) fn_rms_norm_batch: CudaFunction,
    pub(crate) fn_matvec_bf16: CudaFunction,
    pub(crate) fn_matvec_q4: CudaFunction,
    pub(crate) fn_gemm_bf16: CudaFunction,
    pub(crate) fn_gemm_q4: CudaFunction,
    pub(crate) fn_rope: CudaFunction,
    pub(crate) fn_rope_batch: CudaFunction,
    pub(crate) fn_rope_partial: CudaFunction,
    #[allow(dead_code)]
    pub(crate) fn_attention: CudaFunction,
    #[allow(dead_code)]
    pub(crate) fn_attention_hd256: CudaFunction,
    #[allow(dead_code)]
    pub(crate) fn_copy_kv: CudaFunction,
    pub(crate) fn_paged_copy_kv: CudaFunction,
    pub(crate) fn_paged_attention: CudaFunction,
    pub(crate) fn_paged_attention_hd256: CudaFunction,
    pub(crate) fn_paged_attention_fused: CudaFunction,
    pub(crate) fn_paged_attention_fused_hd256: CudaFunction,
    pub(crate) fn_paged_copy_kv_batch: CudaFunction,
    pub(crate) fn_prefill_attention: CudaFunction,
    pub(crate) fn_prefill_attention_hd256: CudaFunction,
    pub(crate) fn_silu_mul: CudaFunction,
    pub(crate) fn_gelu_mul: CudaFunction,
    pub(crate) fn_scalar_mul: CudaFunction,
    pub(crate) fn_add: CudaFunction,
    pub(crate) fn_bias_add: CudaFunction,
    pub(crate) fn_scale_add: CudaFunction,
    pub(crate) fn_fill_zero: CudaFunction,
    pub(crate) fn_top_k_softmax: CudaFunction,
    pub(crate) fn_embed_lookup: CudaFunction,
    pub(crate) fn_embed_lookup_batch: CudaFunction,

    // DeltaNet kernels (Qwen 3.5 hybrid attention).
    pub(crate) fn_conv1d_depthwise: CudaFunction,
    pub(crate) fn_conv1d_shift: CudaFunction,
    pub(crate) fn_l2_normalize: CudaFunction,
    pub(crate) fn_sigmoid: CudaFunction,
    pub(crate) fn_sigmoid_bf16: CudaFunction,
    pub(crate) fn_decay_gate: CudaFunction,
    pub(crate) fn_silu_elem: CudaFunction,
    pub(crate) fn_mul_elem: CudaFunction,
    pub(crate) fn_deltanet_step: CudaFunction,
}

impl CudaBackend {
    pub fn new() -> anyhow::Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
        let stream = ctx.default_stream();

        // Query device name via the CudaContext API.
        let name = ctx.name()
            .unwrap_or_else(|_| "NVIDIA GPU".to_string());

        // Compile shader sources via NVRTC.
        // NVRTC needs the CUDA include path for headers like cuda_bf16.h.
        let cuda_include = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let include_path = format!("{cuda_include}/include");

        let compile = |source: &str| -> anyhow::Result<cudarc::nvrtc::Ptx> {
            let opts = cudarc::nvrtc::CompileOptions {
                options: vec![format!("-I{include_path}")],
                ..Default::default()
            };
            cudarc::nvrtc::compile_ptx_with_opts(source, opts)
                .map_err(|e| anyhow!("NVRTC compilation failed: {e}"))
        };

        // Attention needs two specialisations for different MAX_HEAD_DIM values.
        let attn_src_128 = format!("#define MAX_HEAD_DIM 128\n{CUDA_SOURCE_ATTENTION}");
        let attn_src_256 = format!("#define MAX_HEAD_DIM 256\n{CUDA_SOURCE_ATTENTION}");

        let ptx_rms_norm = compile(CUDA_SOURCE_RMS_NORM)?;
        let ptx_matmul = compile(CUDA_SOURCE_MATMUL)?;
        let ptx_rope = compile(CUDA_SOURCE_ROPE)?;
        let ptx_attn_128 = compile(&attn_src_128)?;
        let ptx_attn_256 = compile(&attn_src_256)?;
        let ptx_elementwise = compile(CUDA_SOURCE_ELEMENTWISE)?;
        let ptx_embed = compile(CUDA_SOURCE_EMBED)?;
        let ptx_deltanet = compile(CUDA_SOURCE_DELTANET)?;

        // Load modules and extract function handles.
        let load = |ptx: cudarc::nvrtc::Ptx| -> anyhow::Result<Arc<CudaModule>> {
            ctx.load_module(ptx)
                .map_err(|e| anyhow!("failed to load CUDA module: {e}"))
        };

        let mod_rms_norm = load(ptx_rms_norm)?;
        let mod_matmul = load(ptx_matmul)?;
        let mod_rope = load(ptx_rope)?;
        let mod_attn_128 = load(ptx_attn_128)?;
        let mod_attn_256 = load(ptx_attn_256)?;
        let mod_elementwise = load(ptx_elementwise)?;
        let mod_embed = load(ptx_embed)?;
        let mod_deltanet = load(ptx_deltanet)?;

        let func = |module: &Arc<CudaModule>, name: &str| -> anyhow::Result<CudaFunction> {
            module.load_function(name)
                .map_err(|e| anyhow!("function '{name}' not found: {e}"))
        };

        Ok(Self {
            ctx,
            stream,
            name,

            // RMSNorm
            fn_rms_norm: func(&mod_rms_norm, "rms_norm")?,
            fn_rms_norm_batch: func(&mod_rms_norm, "rms_norm_batch")?,

            // Matmul
            fn_matvec_bf16: func(&mod_matmul, "matvec_bf16")?,
            fn_matvec_q4: func(&mod_matmul, "matvec_q4")?,
            fn_gemm_bf16: func(&mod_matmul, "gemm_bf16")?,
            fn_gemm_q4: func(&mod_matmul, "gemm_q4")?,

            // RoPE
            fn_rope: func(&mod_rope, "rotary_embedding")?,
            fn_rope_batch: func(&mod_rope, "rotary_embedding_batch")?,
            fn_rope_partial: func(&mod_rope, "rotary_embedding_partial")?,

            // Attention (HD128)
            fn_attention: func(&mod_attn_128, "attention")?,
            fn_copy_kv: func(&mod_attn_128, "copy_to_kv_cache")?,
            fn_paged_copy_kv: func(&mod_attn_128, "copy_to_paged_kv_cache")?,
            fn_paged_attention: func(&mod_attn_128, "paged_attention")?,
            fn_paged_attention_fused: func(&mod_attn_128, "paged_attention_fused")?,
            fn_paged_copy_kv_batch: func(&mod_attn_128, "copy_to_paged_kv_cache_batch")?,
            fn_prefill_attention: func(&mod_attn_128, "prefill_attention")?,

            // Attention (HD256)
            fn_attention_hd256: func(&mod_attn_256, "attention")?,
            fn_paged_attention_hd256: func(&mod_attn_256, "paged_attention")?,
            fn_paged_attention_fused_hd256: func(&mod_attn_256, "paged_attention_fused")?,
            fn_prefill_attention_hd256: func(&mod_attn_256, "prefill_attention")?,

            // Elementwise
            fn_silu_mul: func(&mod_elementwise, "silu_mul")?,
            fn_gelu_mul: func(&mod_elementwise, "gelu_mul")?,
            fn_scalar_mul: func(&mod_elementwise, "scalar_mul")?,
            fn_add: func(&mod_elementwise, "add_tensors")?,
            fn_bias_add: func(&mod_elementwise, "bias_add")?,
            fn_scale_add: func(&mod_elementwise, "scale_add")?,
            fn_fill_zero: func(&mod_elementwise, "fill_zero")?,
            fn_top_k_softmax: func(&mod_elementwise, "top_k_softmax")?,

            // Embed
            fn_embed_lookup: func(&mod_embed, "embed_lookup")?,
            fn_embed_lookup_batch: func(&mod_embed, "embed_lookup_batch")?,

            // DeltaNet
            fn_conv1d_depthwise: func(&mod_deltanet, "conv1d_depthwise_single")?,
            fn_conv1d_shift: func(&mod_deltanet, "conv1d_shift_history")?,
            fn_l2_normalize: func(&mod_deltanet, "l2_normalize_heads")?,
            fn_sigmoid: func(&mod_deltanet, "sigmoid_kernel")?,
            fn_sigmoid_bf16: func(&mod_deltanet, "sigmoid_bf16")?,
            fn_decay_gate: func(&mod_deltanet, "deltanet_decay_gate")?,
            fn_silu_elem: func(&mod_deltanet, "silu_elementwise")?,
            fn_mul_elem: func(&mod_deltanet, "mul_elementwise")?,
            fn_deltanet_step: func(&mod_deltanet, "deltanet_step")?,
        })
    }

    // =======================================================================
    // LAUNCH HELPERS
    //
    // CUDA grid/block calculation differs from Metal:
    //   Metal:  grid_size = total_threads, threadgroup_size = block_size
    //   CUDA:   grid_dim  = num_blocks,    block_dim = threads_per_block
    //
    // `launch_1d(total, block)` → grid = ceil(total / block)
    // `launch_blocks(blocks, block)` → grid = blocks
    // =======================================================================

    /// LaunchConfig for `total` threads with `block_size` threads per block.
    pub(crate) fn cfg_1d(total: u32, block_size: u32) -> LaunchConfig {
        let grid = (total + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// LaunchConfig for `num_blocks` blocks of `block_size` threads.
    pub(crate) fn cfg_blocks(num_blocks: u32, block_size: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Query total device memory in bytes.
    pub(crate) fn device_total_memory(&self) -> u64 {
        let (_, total) = cudarc::driver::result::mem_get_info()
            .unwrap_or((0, 0));
        total as u64
    }
}
