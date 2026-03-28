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
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, LaunchConfig,
};

// ---------------------------------------------------------------------------
// Embedded shader sources.
// ---------------------------------------------------------------------------

const CUDA_SOURCE_RMS_NORM: &str = include_str!("shaders/rms_norm.cu");
const CUDA_SOURCE_MATMUL: &str = include_str!("shaders/matmul.cu");
const CUDA_SOURCE_MATMUL_TC: &str = include_str!("shaders/matmul_tc.cu");
const CUDA_SOURCE_ROPE: &str = include_str!("shaders/rope.cu");
const CUDA_SOURCE_ATTENTION: &str = include_str!("shaders/attention.cu");
const CUDA_SOURCE_ELEMENTWISE: &str = include_str!("shaders/elementwise.cu");
const CUDA_SOURCE_EMBED: &str = include_str!("shaders/embed.cu");
const CUDA_SOURCE_DELTANET: &str = include_str!("shaders/deltanet.cu");
const CUDA_SOURCE_MOE: &str = include_str!("shaders/moe.cu");
const CUDA_SOURCE_MAMBA2: &str = include_str!("shaders/mamba2.cu");
const CUDA_SOURCE_VISION: &str = include_str!("shaders/vision.cu");
const CUDA_SOURCE_TURBOQUANT: &str = include_str!("shaders/turboquant.cu");

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

    // Dedicated transfer stream for async HtoD copies (expert streaming).
    //
    // Separate from the compute stream so DMA and kernel execution overlap.
    // sync_transfers() records an event here, then makes the compute stream
    // wait — GPU-side only, CPU doesn't block.
    pub(crate) transfer_stream: Arc<CudaStream>,
    pub(crate) transfer_event: cudarc::driver::CudaEvent,

    // Multi-GPU tensor parallelism fields.
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) nccl_comm: Option<Arc<super::nccl::NcclComm>>,

    // GPU compute capability (major, minor). Used to select tensor-core vs
    // scalar GEMM paths.  bf16 WMMA requires sm_80+ (A100/H100).
    pub(crate) compute_capability: (i32, i32),

    // Compiled kernel functions — one per kernel entry point.
    pub(crate) fn_rms_norm: CudaFunction,
    pub(crate) fn_rms_norm_batch: CudaFunction,
    pub(crate) fn_matvec_bf16: CudaFunction,
    pub(crate) fn_matvec_q4: CudaFunction,
    pub(crate) fn_gemm_bf16: CudaFunction,
    pub(crate) fn_gemm_q4: CudaFunction,

    // Tensor-core WMMA GEMM kernels (sm_80+ only).  None on older GPUs,
    // which fall back to the scalar gemm_bf16/gemm_q4 above.
    pub(crate) fn_gemm_bf16_tc: Option<CudaFunction>,
    pub(crate) fn_gemm_q4_tc: Option<CudaFunction>,
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

    // Fused MoE kernels (gate+up+SwiGLU, combine+residual).
    pub(crate) fn_fused_gate_up_swiglu_bf16: CudaFunction,
    pub(crate) fn_fused_gate_up_swiglu_q4: CudaFunction,
    pub(crate) fn_moe_combine_residual: CudaFunction,

    // GPT-OSS kernels.
    pub(crate) fn_silu_mul_clamp: CudaFunction,
    pub(crate) fn_rope_yarn: CudaFunction,
    pub(crate) fn_rope_yarn_batch: CudaFunction,
    pub(crate) fn_gpt_oss_gated_act: CudaFunction,

    // New elementwise kernels (Nemotron-H, vision).
    pub(crate) fn_gelu_act: CudaFunction,
    pub(crate) fn_relu_squared: CudaFunction,
    pub(crate) fn_top_k_sigmoid: CudaFunction,

    // LayerNorm (vision encoder).
    pub(crate) fn_layer_norm_batch: CudaFunction,

    // Q8 matmul kernels.
    pub(crate) fn_matvec_q8: CudaFunction,
    pub(crate) fn_gemm_q8: CudaFunction,
    pub(crate) fn_gemm_q8_tc: Option<CudaFunction>,

    // Q8 MoE kernel.
    pub(crate) fn_fused_gate_up_swiglu_q8: CudaFunction,

    // Mamba2 kernels (Nemotron-H).
    pub(crate) fn_mamba2_conv1d_silu: CudaFunction,
    pub(crate) fn_mamba2_ssm_step: CudaFunction,
    pub(crate) fn_mamba2_gated_rms_norm: CudaFunction,

    // Vision encoder kernels.
    pub(crate) fn_spatial_merge: CudaFunction,
    pub(crate) fn_spatial_merge_norm: CudaFunction,
    pub(crate) fn_scatter_vision_tokens: CudaFunction,

    // Fused QKV attention (vision encoder).
    pub(crate) fn_prefill_attention_fused_qkv: CudaFunction,

    // TurboQuant KV cache quantization kernels.
    pub(crate) fn_turbo_quantize_paged: CudaFunction,
    pub(crate) fn_turbo_quantize_paged_batch: CudaFunction,
    pub(crate) fn_turbo_rotate_q: CudaFunction,
    pub(crate) fn_turbo_paged_attention: CudaFunction,
}

impl CudaBackend {
    /// Single-GPU convenience constructor (rank=0, world_size=1, no NCCL).
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with_device(0, 0, 1, None)
    }

    /// Multi-GPU constructor: creates a backend for a specific device with NCCL.
    pub fn new_with_device(
        device_id: usize,
        rank: usize,
        world_size: usize,
        nccl_comm: Option<Arc<super::nccl::NcclComm>>,
    ) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| anyhow!("failed to create CUDA context on device {device_id}: {e}"))?;
        let stream = ctx.default_stream();

        // Dedicated transfer stream for async HtoD copies (expert streaming).
        // new_stream() creates a CU_STREAM_NON_BLOCKING stream — prevents
        // implicit synchronisation with the default stream so transfers and
        // compute can run concurrently.
        let transfer_stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("failed to create CUDA transfer stream: {e}"))?;

        // Event for transfer→compute synchronisation.
        // CU_EVENT_DISABLE_TIMING (default): we don't need timing, saves overhead.
        let transfer_event = ctx
            .new_event(None)
            .map_err(|e| anyhow!("failed to create CUDA transfer event: {e}"))?;

        // Query device name via the CudaContext API.
        let name = ctx.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());

        // Query compute capability to decide whether tensor-core WMMA GEMM
        // kernels are available.  bf16 WMMA requires sm_80+ (A100/H100).
        let compute_capability = ctx
            .compute_capability()
            .unwrap_or((0, 0));

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
        let ptx_moe = compile(CUDA_SOURCE_MOE)?;
        let ptx_mamba2 = compile(CUDA_SOURCE_MAMBA2)?;
        let ptx_vision = compile(CUDA_SOURCE_VISION)?;
        let ptx_turboquant = compile(CUDA_SOURCE_TURBOQUANT)?;

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
        let mod_moe = load(ptx_moe)?;
        let mod_mamba2 = load(ptx_mamba2)?;
        let mod_vision = load(ptx_vision)?;
        let mod_turboquant = load(ptx_turboquant)?;

        // Tensor-core WMMA module (sm_80+ only).
        // Compiled with -arch=compute_80 so NVRTC emits WMMA (mma.h) instructions.
        // On older GPUs, we skip this entirely and use the scalar GEMM fallback.
        let mod_matmul_tc = if compute_capability.0 >= 8 {
            let ptx_tc = {
                let opts = cudarc::nvrtc::CompileOptions {
                    options: vec![
                        format!("-I{include_path}"),
                        "-arch=compute_80".to_string(),
                    ],
                    ..Default::default()
                };
                cudarc::nvrtc::compile_ptx_with_opts(CUDA_SOURCE_MATMUL_TC, opts)
                    .map_err(|e| anyhow!("NVRTC TC compilation failed: {e}"))?
            };
            Some(load(ptx_tc)?)
        } else {
            None
        };

        let func = |module: &Arc<CudaModule>, name: &str| -> anyhow::Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| anyhow!("function '{name}' not found: {e}"))
        };

        Ok(Self {
            ctx,
            stream,
            transfer_stream,
            transfer_event,
            name,
            rank,
            world_size,
            nccl_comm,
            compute_capability,

            // RMSNorm
            fn_rms_norm: func(&mod_rms_norm, "rms_norm")?,
            fn_rms_norm_batch: func(&mod_rms_norm, "rms_norm_batch")?,

            // Matmul (scalar — used for matvec always, GEMM on sm < 80)
            fn_matvec_bf16: func(&mod_matmul, "matvec_bf16")?,
            fn_matvec_q4: func(&mod_matmul, "matvec_q4")?,
            fn_gemm_bf16: func(&mod_matmul, "gemm_bf16")?,
            fn_gemm_q4: func(&mod_matmul, "gemm_q4")?,

            // Matmul (tensor-core WMMA — sm_80+ only)
            fn_gemm_bf16_tc: mod_matmul_tc.as_ref().map(|m| func(m, "gemm_bf16_tc")).transpose()?,
            fn_gemm_q4_tc: mod_matmul_tc.as_ref().map(|m| func(m, "gemm_q4_tc")).transpose()?,

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

            // Fused MoE
            fn_fused_gate_up_swiglu_bf16: func(&mod_moe, "fused_gate_up_swiglu_bf16")?,
            fn_fused_gate_up_swiglu_q4: func(&mod_moe, "fused_gate_up_swiglu_q4")?,
            fn_moe_combine_residual: func(&mod_moe, "moe_combine_residual")?,

            // GPT-OSS kernels.
            fn_silu_mul_clamp: func(&mod_elementwise, "silu_mul_clamp")?,
            fn_rope_yarn: func(&mod_rope, "rotary_embedding_yarn")?,
            fn_rope_yarn_batch: func(&mod_rope, "rotary_embedding_yarn_batch")?,
            fn_gpt_oss_gated_act: func(&mod_elementwise, "gpt_oss_gated_act")?,

            // New elementwise kernels.
            fn_gelu_act: func(&mod_elementwise, "gelu_act")?,
            fn_relu_squared: func(&mod_elementwise, "relu_squared")?,
            fn_top_k_sigmoid: func(&mod_elementwise, "top_k_sigmoid")?,

            // LayerNorm (vision encoder).
            fn_layer_norm_batch: func(&mod_rms_norm, "layer_norm_batch")?,

            // Q8 matmul kernels.
            fn_matvec_q8: func(&mod_matmul, "matvec_q8")?,
            fn_gemm_q8: func(&mod_matmul, "gemm_q8")?,
            fn_gemm_q8_tc: mod_matmul_tc.as_ref().map(|m| func(m, "gemm_q8_tc")).transpose()?,

            // Q8 MoE kernel.
            fn_fused_gate_up_swiglu_q8: func(&mod_moe, "fused_gate_up_swiglu_q8")?,

            // Mamba2 kernels.
            fn_mamba2_conv1d_silu: func(&mod_mamba2, "mamba2_conv1d_silu")?,
            fn_mamba2_ssm_step: func(&mod_mamba2, "mamba2_ssm_step")?,
            fn_mamba2_gated_rms_norm: func(&mod_mamba2, "mamba2_gated_rms_norm")?,

            // Vision encoder kernels.
            fn_spatial_merge: func(&mod_vision, "spatial_merge")?,
            fn_spatial_merge_norm: func(&mod_vision, "spatial_merge_norm")?,
            fn_scatter_vision_tokens: func(&mod_vision, "scatter_vision_tokens")?,

            // Fused QKV attention (vision encoder).
            fn_prefill_attention_fused_qkv: func(&mod_attn_128, "prefill_attention_fused_qkv")?,

            // TurboQuant kernels.
            fn_turbo_quantize_paged: func(&mod_turboquant, "turbo_quantize_paged")?,
            fn_turbo_quantize_paged_batch: func(&mod_turboquant, "turbo_quantize_paged_batch")?,
            fn_turbo_rotate_q: func(&mod_turboquant, "turbo_rotate_q")?,
            fn_turbo_paged_attention: func(&mod_turboquant, "turbo_paged_attention")?,
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

    /// LaunchConfig with 2D grid and explicit shared memory allocation.
    /// Used by tensor-core GEMM which tiles over two output dimensions.
    pub(crate) fn cfg_2d_smem(
        grid_x: u32,
        grid_y: u32,
        block_size: u32,
        shared_mem_bytes: u32,
    ) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes,
        }
    }

    /// Query total device memory in bytes.
    pub(crate) fn device_total_memory(&self) -> u64 {
        let (_, total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
        total as u64
    }
}

