// ===========================================================================
// Engine factory — loads a model and constructs an InferenceEngine.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Provides load_and_run(), the single entry point for loading a model and
//   running inference.  Handles both single-GPU (tp=1) and multi-GPU (tp>1)
//   paths.  Callers provide two callbacks:
//     - on_ready: called with the tokenizer and arch after loading, before
//       the inference loop starts (used by the API to signal readiness)
//     - run: called with the constructed InferenceEngine (the caller's loop)
//
// Why callbacks instead of returning the engine?
//   The Engine<'a, B> borrows the GPU backend (&'a B), which is created on
//   the same thread.  Returning the engine would require the backend to
//   outlive the function, but it's a stack local.  The callback pattern
//   keeps everything on the same stack frame — the backend, model, KV pool,
//   and engine all live together and the borrow checker is happy.
//
// Related files:
//   - engine/mod.rs       — Engine, SingleGpuDispatch, InferenceEngine
//   - engine/multi_gpu.rs — MultiGpuEngine, MultiGpuDispatch
//   - api/mod.rs          — spawns a thread, calls load_and_run()
//   - commands/batch.rs   — calls load_and_run() directly (no thread)
// ===========================================================================

use std::path::Path;
use std::time::Instant;

use tracing::{info, warn};

use crate::gpu::{self, GpuBackend, GpuCore, GpuElementwise, TensorDtype};
use crate::model;
use crate::model::config::{ModelArch, ModelConfig};
use crate::model::forward::{MoeBuffers, DeltaNetBuffers, Mamba2Buffers, ModelForward};
use crate::model::registry;
use crate::model::tokenizer::Tokenizer;
use crate::model::turboquant::{KvQuantMode, KvQuantPair};
use crate::model::{kv_cache, loader};

use super::InferenceEngine;

/// Allocate MoE scratch buffers for MoE architectures.
///
/// With expert parallelism (EP/Hybrid strategy), each rank owns whole experts
/// so `moe_inter` is NOT divided.  With tensor parallelism, each expert is
/// split across ranks so `moe_inter` is divided by `world_size`.
fn alloc_moe_buffers<B: GpuCore>(
    backend: &B,
    config: &ModelConfig,
    world_size: usize,
    rank: usize,
    use_ep: bool,
) -> MoeBuffers<B> {
    let hidden = config.hidden_size;
    let num_experts = config.num_experts;
    let k = config.num_experts_per_tok;

    let (moe_inter, local_expert_start, local_expert_count) = if use_ep && world_size > 1 {
        // EP: whole experts per rank, no intermediate-dim splitting.
        let experts_per_rank = num_experts / world_size;
        (config.moe_intermediate_size, rank * experts_per_rank, experts_per_rank)
    } else {
        // TP or single-GPU: each expert is split, all experts on every rank.
        (config.moe_intermediate_size / world_size, 0, num_experts)
    };

    MoeBuffers {
        router_logits: backend.alloc_tensor(&[num_experts], TensorDtype::F32),
        moe_gate_buf: backend.alloc_tensor(&[moe_inter], TensorDtype::BF16),
        moe_up_buf: backend.alloc_tensor(&[moe_inter], TensorDtype::BF16),
        moe_output: backend.alloc_tensor(&[hidden], TensorDtype::BF16),
        routing_output: backend.alloc_tensor(&[2 * k], TensorDtype::F32),
        expert_streamer: None,
        local_expert_start,
        local_expert_count,
    }
}

/// Allocate DeltaNet recurrent state and scratch buffers (Qwen 3.5 only).
fn alloc_deltanet_buffers<B: GpuCore + GpuElementwise>(backend: &B, config: &ModelConfig, world_size: usize) -> DeltaNetBuffers<B> {
    let num_qk_heads = config.linear_num_key_heads / world_size;
    let num_v_heads = config.linear_num_value_heads / world_size;
    let hd = config.linear_key_head_dim;
    let v_per_qk = num_v_heads / num_qk_heads;
    let v_dim = num_v_heads * config.linear_value_head_dim;
    let qk_dim = num_qk_heads * hd;
    let conv_dim = qk_dim * 2 + v_dim;
    let kernel_size = config.linear_conv_kernel_dim;

    let num_dn_layers = config.layer_types.iter()
        .filter(|t| t.as_str() == "linear_attention").count();
    let state_size = num_qk_heads * v_per_qk * hd * hd;

    let mut states = Vec::with_capacity(num_dn_layers);
    let mut conv_history = Vec::with_capacity(num_dn_layers);
    for _ in 0..num_dn_layers {
        states.push(backend.alloc_tensor(&[state_size], TensorDtype::F32));
        conv_history.push(backend.alloc_tensor(&[(kernel_size - 1) * conv_dim], TensorDtype::BF16));
    }
    for s in &states { backend.fill_zero(s, state_size as u32); }
    for h in &conv_history { backend.fill_zero(h, ((kernel_size - 1) * conv_dim) as u32); }

    DeltaNetBuffers {
        states,
        conv_history,
        qkv_buf: backend.alloc_tensor(&[conv_dim], TensorDtype::BF16),
        alpha_buf: backend.alloc_tensor(&[num_v_heads], TensorDtype::F32),
        beta_buf: backend.alloc_tensor(&[num_v_heads], TensorDtype::F32),
        z_buf: backend.alloc_tensor(&[v_dim], TensorDtype::BF16),
        conv_out: backend.alloc_tensor(&[conv_dim], TensorDtype::BF16),
        attn_out: backend.alloc_tensor(&[v_dim], TensorDtype::BF16),
        norm_out: backend.alloc_tensor(&[v_dim], TensorDtype::BF16),
    }
}

/// Allocate Mamba-2 SSM recurrent state and scratch buffers (Nemotron-H only).
fn alloc_mamba2_buffers<B: GpuCore + GpuElementwise>(backend: &B, config: &ModelConfig) -> Mamba2Buffers<B> {
    let d_inner = config.mamba2_d_inner();
    let conv_dim = config.mamba2_conv_dim();
    let in_proj_dim = config.mamba2_in_proj_dim();
    let ks = config.mamba_conv_kernel;
    let n_heads = config.mamba_num_heads;
    let hd = config.mamba_head_dim;
    let ss = config.ssm_state_size;

    let num_m2_layers = config.layer_types.iter()
        .filter(|t| t.as_str() == "mamba2").count();
    let state_size = n_heads * hd * ss;

    let mut states = Vec::with_capacity(num_m2_layers);
    let mut conv_history = Vec::with_capacity(num_m2_layers);
    for _ in 0..num_m2_layers {
        states.push(backend.alloc_tensor(&[state_size], TensorDtype::F32));
        conv_history.push(backend.alloc_tensor(&[(ks - 1) * conv_dim], TensorDtype::BF16));
    }
    for s in &states { backend.fill_zero(s, state_size as u32); }
    for h in &conv_history { backend.fill_zero(h, ((ks - 1) * conv_dim) as u32); }

    Mamba2Buffers {
        states,
        conv_history,
        in_proj_buf: backend.alloc_tensor(&[in_proj_dim], TensorDtype::BF16),
        conv_out: backend.alloc_tensor(&[conv_dim], TensorDtype::BF16),
        ssm_out: backend.alloc_tensor(&[d_inner], TensorDtype::BF16),
    }
}

/// Construct the architecture-specific forward pass from ModelArch.
///
/// Allocates arch-specific buffers (MoE, DeltaNet, Mamba-2) on the appropriate
/// Forward struct.  The Model struct only holds universal buffers.
pub(crate) fn create_forward<B: GpuBackend + 'static>(
    arch: ModelArch,
    config: &ModelConfig,
    backend: &B,
    world_size: usize,
    rank: usize,
    use_ep: bool,
    expert_streamer: Option<model::expert_stream::ExpertStreamer<B>>,
) -> Box<dyn ModelForward<B>> {
    match arch {
        ModelArch::Llama => Box::new(registry::llama::LlamaForward::new(false)),
        ModelArch::Mistral => Box::new(registry::llama::LlamaForward::new(false)),
        ModelArch::Phi => Box::new(registry::llama::LlamaForward::new(false)),
        ModelArch::Qwen2 => Box::new(registry::llama::LlamaForward::new(true)),
        ModelArch::Gemma3 => Box::new(registry::gemma::GemmaForward),
        ModelArch::Mixtral => {
            let mut moe = alloc_moe_buffers(backend, config, world_size, rank, use_ep);
            moe.expert_streamer = expert_streamer;
            Box::new(registry::mixtral::MixtralForward { moe })
        }
        ModelArch::Qwen3Moe => {
            let mut moe = alloc_moe_buffers(backend, config, world_size, rank, use_ep);
            moe.expert_streamer = expert_streamer;
            Box::new(registry::qwen3_moe::Qwen3MoeForward { moe })
        }
        ModelArch::Qwen3_5 => {
            let moe = if config.is_moe() {
                let mut moe = alloc_moe_buffers(backend, config, world_size, rank, use_ep);
                moe.expert_streamer = expert_streamer;
                Some(moe)
            } else {
                None
            };
            Box::new(registry::qwen3_5::Qwen35Forward {
                moe,
                dn: alloc_deltanet_buffers(backend, config, world_size),
            })
        }
        ModelArch::GptOss => {
            let mut moe = alloc_moe_buffers(backend, config, world_size, rank, use_ep);
            moe.expert_streamer = expert_streamer;
            Box::new(registry::gpt_oss::GptOssForward { moe })
        }
        ModelArch::NemotronH => {
            let mut moe = alloc_moe_buffers(backend, config, world_size, rank, use_ep);
            moe.expert_streamer = expert_streamer;
            Box::new(registry::nemotron_h::NemotronForward {
                moe,
                mamba2: alloc_mamba2_buffers(backend, config),
            })
        }
    }
}

/// Load a model and run a function with the constructed InferenceEngine.
///
/// Handles both single-GPU (`tp == 1`) and multi-GPU (`tp > 1`) paths.
/// The caller doesn't need to know which path is taken.
pub(crate) fn load_and_run_ext(
    model_dir: &Path,
    stream_experts: bool,
    tp: usize,
    kv_quant: KvQuantPair,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    if tp > 1 {
        load_and_run_multi_gpu(model_dir, tp, max_active, on_ready, run)
    } else {
        load_and_run_single_gpu(model_dir, stream_experts, kv_quant, max_active, on_ready, run)
    }
}

/// Single-GPU path: one backend, one model, one KV pool.
fn load_and_run_single_gpu(
    model_dir: &Path,
    stream_experts: bool,
    kv_quant: KvQuantPair,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let t_start = Instant::now();

    let backend = gpu::create_backend()?;
    info!(gpu = %backend.device_name(), "GPU initialized");

    let t_load = Instant::now();
    let loader::LoadedModel {
        config,
        arch,
        tokenizer,
        weights,
        expert_index,
        is_quantized,
        vision_weights,
    } = loader::load_model(&backend, model_dir, stream_experts)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    info!(load_time_s = format_args!("{:.2}", load_secs), "model loaded");

    let mut model = model::Model::new(config.clone(), weights, &backend)?;

    // Auto-adjust TurboQuant for K-intolerant architectures.
    //
    // Some models have properties that make K quantization produce correlated
    // errors that softmax amplifies:
    //   - QKV bias (Qwen2, GPT-OSS): learned bias reduces angular diversity
    //     of normalised K vectors.
    //   - Sparse attention hybrids (Qwen 3.5, Nemotron-H): few attention layers
    //     mean K errors propagate through many non-attention layers uncorrected.
    //
    // V is tolerant of quantisation error in both cases because it's a weighted
    // sum — errors average out across positions.  On Metal (where the V-only
    // kernel exists), we keep K at BF16 and turbo-quantise V only ("asymmetric
    // mode").  On CUDA the V-only kernel doesn't exist yet, so we fully disable.
    let needs_asymmetric = arch.has_qkv_bias() || config.has_sparse_attention();
    let kv_quant = if kv_quant.is_any_quantized() && needs_asymmetric {
        #[cfg(target_os = "macos")]
        {
            let pair = KvQuantPair::asymmetric(KvQuantMode::None, kv_quant.v);
            if arch.has_qkv_bias() {
                warn!(
                    arch = ?arch,
                    k = "BF16",
                    v = ?kv_quant.v,
                    "TurboQuant asymmetric mode (K=BF16, V=turbo) — QKV bias",
                );
            } else {
                warn!(
                    arch = ?arch,
                    kv_layers = config.num_kv_layers(),
                    total_layers = config.num_hidden_layers,
                    k = "BF16",
                    v = ?kv_quant.v,
                    "TurboQuant asymmetric mode (K=BF16, V=turbo) — sparse attention",
                );
            }
            pair
        }
        #[cfg(not(target_os = "macos"))]
        {
            warn!(
                arch = ?arch,
                "TurboQuant disabled (V-only kernel unavailable on CUDA); using BF16 KV cache"
            );
            KvQuantPair::symmetric(KvQuantMode::None)
        }
    } else {
        kv_quant
    };

    // Set up TurboQuant KV cache quantization if any pool is quantized.
    // In asymmetric mode (K=BF16, V=Turbo), V still needs rotation + centroids.
    //
    // Boundary layer protection (TurboQuant+): at aggressive compression (turbo2,
    // turbo3), first/last 2 layers are protected at turbo4 precision.  This
    // recovers 37-91% of the quality gap with zero speed penalty.
    let boundary = model::turboquant::BoundaryConfig::default_for(kv_quant.v);
    if kv_quant.is_any_quantized() {
        if let Some(ref bc) = boundary {
            info!(
                first_n = bc.first_n,
                last_n = bc.last_n,
                boundary_mode = ?bc.boundary_mode,
                "TurboQuant+ boundary layer protection enabled",
            );
        }
        model.turbo_ctx = Some(model::turboquant::TurboContext::new(
            &backend,
            kv_quant,
            config.head_dim,
            config.num_attention_heads,
            boundary,
        ));
    }

    // Set up vision encoder if VLM weights were loaded.
    if let Some(vw) = vision_weights {
        if let Some(vc) = &config.vision {
            // Size buffers for max_pixels from config (e.g. 401408 pixels).
            // max_patches = max_pixels / (patch_size²)
            let max_patches = vc.max_pixels / (vc.patch_size * vc.patch_size);
            let bufs = model::vision::alloc_vision_buffers(&backend, vc, max_patches);
            model.vision_weights = Some(vw);
            model.vision_bufs = Some(bufs);
            info!(
                blocks = vc.depth,
                hidden = vc.hidden_size,
                max_patches = max_patches,
                "vision encoder ready",
            );
        }
    }

    // Dynamic KV cache sizing based on remaining GPU memory.
    let gpu_budget = backend.recommended_max_memory();
    let qpb = |m, k| backend.quantized_weight_bytes(m, k);
    let num_blocks = config.recommended_kv_blocks(gpu_budget, is_quantized, kv_quant, &qpb);
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let kv_pool = kv_cache::KvPool::new(
        &backend, num_blocks, kv_dim, config.num_kv_layers(), kv_quant, config.head_dim,
        boundary,
    );

    let weight_mb = config.estimate_weight_bytes(is_quantized, &qpb) as f64 / (1024.0 * 1024.0);
    let kv_mb = kv_pool.total_memory_bytes() as f64 / (1024.0 * 1024.0);
    let max_tokens = kv_pool.max_tokens();
    let bf16_bytes_per_pos = (kv_dim * 2) as f64;
    if kv_quant.is_asymmetric() {
        let v_compression = bf16_bytes_per_pos / kv_pool.v_bytes_per_position() as f64;
        info!(
            k = "BF16",
            v_bits = kv_quant.v.bits(),
            v_compression = format_args!("{:.1}x", v_compression),
            "kv cache: TurboQuant asymmetric (K=BF16, V=turbo)",
        );
    } else if kv_quant.is_any_quantized() {
        let compression = bf16_bytes_per_pos / kv_pool.v_bytes_per_position() as f64;
        info!(bits = kv_quant.v.bits(),
            compression = format_args!("{:.1}x", compression),
            "kv cache: TurboQuant",
        );
    }
    info!(
        weight_mb = format_args!("{:.0}", weight_mb),
        kv_mb = format_args!("{:.0}", kv_mb),
        blocks = num_blocks,
        max_tokens = max_tokens,
        gpu_budget_mb = format_args!("{:.0}", gpu_budget as f64 / (1024.0 * 1024.0)),
        "memory allocation",
    );
    info!(max_active = max_active, "max concurrent sequences");
    info!(ready_s = format_args!("{:.2}", t_start.elapsed().as_secs_f64()), "ready");

    on_ready(&tokenizer, arch);

    let expert_streamer = expert_index.map(|index| {
        let k = config.num_experts_per_tok;
        model::expert_stream::ExpertStreamer::new(&backend, index, k)
    });
    let forward = create_forward(arch, &config, &backend, 1, 0, false, expert_streamer);
    let mut eng = super::Engine::new(model, forward, kv_pool, tokenizer, &backend, max_active);
    run(&mut eng)
}

/// Multi-GPU path: N backends with NCCL, sharded weights across ranks.
#[cfg(feature = "cuda")]
fn load_and_run_multi_gpu(
    model_dir: &Path,
    tp: usize,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    use super::multi_gpu::MultiGpuEngine;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;
    use crate::model::{config, tokenizer};

    let t_start = Instant::now();

    info!(gpus = tp, "tensor parallelism");

    let config = config::ModelConfig::from_file(&model_dir.join("config.json"))?;

    let arch = config.arch()?;
    let tok = tokenizer::Tokenizer::from_file(&model_dir.join("tokenizer.json"), arch)?;

    let t_load = Instant::now();
    let num_blocks = 256;
    let is_prequantized = loader::is_prequantized_model(model_dir);
    let multi = MultiGpuInference::new(model_dir, config.clone(), is_prequantized, tp, num_blocks)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    info!(load_time_s = format_args!("{:.2}", load_secs), "model loaded");
    info!(
        ranks = tp,
        max_active = max_active,
        "multi-GPU inference ready",
    );
    info!(ready_s = format_args!("{:.2}", t_start.elapsed().as_secs_f64()), "ready");

    on_ready(&tok, arch);

    let mut engine = MultiGpuEngine::new(multi, tok, max_active);
    run(&mut engine)
}

/// Non-CUDA stub — multi-GPU requires CUDA + NCCL.
#[cfg(not(feature = "cuda"))]
fn load_and_run_multi_gpu(
    _model_dir: &Path,
    _tp: usize,
    _max_active: usize,
    _on_ready: impl FnOnce(&Tokenizer, ModelArch),
    _run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    anyhow::bail!("multi-GPU tensor parallelism requires the cuda feature")
}
