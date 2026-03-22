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

use crate::gpu::{self, GpuCore};
use crate::model;
use crate::model::config::ModelArch;
use crate::model::tokenizer::Tokenizer;
use crate::model::{kv_cache, loader};

use super::InferenceEngine;

/// Load a model and run a function with the constructed InferenceEngine.
///
/// Handles both single-GPU (`tp == 1`) and multi-GPU (`tp > 1`) paths.
/// The caller doesn't need to know which path is taken.
///
/// - `on_ready` is called after the model is loaded but before the loop starts.
///   The API server uses this to send the tokenizer/arch back to the main thread.
/// - `run` is called with the engine — this is the caller's inference loop.
pub(crate) fn load_and_run(
    model_dir: &Path,
    tp: usize,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    load_and_run_ext(model_dir, false, tp, max_active, on_ready, run)
}

/// Extended version with expert streaming support.
pub(crate) fn load_and_run_ext(
    model_dir: &Path,
    stream_experts: bool,
    tp: usize,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    if tp > 1 {
        load_and_run_multi_gpu(model_dir, tp, max_active, on_ready, run)
    } else {
        load_and_run_single_gpu(model_dir, stream_experts, max_active, on_ready, run)
    }
}

/// Single-GPU path: one backend, one model, one KV pool.
fn load_and_run_single_gpu(
    model_dir: &Path,
    stream_experts: bool,
    max_active: usize,
    on_ready: impl FnOnce(&Tokenizer, ModelArch),
    run: impl FnOnce(&mut dyn InferenceEngine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let t_start = Instant::now();

    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    let t_load = Instant::now();
    let loader::LoadedModel {
        config,
        arch,
        tokenizer,
        weights,
        expert_index,
        is_quantized,
    } = loader::load_model(&backend, model_dir, stream_experts)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("loaded in {:.2}s", load_secs);

    let mut model = model::Model::new(config.clone(), weights, &backend)?;

    // Set up expert streamer if SSD streaming was requested.
    if let Some(index) = expert_index {
        let k = config.num_experts_per_tok;
        let streamer = model::expert_stream::ExpertStreamer::new(&backend, index, k);
        model.expert_streamer = Some(streamer);
        eprintln!(
            "expert streaming: {} slots, {} MB per expert load",
            k,
            config.moe_intermediate_size * config.hidden_size * 2 * 3 / 1024 / 1024,
        );
    }

    // Dynamic KV cache sizing based on remaining GPU memory.
    let gpu_budget = backend.recommended_max_memory();
    let qpb = |m, k| backend.quantized_weight_bytes(m, k);
    let num_blocks = config.recommended_kv_blocks(gpu_budget, is_quantized, &qpb);
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_kv_layers());

    let weight_mb = config.estimate_weight_bytes(is_quantized, &qpb) as f64 / (1024.0 * 1024.0);
    let kv_mb = kv_pool.total_memory_bytes() as f64 / (1024.0 * 1024.0);
    let max_tokens = kv_pool.max_tokens();
    eprintln!(
        "memory: {:.0} MB weights, {:.0} MB KV cache ({} blocks, {} max tokens), {:.0} MB GPU budget",
        weight_mb,
        kv_mb,
        num_blocks,
        max_tokens,
        gpu_budget as f64 / (1024.0 * 1024.0),
    );
    eprintln!("max {} concurrent sequences", max_active);
    eprintln!("ready in {:.2}s", t_start.elapsed().as_secs_f64());

    on_ready(&tokenizer, arch);

    let mut eng = super::Engine::new(model, kv_pool, tokenizer, &backend, max_active);
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

    eprintln!("tensor parallelism: {} GPUs", tp);

    let config = config::ModelConfig::from_file(&model_dir.join("config.json"))?;

    let arch = config.arch()?;
    let tok = tokenizer::Tokenizer::from_file(&model_dir.join("tokenizer.json"), arch)?;

    let t_load = Instant::now();
    let num_blocks = 256;
    let is_prequantized = loader::is_prequantized_model(model_dir);
    let multi = MultiGpuInference::new(model_dir, config.clone(), is_prequantized, tp, num_blocks)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("loaded in {:.2}s", load_secs);
    eprintln!(
        "multi-GPU inference ready ({} ranks, max {} concurrent sequences)",
        tp, max_active,
    );
    eprintln!("ready in {:.2}s", t_start.elapsed().as_secs_f64());

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
