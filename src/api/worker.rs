// ===========================================================================
// Inference worker — spawns the model-loading thread for single- or multi-GPU.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Provides a single spawn_worker() entry point that handles both single-GPU
//   and multi-GPU (tensor parallel) inference.  The caller passes `tp` (tensor
//   parallelism degree): tp=1 uses a single GPU, tp>1 fans out across N GPUs
//   via NCCL.  The API server and batch command don't need to know which path
//   is taken — they get back the same WorkerHandle either way.
//
// Why a dedicated worker thread?
//   The model borrows the GPU backend and has mutable scratch buffers.
//   Rather than fighting Arc<Mutex<...>>, we keep ALL GPU state on one
//   std::thread and communicate via channels.  The worker thread outlives
//   the GPU resources it creates (backend, model, KV pool), so Rust's
//   borrow checker is satisfied without any unsafe lifetime tricks.
//
// Related files:
//   - api/mod.rs       — run_worker_loop(), serve(), shared types
//   - engine/mod.rs    — Engine, SingleGpuDispatch, InferenceEngine trait
//   - engine/multi_gpu.rs — MultiGpuEngine, MultiGpuDispatch
// ===========================================================================

use std::path::PathBuf;
use std::sync::Arc;

use crate::engine;
use crate::gpu::{self, GpuCore};
use crate::model;
use crate::model::loader;
use crate::model::{config, kv_cache, tokenizer};

use super::{WorkerHandle, WorkerRequest};

/// Spawn the inference worker thread.
///
/// - `tp == 1`: single-GPU path (Metal or CUDA).
/// - `tp > 1`: multi-GPU tensor parallelism via NCCL (CUDA only).
///
/// Returns a WorkerHandle with the request channel, shared tokenizer, and
/// model architecture.  The caller doesn't need to know which GPU path was taken.
pub(super) fn spawn_worker(
    model_dir: PathBuf,
    quantize: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    if tp > 1 {
        spawn_multi_gpu(model_dir, quantize, tp)
    } else {
        spawn_single_gpu(model_dir, quantize)
    }
}

/// Single-GPU worker: loads one backend, one model, one KV pool.
fn spawn_single_gpu(model_dir: PathBuf, quantize: bool) -> anyhow::Result<WorkerHandle> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>,
    >(1);

    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<()> {
            let backend = gpu::create_backend()?;
            eprintln!("gpu: {}", backend.device_name());

            let loader::LoadedModel {
                config,
                arch,
                tokenizer,
                weights,
            } = loader::load_model(&backend, &model_dir, quantize)?;

            let tokenizer = Arc::new(tokenizer);
            let _ = ready_tx.send(Ok((Arc::clone(&tokenizer), arch)));

            let engine_tokenizer = (*tokenizer).clone();
            let model = model::Model::new(config.clone(), weights, &backend)?;

            // Dynamic KV cache sizing.
            let gpu_budget = backend.recommended_max_memory();
            let qpb = |m, k| backend.quantized_weight_bytes(m, k);
            let num_blocks = config.recommended_kv_blocks(gpu_budget, quantize, &qpb);
            let kv_dim = config.num_key_value_heads * config.head_dim;
            let kv_pool =
                kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_kv_layers());

            let weight_mb =
                config.estimate_weight_bytes(quantize, &qpb) as f64 / (1024.0 * 1024.0);
            let kv_mb = kv_pool.total_memory_bytes() as f64 / (1024.0 * 1024.0);
            let max_tokens = kv_pool.max_tokens();
            eprintln!(
                "memory: {:.0} MB weights, {:.0} MB KV cache ({} blocks, {} max tokens), {:.0} MB GPU budget",
                weight_mb, kv_mb, num_blocks, max_tokens,
                gpu_budget as f64 / (1024.0 * 1024.0),
            );

            let max_active = 32;
            eprintln!("max {} concurrent sequences", max_active);

            let mut eng =
                engine::Engine::new(model, kv_pool, engine_tokenizer, &backend, max_active);

            super::run_worker_loop(&mut eng, request_rx)
        })();

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

    recv_worker_handle(ready_rx, request_tx)
}

/// Multi-GPU worker: loads N backends with NCCL, shards weights across ranks.
#[cfg(feature = "cuda")]
fn spawn_multi_gpu(
    model_dir: PathBuf,
    quantize: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    use crate::engine::multi_gpu::MultiGpuEngine;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;

    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>,
    >(1);

    std::thread::spawn(move || {
        let result = (|| -> anyhow::Result<()> {
            eprintln!("tensor parallelism: {} GPUs", tp);

            let config = config::ModelConfig::from_file(&model_dir.join("config.json"))?;
            let arch = config.arch()?;
            let tokenizer = tokenizer::Tokenizer::from_file(
                &model_dir.join("tokenizer.json"),
                arch,
            )?;
            let tokenizer = Arc::new(tokenizer);

            let num_blocks = 256;
            let multi =
                MultiGpuInference::new(&model_dir, config.clone(), quantize, tp, num_blocks)?;

            let max_active: usize = 32;
            eprintln!(
                "multi-GPU inference ready ({} ranks, max {} concurrent sequences)",
                tp, max_active,
            );

            let _ = ready_tx.send(Ok((Arc::clone(&tokenizer), arch)));

            let engine_tokenizer = (*tokenizer).clone();
            let mut engine = MultiGpuEngine::new(multi, engine_tokenizer, max_active);

            super::run_worker_loop(&mut engine, request_rx)
        })();

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

    recv_worker_handle(ready_rx, request_tx)
}

/// Non-CUDA stub — multi-GPU requires CUDA + NCCL.
#[cfg(not(feature = "cuda"))]
fn spawn_multi_gpu(
    _model_dir: PathBuf,
    _quantize: bool,
    _tp: usize,
) -> anyhow::Result<WorkerHandle> {
    anyhow::bail!("multi-GPU tensor parallelism requires the cuda feature")
}

/// Shared helper: wait for the worker thread to report ready (or error).
fn recv_worker_handle(
    ready_rx: std::sync::mpsc::Receiver<
        Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>,
    >,
    request_tx: std::sync::mpsc::SyncSender<WorkerRequest>,
) -> anyhow::Result<WorkerHandle> {
    match ready_rx.recv() {
        Ok(Ok((tokenizer, arch))) => Ok(WorkerHandle {
            request_tx,
            tokenizer,
            arch,
        }),
        Ok(Err(e)) => anyhow::bail!("worker failed to start: {e}"),
        Err(_) => anyhow::bail!("worker thread died during startup"),
    }
}
