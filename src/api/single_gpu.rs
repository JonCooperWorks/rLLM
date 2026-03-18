// ===========================================================================
// Single-GPU inference worker for the API server (--tp 1 or Metal).
//
// Spawns a worker thread that creates a GPU backend, loads the model,
// builds an Engine<B> (which implements InferenceEngine), and hands it
// to the shared run_worker_loop().  Returns a WorkerHandle so the HTTP
// server can send pre-tokenized requests without knowing the GPU details.
//
// Related files:
//   - api/mod.rs          — run_worker_loop (shared), serve(), shared types
//   - api/multi_gpu.rs    — multi-GPU equivalent (MultiGpuEngine)
//   - engine/mod.rs       — Engine + InferenceEngine trait
// ===========================================================================

use std::path::PathBuf;
use std::sync::Arc;

use crate::engine;
use crate::gpu::{self, GpuCore};
use crate::model;
use crate::model::loader;
use crate::model::{config, kv_cache, tokenizer};

use super::{WorkerHandle, WorkerRequest};

/// Spawn the single-GPU inference worker thread.
///
/// Loads the model on a dedicated thread (GPU resources tied to thread
/// lifetime), builds an Engine, and runs the shared worker loop.
pub(super) fn spawn_worker(model_dir: PathBuf, quantize: bool) -> anyhow::Result<WorkerHandle> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>>(1);

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

            let weight_mb = config.estimate_weight_bytes(quantize, &qpb) as f64 / (1024.0 * 1024.0);
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

            let max_active = 32;
            eprintln!("max {} concurrent sequences", max_active);

            let mut eng = engine::Engine::new(
                model,
                kv_pool,
                engine_tokenizer,
                &backend,
                max_active,
            );

            super::run_worker_loop(&mut eng, request_rx)
        })();

        if let Err(e) = result {
            let _ = ready_tx.send(Err(format!("{e:#}")));
        }
    });

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
