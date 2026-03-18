// ===========================================================================
// Multi-GPU inference worker for the API server (--tp > 1).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Spawns the multi-GPU inference worker thread.  Constructs a
//   MultiGpuEngine (which wraps MultiGpuInference behind the unified
//   InferenceEngine trait) and hands it to the shared worker loop in
//   api/mod.rs.
//
//   The continuous batching logic (drain requests, prefill, decode, stream
//   tokens, clean up) lives in the shared run_worker_loop() — the same
//   code path as single-GPU.  This module only handles multi-GPU setup:
//   creating N backends with NCCL communicators, loading sharded weights,
//   and building the MultiGpuEngine.
//
// Architecture:
//
//   HTTP handlers → [request channel] → worker thread
//                                             ↓
//                                  run_worker_loop(&mut MultiGpuEngine, rx)
//                                             ↓
//                                  MultiGpuEngine (InferenceEngine trait)
//                                             ↓
//                                  MultiGpuInference (N GPUs via NCCL)
//
// Related files:
//   - engine/multi_gpu.rs — MultiGpuEngine: InferenceEngine impl for N GPUs
//   - engine/mod.rs       — InferenceEngine trait, shared by single + multi
//   - gpu/multi_gpu.rs    — MultiGpuInference: fan-out, NCCL, per-rank state
//   - api/mod.rs          — run_worker_loop (shared), spawn_worker (single-GPU)
// ===========================================================================

use std::path::PathBuf;

use super::WorkerHandle;

/// Spawn the multi-GPU inference worker thread.
///
/// Returns the request channel, shared tokenizer, and model architecture —
/// the same WorkerHandle as the single-GPU path, so the HTTP handlers don't
/// know or care which backend is running.
#[cfg(feature = "cuda")]
pub(crate) fn spawn_worker(
    model_dir: PathBuf,
    quantize: bool,
    tp: usize,
) -> anyhow::Result<WorkerHandle> {
    use std::sync::Arc;
    use crate::engine::multi_gpu::MultiGpuEngine;
    use crate::gpu::multi_gpu::tp::MultiGpuInference;
    use crate::model::{config, tokenizer};
    use super::WorkerRequest;

    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>>(1);

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
            let multi = MultiGpuInference::new(
                &model_dir, config.clone(), quantize, tp, num_blocks,
            )?;

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

/// Non-CUDA stub — multi-GPU requires CUDA + NCCL.
#[cfg(not(feature = "cuda"))]
pub(crate) fn spawn_worker(
    _model_dir: PathBuf,
    _quantize: bool,
    _tp: usize,
) -> anyhow::Result<WorkerHandle> {
    anyhow::bail!("multi-GPU tensor parallelism requires the cuda feature")
}
