// ===========================================================================
// API worker thread — spawns the inference worker and returns a WorkerHandle.
//
// This is a thin wrapper around engine::worker::load_and_run().  It handles
// only the API-specific concerns: thread spawning, channel plumbing, and the
// ready signal.  All model loading and engine construction lives in the
// engine module.
//
// Related files:
//   - engine/worker.rs — load_and_run() (model loading + engine construction)
//   - api/mod.rs       — run_worker_loop(), serve(), shared types
// ===========================================================================

use std::path::PathBuf;
use std::sync::Arc;

use crate::model::{config, tokenizer};

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
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<WorkerRequest>(8);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<
        Result<(Arc<tokenizer::Tokenizer>, config::ModelArch), String>,
    >(1);

    std::thread::spawn(move || {
        let max_active = 32;

        let result = crate::engine::worker::load_and_run(
            &model_dir,
            quantize,
            tp,
            max_active,
            |tok, arch| {
                let _ = ready_tx.send(Ok((Arc::new(tok.clone()), arch)));
            },
            |eng| super::run_worker_loop(eng, request_rx),
        );

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
