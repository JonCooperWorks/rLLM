// ===========================================================================
// Shared model loading — eliminates duplication across run, batch, and serve.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Every inference path (single-prompt, batch, API server) needs the same
//   four things: model config, architecture tag, tokenizer, and GPU-resident
//   weights.  Before this module existed, that ~20-line loading sequence was
//   copy-pasted in three places.  Now it lives here once.
//
// Why the backend isn't included:
//   Model<'a, B> borrows the backend with lifetime 'a.  The backend must
//   live on the caller's stack (in run()/run_batched()) or on the worker
//   thread's stack (in serve()).  If we created it inside load_model() and
//   tried to return it alongside things that borrow it, we'd fight the
//   borrow checker.  So the caller creates the backend first, then passes
//   a reference here.
// ===========================================================================

use std::path::Path;

use crate::config::{ModelArch, ModelConfig};
use crate::gpu::GpuBackend;
use crate::loader::{self, ModelWeights};
use crate::tokenizer::Tokenizer;

/// Everything needed to run inference, loaded from a model directory.
pub(crate) struct LoadedModel<B: GpuBackend> {
    pub config: ModelConfig,
    pub arch: ModelArch,
    pub tokenizer: Tokenizer,
    pub weights: ModelWeights<B>,
}

/// Load config, tokenizer, and weights from a model directory.
///
/// The backend must be created by the caller and passed in (see module
/// doc for why).  Logs progress to stderr so the user sees what's happening.
pub(crate) fn load_model<B: GpuBackend>(
    backend: &B,
    model_dir: &Path,
    quantize: bool,
) -> anyhow::Result<LoadedModel<B>> {
    let config = ModelConfig::from_file(&model_dir.join("config.json"))?;
    let arch = config.arch()?;
    eprintln!(
        "loaded config: {:?}, {} layers, {} heads, hidden_size={}",
        arch, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"), arch)?;
    eprintln!("tokenizer loaded");

    let weights = loader::load_weights(backend, model_dir, &config, quantize)?;
    eprintln!(
        "weights loaded{}",
        if quantize { " (Q4 quantised)" } else { "" }
    );

    Ok(LoadedModel { config, arch, tokenizer, weights })
}
