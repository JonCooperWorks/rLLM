// ===========================================================================
// CLI subcommands — one module per command.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the shared infrastructure that all inference commands use:
//   ModelArgs (common CLI flags), load_model_and_run() (model loading +
//   engine setup), and encode_prompt() (prompt → token IDs with chat
//   templates and vision support).
//
//   Individual commands only contain their unique args and generation
//   callback — no duplicated setup logic.
//
// Command modules:
//   run.rs      — `rllm run`      — single-prompt, streaming output
//   batch.rs    — `rllm batch`    — multi-prompt from file, batch output
//   bench.rs    — `rllm bench`    — prefix cache benchmark with stats
//   serve.rs    — `rllm serve`    — HTTP API server
//   quantize.rs — `rllm quantize` — offline weight quantization (bf16 → Q4)
// ===========================================================================

use std::path::PathBuf;

use crate::engine;
use crate::gpu;
use crate::model::config::ModelArch;
use crate::model::vision::ProcessedImage;

mod batch;
mod bench;
mod quantize;
mod run;
mod serve;

pub(crate) use serve::ServeArgs;

#[derive(clap::Subcommand)]
pub(crate) enum Command {
    /// Run inference on a single prompt.
    Run(run::RunArgs),
    /// Run batched inference from a file of prompts.
    Batch(batch::BatchArgs),
    /// Benchmark prefix caching with system prompts.
    Bench(bench::BenchArgs),
    /// Start an OpenAI/Anthropic-compatible API server.
    Serve(serve::ServeArgs),
    /// Pre-quantize model weights from bf16 to Q4 on disk.
    Quantize(quantize::QuantizeArgs),
}

impl Command {
    pub(crate) fn exec(self) -> anyhow::Result<()> {
        match self {
            Command::Run(args) => run::exec(args),
            Command::Batch(args) => batch::exec(args),
            Command::Bench(args) => bench::exec(args),
            Command::Serve(args) => serve::exec(args),
            Command::Quantize(args) => quantize::exec(args),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared model args — common CLI flags for all inference commands.
//
// Each command embeds this via `#[command(flatten)]` and only adds its own
// unique flags (e.g. `--batch-file`, `--prompt`).  Keeps arg definitions
// and their help text in one place.
// ---------------------------------------------------------------------------

#[derive(clap::Args, Clone)]
pub(crate) struct ModelArgs {
    /// Path to model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    pub model: PathBuf,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value = "128")]
    pub max_tokens: usize,

    /// Sampling temperature.  T=0 → greedy (deterministic), T=1 → natural,
    /// T>1 → more random.  Default 1.0.
    #[arg(long, default_value = "1.0")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.  Only sample from tokens whose
    /// cumulative probability mass is within the top p.  Default 0.9.
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Enable chat mode for instruct models.  Wraps prompts in the model's
    /// chat template (auto-detected: Llama 3, ChatML, Gemma 3, Phi, or Mistral).
    #[arg(long)]
    pub chat: bool,

    /// System prompt for chat mode (requires --chat).
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        requires = "chat"
    )]
    pub system: String,

    /// Tensor parallelism: number of GPUs (0 = auto-detect all available).
    #[arg(long, default_value = "0")]
    pub tp: usize,

    /// Stream MoE expert weights from SSD instead of loading all into GPU memory.
    /// Enables running large MoE models (e.g. 397B) that don't fit in VRAM.
    #[arg(long)]
    pub stream_experts: bool,

    /// Path to an image file (JPEG, PNG, or WebP) to include with the prompt.
    /// Requires --chat mode.  The image is attached to the user message so
    /// vision models can process it alongside the text prompt.
    #[arg(long, requires = "chat")]
    pub image: Option<PathBuf>,

    /// KV cache quantization mode.  TurboQuant applies a random orthogonal
    /// rotation followed by optimal scalar quantization (Max-Lloyd) per
    /// coordinate, achieving near-optimal distortion rates.
    /// "turbo4" (default): 4-bit, ~4x compression, quality-neutral.
    /// "turbo3": 3-bit, ~5x compression, near-lossless.
    /// "turbo2": 2-bit, ~7.5x compression, marginal quality loss.
    /// "none": BF16 (no quantization, for debugging/benchmarking).
    #[arg(long, default_value = "turbo4")]
    pub kv_quant: String,
}

// ---------------------------------------------------------------------------
// Shared setup — model loading, engine construction, prompt encoding.
//
// Every inference command calls load_model_and_run(), which handles:
//   1. Parse kv_quant string → KvQuantMode
//   2. Resolve TP (auto-detect GPU count, macOS single-GPU warning)
//   3. Log sampling strategy
//   4. Preprocess vision images if --image provided
//   5. Call engine::loader::load_and_run_ext() with the caller's closure
//
// The caller's closure receives the engine, the model arch, and any
// preprocessed images — it only needs to encode prompts and drive generation.
// ---------------------------------------------------------------------------

/// Load a model and run a generation callback.
///
/// Handles all shared setup (kv_quant parsing, TP resolution, image
/// preprocessing, engine construction).  The callback receives the engine,
/// architecture, and preprocessed images.
pub(crate) fn load_model_and_run(
    args: &ModelArgs,
    max_active: usize,
    run: impl FnOnce(&mut dyn engine::InferenceEngine, ModelArch, &[ProcessedImage]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mut tp = args.tp;

    // Resolve --tp 0 → auto-detect available GPUs.
    if tp == 0 {
        tp = gpu::device_count();
        eprintln!("auto-detected {} GPU(s)", tp);
    }

    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    #[cfg(not(feature = "cuda"))]
    if tp > 1 {
        eprintln!(
            "warning: --tp {} ignored (multi-GPU requires CUDA + NCCL), using single GPU",
            tp
        );
        tp = 1;
    }

    // Log sampling strategy so the user knows what to expect.
    if args.temperature == 0.0 {
        eprintln!("sampling: greedy (temperature=0)");
    } else {
        eprintln!(
            "sampling: temperature={}, top_p={}",
            args.temperature, args.top_p
        );
    }

    let kv_quant = crate::model::turboquant::KvQuantMode::from_str(&args.kv_quant)
        .unwrap_or_else(|| {
            eprintln!(
                "error: invalid --kv-quant value '{}', expected: turbo4, turbo3, turbo2, none",
                args.kv_quant
            );
            std::process::exit(1);
        });

    // Read and preprocess image file if provided.
    let processed_images: Vec<ProcessedImage> = if let Some(ref image_path) = args.image {
        let data = std::fs::read(image_path).map_err(|e| {
            anyhow::anyhow!("failed to read image '{}': {e}", image_path.display())
        })?;
        eprintln!("image: {} ({} bytes)", image_path.display(), data.len());
        let vc = crate::model::config::ModelConfig::from_file(
            &args.model.join("config.json"),
        )?
        .vision
        .ok_or_else(|| {
            anyhow::anyhow!("--image requires a vision model (no vision_config in config.json)")
        })?;
        let processed = crate::model::vision::preprocess_image(&data, &vc)?;
        eprintln!(
            "image preprocessed: {}x{} patches, {} vision tokens",
            processed.grid_h, processed.grid_w, processed.num_vision_tokens
        );
        vec![processed]
    } else {
        Vec::new()
    };

    use std::cell::Cell;
    let arch_cell: Cell<Option<ModelArch>> = Cell::new(None);

    engine::loader::load_and_run_ext(
        &args.model,
        args.stream_experts,
        tp,
        kv_quant,
        max_active,
        |_tok, arch| {
            arch_cell.set(Some(arch));
        },
        |eng| {
            let arch = arch_cell.get().unwrap();
            run(eng, arch, &processed_images)
        },
    )
}

/// Encode a prompt into token IDs with optional chat template and vision.
///
/// Handles three cases:
///   1. Chat mode with image → build Message array with ImageData
///   2. Chat mode without image → use encode_prompt with system
///   3. Raw mode → encode_prompt without system
pub(crate) fn encode_prompt(
    eng: &dyn engine::InferenceEngine,
    arch: ModelArch,
    prompt: &str,
    chat: bool,
    system: &str,
    image_data: Option<&[u8]>,
) -> anyhow::Result<Vec<u32>> {
    if chat && image_data.is_some() {
        use crate::model::chat::{self, ImageData, Message};
        let messages = vec![
            Message {
                role: "system".into(),
                content: system.to_string(),
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
            Message {
                role: "user".into(),
                content: prompt.to_string(),
                tool_calls: None,
                tool_call_id: None,
                images: Some(vec![ImageData {
                    data: image_data.unwrap().to_vec(),
                }]),
            },
        ];
        let formatted = chat::format_chat(arch, &messages);
        eng.tokenizer().encode_chat(&formatted)
    } else {
        let system_ref = chat.then(|| system);
        eng.tokenizer().encode_prompt(prompt, arch, system_ref)
    }
}
