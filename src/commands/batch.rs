// ===========================================================================
// `rllm batch` — Batched inference from a file of prompts.
//
// Reads prompts (one per line), submits them all to the engine, and runs
// the continuous batching loop until all sequences are complete.  Uses the
// scheduler to manage concurrent sequences and the paged KV cache for
// memory-efficient multi-sequence inference.
// ===========================================================================

use std::path::PathBuf;

use crate::engine;
use crate::engine::scheduler;
use crate::gpu::{self, GpuCore};
use crate::model;
use crate::model::{kv_cache, loader};

#[derive(clap::Args)]
pub(crate) struct BatchArgs {
    /// Path to model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,

    /// Path to a batch file (one prompt per line).
    #[arg(long)]
    batch_file: PathBuf,

    /// Maximum number of tokens to generate per prompt.
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Quantise weights to Q4 on load (reduces memory ~3.2x, speeds up matmul).
    #[arg(long)]
    quantize: bool,

    /// Sampling temperature.  T=0 → greedy (deterministic), T=1 → natural,
    /// T>1 → more random.  Default 1.0.
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.  Only sample from tokens whose
    /// cumulative probability mass is within the top p.  Default 0.9.
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Enable chat mode for instruct models.  Wraps prompts in the model's
    /// chat template (auto-detected: Llama 3 or ChatML for Qwen 2.5).
    #[arg(long)]
    chat: bool,

    /// System prompt for chat mode (requires --chat).
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        requires = "chat"
    )]
    system: String,
}

pub(crate) fn exec(args: BatchArgs) -> anyhow::Result<()> {
    // --- Load GPU backend + model ---
    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    let loader::LoadedModel {
        config,
        arch,
        tokenizer,
        weights,
    } = loader::load_model(&backend, &args.model, args.quantize)?;

    // Read prompts from file.
    let prompts_text = std::fs::read_to_string(&args.batch_file)?;
    let prompts: Vec<&str> = prompts_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    eprintln!(
        "batch: {} prompts from {}",
        prompts.len(),
        args.batch_file.display()
    );

    // Create paged KV pool: enough blocks for all prompts + generation.
    // Each prompt needs ceil(prompt_len / 16) blocks for prefill, plus
    // ceil(max_tokens / 16) blocks for generation.
    let max_blocks_per_seq =
        (512 + args.max_tokens + kv_cache::BLOCK_SIZE - 1) / kv_cache::BLOCK_SIZE;
    let num_blocks = prompts.len() * max_blocks_per_seq;
    eprintln!(
        "KV pool: {} blocks ({} per sequence)",
        num_blocks, max_blocks_per_seq
    );

    let kv_dim = config.num_key_value_heads * config.head_dim;
    let kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_kv_layers());

    let model = model::Model::new(config, weights, &backend)?;

    // Create scheduler and engine.
    let sched = scheduler::Scheduler::new(kv_pool, prompts.len());
    let mut eng = engine::Engine::new(
        model,
        sched,
        tokenizer,
        &backend,
    );

    // Submit all prompts.
    let system = args.chat.then(|| args.system.as_str());
    for prompt_text in &prompts {
        let tokens = eng.tokenizer.encode_prompt(prompt_text, arch, system)?;
        eng.add_request(tokens, args.max_tokens, args.temperature, args.top_p);
    }

    // Run engine loop.
    let start = std::time::Instant::now();
    let mut total_generated = 0usize;

    while eng.has_work() {
        let output = eng.step()?;
        for seq in &output.finished {
            total_generated += seq.tokens.len();
            println!("--- sequence {} ({} tokens) ---", seq.id, seq.tokens.len());
            println!("{}", seq.text);
        }
    }

    let elapsed = start.elapsed();
    let tps = total_generated as f64 / elapsed.as_secs_f64();
    eprintln!(
        "batch complete: {} tokens from {} sequences in {:.1?} ({:.1} tok/s total throughput)",
        total_generated,
        prompts.len(),
        elapsed,
        tps
    );

    Ok(())
}
