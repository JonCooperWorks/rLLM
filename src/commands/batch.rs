// ===========================================================================
// `rllm batch` — Batched inference from a file of prompts.
//
// Reads prompts (one per line), submits them all to the engine, and runs
// the continuous batching loop until all sequences are complete.  Uses
// engine::loader::load_and_run() for model loading and engine construction.
// ===========================================================================

use std::path::PathBuf;

use std::cell::Cell;

use crate::engine;
use crate::model::config::ModelArch;

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
    /// chat template (auto-detected: Llama 3, ChatML, Gemma 3, Phi, or Mistral).
    #[arg(long)]
    chat: bool,

    /// System prompt for chat mode (requires --chat).
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        requires = "chat"
    )]
    system: String,

    /// Tensor parallelism: number of GPUs to use (default 1).
    #[arg(long, default_value = "1")]
    tp: usize,
}

pub(crate) fn exec(args: BatchArgs) -> anyhow::Result<()> {
    // Read prompts from file.
    let prompts_text = std::fs::read_to_string(&args.batch_file)?;
    let prompts: Vec<String> = prompts_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    eprintln!(
        "batch: {} prompts from {}",
        prompts.len(),
        args.batch_file.display()
    );

    let max_active = prompts.len();
    let max_tokens = args.max_tokens;
    let temperature = args.temperature;
    let top_p = args.top_p;
    let chat = args.chat;
    let system = args.system.clone();

    let arch_cell: Cell<Option<ModelArch>> = Cell::new(None);

    engine::loader::load_and_run(
        &args.model,
        args.quantize,
        args.tp,
        max_active,
        |_tok, arch| { arch_cell.set(Some(arch)); },
        |eng| {
            let arch = arch_cell.get().unwrap();
            // Submit all prompts.
            let system_ref = chat.then(|| system.as_str());
            for prompt_text in &prompts {
                let tokens = eng.tokenizer().encode_prompt(prompt_text, arch, system_ref)?;
                eng.add_request(tokens, max_tokens, temperature, top_p);
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
        },
    )
}
