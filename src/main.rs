// ===========================================================================
// rLLM — Rust LLM inference engine.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Entry point for the inference engine.  Parses CLI arguments, wires up all
//   components (config, backend, tokenizer, weights, model), runs the forward
//   pass for each prompt token (prefill), then enters the generation loop.
//
// Inference pipeline (high level):
//   1. Load config.json → model architecture parameters
//   2. Create GPU backend → compiles Metal shaders, creates pipeline states
//   3. Load tokenizer.json → BPE encoder/decoder
//   4. Load model.safetensors → upload ~2.6 GB of weights to GPU
//   5. Encode prompt → token IDs (e.g. "Hello" → [128000, 9906])
//   6. Prefill: forward pass each prompt token (builds up KV cache)
//   7. Generate: forward → sample → decode → print, repeat until EOS/max
//
// Prefill vs. generation:
//   Prefill processes the prompt tokens one at a time.  Each token's K and V
//   vectors are stored in the KV cache but only the LAST token's logits matter
//   (they predict the first generated token).  This is O(n²) in prompt length
//   because each successive token attends to all previous ones.
//
//   Generation processes one token at a time.  Each forward pass attends to
//   the full KV cache (all prompt + generated tokens so far).  The sampler
//   picks the next token (greedy if T=0, or temperature + top-p sampling).
//
// Streaming output:
//   Tokens are printed as they're generated (flush after each token), giving
//   the appearance of the model "typing".  This is the standard UX for LLM
//   inference — users see output appearing word by word.
// ===========================================================================

mod chat;
mod config;
mod gpu;
mod loader;
mod model;
mod sampler;
mod tokenizer;

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use gpu::GpuBackend; // Import the trait so we can call .device_name() on the backend.

/// CLI argument definition using clap's derive API.
#[derive(Parser)]
#[command(name = "rllm", about = "Rust LLM inference engine")]
struct Args {
    /// Path to model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,

    /// Prompt text to complete.
    #[arg(long)]
    prompt: String,

    /// Maximum number of tokens to generate.
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

    /// Enable chat mode for instruct models.  Wraps --prompt in the Llama 3
    /// chat template with role markers and a generation prompt.
    #[arg(long)]
    chat: bool,

    /// System prompt for chat mode (ignored without --chat).
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Main inference pipeline.  Separated from main() for clean error propagation
/// via anyhow (main() returns ExitCode, not Result).
fn run(args: Args) -> anyhow::Result<()> {
    // --- Step 1: Load model configuration ---
    let config = config::LlamaConfig::from_file(&args.model.join("config.json"))?;
    eprintln!(
        "loaded config: {} layers, {} heads, hidden_size={}",
        config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // --- Step 2: Initialise GPU backend ---
    // On macOS: creates a Metal device, command queue, and compiles all 8 shaders.
    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    // --- Step 3: Load tokenizer ---
    let tokenizer = tokenizer::Tokenizer::from_file(&args.model.join("tokenizer.json"))?;
    eprintln!("tokenizer loaded");

    // --- Step 4: Load model weights ---
    // mmap the safetensors file and upload all tensors to GPU memory.
    let weights = loader::load_weights(&backend, &args.model, &config, args.quantize)?;
    eprintln!(
        "weights loaded{}",
        if args.quantize { " (Q4 quantised)" } else { "" }
    );

    // Log sampling strategy so the user knows what to expect.
    if args.temperature == 0.0 {
        eprintln!("sampling: greedy (temperature=0)");
    } else {
        eprintln!("sampling: temperature={}, top_p={}", args.temperature, args.top_p);
    }

    // --- Step 5: Create model (allocates KV cache + scratch buffers) ---
    let mut llama = model::LlamaModel::new(config, weights, &backend)?;

    // --- Step 6: Encode prompt ---
    // In chat mode, wrap the user's prompt in the Llama 3 chat template
    // before tokenising.  This adds role markers and special tokens that
    // instruct models expect.  Without --chat, the raw prompt is used
    // (suitable for base/completion models).
    let prompt_tokens = if args.chat {
        let messages = vec![
            chat::Message { role: "system".into(), content: args.system.clone() },
            chat::Message { role: "user".into(), content: args.prompt.clone() },
        ];
        let formatted = chat::format_llama3(&messages);
        eprintln!("chat template applied");
        tokenizer.encode_chat(&formatted)?
    } else {
        tokenizer.encode(&args.prompt)?
    };
    eprintln!("prompt tokens: {:?} ({})", &prompt_tokens, prompt_tokens.len());

    // --- Step 7: Prefill ---
    // Process each prompt token through the model.  This populates the KV
    // cache with the prompt's K/V vectors.  Only the logits from the LAST
    // prompt token are used (they predict the first generated token).
    let prefill_start = std::time::Instant::now();
    for &token_id in &prompt_tokens {
        llama.forward(token_id)?;
    }
    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tps = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
    eprintln!(
        "prefill: {} tokens in {:.1?} ({:.1} tok/s)",
        prompt_tokens.len(), prefill_elapsed, prefill_tps
    );

    // --- Step 8: Generation loop ---
    // Sample the first token from the prefill logits, then enter the
    // autoregressive loop: forward → sample → print → repeat.
    let gen_start = std::time::Instant::now();
    let mut gen_count: usize = 0;
    let mut rng = rand::rng();
    let mut next_token = sampler::sample(
        &backend, llama.logits(), args.temperature, args.top_p, &mut rng,
    )?;
    for _ in 0..args.max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }
        gen_count += 1;
        // Decode and print the token immediately (streaming output).
        let text = tokenizer.decode(&[next_token])?;
        print!("{text}");
        io::stdout().flush()?;

        // Forward pass for the new token (updates KV cache).
        llama.forward(next_token)?;
        // Sample the next token from the updated logits.
        next_token = sampler::sample(
            &backend, llama.logits(), args.temperature, args.top_p, &mut rng,
        )?;
    }
    println!();
    let gen_elapsed = gen_start.elapsed();
    let gen_tps = gen_count as f64 / gen_elapsed.as_secs_f64();
    eprintln!(
        "generation: {} tokens in {:.1?} ({:.1} tok/s)",
        gen_count, gen_elapsed, gen_tps
    );

    Ok(())
}
