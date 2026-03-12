// ===========================================================================
// rLLM — Rust LLM inference engine.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Entry point for the inference engine.  Parses CLI arguments, wires up all
//   components (config, backend, tokenizer, weights, model), and runs inference.
//
// Three modes:
//
//   1. SINGLE-SEQUENCE MODE (`rllm run --prompt "..."`)
//      Processes one prompt, generates tokens, streams output.  Uses:
//        - Batched prefill: entire prompt in one GEMM forward pass (fast)
//        - Single-token decode: one mat-vec forward per generated token
//        - Paged KV cache: allocates memory on demand in 16-token blocks
//
//   2. BATCHED MODE (`rllm run --batch-file prompts.txt`)
//      Processes multiple prompts concurrently via the engine's continuous
//      batching loop.  Uses the scheduler to manage sequences.
//
//   3. SERVER MODE (`rllm serve --model path --port 8080`)
//      Starts an HTTP server with OpenAI and Anthropic compatible APIs.
//      Supports both streaming (SSE) and non-streaming responses.
//
// Prefill vs. generation (the two phases of inference):
//
//   PREFILL processes all prompt tokens through the model to build up the
//   KV cache.  Using batched prefill (GEMM), the entire prompt is processed
//   in one forward pass — shifting from bandwidth-bound (mat-vec) to
//   compute-bound (mat-mat).  This is 3-10x faster than token-by-token.
//   Only the LAST token's logits are needed (they predict the first
//   generated token).
//
//   GENERATION produces one new token at a time.  Each forward pass runs a
//   single token through the model (mat-vec projections), attending to the
//   full KV cache.  The sampler picks the next token, which is printed
//   immediately (streaming output).
//
// Why batched prefill matters:
//   For a 100-token prompt with mat-vec: 100 separate forward passes, each
//   loading the full weight matrix from memory.  Total memory traffic:
//   100 × (weights_size).  Each pass does M×K multiply-adds → O(1) FLOPs/byte.
//
//   With GEMM: ONE forward pass, loading the weight matrix once.  The input
//   is [100, K] instead of [K].  Arithmetic intensity: 100 FLOPs/byte.
//   The GPU shifts from waiting on memory to actually computing — the
//   weight matrix is reused across all 100 input rows.
// ===========================================================================

mod api;
mod chat;
mod config;
mod engine;
mod gpu;
mod kv_cache;
mod loader;
mod model;
mod sampler;
mod scheduler;
mod tokenizer;

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use gpu::GpuBackend; // Import the trait so we can call .device_name() on the backend.

/// CLI definition using clap subcommands.
#[derive(Parser)]
#[command(name = "rllm", about = "Rust LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run inference on a single prompt or batch file.
    Run(RunArgs),
    /// Start an OpenAI/Anthropic-compatible API server.
    Serve(api::ServeArgs),
}

/// Arguments for the `run` subcommand (single-sequence and batched inference).
#[derive(clap::Args)]
struct RunArgs {
    /// Path to model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,

    /// Prompt text to complete (required unless --batch-file is used).
    #[arg(long, default_value = "")]
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

    /// Enable chat mode for instruct models.  Wraps --prompt in the model's
    /// chat template (auto-detected: Llama 3 or ChatML for Qwen 2.5).
    #[arg(long)]
    chat: bool,

    /// System prompt for chat mode (ignored without --chat).
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Path to a batch file (one prompt per line) for continuous batching.
    /// Each line is processed concurrently through the engine.
    #[arg(long)]
    batch_file: Option<std::path::PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Run(args) => {
            if args.batch_file.is_some() {
                run_batched(args)
            } else {
                run(args)
            }
        }
        Command::Serve(args) => api::serve(args),
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Main inference pipeline.  Separated from main() for clean error propagation
/// via anyhow (main() returns ExitCode, not Result).
fn run(args: RunArgs) -> anyhow::Result<()> {
    // --- Step 1: Load model configuration + detect architecture ---
    let config = config::ModelConfig::from_file(&args.model.join("config.json"))?;
    let arch = config.arch()?;
    eprintln!(
        "loaded config: {:?}, {} layers, {} heads, hidden_size={}",
        arch, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // --- Step 2: Initialise GPU backend ---
    // On macOS: creates a Metal device, command queue, and compiles all 8 shaders.
    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    // --- Step 3: Load tokenizer (model-aware for BOS/EOS tokens) ---
    let tokenizer = tokenizer::Tokenizer::from_file(&args.model.join("tokenizer.json"), arch)?;
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

    // --- Step 5: Create model + paged KV cache + prefill buffers ---
    //
    // The model holds weights and single-token scratch buffers (for decode).
    // The paged KV cache allocates memory on demand in 16-token blocks —
    // no wasted memory for short sequences, and it enables batched prefill.
    //
    // PrefillBuffers are batch-sized scratch tensors (up to 1024 tokens)
    // used by the GEMM-based prefill forward pass.  Allocated once, reused.
    let mut model = model::Model::new(config.clone(), weights, &backend)?;

    let num_blocks = 256; // 256 blocks × 16 tokens/block = 4096 max positions.
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let mut kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_hidden_layers);
    let mut seq_state = kv_pool.new_sequence(&backend);
    let prefill_bufs = model::PrefillBuffers::new(&backend, &config, 1024);
    eprintln!(
        "KV cache: {} blocks × {} tokens/block, batched prefill up to 1024 tokens",
        num_blocks, kv_cache::BLOCK_SIZE
    );

    // --- Step 6: Encode prompt ---
    // In chat mode, wrap the user's prompt in the model's chat template
    // before tokenising.  The template is auto-detected from the architecture
    // (Llama 3 format or ChatML for Qwen 2.5).  Without --chat, the raw
    // prompt is used (suitable for base/completion models).
    let prompt_tokens = if args.chat {
        let messages = vec![
            chat::Message { role: "system".into(), content: args.system.clone() },
            chat::Message { role: "user".into(), content: args.prompt.clone() },
        ];
        let formatted = chat::format_chat(arch, &messages);
        eprintln!("chat template applied ({:?})", arch);
        tokenizer.encode_chat(&formatted)?
    } else {
        tokenizer.encode(&args.prompt)?
    };
    eprintln!("prompt tokens: {:?} ({})", &prompt_tokens, prompt_tokens.len());

    // --- Step 7: Prefill ---
    // Process the entire prompt in one batched forward pass using GEMM.
    //
    // Three steps: allocate KV blocks → upload block table → GEMM forward.
    //
    //   ensure_slots: pre-allocates enough physical KV blocks for the whole
    //     prompt (ceil(prompt_len / 16) blocks from the free list).
    //   sync_block_table: uploads the logical→physical block mapping to GPU
    //     so the attention kernel can find the right memory locations.
    //   forward_prefill_paged: the actual GEMM-based forward pass — all
    //     projections use mat-mat instead of mat-vec.
    //   advance_by: records that these positions are now filled in the cache.
    let prefill_start = std::time::Instant::now();
    seq_state.ensure_slots(&mut kv_pool, prompt_tokens.len())?;
    seq_state.sync_block_table(&backend);
    model.forward_prefill_paged(&prompt_tokens, &kv_pool, &seq_state, &prefill_bufs)?;
    seq_state.advance_by(prompt_tokens.len());

    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tps = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
    eprintln!(
        "prefill: {} tokens in {:.1?} ({:.1} tok/s)",
        prompt_tokens.len(), prefill_elapsed, prefill_tps
    );

    // --- Step 8: Generation loop ---
    // Sample the first token from the prefill logits, then enter the
    // autoregressive decode loop: forward → sample → print → repeat.
    //
    // Each iteration does:
    //   1. ensure_slot: allocate a new KV block if the current one is full
    //   2. forward_single_paged: mat-vec forward pass for one token
    //   3. advance: mark this position as filled in the KV cache
    //   4. sample: pick the next token from the logits distribution
    //
    // Generation is memory-bound (mat-vec), not compute-bound (mat-mat),
    // because each token is a single vector — no batch dimension to amortise
    // the weight matrix loads.  This is inherent to autoregressive decoding.
    let gen_start = std::time::Instant::now();
    let mut gen_count: usize = 0;
    let mut rng = rand::rng();
    let mut next_token = sampler::sample(
        &backend, model.logits(), args.temperature, args.top_p, &mut rng,
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

        // Single-token forward pass into the paged KV cache.
        seq_state.ensure_slot(&mut kv_pool)?;
        seq_state.sync_block_table(&backend);
        model.forward_single_paged(next_token, &kv_pool, &seq_state)?;
        seq_state.advance();

        // Sample the next token from the updated logits.
        next_token = sampler::sample(
            &backend, model.logits(), args.temperature, args.top_p, &mut rng,
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

/// Batched inference pipeline: process multiple prompts concurrently.
///
/// Reads prompts from a file (one per line), submits them all to the engine,
/// and runs the continuous batching loop until all sequences are complete.
fn run_batched(args: RunArgs) -> anyhow::Result<()> {
    let batch_file = args.batch_file.as_ref().unwrap();

    // Load config, backend, tokenizer, weights (same as single mode).
    let config = config::ModelConfig::from_file(&args.model.join("config.json"))?;
    let arch = config.arch()?;
    eprintln!(
        "loaded config: {:?}, {} layers, {} heads, hidden_size={}",
        arch, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    let tokenizer = tokenizer::Tokenizer::from_file(&args.model.join("tokenizer.json"), arch)?;
    eprintln!("tokenizer loaded");

    let weights = loader::load_weights(&backend, &args.model, &config, args.quantize)?;
    eprintln!(
        "weights loaded{}",
        if args.quantize { " (Q4 quantised)" } else { "" }
    );

    // Read prompts from file.
    let prompts_text = std::fs::read_to_string(batch_file)?;
    let prompts: Vec<&str> = prompts_text.lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    eprintln!("batch: {} prompts from {}", prompts.len(), batch_file.display());

    // Create paged KV pool: enough blocks for all prompts + generation.
    // Each prompt needs ceil(prompt_len / 16) blocks for prefill, plus
    // ceil(max_tokens / 16) blocks for generation.
    let max_blocks_per_seq = (512 + args.max_tokens + kv_cache::BLOCK_SIZE - 1) / kv_cache::BLOCK_SIZE;
    let num_blocks = prompts.len() * max_blocks_per_seq;
    eprintln!("KV pool: {} blocks ({} per sequence)", num_blocks, max_blocks_per_seq);

    let kv_dim = config.num_key_value_heads * config.head_dim;
    let kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_hidden_layers);

    // Create model with a dummy KV mode (engine uses forward_single_paged).
    let model = model::Model::new(config, weights, &backend)?;

    // Create scheduler and engine.
    let sched = scheduler::Scheduler::new(kv_pool, prompts.len());
    let mut eng = engine::Engine::new(
        model, sched, tokenizer, &backend, args.temperature, args.top_p,
    );

    // Submit all prompts.
    for prompt_text in &prompts {
        let tokens = if args.chat {
            let messages = vec![
                chat::Message { role: "system".into(), content: args.system.clone() },
                chat::Message { role: "user".into(), content: prompt_text.to_string() },
            ];
            let formatted = chat::format_chat(arch, &messages);
            eng.tokenizer.encode_chat(&formatted)?
        } else {
            eng.tokenizer.encode(prompt_text)?
        };
        eng.add_request(tokens, args.max_tokens);
    }

    // Run engine loop.
    let start = std::time::Instant::now();
    let mut total_generated = 0usize;

    while eng.has_work() {
        let finished = eng.step()?;
        for seq in &finished {
            total_generated += seq.tokens.len();
            println!("--- sequence {} ({} tokens) ---", seq.id, seq.tokens.len());
            println!("{}", seq.text);
        }
    }

    let elapsed = start.elapsed();
    let tps = total_generated as f64 / elapsed.as_secs_f64();
    eprintln!(
        "batch complete: {} tokens from {} sequences in {:.1?} ({:.1} tok/s total throughput)",
        total_generated, prompts.len(), elapsed, tps
    );

    Ok(())
}
