// ===========================================================================
// `rllm run` — Single-prompt inference.
//
// Processes one prompt, generates tokens, and streams output to stdout.
// Uses batched prefill (GEMM) for the prompt, then single-token decode
// (mat-vec) for generation.  Paged KV cache allocates memory on demand.
// ===========================================================================

use std::io::{self, Write};
use std::path::PathBuf;

use crate::gpu::{self, GpuBackend};
use crate::model;
use crate::model::{kv_cache, loader, sampler};

#[derive(clap::Args)]
pub(crate) struct RunArgs {
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

    /// Enable chat mode for instruct models.  Wraps --prompt in the model's
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

pub(crate) fn exec(args: RunArgs) -> anyhow::Result<()> {
    // --- Load GPU backend + model ---
    let backend = gpu::create_backend()?;
    eprintln!("gpu: {}", backend.device_name());

    let loader::LoadedModel {
        config,
        arch,
        tokenizer,
        weights,
    } = loader::load_model(&backend, &args.model, args.quantize)?;

    // Log sampling strategy so the user knows what to expect.
    if args.temperature == 0.0 {
        eprintln!("sampling: greedy (temperature=0)");
    } else {
        eprintln!(
            "sampling: temperature={}, top_p={}",
            args.temperature, args.top_p
        );
    }

    // --- Create model + paged KV cache + prefill buffers ---
    //
    // The model holds weights and single-token scratch buffers (for decode).
    // The paged KV cache allocates memory on demand in 16-token blocks —
    // no wasted memory for short sequences, and it enables batched prefill.
    //
    // PrefillBuffers are batch-sized scratch tensors (up to 4096 tokens)
    // used by the GEMM-based prefill forward pass.  Allocated once, reused.
    let mut model = model::Model::new(config.clone(), weights, &backend)?;

    let num_blocks = 8192; // 8192 blocks × 16 tokens/block = 131072 (128K) max positions.
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let mut kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_hidden_layers);
    let mut seq_state = kv_pool.new_sequence(&backend);
    let max_prefill = 4096;
    let prefill_bufs = model::PrefillBuffers::new(&backend, &config, max_prefill);
    eprintln!(
        "KV cache: {} blocks ({} max tokens), prefill up to {} tokens",
        num_blocks,
        num_blocks * kv_cache::BLOCK_SIZE,
        max_prefill
    );

    // --- Encode prompt ---
    // In chat mode, wraps the prompt in the model's chat template (auto-detected
    // from the architecture).  Without --chat, encodes the raw prompt directly.
    let system = args.chat.then(|| args.system.as_str());
    let prompt_tokens = tokenizer.encode_prompt(&args.prompt, arch, system)?;
    if args.chat {
        eprintln!("chat template applied ({:?})", arch);
    }
    eprintln!(
        "prompt tokens: {:?} ({})",
        &prompt_tokens,
        prompt_tokens.len()
    );

    // --- Prefill ---
    // Process the entire prompt in one batched forward pass using GEMM.
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
        prompt_tokens.len(),
        prefill_elapsed,
        prefill_tps
    );

    // --- Generation loop ---
    // Sample the first token from the prefill logits, then enter the
    // autoregressive decode loop: forward → sample → print → repeat.
    //
    // Generation is memory-bound (mat-vec), not compute-bound (mat-mat),
    // because each token is a single vector — no batch dimension to amortise
    // the weight matrix loads.  This is inherent to autoregressive decoding.
    let gen_start = std::time::Instant::now();
    let mut gen_count: usize = 0;
    let mut rng = rand::rng();
    let mut next_token = sampler::sample(
        &backend,
        model.logits(),
        args.temperature,
        args.top_p,
        &mut rng,
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
            &backend,
            model.logits(),
            args.temperature,
            args.top_p,
            &mut rng,
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
