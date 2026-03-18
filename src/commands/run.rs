// ===========================================================================
// `rllm run` — Single-prompt inference.
//
// Processes one prompt, generates tokens, and streams output to stdout.
// Uses batched prefill (GEMM) for the prompt, then single-token decode
// (mat-vec) for generation.  Paged KV cache allocates memory on demand.
// ===========================================================================

use std::io::{self, Write};
use std::path::PathBuf;

use crate::gpu::{self, GpuCore};
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

pub(crate) fn exec(mut args: RunArgs) -> anyhow::Result<()> {
    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    // On macOS (Metal), fall back to single GPU with a warning.
    #[cfg(not(feature = "cuda"))]
    if args.tp > 1 {
        eprintln!("warning: --tp {} ignored (multi-GPU requires CUDA + NCCL), using single GPU", args.tp);
        args.tp = 1;
    }

    if args.tp > 1 {
        return exec_multi_gpu(args);
    }

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
    let model = model::Model::new(config.clone(), weights, &backend)?;

    // Dynamic KV cache sizing based on remaining GPU memory.
    let gpu_budget = backend.recommended_max_memory();
    let qpb = |m, k| backend.quantized_weight_bytes(m, k);
    let num_blocks = config.recommended_kv_blocks(gpu_budget, args.quantize, &qpb);
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let mut kv_pool = kv_cache::KvPool::new(&backend, num_blocks, kv_dim, config.num_kv_layers());
    let mut seq_state = kv_pool.new_sequence(&backend);
    let max_prefill = 4096;
    let prefill_bufs = model::PrefillBuffers::new(&backend, &config, max_prefill);

    let weight_mb = config.estimate_weight_bytes(args.quantize, &qpb) as f64 / (1024.0 * 1024.0);
    let kv_mb = kv_pool.total_memory_bytes() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "memory: {:.0} MB weights, {:.0} MB KV cache ({} blocks, {} max tokens), {:.0} MB GPU budget",
        weight_mb,
        kv_mb,
        num_blocks,
        kv_pool.max_tokens(),
        gpu_budget as f64 / (1024.0 * 1024.0),
    );

    // --- Encode prompt ---
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
        let text = tokenizer.decode(&[next_token])?;
        print!("{text}");
        io::stdout().flush()?;

        seq_state.ensure_slot(&mut kv_pool)?;
        seq_state.sync_block_table(&backend);
        model.forward_single_paged(next_token, &kv_pool, &seq_state)?;
        seq_state.advance();
        crate::model::profile::tick();

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

/// Multi-GPU inference path (--tp > 1).
#[cfg(feature = "cuda")]
fn exec_multi_gpu(args: RunArgs) -> anyhow::Result<()> {
    use crate::gpu::multi_gpu::tp::MultiGpuInference;

    let tp = args.tp;
    eprintln!("tensor parallelism: {} GPUs", tp);

    // Load config and tokenizer (these don't need GPU).
    let config = model::config::ModelConfig::from_file(&args.model.join("config.json"))?;
    let arch = config.arch()?;
    let tokenizer = model::tokenizer::Tokenizer::from_file(&args.model.join("tokenizer.json"), arch)?;

    eprintln!(
        "loaded config: {:?}, {} layers, {} heads, hidden_size={}",
        arch, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // Estimate KV blocks — use 256 as reasonable default for TP.
    let num_blocks = 256;

    let mut multi = MultiGpuInference::new(
        &args.model, config.clone(), args.quantize, tp, num_blocks,
    )?;
    eprintln!("multi-GPU inference ready ({} ranks)", tp);

    // Log sampling strategy.
    if args.temperature == 0.0 {
        eprintln!("sampling: greedy (temperature=0)");
    } else {
        eprintln!("sampling: temperature={}, top_p={}", args.temperature, args.top_p);
    }

    // --- Encode prompt ---
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
    let prefill_start = std::time::Instant::now();
    multi.ensure_slots(prompt_tokens.len())?;
    multi.forward_prefill_paged(&prompt_tokens)?;
    multi.advance_by(prompt_tokens.len());

    let prefill_elapsed = prefill_start.elapsed();
    let prefill_tps = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
    eprintln!(
        "prefill: {} tokens in {:.1?} ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_elapsed,
        prefill_tps
    );

    // --- Generation loop ---
    let gen_start = std::time::Instant::now();
    let mut gen_count: usize = 0;
    let mut rng = rand::rng();
    let mut next_token = sampler::sample(
        multi.backend(),
        multi.logits(),
        args.temperature,
        args.top_p,
        &mut rng,
    )?;
    for _ in 0..args.max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }
        gen_count += 1;
        let text = tokenizer.decode(&[next_token])?;
        print!("{text}");
        io::stdout().flush()?;

        multi.ensure_slot()?;
        multi.forward_single_paged(next_token)?;
        multi.advance();
        crate::model::profile::tick();

        next_token = sampler::sample(
            multi.backend(),
            multi.logits(),
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

#[cfg(not(feature = "cuda"))]
fn exec_multi_gpu(_args: RunArgs) -> anyhow::Result<()> {
    anyhow::bail!("multi-GPU tensor parallelism requires the cuda feature")
}
