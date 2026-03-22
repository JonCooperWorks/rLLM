// ===========================================================================
// `rllm run` — Single-prompt inference.
//
// Processes one prompt, generates tokens, and streams output to stdout.
// Uses engine::loader::load_and_run() for model loading and engine
// construction, then drives generation via the InferenceEngine trait.
// ===========================================================================

use std::io::{self, Write};
use std::path::PathBuf;

use crate::engine;
use crate::gpu;

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

    /// Stream MoE expert weights from SSD instead of loading all into GPU memory.
    /// Enables running large MoE models (e.g. 397B) that don't fit in VRAM.
    #[arg(long)]
    stream_experts: bool,

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

    /// Tensor parallelism: number of GPUs (0 = auto-detect all available).
    #[arg(long, default_value = "0")]
    tp: usize,
}

pub(crate) fn exec(mut args: RunArgs) -> anyhow::Result<()> {
    // Resolve --tp 0 → auto-detect available GPUs.
    if args.tp == 0 {
        args.tp = gpu::device_count();
        eprintln!("auto-detected {} GPU(s)", args.tp);
    }

    // Multi-GPU tensor parallelism is CUDA-only (requires NCCL).
    // On macOS (Metal), fall back to single GPU with a warning.
    #[cfg(not(feature = "cuda"))]
    if args.tp > 1 {
        eprintln!(
            "warning: --tp {} ignored (multi-GPU requires CUDA + NCCL), using single GPU",
            args.tp
        );
        args.tp = 1;
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

    let prompt = args.prompt.clone();
    let max_tokens = args.max_tokens;
    let temperature = args.temperature;
    let top_p = args.top_p;
    let chat = args.chat;
    let system = args.system.clone();

    use crate::model::config::ModelArch;
    use std::cell::Cell;
    let arch_cell: Cell<Option<ModelArch>> = Cell::new(None);

    engine::loader::load_and_run_ext(
        &args.model,
        args.stream_experts,
        args.tp,
        1, // single sequence
        |_tok, arch| {
            arch_cell.set(Some(arch));
        },
        |eng| {
            let arch = arch_cell.get().unwrap();

            // Encode prompt.
            let system_ref = chat.then(|| system.as_str());
            let prompt_tokens = eng.tokenizer().encode_prompt(&prompt, arch, system_ref)?;
            if chat {
                eprintln!("chat template applied ({:?})", arch);
            }
            eprintln!(
                "prompt tokens: {:?} ({})",
                &prompt_tokens,
                prompt_tokens.len()
            );

            // Submit request and run generation loop.
            let prompt_len = prompt_tokens.len();
            eng.add_request(prompt_tokens, max_tokens, temperature, top_p);

            let start = std::time::Instant::now();
            let mut gen_count = 0usize;
            let mut prev_text_len = 0usize;
            let mut all_token_ids: Vec<u32> = Vec::new();
            let mut prefill_reported = false;

            while eng.has_work() {
                let output = eng.step()?;

                // Report prefill (TTFT) after the first token arrives.
                if !prefill_reported && !output.tokens.is_empty() {
                    let ttft = start.elapsed();
                    let prefill_tps = prompt_len as f64 / ttft.as_secs_f64();
                    eprintln!(
                        "prefill: {} tokens in {:.1?} ({:.1} tok/s)",
                        prompt_len, ttft, prefill_tps
                    );
                    prefill_reported = true;
                }

                // Stream tokens to stdout as they're generated.
                for &(_seq_id, token_id) in &output.tokens {
                    all_token_ids.push(token_id);
                    gen_count += 1;

                    // Incremental decode: decode all tokens and emit only new text.
                    // This avoids SentencePiece Strip decoder issues where single-token
                    // decode strips leading spaces (affects Mistral/Mixtral).
                    let full_text = eng.tokenizer().decode(&all_token_ids).unwrap_or_default();
                    let new_text = &full_text[prev_text_len..];
                    if !new_text.is_empty() {
                        print!("{new_text}");
                        io::stdout().flush()?;
                    }
                    prev_text_len = full_text.len();
                }
            }
            println!();

            let elapsed = start.elapsed();
            let tps = gen_count as f64 / elapsed.as_secs_f64();
            eprintln!(
                "generation: {} tokens in {:.1?} ({:.1} tok/s)",
                gen_count, elapsed, tps
            );

            Ok(())
        },
    )
}
