// ===========================================================================
// `rllm run` — Single-prompt inference with streaming output.
//
// Processes one prompt, generates tokens, and streams output to stdout.
// All model loading and engine setup is handled by load_model_and_run()
// in commands/mod.rs — this file only contains the args and generation loop.
// ===========================================================================

use std::io::{self, Write};

use super::ModelArgs;

#[derive(clap::Args)]
pub(crate) struct RunArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Prompt text to complete.
    #[arg(long)]
    prompt: String,
}

pub(crate) fn exec(args: RunArgs) -> anyhow::Result<()> {
    let prompt = args.prompt.clone();
    let chat = args.model.chat;
    let system = args.model.system.clone();
    let max_tokens = args.model.max_tokens;
    let temperature = args.model.temperature;
    let top_p = args.model.top_p;

    // Read raw image bytes for vision encoding (preprocessing is in load_model_and_run).
    let image_data: Option<Vec<u8>> = args
        .model
        .image
        .as_ref()
        .map(|p| std::fs::read(p))
        .transpose()?;

    super::load_model_and_run(&args.model, 1, |eng, arch, processed_images, image_token_id| {
        let tokens = super::encode_prompt(
            eng,
            arch,
            &prompt,
            chat,
            &system,
            image_data.as_deref(),
            image_token_id,
            processed_images,
        )?;
        if chat {
            eprintln!("chat template applied ({:?})", arch);
        }
        eprintln!("prompt tokens: {:?} ({})", &tokens, tokens.len());

        let prompt_len = tokens.len();
        eng.add_request(
            tokens,
            max_tokens,
            temperature,
            top_p,
            processed_images.to_vec(),
            None,
        );

        // Stream tokens to stdout as they're generated.
        let start = std::time::Instant::now();
        let mut gen_count = 0usize;
        let mut prev_text_len = 0usize;
        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut prefill_reported = false;

        while eng.has_work() {
            let output = eng.step()?;

            if !prefill_reported && !output.tokens.is_empty() {
                let ttft = start.elapsed();
                let prefill_tps = prompt_len as f64 / ttft.as_secs_f64();
                eprintln!(
                    "prefill: {} tokens in {:.1?} ({:.1} tok/s)",
                    prompt_len, ttft, prefill_tps
                );
                prefill_reported = true;
            }

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
    })
}
