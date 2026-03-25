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
        args.tp,
        max_active,
        |_tok, arch| {
            arch_cell.set(Some(arch));
        },
        |eng| {
            let arch = arch_cell.get().unwrap();
            // Submit all prompts.
            let system_ref = chat.then(|| system.as_str());
            for prompt_text in &prompts {
                let tokens = eng
                    .tokenizer()
                    .encode_prompt(prompt_text, arch, system_ref)?;
                eng.add_request(tokens, max_tokens, temperature, top_p, Vec::new());
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    /// Helper: replicate the prompt-parsing logic from exec() lines 65-69.
    fn parse_prompts(text: &str) -> Vec<String> {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect()
    }

    #[test]
    fn test_prompt_parsing_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "What is 2+2?").unwrap();
        writeln!(f, "Tell me a joke.").unwrap();
        writeln!(f, "Summarize this.").unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let prompts = parse_prompts(&text);
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts[0], "What is 2+2?");
        assert_eq!(prompts[1], "Tell me a joke.");
        assert_eq!(prompts[2], "Summarize this.");
    }

    #[test]
    fn test_prompt_parsing_skips_empty_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "First prompt").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "   ").unwrap();
        writeln!(f, "Second prompt").unwrap();
        writeln!(f).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let prompts = parse_prompts(&text);
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0], "First prompt");
        assert_eq!(prompts[1], "Second prompt");
    }

    #[test]
    fn test_prompt_parsing_trims_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "  leading spaces").unwrap();
        writeln!(f, "trailing spaces   ").unwrap();
        writeln!(f, "  both sides  ").unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let prompts = parse_prompts(&text);
        // The current code does NOT trim — it only filters empty-after-trim.
        // So whitespace is preserved in the output strings.
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts[0], "  leading spaces");
        assert_eq!(prompts[1], "trailing spaces   ");
        assert_eq!(prompts[2], "  both sides  ");
    }
}
