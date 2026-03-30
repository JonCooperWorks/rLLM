// ===========================================================================
// `rllm batch` — Batched inference from a file of prompts.
//
// Reads prompts (one per line), submits them all to the engine, and runs
// the continuous batching loop until all sequences are complete.  All model
// loading and engine setup is handled by load_model_and_run() in
// commands/mod.rs.
// ===========================================================================

use std::path::PathBuf;

use tracing::info;

use super::ModelArgs;

#[derive(clap::Args)]
pub(crate) struct BatchArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Path to a batch file (one prompt per line).
    #[arg(long)]
    batch_file: PathBuf,
}

pub(crate) fn exec(args: BatchArgs) -> anyhow::Result<()> {
    let prompts_text = std::fs::read_to_string(&args.batch_file)?;
    let prompts: Vec<String> = prompts_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    info!(
        prompts = prompts.len(),
        file = %args.batch_file.display(),
        "batch loaded"
    );

    let max_active = prompts.len();
    let chat = args.model.chat;
    let system = args.model.system.clone();
    let max_tokens = args.model.max_tokens;
    let temperature = args.model.temperature;
    let top_p = args.model.top_p;

    super::load_model_and_run(&args.model, max_active, |eng, arch, _images, _image_token_id| {
        // Submit all prompts.
        for prompt_text in &prompts {
            let tokens = super::encode_prompt(eng, arch, prompt_text, chat, &system, None, None, &[])?;
            eng.add_request(
                tokens,
                max_tokens,
                crate::model::sampler::SampleParams {
                    temperature,
                    top_p,
                    ..crate::model::sampler::SampleParams::default()
                },
                Vec::new(),
                None,
                None,
                std::collections::HashMap::new(),
            );
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
        info!(
            tokens = total_generated,
            sequences = prompts.len(),
            elapsed = ?elapsed,
            tok_per_sec = format_args!("{:.1}", tps),
            "batch complete"
        );

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    /// Helper: replicate the prompt-parsing logic from exec().
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
