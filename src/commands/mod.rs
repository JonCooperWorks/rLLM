// ===========================================================================
// CLI subcommands — one module per command.
//
//   run.rs      — `rllm run`      — single-prompt inference
//   batch.rs    — `rllm batch`    — batched inference from a file
//   serve.rs    — `rllm serve`    — HTTP API server
//   quantize.rs — `rllm quantize` — offline weight quantization (bf16 → Q4)
// ===========================================================================

mod batch;
mod quantize;
mod run;
mod serve;

pub(crate) use serve::ServeArgs;

#[derive(clap::Subcommand)]
pub(crate) enum Command {
    /// Run inference on a single prompt.
    Run(run::RunArgs),
    /// Run batched inference from a file of prompts.
    Batch(batch::BatchArgs),
    /// Start an OpenAI/Anthropic-compatible API server.
    Serve(serve::ServeArgs),
    /// Pre-quantize model weights from bf16 to Q4 on disk.
    Quantize(quantize::QuantizeArgs),
}

impl Command {
    pub(crate) fn exec(self) -> anyhow::Result<()> {
        match self {
            Command::Run(args) => run::exec(args),
            Command::Batch(args) => batch::exec(args),
            Command::Serve(args) => serve::exec(args),
            Command::Quantize(args) => quantize::exec(args),
        }
    }
}
