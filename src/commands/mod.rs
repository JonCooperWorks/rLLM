// ===========================================================================
// CLI subcommands — one module per command.
//
//   run.rs   — `rllm run`   — single-prompt inference
//   batch.rs — `rllm batch` — batched inference from a file
//   serve.rs — `rllm serve` — HTTP API server
// ===========================================================================

mod batch;
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
}

impl Command {
    pub(crate) fn exec(self) -> anyhow::Result<()> {
        match self {
            Command::Run(args) => run::exec(args),
            Command::Batch(args) => batch::exec(args),
            Command::Serve(args) => serve::exec(args),
        }
    }
}
