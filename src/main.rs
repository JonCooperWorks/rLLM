// ===========================================================================
// rLLM — Rust LLM inference engine.
//
// Entry point.  Parses CLI arguments and dispatches to the appropriate
// subcommand.  Each command lives in its own module under `commands/`:
//
//   rllm run   — single-prompt inference    (commands/run.rs)
//   rllm batch — batched inference from file (commands/batch.rs)
//   rllm serve — HTTP API server            (commands/serve.rs)
// ===========================================================================

mod api;
mod chat;
mod commands;
mod config;
mod engine;
mod gpu;
mod kv_cache;
mod loader;
mod model;
mod sampler;
mod scheduler;
mod setup;
mod tokenizer;

// Re-export for use by api module.
pub(crate) use commands::ServeArgs;

use std::process::ExitCode;
use clap::Parser;

#[derive(Parser)]
#[command(name = "rllm", about = "Rust LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: commands::Command,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = cli.command.exec() {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
