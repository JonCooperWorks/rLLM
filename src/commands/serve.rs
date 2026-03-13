// ===========================================================================
// `rllm serve` — HTTP API server.
//
// Starts an HTTP server with OpenAI and Anthropic compatible APIs.
// Supports both streaming (SSE) and non-streaming responses, with
// optional TLS via manual certificates or Let's Encrypt.
//
// The actual server implementation lives in the `api` module — this file
// just defines the CLI arguments and delegates.
// ===========================================================================

use std::path::PathBuf;

#[derive(clap::Args)]
pub(crate) struct ServeArgs {
    /// Path to model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    pub model: PathBuf,

    /// Port to listen on.
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Quantise weights to Q4 on load.
    #[arg(long)]
    pub quantize: bool,

    /// Domain name (for Let's Encrypt certificate provisioning).
    #[arg(long)]
    pub domain: Option<String>,

    /// Use Let's Encrypt for automatic TLS certificates (TLS-ALPN-01 challenge).
    #[arg(long, requires = "domain")]
    pub letsencrypt: bool,

    /// Path to TLS certificate chain (PEM format).  Use with --private-key.
    #[arg(long, requires = "private_key", conflicts_with = "letsencrypt")]
    pub cert: Option<PathBuf>,

    /// Path to TLS private key (PEM format).  Use with --cert.
    #[arg(long, requires = "cert", conflicts_with = "letsencrypt")]
    pub private_key: Option<PathBuf>,

    /// Contact email for Let's Encrypt registration.
    #[arg(long, requires = "letsencrypt")]
    pub letsencrypt_email: Option<String>,

    /// Directory to cache Let's Encrypt certificates.
    #[arg(long, requires = "letsencrypt", default_value = ".rllm-certs")]
    pub cert_cache_dir: PathBuf,

    /// Allow running without TLS (plain HTTP).  Required when no TLS config is set.
    #[arg(long)]
    pub dangerous_no_tls: bool,
}

pub(crate) fn exec(args: ServeArgs) -> anyhow::Result<()> {
    crate::api::serve(args)
}
