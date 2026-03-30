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

    /// Stream MoE expert weights from SSD instead of loading all into GPU memory.
    /// Enables serving large MoE models (e.g. 397B) that don't fit in VRAM.
    #[arg(long)]
    pub stream_experts: bool,

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

    /// Allow running without TLS (plain HTTP) on external interfaces.
    /// Not required for localhost/127.0.0.1/::1 — those skip TLS automatically.
    #[arg(long)]
    pub dangerous_no_tls: bool,

    /// Allow running without authentication on external interfaces.
    /// Not required for localhost/127.0.0.1/::1 — those skip auth automatically.
    #[arg(long)]
    pub dangerous_no_auth: bool,

    /// Tensor parallelism: number of GPUs (0 = auto-detect all available).
    #[arg(long, default_value = "0")]
    pub tp: usize,

    /// Path to auth config file (JSON).  Enables pluggable authentication.
    /// When omitted, no auth is applied (all requests allowed).
    /// See docs/threat-model.md for config format and deployment guidance.
    #[arg(long)]
    pub auth_config: Option<PathBuf>,

    /// KV cache quantization mode.  TurboQuant applies a random orthogonal
    /// rotation followed by optimal scalar quantization (Max-Lloyd) per
    /// coordinate.  See `rllm run --help` for details.
    #[arg(long, default_value = "turbo4")]
    pub kv_quant: String,

    /// Maximum number of requests queued between HTTP handlers and the worker.
    /// When the queue is full, new requests receive 503 Service Unavailable.
    #[arg(long, default_value = "8")]
    pub max_pending: usize,

    /// Maximum number of sequences actively being processed by the engine.
    /// Additional sequences wait in the scheduler queue until a slot opens.
    #[arg(long, default_value = "32")]
    pub max_active: usize,

    /// Per-request timeout in seconds.  Requests that exceed this deadline are
    /// aborted and the client receives an error.  0 = no timeout (not recommended).
    #[arg(long, default_value = "300")]
    pub request_timeout: u64,
}

pub(crate) fn exec(args: ServeArgs) -> anyhow::Result<()> {
    crate::api::serve(args)
}
