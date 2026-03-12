// ===========================================================================
// Model configuration — deserialized from HuggingFace config.json.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Parses the model's `config.json` file into a Rust struct.  Every HuggingFace
//   model ships with a config.json that describes the model's architecture:
//   how many layers, heads, the hidden dimension, vocabulary size, etc.
//
//   This is the ONLY place where model hyperparameters live.  Every other file
//   reads from `LlamaConfig` rather than hardcoding numbers.
//
// Llama 3.2 1B config values:
//   hidden_size:             2048   — dimension of the residual stream
//   num_hidden_layers:       16     — number of transformer layers
//   num_attention_heads:     32     — query heads
//   num_key_value_heads:     8      — KV heads (GQA with 4:1 ratio)
//   head_dim:                64     — dimension per attention head
//   intermediate_size:       8192   — FFN hidden dimension (4× expansion)
//   vocab_size:              128256 — number of tokens in the vocabulary
//   max_position_embeddings: 131072 — maximum context length
//   rope_theta:              500000 — RoPE base frequency
//   rms_norm_eps:            1e-5   — epsilon for RMSNorm numerical stability
//   tie_word_embeddings:     true   — lm_head reuses embed_tokens
//
// RoPE scaling:
//   Llama 3.2 uses a custom RoPE scaling scheme for long contexts (>8192).
//   The `RopeScaling` struct captures these parameters, but Phase 1 doesn't
//   apply them — they only matter for sequences longer than the original
//   training length.  The scaling is a no-op for our 4096-token limit.
// ===========================================================================

use serde::Deserialize;

/// Llama model configuration, deserialized from `config.json`.
///
/// Uses serde's `Deserialize` trait — any field not present in the JSON
/// that has a `#[serde(default)]` attribute will use its type's Default value.
#[derive(Debug, Deserialize)]
pub struct LlamaConfig {
    /// Dimension of the residual stream (2048).
    pub hidden_size: usize,
    /// Number of transformer layers (16).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (32).
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (8).
    /// Fewer than query heads = Grouped-Query Attention (GQA).
    pub num_key_value_heads: usize,
    /// Dimension per attention head (64).
    /// Invariant: num_attention_heads × head_dim = hidden_size.
    pub head_dim: usize,
    /// FFN hidden dimension (8192).
    /// The FFN expands to this size then contracts back to hidden_size.
    pub intermediate_size: usize,
    /// Number of tokens in the vocabulary (128256).
    pub vocab_size: usize,
    /// Maximum sequence length the model supports (131072).
    pub max_position_embeddings: usize,
    /// RoPE base frequency (500000.0).
    /// Higher theta = slower rotation = longer effective context.
    pub rope_theta: f64,
    /// Epsilon for RMSNorm numerical stability (1e-5).
    /// Prevents division by zero when the input vector is near-zero.
    pub rms_norm_eps: f64,
    /// Whether the LM head shares weights with the embedding table.
    /// When true, there is no separate `lm_head.weight` in the checkpoint.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Optional RoPE frequency scaling for long contexts.
    pub rope_scaling: Option<RopeScaling>,
}

/// RoPE frequency scaling parameters for extended context lengths.
///
/// Learning note: standard RoPE with theta=500000 works well up to ~8192
/// tokens.  Beyond that, the model applies frequency-dependent scaling
/// to some RoPE dimensions, allowing extrapolation to 131072 tokens.
/// This is unused in Phase 1 (max 4096 tokens).
#[derive(Debug, Deserialize)]
pub struct RopeScaling {
    /// Scaling algorithm type (e.g. "llama3").
    pub rope_type: String,
    /// Overall scaling factor.
    pub factor: f64,
    /// Factor for high-frequency RoPE dimensions.
    pub high_freq_factor: f64,
    /// Factor for low-frequency RoPE dimensions.
    pub low_freq_factor: f64,
    /// Training context length before scaling was applied.
    pub original_max_position_embeddings: usize,
}

impl LlamaConfig {
    /// Load config from a JSON file.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// How many query heads share each KV head.
    /// For Llama 3.2 1B: 32 / 8 = 4 query heads per KV group.
    pub fn num_heads_per_kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
