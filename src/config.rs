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
//   reads from `ModelConfig` rather than hardcoding numbers.
//
// Model architecture detection:
//   HuggingFace config.json files contain a `model_type` field (e.g. "llama",
//   "qwen2").  We parse this into a `ModelArch` enum that encodes the small set
//   of model-specific behaviors: whether QKV projections have bias, which chat
//   template to use, and what stop tokens to use.
//
// Example config values (Llama 3.2 1B / Qwen 2.5 3B):
//   hidden_size:             2048   / 2048
//   num_hidden_layers:       16     / 36
//   num_attention_heads:     32     / 16
//   num_key_value_heads:     8      / 2
//   head_dim:                64     / 128
//   intermediate_size:       8192   / 11008
//   vocab_size:              128256 / 152064
//   rope_theta:              500000 / 10000
//   rms_norm_eps:            1e-5   / 1e-6
//   tie_word_embeddings:     true   / false
//   model_type:              llama  / qwen2
//
// Both families use the same architecture (RMSNorm, GQA, SwiGLU, RoPE).
// The main difference is Qwen adds bias to Q/K/V attention projections.
//
// RoPE scaling:
//   Llama 3.2 uses a custom RoPE scaling scheme for long contexts (>8192).
//   The `RopeScaling` struct captures these parameters, but Phase 1 doesn't
//   apply them — they only matter for sequences longer than the original
//   training length.  The scaling is a no-op for our 4096-token limit.
// ===========================================================================

use serde::Deserialize;

// ===========================================================================
// Model architecture enum — detected from config.json's `model_type` field.
//
// Why an enum instead of a trait?
//   When two models share 95% of their architecture (same RMSNorm, GQA,
//   SwiGLU, RoPE), a trait with dynamic dispatch adds complexity without
//   benefit.  An enum lets us encode the small set of differences (QKV bias,
//   chat template, stop tokens) as simple method calls.  Adding a third
//   model?  Just add a variant and fill in the methods.
//
//   If models ever diverge significantly (different forward pass structure,
//   different attention mechanism), THEN a trait would be justified.  But
//   for the Llama/Qwen family, enum dispatch keeps things flat and readable.
// ===========================================================================

/// Supported model architectures, detected from config.json's `model_type`.
///
/// Each variant encodes model-specific behavior without changing the
/// shared forward pass.  The forward pass in model.rs checks these
/// properties at the right points (e.g., bias-add after QKV matmul).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    /// Llama 3.x family (1B, 3B, 8B, 70B).
    /// No biases anywhere.  Chat template uses <|start_header_id|> markers.
    Llama,
    /// Qwen 2.5 family (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B).
    /// QKV projections have bias (NOT O projection or FFN).  Chat uses ChatML.
    Qwen2,
}

impl ModelArch {
    /// Whether Q, K, V linear projections have bias vectors.
    ///
    /// Learning note: bias in a linear layer means output = W @ x + b instead
    /// of just W @ x.  Llama omits all biases (simpler, fewer parameters).
    /// Qwen adds bias to Q/K/V projections (but NOT O projection or FFN),
    /// which gives the attention mechanism a learnable offset — empirically
    /// this helps at smaller model sizes.
    pub fn has_qkv_bias(&self) -> bool {
        match self {
            ModelArch::Llama => false,
            ModelArch::Qwen2 => true,
        }
    }

    /// Detect model architecture from config.json's `model_type` field.
    pub fn from_model_type(model_type: &str) -> anyhow::Result<Self> {
        match model_type {
            "llama" => Ok(ModelArch::Llama),
            "qwen2" => Ok(ModelArch::Qwen2),
            other => anyhow::bail!(
                "unsupported model_type '{}' (expected 'llama' or 'qwen2')",
                other
            ),
        }
    }
}

/// Model configuration, deserialized from `config.json`.
///
/// These fields are shared across Llama 3 and Qwen 2.5 — both use the same
/// HuggingFace field names for all architecture parameters.
///
/// Uses serde's `Deserialize` trait — any field not present in the JSON
/// that has a `#[serde(default)]` attribute will use its type's Default value.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Architecture identifier (e.g. "llama", "qwen2").
    /// Detected automatically from config.json.
    #[serde(default)]
    pub model_type: String,
    /// Dimension of the residual stream.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads.
    /// Fewer than query heads = Grouped-Query Attention (GQA).
    pub num_key_value_heads: usize,
    /// Dimension per attention head.
    /// Invariant: num_attention_heads × head_dim = hidden_size.
    /// Some models (Qwen 2.5) omit this — computed from hidden_size / num_heads.
    #[serde(default)]
    pub head_dim: usize,
    /// FFN hidden dimension.
    /// The FFN expands to this size then contracts back to hidden_size.
    pub intermediate_size: usize,
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Maximum sequence length the model supports.
    pub max_position_embeddings: usize,
    /// RoPE base frequency.
    /// Higher theta = slower rotation = longer effective context.
    pub rope_theta: f64,
    /// Epsilon for RMSNorm numerical stability.
    /// Prevents division by zero when the input vector is near-zero.
    pub rms_norm_eps: f64,
    /// Whether the LM head shares weights with the embedding table.
    /// When true, there is no separate `lm_head.weight` in the checkpoint.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Optional RoPE frequency scaling for long contexts (Llama 3.2 only).
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

impl ModelConfig {
    /// Load config from a JSON file.
    ///
    /// If `head_dim` is missing from the JSON (e.g. Qwen 2.5), it's computed
    /// as hidden_size / num_attention_heads.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&contents)?;
        if config.head_dim == 0 {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        Ok(config)
    }

    /// Detect which model architecture this config belongs to.
    pub fn arch(&self) -> anyhow::Result<ModelArch> {
        ModelArch::from_model_type(&self.model_type)
    }

    /// How many query heads share each KV head.
    /// For Llama 3.2 1B: 32 / 8 = 4 query heads per KV group.
    pub fn num_heads_per_kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
