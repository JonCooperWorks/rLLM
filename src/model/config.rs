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
    /// Qwen 3 Mixture-of-Experts family (e.g. Qwen3-Coder-30B-A3B).
    ///
    /// Learning note: MoE models have many "expert" FFN sub-networks per layer
    /// but only activate a small subset (top-k) for each token.  This gives the
    /// model capacity of a large dense model (30B total params) with the compute
    /// cost of a small one (3B active params per token).
    ///
    /// Attention is identical to Llama (no QKV bias), but adds QK-norm (RMSNorm
    /// on Q and K after projection, before RoPE) and uses ChatML chat template.
    Qwen3Moe,
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
            ModelArch::Qwen3Moe => false,
        }
    }

    /// Whether Q and K projections have per-head RMSNorm applied before RoPE.
    ///
    /// Learning note: QK-norm stabilises attention by normalising Q and K
    /// vectors before computing dot products.  Without it, attention logits
    /// can grow large in deeper layers, causing sharp softmax distributions
    /// that hurt training stability.  Qwen 3 adds this; Llama and Qwen 2.5
    /// rely on the implicit normalisation from RMSNorm on the hidden state.
    pub fn has_qk_norm(&self) -> bool {
        match self {
            ModelArch::Llama | ModelArch::Qwen2 => false,
            ModelArch::Qwen3Moe => true,
        }
    }

    /// Detect model architecture from config.json's `model_type` field.
    pub fn from_model_type(model_type: &str) -> anyhow::Result<Self> {
        match model_type {
            "llama" => Ok(ModelArch::Llama),
            "qwen2" => Ok(ModelArch::Qwen2),
            "qwen3_moe" => Ok(ModelArch::Qwen3Moe),
            other => anyhow::bail!(
                "unsupported model_type '{}' (expected 'llama', 'qwen2', or 'qwen3_moe')",
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
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
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

    // --- MoE (Mixture of Experts) fields ---
    //
    // Learning note: MoE replaces the single dense FFN per layer with many
    // small "expert" FFNs.  A learned router picks the top-k experts per
    // token.  Only activated experts consume compute, so a 30B-parameter
    // model can run at the cost of a ~3B model — the key insight behind
    // efficient scaling.
    //
    // These fields are only present in MoE configs (e.g. qwen3_moe).
    // For dense models they default to 0.

    /// Total number of expert FFN sub-networks per layer.
    /// Qwen3-Coder-30B-A3B: 128 experts per layer.
    #[serde(default)]
    pub num_experts: usize,
    /// How many experts are activated (routed to) per token.
    /// Qwen3-Coder-30B-A3B: top-8 experts selected per token.
    #[serde(default)]
    pub num_experts_per_tok: usize,
    /// Hidden dimension of each expert's FFN (gate/up/down projections).
    /// Much smaller than `intermediate_size` because there are many experts.
    /// Qwen3-Coder-30B-A3B: 768 per expert (vs 6144 dense intermediate_size).
    #[serde(default)]
    pub moe_intermediate_size: usize,
}

/// RoPE frequency scaling parameters for extended context lengths.
///
/// Learning note: standard RoPE with theta=500000 works well up to ~8192
/// tokens.  Beyond that, the model applies frequency-dependent scaling
/// to some RoPE dimensions, allowing extrapolation to 131072 tokens.
/// This is unused in Phase 1 (max 4096 tokens).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn num_heads_per_kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this model uses Mixture of Experts instead of dense FFN.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}
