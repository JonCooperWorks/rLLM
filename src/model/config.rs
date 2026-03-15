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
// All supported architectures (Llama, Qwen, Phi, Gemma, Mistral, Qwen3 MoE,
// Qwen 3.5) share the same core building blocks (RMSNorm, GQA, SwiGLU, RoPE).
// Differences are captured by ModelArch: QKV bias (Qwen), QK-norm (Qwen3 MoE),
// fused QKV (Gemma), chat template format, and MoE vs dense FFN.
//
// RoPE scaling:
//   Llama 3.2 uses a custom RoPE scaling scheme for long contexts (>8192).
//   The `RopeScaling` struct captures these parameters, but Phase 1 doesn't
//   apply them — they only matter for sequences longer than the original
//   training length.  The scaling is a no-op for our 4096-token limit.
// ===========================================================================

use crate::model::kv_cache;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Deserialize a `usize` that may be `null` in JSON (treat null as 0).
///
/// Many HuggingFace configs set optional numeric fields to `null` rather than
/// omitting them.  `#[serde(default)]` handles the absent case but not null.
fn null_as_zero<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    Ok(Option::<usize>::deserialize(d)?.unwrap_or(0))
}

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
    /// Qwen 3.5 hybrid architecture (e.g. Qwen3.5-27B).
    ///
    /// Learning note: Qwen 3.5 uses a HYBRID attention design — 75% of layers
    /// use Gated DeltaNet (linear attention with recurrent state) and 25% use
    /// standard GQA softmax attention.  This is fundamentally different from all
    /// other supported architectures.
    ///
    /// DeltaNet layers maintain a fixed-size [head_dim, head_dim] state matrix
    /// per head, updated via a delta rule with exponential gating.  This gives
    /// O(1) per-token cost regardless of sequence length.  GQA layers use a
    /// standard paged KV cache for full-precision retrieval.
    ///
    /// The model is multimodal (VLM), but we only use the language model portion.
    /// The config.json has text parameters nested under `text_config`.
    /// Weight tensors are prefixed with `model.language_model.` instead of `model.`.
    Qwen3_5,
    /// Mistral family (Mistral AI, 7B).
    ///
    /// Learning note: Mistral 7B is architecturally identical to Llama (RMSNorm,
    /// GQA, SwiGLU, RoPE).  No QKV bias, no QK-norm, no fused weights.  The only
    /// differences are the chat template ([INST]/[/INST] markers) and stop tokens.
    /// The forward pass reuses Llama's implementation directly.
    Mistral,
    /// Mixtral family (Mistral AI, 8x7B sparse MoE).
    ///
    /// Learning note: Mixtral is Mistral's attention (identical to Llama: GQA,
    /// RoPE, SwiGLU, no bias, no QK-norm) combined with a Mixture-of-Experts FFN.
    /// Each layer has 8 expert FFNs with top-2 routing — only 2 of 8 experts
    /// activate per token, giving ~46.7B total params but ~12.9B active.
    ///
    /// Config differences from Qwen3-MoE:
    ///   - Uses `num_local_experts` (not `num_experts`) in config.json
    ///   - Expert FFN size is `intermediate_size` (not separate `moe_intermediate_size`)
    ///   - Weight paths use `block_sparse_moe` prefix with w1/w2/w3 naming
    ///
    /// Chat template: [INST]/[/INST] (same as Mistral).
    Mixtral,
    /// Phi family (Microsoft Phi-3, Phi-4).
    ///
    /// Learning note: Phi models are architecturally near-identical to Llama
    /// (RMSNorm, GQA, SwiGLU, RoPE) but use FUSED weight tensors:
    ///   - `qkv_proj` instead of separate q/k/v projections
    ///   - `gate_up_proj` instead of separate gate/up projections
    /// These are split on-load into the standard separate tensors.
    ///
    /// No QKV bias, no QK-norm.  Chat template uses `<|im_start|>`/`<|im_sep|>`
    /// markers (similar to ChatML but with a separator token).
    Phi,
    /// Gemma 3 family (Google, 1B, 4B, 12B, 27B).
    ///
    /// Learning note: Gemma 3 shares the Llama backbone (GQA, RoPE, gated FFN)
    /// but has several architectural innovations:
    ///
    ///   1. **Sliding window attention**: alternating local (sliding window) and
    ///      global (full context) layers.  Pattern: 5 local + 1 global, repeating.
    ///      Local layers only attend to the last `sliding_window` tokens (512–4096),
    ///      saving KV cache memory without degrading quality — nearby tokens matter
    ///      most for local patterns, while periodic global layers capture long-range
    ///      dependencies.
    ///
    ///   2. **Sandwich norms**: 4 RMSNorm layers per decoder layer instead of
    ///      Llama's 2.  Pre-norm AND post-norm around both attention and FFN.
    ///      The post-norms stabilize training for deeper models by controlling
    ///      the scale of sub-layer outputs before they enter the residual stream.
    ///
    ///   3. **GeGLU**: GELU-gated linear unit instead of SiLU (SwiGLU).
    ///      `gelu(gate) * up` replaces `silu(gate) * up`.  GELU's smoother
    ///      gradient landscape empirically helps with training stability.
    ///
    ///   4. **Dual RoPE bases**: local layers use rope_local_base_freq (10000),
    ///      global layers use rope_theta (1000000).  Lower theta for local layers
    ///      gives sharper positional encoding for nearby tokens; higher theta for
    ///      global layers supports long-range position discrimination.
    ///
    ///   5. **Embedding scaling**: embeddings multiplied by √hidden_size after
    ///      lookup, instead of a learned norm.  This ensures the embedding vectors
    ///      start at the right magnitude for the residual stream.
    ///
    ///   6. **Offset RMSNorm**: `(1 + w) * normalized(x)` — weights init to 0,
    ///      effective scale starts at 1.0.  Same as Qwen 3.5.
    ///
    ///   7. **Custom attention scale**: `query_pre_attn_scalar` replaces head_dim
    ///      for computing 1/√(scale).  Decouples the scale from head dimension.
    ///
    /// No QKV bias, no QK-norm.  Chat uses `<start_of_turn>`/`<end_of_turn>`.
    Gemma3,
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
            ModelArch::Llama | ModelArch::Mistral | ModelArch::Mixtral | ModelArch::Phi | ModelArch::Gemma3 => false,
            ModelArch::Qwen2 => true,
            ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => false,
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
            ModelArch::Llama | ModelArch::Mistral | ModelArch::Mixtral | ModelArch::Qwen2 | ModelArch::Phi => false,
            ModelArch::Gemma3 => true,  // Both 4B and 27B have q_norm/k_norm weights
            ModelArch::Qwen3_5 => true,  // GQA layers have QK-norm
            ModelArch::Qwen3Moe => true,
        }
    }

    /// Whether the model uses fused QKV and gate_up weight tensors.
    ///
    /// Learning note: Phi models store Q, K, V as a single concatenated tensor
    /// `qkv_proj` of shape [q_dim + 2*kv_dim, hidden] and gate+up as a single
    /// `gate_up_proj` of shape [2*inter_size, hidden].  This saves a tiny amount
    /// of overhead in the original PyTorch implementation (one matmul instead
    /// of three).  We split them on-load so the forward pass stays generic.
    pub fn has_fused_qkv(&self) -> bool {
        matches!(self, ModelArch::Phi)
    }

    /// Detect model architecture from config.json's `model_type` field.
    pub fn from_model_type(model_type: &str) -> anyhow::Result<Self> {
        match model_type {
            "llama" => Ok(ModelArch::Llama),
            "mistral" => Ok(ModelArch::Mistral),
            "mixtral" => Ok(ModelArch::Mixtral),
            "qwen2" => Ok(ModelArch::Qwen2),
            "qwen3_moe" => Ok(ModelArch::Qwen3Moe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(ModelArch::Qwen3_5),
            "phi3" | "phi4" => Ok(ModelArch::Phi),
            "gemma3_text" | "gemma3" => Ok(ModelArch::Gemma3),
            other => anyhow::bail!(
                "unsupported model_type '{}' (expected 'llama', 'mistral', 'mixtral', 'qwen2', 'qwen3_moe', 'qwen3_5', 'phi3', or 'gemma3_text')",
                other
            ),
        }
    }
}

/// Model configuration, deserialized from `config.json`.
///
/// These fields are shared across all supported architectures — they use the same
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
    /// Gemma 3 4B omits this from text_config — filled in by apply_gemma3_defaults().
    #[serde(default)]
    pub num_attention_heads: usize,
    /// Number of key/value attention heads.
    /// Fewer than query heads = Grouped-Query Attention (GQA).
    #[serde(default)]
    pub num_key_value_heads: usize,
    /// Dimension per attention head.
    /// Invariant: num_attention_heads × head_dim = hidden_size.
    /// Some models (Qwen 2.5) omit this — computed from hidden_size / num_heads.
    #[serde(default)]
    pub head_dim: usize,
    /// FFN hidden dimension (dense models).
    /// The FFN expands to this size then contracts back to hidden_size.
    /// MoE-only models (e.g. Qwen3.5) have no dense FFN — this is 0.
    #[serde(default)]
    pub intermediate_size: usize,
    /// Number of tokens in the vocabulary.
    #[serde(default)]
    pub vocab_size: usize,
    /// Maximum sequence length the model supports.
    #[serde(default)]
    pub max_position_embeddings: usize,
    /// RoPE base frequency.
    /// Higher theta = slower rotation = longer effective context.
    #[serde(default)]
    pub rope_theta: f64,
    /// Epsilon for RMSNorm numerical stability.
    /// Prevents division by zero when the input vector is near-zero.
    #[serde(default)]
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
    /// Mixtral uses `num_local_experts` in config.json (serde alias handles this).
    #[serde(default, alias = "num_local_experts")]
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
    /// Hidden dimension of the shared (always-active) expert's FFN.
    /// Qwen3.5-35B-A3B: 512.  Only present in models with a shared expert.
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,

    // --- Qwen 3.5 hybrid DeltaNet + GQA fields ---
    //
    // Learning note: Qwen 3.5 uses a hybrid architecture where 75% of layers
    // use Gated DeltaNet (linear attention with a fixed-size recurrent state
    // matrix) and 25% use standard GQA softmax attention.  DeltaNet layers
    // have different head configurations than GQA layers.
    //
    // DeltaNet maintains a [head_dim, head_dim] = [128, 128] state matrix per
    // QK-head that's updated at each token.  This gives O(1) per-token cost
    // regardless of sequence length — no growing KV cache.  The trade-off is
    // that the fixed-size state can't represent every detail of long contexts,
    // which is why some layers still use full attention.

    /// Number of key heads in DeltaNet (linear attention) layers.
    /// Qwen3.5-27B: 16 QK-heads with head_dim=128.
    #[serde(default)]
    pub linear_num_key_heads: usize,
    /// Number of value heads in DeltaNet layers.
    /// Qwen3.5-27B: 48 V-heads — 3 V-heads per QK-head (GQA-style grouping).
    #[serde(default)]
    pub linear_num_value_heads: usize,
    /// Head dimension for keys in DeltaNet layers.
    #[serde(default)]
    pub linear_key_head_dim: usize,
    /// Head dimension for values in DeltaNet layers.
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// Kernel size for the causal depthwise Conv1D in DeltaNet layers.
    /// Conv1D provides local positional information (DeltaNet has no RoPE).
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    /// How often a full-attention layer appears (e.g. 4 = every 4th layer).
    /// Layer pattern: [DeltaNet, DeltaNet, DeltaNet, GQA, DeltaNet, ...].
    #[serde(default)]
    pub full_attention_interval: usize,
    /// Per-layer attention type: "linear_attention" or "full_attention".
    /// This is the authoritative source for which layers use which mechanism.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Fraction of head_dim that gets RoPE in full-attention layers.
    /// Qwen3.5: 0.25, meaning only 64 of 256 dims get rotary embeddings.
    #[serde(default)]
    pub partial_rotary_factor: f64,
    /// RoPE parameters for models with nested rope config (Qwen 3.5).
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    /// Whether full-attention layers use output gating (Qwen 3.5).
    /// When true, q_proj produces [Q, Z] concatenated (2× q_dim), and the
    /// attention output is gated: out = rmsnorm_no_weight(attn_out) * silu(Z).
    #[serde(default)]
    pub attn_output_gate: bool,

    // --- Gemma 3 fields ---
    //
    // Learning note: Gemma 3 uses sliding window attention where most layers
    // only attend to a limited window of recent tokens, with periodic "global"
    // layers that see the full context.  This is a memory-efficiency trick:
    // local patterns (syntax, coreference) are captured by sliding window layers,
    // while long-range dependencies (document structure, early context) are
    // handled by global layers.  The pattern is typically 5 local + 1 global.
    //
    // Gemma 3 also uses two different RoPE frequencies — a lower base for
    // local layers (where precise nearby-token positioning matters) and a
    // higher base for global layers (where long-range discrimination matters).

    /// Sliding window size for local attention layers (tokens).
    /// Gemma 3 1B: 512, larger models: up to 4096.
    /// Only used when layer_types contains "sliding_attention".
    #[serde(default, deserialize_with = "null_as_zero")]
    pub sliding_window: usize,
    /// How often a global (full-attention) layer appears in the interleaved
    /// pattern.  Gemma 3: 6 (every 6th layer is global, i.e. 5 local + 1 global).
    /// Used to auto-generate layer_types when the config omits them.
    #[serde(default)]
    pub sliding_window_pattern: usize,
    /// Custom attention scale factor.  When > 0, attention uses
    /// 1/√(query_pre_attn_scalar) instead of the default 1/√(head_dim).
    /// Gemma 3: typically set to head_dim (256), making it equivalent, but
    /// this decouples the scale from the actual head dimension.
    #[serde(default)]
    pub query_pre_attn_scalar: f64,
    /// RoPE base frequency for LOCAL (sliding window) attention layers.
    /// Gemma 3: 10000 (standard).  Global layers use `rope_theta` (1000000)
    /// for better long-range position discrimination.
    #[serde(default)]
    pub rope_local_base_freq: f64,
    /// FFN hidden activation function name (from HuggingFace config).
    /// "gelu_pytorch_tanh" = GeGLU (Gemma 3), empty or absent = SwiGLU (default).
    ///
    /// Learning note: GeGLU uses gelu(gate) × up, while SwiGLU uses silu(gate) × up.
    /// GELU has a smoother gradient landscape which empirically helps training stability.
    /// The "pytorch_tanh" suffix refers to PyTorch's tanh-approximated GELU variant.
    #[serde(default)]
    pub hidden_activation: String,

    // --- Weight prefix ---
    // Multimodal models (Qwen 3.5) prefix layer weights with
    // "model.language_model." instead of "model.".  This field is set
    // during config loading, not deserialized from JSON.
    #[serde(skip)]
    pub weight_prefix: String,
}

/// RoPE parameters for models with nested rope configuration (Qwen 3.5).
///
/// Unlike the older RopeScaling struct, this encodes the base theta and
/// partial rotary factor directly, used by models with non-standard RoPE.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: f64,
}

fn default_rope_theta() -> f64 {
    10000.0
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
    /// Scaling algorithm type (e.g. "llama3", "linear").
    pub rope_type: String,
    /// Overall scaling factor.
    pub factor: f64,
    /// Factor for high-frequency RoPE dimensions (Llama 3 only).
    #[serde(default)]
    pub high_freq_factor: f64,
    /// Factor for low-frequency RoPE dimensions (Llama 3 only).
    #[serde(default)]
    pub low_freq_factor: f64,
    /// Training context length before scaling was applied (Llama 3 only).
    #[serde(default)]
    pub original_max_position_embeddings: usize,
}

impl ModelConfig {
    /// Load config from a JSON file.
    ///
    /// Handles two config layouts:
    ///   1. Flat configs (Llama, Qwen2, Qwen3-MoE): fields at top level.
    ///   2. Nested configs (Qwen 3.5 VLM): text model fields inside `text_config`,
    ///      with `tie_word_embeddings` at the outer level.
    ///
    /// If `head_dim` is missing from the JSON (e.g. Qwen 2.5), it's computed
    /// as hidden_size / num_attention_heads.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let raw: Value = serde_json::from_str(&contents)?;

        // Detect nested VLM config: if `text_config` exists, extract it and
        // merge top-level fields (like `tie_word_embeddings`) into it.
        let (config_value, weight_prefix) = if let Some(text_config) = raw.get("text_config") {
            let mut merged = text_config.clone();
            // Promote top-level `tie_word_embeddings` into text_config if not already set.
            if let Some(tie) = raw.get("tie_word_embeddings") {
                if merged.get("tie_word_embeddings").is_none() {
                    merged.as_object_mut().unwrap().insert(
                        "tie_word_embeddings".to_string(),
                        tie.clone(),
                    );
                }
            }
            // Detect which VLM wrapper this is.  Gemma 3 text_config has
            // model_type="gemma3_text" and keeps it; Qwen 3.5 VLMs need
            // the type forced to "qwen3_5".
            let text_model_type = merged.get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let weight_prefix = if text_model_type.starts_with("gemma3") {
                // Gemma 3 VLM: text weights under "language_model.model."
                "language_model.model.".to_string()
            } else {
                // Qwen 3.5 VLM: force model_type and use language_model prefix.
                merged.as_object_mut().unwrap().insert(
                    "model_type".to_string(),
                    Value::String("qwen3_5".to_string()),
                );
                "model.language_model.".to_string()
            };
            // Extract rope_theta from nested rope_parameters if present.
            // Clone values first to avoid borrow conflicts.
            let rope_theta_val = merged.get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .cloned();
            let prf_val = merged.get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .cloned();
            if let Some(theta) = rope_theta_val {
                if merged.get("rope_theta").is_none() {
                    merged.as_object_mut().unwrap().insert("rope_theta".to_string(), theta);
                }
            }
            if let Some(prf) = prf_val {
                if merged.get("partial_rotary_factor").is_none() {
                    merged.as_object_mut().unwrap().insert("partial_rotary_factor".to_string(), prf);
                }
            }
            (merged, weight_prefix)
        } else {
            (raw, "model.".to_string())
        };

        let mut config: Self = serde_json::from_value(config_value)?;
        config.weight_prefix = weight_prefix;

        // Mixtral: expert FFN size equals intermediate_size (no separate
        // moe_intermediate_size field in config.json).  Map it so the MoE
        // code path can uniformly use moe_intermediate_size.
        if config.model_type == "mixtral" && config.moe_intermediate_size == 0 && config.num_experts > 0 {
            config.moe_intermediate_size = config.intermediate_size;
        }

        // Apply Gemma 3 defaults for fields omitted by minimal HF configs.
        // Google's Gemma 3 4B config.json only includes a sparse text_config
        // (hidden_size, intermediate_size, num_hidden_layers, sliding_window).
        // The full defaults come from HF's Gemma3TextConfig class.
        if config.model_type == "gemma3_text" || config.model_type == "gemma3" {
            if config.num_attention_heads == 0 { config.num_attention_heads = 8; }
            if config.num_key_value_heads == 0 { config.num_key_value_heads = 4; }
            if config.head_dim == 0 { config.head_dim = 256; }
            if config.vocab_size == 0 { config.vocab_size = 262208; }
            if config.max_position_embeddings == 0 { config.max_position_embeddings = 131072; }
            if config.rope_theta == 0.0 { config.rope_theta = 1_000_000.0; }
            if config.rms_norm_eps == 0.0 { config.rms_norm_eps = 1e-6; }
            if config.sliding_window_pattern == 0 { config.sliding_window_pattern = 6; }
            if config.hidden_activation.is_empty() {
                config.hidden_activation = "gelu_pytorch_tanh".to_string();
            }
            if config.query_pre_attn_scalar == 0.0 {
                config.query_pre_attn_scalar = config.head_dim as f64;
            }
            if config.sliding_window == 0 { config.sliding_window = 1024; }
            if config.rope_local_base_freq == 0.0 { config.rope_local_base_freq = 10_000.0; }
            // Gemma 3 ties embeddings (no separate lm_head.weight).
            // The top-level config may not specify this, so default to true.
            if !config.tie_word_embeddings {
                config.tie_word_embeddings = true;
            }
        }

        if config.head_dim == 0 {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        // Gemma 3: auto-generate layer_types from sliding_window_pattern if not
        // explicitly provided.  Some HF configs include `layer_types` directly,
        // others only provide `sliding_window_pattern` (e.g. 6 = every 6th layer
        // is global).  We normalise to an explicit per-layer array so the forward
        // pass can just index into it.
        if config.layer_types.is_empty() && config.sliding_window_pattern > 0 {
            config.layer_types = (0..config.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % config.sliding_window_pattern == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect();
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

    /// Whether this model has a shared expert alongside routed experts.
    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert_intermediate_size > 0
    }

    /// Effective intermediate size for buffer allocation.
    ///
    /// Dense models use `intermediate_size`.  MoE-only models have no dense FFN
    /// (intermediate_size=0), but still need scratch buffers sized for the
    /// largest operation — the output gate Z projection uses q_dim, and the
    /// shared expert uses shared_expert_intermediate_size.
    pub fn effective_intermediate_size(&self) -> usize {
        if self.intermediate_size > 0 {
            self.intermediate_size
        } else {
            let q_dim = self.num_attention_heads * self.head_dim;
            q_dim
                .max(self.hidden_size)
                .max(self.shared_expert_intermediate_size)
                .max(self.moe_intermediate_size)
        }
    }

    /// Estimate total GPU memory for model weights in bytes.
    ///
    /// Matches the loading logic in loader.rs: projection weights are Q4 when
    /// `quantize` is true, everything else (embeddings, norms, biases, router
    /// gates) stays BF16.
    pub fn estimate_weight_bytes(&self, quantize: bool) -> usize {
        let hidden = self.hidden_size;
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let inter = self.intermediate_size;

        // Helper: byte count for a [m, k] projection weight.
        let proj = |m: usize, k: usize| -> usize {
            if quantize {
                crate::gpu::q4_byte_count(m, k)
            } else {
                m * k * 2 // bf16
            }
        };

        // Embedding table (always BF16).
        let mut total = self.vocab_size * hidden * 2;

        // LM head (BF16, only if untied).
        if !self.tie_word_embeddings {
            total += self.vocab_size * hidden * 2;
        }

        // Final norm weight (BF16).
        total += hidden * 2;

        // Per-layer weights.
        let per_layer = {
            let mut layer = 0usize;

            // Attention projections (Q4 or BF16).
            layer += proj(q_dim, hidden);  // q_proj
            layer += proj(kv_dim, hidden); // k_proj
            layer += proj(kv_dim, hidden); // v_proj
            layer += proj(hidden, q_dim);  // o_proj

            // Norm weights (always BF16, small).
            // Gemma 3 uses 4 norms per layer (sandwich norms); all others use 2.
            layer += hidden * 2; // input_layernorm
            layer += hidden * 2; // post_attention_layernorm
            if matches!(self.arch(), Ok(ModelArch::Gemma3)) {
                layer += hidden * 2; // pre_feedforward_layernorm
                layer += hidden * 2; // post_feedforward_layernorm
            }

            // QKV bias (BF16, only Qwen2).
            if self.arch().map_or(false, |a| a.has_qkv_bias()) {
                layer += (q_dim + kv_dim + kv_dim) * 2;
            }

            // QK-norm weights (BF16, only Qwen3 MoE).
            if self.arch().map_or(false, |a| a.has_qk_norm()) {
                layer += self.head_dim * 2 * 2; // q_norm + k_norm
            }

            // FFN weights.
            if self.is_moe() {
                // Router gate (always BF16).
                layer += self.num_experts * hidden * 2;
                // Expert weights (Q4 or BF16).
                let moe_inter = self.moe_intermediate_size;
                let per_expert = proj(moe_inter, hidden) // gate_proj
                    + proj(moe_inter, hidden)             // up_proj
                    + proj(hidden, moe_inter);            // down_proj
                layer += self.num_experts * per_expert;
                // Shared expert (if present).
                if self.has_shared_expert() {
                    let se = self.shared_expert_intermediate_size;
                    layer += proj(se, hidden) + proj(se, hidden) + proj(hidden, se);
                    layer += 1 * hidden * 2; // shared_expert_gate [1, hidden] bf16
                }
            } else {
                // Dense FFN: gate/up/down projections.
                layer += proj(inter, hidden); // gate_proj
                layer += proj(inter, hidden); // up_proj
                layer += proj(hidden, inter); // down_proj
            }

            layer
        };

        total += per_layer * self.num_hidden_layers;
        total
    }

    /// Compute the number of KV cache blocks that fit in the remaining GPU
    /// memory after weights are loaded.
    ///
    /// Uses 75% of available memory for KV cache, leaving headroom for scratch
    /// buffers, Metal overhead, and macOS.  Result is clamped to [256, 8192].
    pub fn recommended_kv_blocks(&self, gpu_budget: u64, quantize: bool) -> usize {
        let weight_bytes = self.estimate_weight_bytes(quantize) as u64;
        // Reserve 512 MB for scratch buffers and Metal overhead.
        let scratch_overhead = 512 * 1024 * 1024u64;
        let available = gpu_budget.saturating_sub(weight_bytes + scratch_overhead);

        let kv_dim = (self.num_key_value_heads * self.head_dim) as u64;
        // bytes per block = 2 (K+V) × num_layers × BLOCK_SIZE × kv_dim × 2 (bf16)
        let bytes_per_block =
            2 * self.num_hidden_layers as u64 * kv_cache::BLOCK_SIZE as u64 * kv_dim * 2;

        if bytes_per_block == 0 {
            return 8192;
        }

        // Use 75% of available space for KV cache.
        let num_blocks = (available * 3 / 4 / bytes_per_block) as usize;
        num_blocks.clamp(256, 8192)
    }

    /// Whether this model uses the hybrid DeltaNet + GQA architecture.
    pub fn is_hybrid_deltanet(&self) -> bool {
        !self.layer_types.is_empty()
            && self.layer_types.iter().any(|t| t == "linear_attention")
    }

    /// Whether a given layer uses DeltaNet (linear) attention.
    /// Returns false for standard GQA (full attention) layers.
    pub fn is_linear_attention_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx] == "linear_attention"
        } else {
            false
        }
    }

    /// Whether a given layer uses sliding window (local) attention.
    ///
    /// Learning note: Gemma 3 interleaves local layers (attending to the last
    /// `sliding_window` tokens) with global layers (full context).  This saves
    /// memory — local layers don't need to store KV for the entire sequence —
    /// while periodic global layers prevent information loss.
    pub fn is_sliding_attention_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx] == "sliding_attention"
        } else {
            false
        }
    }

    /// Whether the model uses GeGLU (GELU-gated) instead of SwiGLU (SiLU-gated) FFN.
    ///
    /// Learning note: both are gated linear units — gate(x) * up(x) — but differ
    /// in the gate activation.  SiLU (x * sigmoid(x)) is sharper; GELU
    /// (0.5x * (1 + tanh(√(2/π)(x + 0.044715x³)))) is smoother.  Gemma 3 uses
    /// GELU; most other models (Llama, Qwen, Phi) use SiLU.
    pub fn uses_geglu(&self) -> bool {
        self.hidden_activation.contains("gelu")
    }

    /// Whether this model uses sliding window attention (Gemma 3 pattern).
    #[cfg(test)]
    pub fn has_sliding_window(&self) -> bool {
        self.sliding_window > 0
    }

    /// Number of RoPE dimensions for full-attention layers.
    /// Qwen 3.5: partial_rotary_factor=0.25, head_dim=256 → 64 RoPE dims.
    pub fn rotary_dim(&self) -> usize {
        if self.partial_rotary_factor > 0.0 && self.partial_rotary_factor < 1.0 {
            (self.head_dim as f64 * self.partial_rotary_factor) as usize
        } else {
            self.head_dim
        }
    }

    /// Number of layers that need KV cache.
    ///
    /// For Qwen 3.5 hybrid models, only GQA ("full_attention") layers need a
    /// KV cache — DeltaNet ("linear_attention") layers use a fixed-size state.
    /// For Gemma 3, ALL layers need KV cache (both "sliding_attention" and
    /// "full_attention" use standard softmax attention, just with different
    /// context windows).
    pub fn num_kv_layers(&self) -> usize {
        if self.layer_types.is_empty() {
            self.num_hidden_layers
        } else {
            // Only DeltaNet layers skip KV cache.  Sliding attention and full
            // attention both need it.
            self.layer_types
                .iter()
                .filter(|t| t.as_str() != "linear_attention")
                .count()
        }
    }

    /// Build a mapping from layer_idx → kv_pool_idx (None for DeltaNet layers).
    ///
    /// Layers using any form of softmax attention (full or sliding window) get
    /// a KV pool slot.  Only DeltaNet (linear_attention) layers return None.
    pub fn kv_layer_map(&self) -> Vec<Option<usize>> {
        if self.layer_types.is_empty() {
            (0..self.num_hidden_layers).map(Some).collect()
        } else {
            let mut idx = 0;
            self.layer_types
                .iter()
                .map(|t| {
                    if t == "linear_attention" {
                        None
                    } else {
                        // Both "full_attention" and "sliding_attention" need KV.
                        let r = Some(idx);
                        idx += 1;
                        r
                    }
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // Helper to load a config from the models directory if it exists.
    fn load_config(subdir: &str) -> Option<ModelConfig> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join(subdir)
            .join("config.json");
        if path.exists() {
            Some(ModelConfig::from_file(&path).expect(&format!("failed to parse {}", path.display())))
        } else {
            None
        }
    }

    #[test]
    fn test_model_arch_from_model_type() {
        assert_eq!(ModelArch::from_model_type("llama").unwrap(), ModelArch::Llama);
        assert_eq!(ModelArch::from_model_type("mistral").unwrap(), ModelArch::Mistral);
        assert_eq!(ModelArch::from_model_type("qwen2").unwrap(), ModelArch::Qwen2);
        assert_eq!(ModelArch::from_model_type("qwen3_moe").unwrap(), ModelArch::Qwen3Moe);
        assert_eq!(ModelArch::from_model_type("qwen3_5").unwrap(), ModelArch::Qwen3_5);
        assert_eq!(ModelArch::from_model_type("qwen3_5_text").unwrap(), ModelArch::Qwen3_5);
        assert_eq!(ModelArch::from_model_type("qwen3_5_moe").unwrap(), ModelArch::Qwen3_5);
        assert_eq!(ModelArch::from_model_type("qwen3_5_moe_text").unwrap(), ModelArch::Qwen3_5);
        assert_eq!(ModelArch::from_model_type("phi3").unwrap(), ModelArch::Phi);
        assert_eq!(ModelArch::from_model_type("phi4").unwrap(), ModelArch::Phi);
        assert_eq!(ModelArch::from_model_type("gemma3_text").unwrap(), ModelArch::Gemma3);
        assert_eq!(ModelArch::from_model_type("gemma3").unwrap(), ModelArch::Gemma3);
        assert!(ModelArch::from_model_type("gpt2").is_err());
        assert!(ModelArch::from_model_type("").is_err());
    }

    #[test]
    fn test_model_arch_has_qkv_bias() {
        assert!(!ModelArch::Llama.has_qkv_bias());
        assert!(!ModelArch::Mistral.has_qkv_bias());
        assert!(ModelArch::Qwen2.has_qkv_bias());
        assert!(!ModelArch::Qwen3Moe.has_qkv_bias());
        assert!(!ModelArch::Qwen3_5.has_qkv_bias());
        assert!(!ModelArch::Phi.has_qkv_bias());
        assert!(!ModelArch::Gemma3.has_qkv_bias());
    }

    #[test]
    fn test_model_arch_has_qk_norm() {
        assert!(!ModelArch::Llama.has_qk_norm());
        assert!(!ModelArch::Mistral.has_qk_norm());
        assert!(!ModelArch::Qwen2.has_qk_norm());
        assert!(ModelArch::Qwen3Moe.has_qk_norm());
        assert!(ModelArch::Qwen3_5.has_qk_norm());
        assert!(!ModelArch::Phi.has_qk_norm());
        assert!(ModelArch::Gemma3.has_qk_norm());
    }

    #[test]
    fn test_model_arch_has_fused_qkv() {
        assert!(!ModelArch::Llama.has_fused_qkv());
        assert!(!ModelArch::Mistral.has_fused_qkv());
        assert!(!ModelArch::Qwen2.has_fused_qkv());
        assert!(ModelArch::Phi.has_fused_qkv());
        assert!(!ModelArch::Gemma3.has_fused_qkv());
    }

    #[test]
    fn test_parse_llama_config() {
        let Some(config) = load_config("llama-3.2-1b") else { return };
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert!(!config.is_moe());
        assert!(!config.is_hybrid_deltanet());
        assert!(!config.uses_geglu());
        assert_eq!(config.weight_prefix, "model.");
    }

    #[test]
    fn test_parse_qwen2_config() {
        let Some(config) = load_config("qwen-2.5-3b-instruct") else { return };
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.arch().unwrap(), ModelArch::Qwen2);
        assert!(config.arch().unwrap().has_qkv_bias());
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
        // head_dim should be computed from hidden_size / num_attention_heads
        assert!(config.head_dim > 0);
        assert_eq!(config.head_dim, config.hidden_size / config.num_attention_heads);
    }

    #[test]
    fn test_parse_phi_config() {
        let Some(config) = load_config("phi-4") else { return };
        assert_eq!(config.arch().unwrap(), ModelArch::Phi);
        assert!(config.arch().unwrap().has_fused_qkv());
        assert!(!config.arch().unwrap().has_qkv_bias());
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_qwen3_moe_config() {
        let Some(config) = load_config("qwen3-coder-30b-a3b-instruct") else { return };
        assert_eq!(config.arch().unwrap(), ModelArch::Qwen3Moe);
        assert!(config.is_moe());
        assert!(config.num_experts > 0);
        assert!(config.num_experts_per_tok > 0);
        assert!(config.moe_intermediate_size > 0);
        assert!(config.arch().unwrap().has_qk_norm());
    }

    #[test]
    fn test_parse_gemma3_4b_config() {
        let Some(config) = load_config("gemma-3-4b-it") else { return };
        assert_eq!(config.arch().unwrap(), ModelArch::Gemma3);
        // Verify defaults were applied for the sparse 4B config
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.vocab_size, 262208);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 34);
        assert!(config.uses_geglu());
        assert!(config.has_sliding_window());
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_gemma3_27b_config() {
        let Some(config) = load_config("gemma-3-27b-it") else { return };
        assert_eq!(config.arch().unwrap(), ModelArch::Gemma3);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_size, 5376);
        assert_eq!(config.num_hidden_layers, 62);
        assert!(config.uses_geglu());
        assert!(config.has_sliding_window());
    }

    #[test]
    fn test_is_moe() {
        // A config with num_experts=0 is not MoE
        let mut config = minimal_config();
        assert!(!config.is_moe());
        config.num_experts = 8;
        assert!(config.is_moe());
    }

    #[test]
    fn test_has_shared_expert() {
        let mut config = minimal_config();
        assert!(!config.has_shared_expert());
        config.shared_expert_intermediate_size = 512;
        assert!(config.has_shared_expert());
    }

    #[test]
    fn test_effective_intermediate_size_dense() {
        let mut config = minimal_config();
        config.intermediate_size = 8192;
        assert_eq!(config.effective_intermediate_size(), 8192);
    }

    #[test]
    fn test_effective_intermediate_size_moe_only() {
        let mut config = minimal_config();
        config.intermediate_size = 0;
        config.num_attention_heads = 16;
        config.head_dim = 128;
        config.hidden_size = 2048;
        config.moe_intermediate_size = 768;
        config.shared_expert_intermediate_size = 512;
        // q_dim = 16 * 128 = 2048, max(2048, 2048, 512, 768) = 2048
        assert_eq!(config.effective_intermediate_size(), 2048);
    }

    #[test]
    fn test_rotary_dim_full() {
        let mut config = minimal_config();
        config.head_dim = 128;
        config.partial_rotary_factor = 0.0;
        assert_eq!(config.rotary_dim(), 128);
    }

    #[test]
    fn test_rotary_dim_partial() {
        let mut config = minimal_config();
        config.head_dim = 256;
        config.partial_rotary_factor = 0.25;
        assert_eq!(config.rotary_dim(), 64);
    }

    #[test]
    fn test_uses_geglu() {
        let mut config = minimal_config();
        assert!(!config.uses_geglu());
        config.hidden_activation = "gelu_pytorch_tanh".to_string();
        assert!(config.uses_geglu());
    }

    #[test]
    fn test_has_sliding_window() {
        let mut config = minimal_config();
        assert!(!config.has_sliding_window());
        config.sliding_window = 4096;
        assert!(config.has_sliding_window());
    }

    #[test]
    fn test_num_kv_layers_dense() {
        let config = minimal_config();
        assert_eq!(config.num_kv_layers(), 4);
    }

    #[test]
    fn test_num_kv_layers_hybrid() {
        let mut config = minimal_config();
        config.layer_types = vec![
            "linear_attention".into(),
            "linear_attention".into(),
            "linear_attention".into(),
            "full_attention".into(),
        ];
        assert_eq!(config.num_kv_layers(), 1);
    }

    #[test]
    fn test_kv_layer_map_dense() {
        let config = minimal_config();
        assert_eq!(config.kv_layer_map(), vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn test_kv_layer_map_hybrid() {
        let mut config = minimal_config();
        config.layer_types = vec![
            "linear_attention".into(),
            "full_attention".into(),
            "linear_attention".into(),
            "full_attention".into(),
        ];
        assert_eq!(config.kv_layer_map(), vec![None, Some(0), None, Some(1)]);
    }

    #[test]
    fn test_is_hybrid_deltanet() {
        let mut config = minimal_config();
        assert!(!config.is_hybrid_deltanet());
        config.layer_types = vec!["full_attention".into()];
        assert!(!config.is_hybrid_deltanet());
        config.layer_types = vec!["linear_attention".into(), "full_attention".into()];
        assert!(config.is_hybrid_deltanet());
    }

    #[test]
    fn test_is_linear_attention_layer() {
        let mut config = minimal_config();
        config.layer_types = vec![
            "linear_attention".into(),
            "full_attention".into(),
        ];
        assert!(config.is_linear_attention_layer(0));
        assert!(!config.is_linear_attention_layer(1));
        assert!(!config.is_linear_attention_layer(99)); // out of bounds
    }

    #[test]
    fn test_is_sliding_attention_layer() {
        let mut config = minimal_config();
        config.layer_types = vec![
            "sliding_attention".into(),
            "full_attention".into(),
        ];
        assert!(config.is_sliding_attention_layer(0));
        assert!(!config.is_sliding_attention_layer(1));
        assert!(!config.is_sliding_attention_layer(99));
    }

    #[test]
    fn test_sliding_window_pattern_generates_layer_types() {
        // Create a JSON config with sliding_window_pattern=6 and 12 layers
        let json = serde_json::json!({
            "model_type": "gemma3_text",
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window_pattern": 6
        });
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json.to_string()).unwrap();
        let config = ModelConfig::from_file(tmp.path()).unwrap();

        assert_eq!(config.layer_types.len(), 12);
        // Every 6th layer (1-indexed) should be "full_attention"
        // Layer 0: (0+1)%6=1 → sliding
        // Layer 5: (5+1)%6=0 → full
        // Layer 11: (11+1)%6=0 → full
        for (i, lt) in config.layer_types.iter().enumerate() {
            if (i + 1) % 6 == 0 {
                assert_eq!(lt, "full_attention", "layer {i} should be full");
            } else {
                assert_eq!(lt, "sliding_attention", "layer {i} should be sliding");
            }
        }
    }

    #[test]
    fn test_head_dim_computed_when_zero() {
        let json = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "intermediate_size": 8192,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6
        });
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), json.to_string()).unwrap();
        let config = ModelConfig::from_file(tmp.path()).unwrap();
        assert_eq!(config.head_dim, 128); // 2048 / 16
    }

    #[test]
    fn test_estimate_weight_bytes_sanity() {
        let Some(config) = load_config("llama-3.2-1b") else { return };
        let bf16_bytes = config.estimate_weight_bytes(false);
        let q4_bytes = config.estimate_weight_bytes(true);
        // BF16 should be larger than Q4
        assert!(bf16_bytes > q4_bytes, "bf16={bf16_bytes} should be > q4={q4_bytes}");
        // Sanity: 1B model should be roughly 2GB bf16
        assert!(bf16_bytes > 1_000_000_000, "bf16 bytes {bf16_bytes} too small for 1B model");
        assert!(bf16_bytes < 5_000_000_000, "bf16 bytes {bf16_bytes} too large for 1B model");
    }

    #[test]
    fn test_num_heads_per_kv_group() {
        let mut config = minimal_config();
        config.num_attention_heads = 32;
        config.num_key_value_heads = 8;
        assert_eq!(config.num_heads_per_kv_group(), 4);
    }

    /// Create a minimal valid ModelConfig for unit testing.
    fn minimal_config() -> ModelConfig {
        ModelConfig {
            model_type: "llama".to_string(),
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 512,
            vocab_size: 1000,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            rope_scaling: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            full_attention_interval: 0,
            layer_types: vec![],
            partial_rotary_factor: 0.0,
            rope_parameters: None,
            attn_output_gate: false,
            sliding_window: 0,
            sliding_window_pattern: 0,
            query_pre_attn_scalar: 0.0,
            rope_local_base_freq: 0.0,
            hidden_activation: String::new(),
            weight_prefix: "model.".to_string(),
        }
    }
}
