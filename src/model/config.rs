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
// All supported architectures (Llama, Qwen, Phi, Gemma, Mistral, Mixtral,
// Qwen3 MoE, Qwen 3.5, GPT-OSS) share the same core building blocks
// (RMSNorm, GQA, SwiGLU, RoPE).
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
    /// GPT-OSS family (OpenAI, sparse MoE).
    ///
    /// Learning note: GPT-OSS models are Mixture-of-Experts models with a
    /// fraction of total parameters active per token (e.g. 32 experts, top-4 routing).
    ///
    /// Key differences from other MoE models:
    ///   1. **QKV AND O-proj bias**: all projection layers have learned bias,
    ///      including the output projection (unique among supported architectures).
    ///   2. **Router bias**: the MoE router gate has a bias vector (uncommon).
    ///   3. **Clamped SwiGLU**: `clamp(silu(gate) * up, -limit, limit)` with
    ///      `swiglu_limit=7.0` to bound expert activations.
    ///   4. **MXFP4 expert weights**: experts stored as microscaling FP4 on disk,
    ///      dequantized to bf16 during loading (then optionally Q4 quantized).
    ///   5. **Sliding window attention**: alternating local (128-token window) and
    ///      global (full context) layers, same mechanism as Gemma 3.
    ///   6. **Expert biases**: per-expert gate_up and down projections have bias.
    ///   7. **YaRN RoPE**: extended context via YaRN frequency scaling.
    ///   8. **q_dim ≠ hidden_size**: 64×64=4096 attention dim vs 2880 hidden.
    ///   9. **Attention sinks**: per-layer learned sink tokens for efficient attention.
    GptOss,
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
    /// NVIDIA Nemotron-H family (Nemotron 3 Nano 30B-A3B, 120B-A22B).
    ///
    /// Learning note: Nemotron-H is a THREE-WAY hybrid architecture that
    /// interleaves three fundamentally different layer types:
    ///
    ///   1. **Mamba-2 SSM layers** (~44% of layers): Selective State Space Model
    ///      with O(1) per-token cost.  Each layer maintains a fixed-size
    ///      [num_heads, head_dim, state_size] = [64, 64, 128] recurrent state
    ///      matrix per sequence.  Uses depthwise Conv1D for local context
    ///      (no RoPE — positional info comes from the convolution) and input-
    ///      dependent discretization via B, C, dt parameters.
    ///
    ///   2. **MoE FFN layers** (~44%): Mixture-of-Experts feed-forward blocks
    ///      with 128 routed experts + 1 shared expert, top-6 routing.  Uses
    ///      relu-squared activation (NOT SwiGLU) — each expert is just
    ///      up_proj → relu² → down_proj.  Routing uses sigmoid scores with
    ///      an additive correction bias (DeepSeek-V3 style).
    ///
    ///   3. **Self-attention layers** (~12%): Standard GQA with 32 Q-heads,
    ///      2 KV-heads, head_dim=128, and full RoPE.  Only these 6 layers
    ///      need KV cache — the rest use recurrent state or are stateless.
    ///
    /// The layer pattern is encoded in `hybrid_override_pattern` from config.json:
    ///   M = Mamba-2, E = MoE (Experts), * = Attention.
    ///   Example (52-layer 30B): "MEMEM*EMEMEM*EMEMEM*..."
    ///
    /// Each layer is purely ONE type — unlike Qwen 3.5 where each layer pairs
    /// attention/DeltaNet with an FFN, Nemotron-H layers are standalone blocks.
    ///
    /// Weight prefix: `backbone.` (not `model.`), with `mixer` subkey.
    /// Chat template: ChatML.
    NemotronH,
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
            ModelArch::Llama
            | ModelArch::Mistral
            | ModelArch::Mixtral
            | ModelArch::Phi
            | ModelArch::Gemma3 => false,
            ModelArch::Qwen2 | ModelArch::GptOss => true,
            ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::NemotronH => false,
        }
    }

    /// Whether the O (output) projection has a bias vector.
    ///
    /// Learning note: most transformer architectures omit bias from the output
    /// projection.  GPT-OSS-20B is the exception — it has bias on ALL projections
    /// (Q, K, V, and O).
    pub fn has_o_proj_bias(&self) -> bool {
        matches!(self, ModelArch::GptOss) // NemotronH has attention_bias: false
    }

    /// Whether the MoE router gate has a bias vector.
    ///
    /// Learning note: most MoE models (Mixtral, Qwen3-MoE) have a simple linear
    /// router: logits = W @ hidden.  GPT-OSS adds a bias: logits = W @ hidden + b.
    pub fn has_router_bias(&self) -> bool {
        matches!(self, ModelArch::GptOss)
    }

    /// Whether MoE expert FFN projections have bias vectors.
    ///
    /// Learning note: GPT-OSS expert gate_up and down projections both have
    /// per-expert bias, which is stored as a fused tensor across all experts.
    pub fn has_expert_bias(&self) -> bool {
        matches!(self, ModelArch::GptOss)
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
            ModelArch::Llama
            | ModelArch::Mistral
            | ModelArch::Mixtral
            | ModelArch::Qwen2
            | ModelArch::Phi
            | ModelArch::GptOss
            | ModelArch::NemotronH => false,
            ModelArch::Gemma3 => true, // Both 4B and 27B have q_norm/k_norm weights
            ModelArch::Qwen3_5 => true, // GQA layers have QK-norm
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
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
                Ok(ModelArch::Qwen3_5)
            }
            "phi3" | "phi4" => Ok(ModelArch::Phi),
            "gemma3_text" | "gemma3" => Ok(ModelArch::Gemma3),
            "gpt_oss" => Ok(ModelArch::GptOss),
            "nemotron_h" => Ok(ModelArch::NemotronH),
            other => anyhow::bail!(
                "unsupported model_type '{}' (expected 'llama', 'mistral', 'mixtral', 'qwen2', \
                 'qwen3_moe', 'qwen3_5', 'phi3', 'gemma3_text', 'gpt_oss', or 'nemotron_h')",
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
    #[serde(default, alias = "norm_eps", alias = "layer_norm_epsilon")]
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
    #[serde(default, alias = "num_local_experts", alias = "n_routed_experts")]
    pub num_experts: usize,
    /// How many experts are activated (routed to) per token.
    /// Qwen3-Coder-30B-A3B: top-8 experts selected per token.
    /// GPT-OSS-20B: top-4 experts (uses `experts_per_token` in config.json;
    /// handled by post-processing since both field names may coexist).
    #[serde(default)]
    pub num_experts_per_tok: usize,
    /// Alternative field name used by GPT-OSS for num_experts_per_tok.
    /// Merged into num_experts_per_tok during from_file().
    #[serde(default)]
    experts_per_token: usize,
    /// Hidden dimension of each expert's FFN (gate/up/down projections).
    /// Much smaller than `intermediate_size` because there are many experts.
    /// Qwen3-Coder-30B-A3B: 768 per expert (vs 6144 dense intermediate_size).
    #[serde(default)]
    pub moe_intermediate_size: usize,
    /// Hidden dimension of the shared (always-active) expert's FFN.
    /// Qwen3.5-35B-A3B: 512.  Only present in models with a shared expert.
    #[serde(default, alias = "moe_shared_expert_intermediate_size")]
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

    // --- Nemotron-H Mamba-2 SSM fields ---
    //
    // Learning note: Mamba-2 (Selective State Space Duality) is a recurrent
    // sequence model that maintains a [num_heads, head_dim, state_size] state
    // matrix per layer.  It's related to continuous-time state space models
    // (S4, S5) but uses input-dependent discretization — the B, C, dt
    // parameters are computed from the input at each step, making the model
    // "selective" (it can decide which information to remember or forget).
    //
    // The recurrent update:
    //   state[h] = dA[h] * state[h] + dt[h] * outer(x[h], B[group(h)])
    //   y[h] = state[h] @ C[group(h)] + D[h] * x[h]
    //
    // where dA = exp(-softplus(dt + dt_bias) * exp(A_log)).  The D parameter
    // provides a skip connection from input to output.
    //
    // "Grouped" B/C: instead of per-head B and C vectors (expensive), Mamba-2
    // shares them across groups of heads (similar to GQA vs MHA).  With
    // n_groups=8 and num_heads=64, each group of 8 heads shares one B/C pair.
    /// Number of heads in Mamba-2 SSM layers.
    /// Nemotron-H 30B: 64 heads × head_dim=64 → d_inner=4096.
    #[serde(default)]
    pub mamba_num_heads: usize,
    /// Head dimension for Mamba-2 SSM layers.
    /// Nemotron-H 30B: 64 (d_inner = mamba_num_heads × mamba_head_dim = 4096).
    #[serde(default)]
    pub mamba_head_dim: usize,
    /// State size (d_state) for the Mamba-2 recurrent state matrix.
    /// Each head maintains a [head_dim, state_size] state in f32.
    /// Nemotron-H 30B: 128.
    #[serde(default)]
    pub ssm_state_size: usize,
    /// Number of groups for shared B/C parameters in Mamba-2.
    /// Reduces parameters by sharing B/C across heads within a group.
    /// Nemotron-H 30B: 8 groups with 64 heads → 8 heads per group.
    #[serde(default, alias = "n_groups")]
    pub mamba_n_groups: usize,
    /// Mamba-2 conv kernel size (causal depthwise Conv1D).
    /// Provides local positional context since Mamba has no RoPE.
    /// Nemotron-H: 4 (same as DeltaNet).
    #[serde(default, alias = "conv_kernel")]
    pub mamba_conv_kernel: usize,
    /// Whether Mamba-2 Conv1D has a bias vector.
    /// Nemotron-H: true (unlike DeltaNet which has no conv bias).
    #[serde(default)]
    pub use_conv_bias: bool,
    /// Whether to scale the pre-norm output by 1/√(2 × num_layers).
    /// Prevents signal amplification through deep networks.  Applied after
    /// RMSNorm and before the mixer block (Mamba, MoE, or attention).
    /// Nemotron-H: true.
    #[serde(default)]
    pub rescale_prenorm_residual: bool,
    /// Hybrid layer pattern string from config.json.
    /// Each character maps to one layer: M=Mamba-2, E=MoE, *=Attention.
    /// Parsed into `layer_types` during `from_file()`.
    /// Example: "MEMEM*EMEMEM*..." (52 chars for 52 layers).
    #[serde(default)]
    pub hybrid_override_pattern: String,

    // --- Nemotron-H MoE routing fields ---
    /// Scaling factor for routed expert outputs (DeepSeek-V3 style).
    /// Combined expert output is multiplied by this before residual add.
    /// Nemotron-H: 2.5.
    #[serde(default)]
    pub routed_scaling_factor: f64,
    /// Whether to normalize top-k routing probabilities to sum to 1.
    /// Nemotron-H: true.
    #[serde(default)]
    pub norm_topk_prob: bool,

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

    /// SwiGLU activation clamp limit (GPT-OSS-20B).
    /// When > 0, the SwiGLU output is clamped to [-limit, limit] before the
    /// down projection.  This bounds expert activations to prevent outliers.
    /// Default 0.0 = no clamping (standard SwiGLU for Llama/Qwen/etc.).
    #[serde(default)]
    pub swiglu_limit: f64,

    // --- Weight prefix ---
    // Multimodal models (Qwen 3.5) prefix layer weights with
    // "model.language_model." instead of "model.".  This field is set
    // during config loading, not deserialized from JSON.
    #[serde(skip)]
    pub weight_prefix: String,

    // --- Vision encoder config ---
    // Present when the model is multimodal (VLM).  Parsed from the top-level
    // `vision_config` field in config.json.  Set during from_file(), not
    // deserialized directly (because the VLM wrapper nests text_config).
    #[serde(skip)]
    pub vision: Option<VisionConfig>,

    /// Token ID for image placeholders in the token sequence.
    /// Each image produces N of these tokens, replaced by vision encoder output.
    #[serde(skip)]
    pub image_token_id: Option<u32>,

    /// Token ID marking the start of a vision segment.
    #[serde(skip)]
    pub vision_start_token_id: Option<u32>,

    /// Token ID marking the end of a vision segment.
    #[serde(skip)]
    pub vision_end_token_id: Option<u32>,
}

/// Vision encoder configuration, parsed from `vision_config` in config.json.
///
/// Both Qwen 3.5 and Gemma 3 VLMs use SigLIP-based ViT encoders with very
/// similar architectures (27 layers, 1152 hidden, 16 heads, LayerNorm).
/// Differences are captured by fields like `fused_qkv` and `spatial_merge_size`.
///
/// LEARNING NOTE: This struct is NOT derived via Deserialize — it's constructed
/// manually in `parse_vision_config()` because HuggingFace config.json uses
/// different field names for different model families (Qwen uses "depth" while
/// Gemma uses "num_hidden_layers", etc.).  The manual construction normalises
/// these differences into a single shared representation.
///
/// The `fused_qkv` and `weight_prefix` / `projector_prefix` fields encode the
/// structural differences that matter at weight-loading time (see loader.rs):
///   - Qwen: fused QKV weight [3*hd, hd], prefix "visual."
///   - Gemma: separate Q/K/V weights, prefix "vision_tower.vision_model."
///
/// Related: model/vision.rs (forward pass), model/loader/ (weight loading),
///          gpu/ops/vision.rs (spatial_merge + scatter kernels)
#[derive(Debug, Clone)]
pub(crate) struct VisionConfig {
    /// Patch size for image tokenization (16 for Qwen 3.5, 14 for Gemma 3).
    /// The image is divided into non-overlapping patch_size×patch_size blocks,
    /// each flattened into a vector of in_channels * patch_size² dimensions.
    pub patch_size: usize,
    /// Number of ViT transformer blocks (typically 27).
    /// Both Qwen and Gemma SigLIP encoders use 27 layers — a sweet spot
    /// between quality and latency for the ~400M parameter vision encoder.
    pub depth: usize,
    /// Hidden dimension of the vision encoder (typically 1152).
    /// This is the internal width of the ViT — distinct from the text model's
    /// hidden_size.  A projection layer bridges the two at the end.
    pub hidden_size: usize,
    /// Number of attention heads (typically 16).
    pub num_heads: usize,
    /// FFN intermediate dimension (typically 4304).
    pub intermediate_size: usize,
    /// Spatial merge factor (Qwen: 2, Gemma: 0 = no spatial merge).
    /// When > 0, every merge_size×merge_size block of tokens is concatenated
    /// into a single token, reducing token count by ms² before feeding into
    /// the text model.  See gpu/ops/vision.rs for the kernel.
    pub spatial_merge_size: usize,
    /// Output hidden dimension after projection, matching text model's hidden_size.
    /// The merger/projector MLP maps from vision hidden_size (or hidden_size * ms²
    /// after spatial merge) down to this dimension.
    pub out_hidden_size: usize,
    /// Number of input channels (3 for RGB).
    pub in_channels: usize,
    /// Whether QKV projections are fused into a single weight (Qwen: true, Gemma: false).
    /// When true, the loader splits [3*hd, hd] into separate Q, K, V tensors
    /// at load time so the forward pass can use the same code path for both.
    pub fused_qkv: bool,
    /// Hidden activation function in the vision FFN.
    /// Typically "gelu_pytorch_tanh" — both Qwen and Gemma vision encoders use
    /// plain GELU (not SwiGLU), unlike the text transformer.
    #[allow(dead_code)] // deserialized from config.json; kept for future activation dispatch
    pub hidden_act: String,
    /// Vision weight tensor prefix in safetensors (e.g. "visual." for Qwen,
    /// "vision_tower.vision_model." for Gemma).  Used by the loader to find
    /// tensors in the safetensors file.
    pub weight_prefix: String,
    /// Projector weight prefix (e.g. "visual.merger." for Qwen,
    /// "multi_modal_projector." for Gemma).  The projector bridges the vision
    /// encoder output to the text model's embedding space.
    pub projector_prefix: String,
    /// Minimum total pixels for an image (Qwen default: 3136 = 4×28²).
    /// Images smaller than this are upscaled to ensure enough patches for
    /// the vision encoder to work with.
    pub min_pixels: usize,
    /// Maximum total pixels for a single tile (Qwen default: 401408 = 28²×16²).
    /// Images larger than this are split into multiple tiles, each processed
    /// independently through the vision encoder.
    pub max_pixels: usize,
    /// Full image size the position embedding table was trained for.
    ///
    /// The position embedding table has (image_size / patch_size)² entries —
    /// one for each possible patch position in a full-resolution image.
    /// For sub-resolution images, we select a 2D subgrid of positions rather
    /// than using contiguous indices.  Gemma 3: 896 (64×64 grid = 4096 positions).
    /// Qwen 3.5: 0 (uses contiguous indices — all images resize to the full grid).
    pub image_size: usize,
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
    /// YaRN: controls wavelength threshold for high-frequency dimensions.
    /// Dimensions with wavelength < 2π/beta_fast use standard RoPE (no scaling).
    #[serde(default)]
    pub beta_fast: f64,
    /// YaRN: controls wavelength threshold for low-frequency dimensions.
    /// Dimensions with wavelength > 2π/beta_slow get full linear scaling.
    #[serde(default)]
    pub beta_slow: f64,
}

impl RopeScaling {
    /// YaRN attention scaling factor.
    ///
    /// HuggingFace's YaRN implementation scales the cos/sin values by an
    /// `attention_scaling` factor, effectively multiplying all Q·K dot products
    /// by `attention_scaling²`.  This compensates for the reduced effective
    /// context length after frequency interpolation.
    ///
    /// Formula: `0.1 * ln(factor) + 1.0` (from the YaRN paper, Section 3.3).
    /// Returns 1.0 for non-YaRN scaling types.
    pub fn attention_scaling(&self) -> f64 {
        if self.rope_type == "yarn" && self.factor > 1.0 {
            0.1 * self.factor.ln() + 1.0
        } else {
            1.0
        }
    }
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
        // Some HuggingFace configs use JavaScript-style `Infinity` / `-Infinity`
        // literals (e.g. Nemotron-H's time_step_limit), which aren't valid JSON.
        // Replace them with large finite values that serde_json can parse.
        // The literal can appear after ": " or standalone on a line (in arrays).
        let contents = contents
            .replace("Infinity", "1e308");
        let raw: Value = serde_json::from_str(&contents)?;

        // Extract vision config and token IDs from the top-level JSON before
        // we destructure `raw` into text_config.  These fields live at the
        // outer level for both Qwen 3.5 and Gemma 3 VLMs.
        let raw_vision_config = raw.get("vision_config").cloned();
        // Qwen uses `image_token_id`, Gemma 3 uses `image_token_index` — check both.
        // Gemma 3 has `"image_token_id": null` alongside `"image_token_index": 262144`,
        // so we must filter null values before falling through to the alternative key.
        let raw_image_token_id = raw.get("image_token_id")
            .filter(|v| !v.is_null())
            .or_else(|| raw.get("image_token_index"))
            .and_then(|v| v.as_u64()).map(|v| v as u32);
        let raw_vision_start = raw.get("vision_start_token_id").and_then(|v| v.as_u64()).map(|v| v as u32);
        let raw_vision_end = raw.get("vision_end_token_id").and_then(|v| v.as_u64()).map(|v| v as u32);

        // Detect nested VLM config: if `text_config` exists, extract it and
        // merge top-level fields (like `tie_word_embeddings`) into it.
        let (mut config_value, weight_prefix) = if let Some(text_config) = raw.get("text_config") {
            let mut merged = text_config.clone();
            // Promote top-level `tie_word_embeddings` into text_config if not already set.
            if let Some(tie) = raw.get("tie_word_embeddings") {
                if merged.get("tie_word_embeddings").is_none() {
                    merged
                        .as_object_mut()
                        .unwrap()
                        .insert("tie_word_embeddings".to_string(), tie.clone());
                }
            }
            // Detect which VLM wrapper this is.  Gemma 3 text_config has
            // model_type="gemma3_text" and keeps it; Qwen 3.5 VLMs need
            // the type forced to "qwen3_5".
            let text_model_type = merged
                .get("model_type")
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
            let rope_theta_val = merged
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .cloned();
            let prf_val = merged
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .cloned();
            if let Some(theta) = rope_theta_val {
                if merged.get("rope_theta").is_none() {
                    merged
                        .as_object_mut()
                        .unwrap()
                        .insert("rope_theta".to_string(), theta);
                }
            }
            if let Some(prf) = prf_val {
                if merged.get("partial_rotary_factor").is_none() {
                    merged
                        .as_object_mut()
                        .unwrap()
                        .insert("partial_rotary_factor".to_string(), prf);
                }
            }
            (merged, weight_prefix)
        } else {
            (raw, "model.".to_string())
        };

        // Nemotron-H has both `norm_eps` and `layer_norm_epsilon` in config.json.
        // Both are serde aliases for `rms_norm_eps`, so having both causes a
        // "duplicate field" error.  Drop `layer_norm_epsilon` when `norm_eps` exists.
        if let Some(obj) = config_value.as_object_mut() {
            if obj.contains_key("norm_eps") {
                obj.remove("layer_norm_epsilon");
            }
        }

        let mut config: Self = serde_json::from_value(config_value)?;
        config.weight_prefix = weight_prefix;

        // GPT-OSS uses `experts_per_token` instead of `num_experts_per_tok`.
        // Both may coexist in the config; prefer num_experts_per_tok if set.
        if config.num_experts_per_tok == 0 && config.experts_per_token > 0 {
            config.num_experts_per_tok = config.experts_per_token;
        }

        // Mixtral and GPT-OSS: expert FFN size equals intermediate_size (no
        // separate moe_intermediate_size field in config.json).  Map it so the
        // MoE code path can uniformly use moe_intermediate_size.
        if (config.model_type == "mixtral" || config.model_type == "gpt_oss")
            && config.moe_intermediate_size == 0
            && config.num_experts > 0
        {
            config.moe_intermediate_size = config.intermediate_size;
        }

        // Nemotron-H: parse hybrid_override_pattern into layer_types and set
        // the weight prefix to "backbone." (not "model.").
        if config.model_type == "nemotron_h" {
            config.weight_prefix = "backbone.".to_string();
            // Also ensure moe_intermediate_size is set from intermediate_size
            // if the config uses the same field for both.
            if config.moe_intermediate_size == 0 && config.num_experts > 0 {
                config.moe_intermediate_size = config.intermediate_size;
            }
            // Parse hybrid_override_pattern: M=mamba2, E=moe, *=attention.
            if !config.hybrid_override_pattern.is_empty() && config.layer_types.is_empty() {
                config.layer_types = config
                    .hybrid_override_pattern
                    .chars()
                    .map(|c| match c {
                        'M' => "mamba2".to_string(),
                        'E' => "moe".to_string(),
                        '*' => "attention".to_string(),
                        _ => panic!(
                            "unknown character '{}' in hybrid_override_pattern",
                            c
                        ),
                    })
                    .collect();
            }
        }

        // Apply Gemma 3 defaults for fields omitted by minimal HF configs.
        // Google's Gemma 3 4B config.json only includes a sparse text_config
        // (hidden_size, intermediate_size, num_hidden_layers, sliding_window).
        // The full defaults come from HF's Gemma3TextConfig class.
        if config.model_type == "gemma3_text" || config.model_type == "gemma3" {
            if config.num_attention_heads == 0 {
                config.num_attention_heads = 8;
            }
            if config.num_key_value_heads == 0 {
                config.num_key_value_heads = 4;
            }
            if config.head_dim == 0 {
                config.head_dim = 256;
            }
            if config.vocab_size == 0 {
                config.vocab_size = 262208;
            }
            if config.max_position_embeddings == 0 {
                config.max_position_embeddings = 131072;
            }
            if config.rope_theta == 0.0 {
                config.rope_theta = 1_000_000.0;
            }
            if config.rms_norm_eps == 0.0 {
                config.rms_norm_eps = 1e-6;
            }
            if config.sliding_window_pattern == 0 {
                config.sliding_window_pattern = 6;
            }
            if config.hidden_activation.is_empty() {
                config.hidden_activation = "gelu_pytorch_tanh".to_string();
            }
            if config.query_pre_attn_scalar == 0.0 {
                config.query_pre_attn_scalar = config.head_dim as f64;
            }
            if config.sliding_window == 0 {
                config.sliding_window = 1024;
            }
            if config.rope_local_base_freq == 0.0 {
                config.rope_local_base_freq = 10_000.0;
            }
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

        // Parse vision encoder config from the top-level vision_config field.
        if let Some(vc) = raw_vision_config {
            let is_gemma = config.model_type == "gemma3_text" || config.model_type == "gemma3";
            let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
            let depth = vc.get("depth").and_then(|v| v.as_u64())
                .or_else(|| vc.get("num_hidden_layers").and_then(|v| v.as_u64()))
                .unwrap_or(27) as usize;
            let hidden_size = vc.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1152) as usize;
            let num_heads = vc.get("num_heads").and_then(|v| v.as_u64())
                .or_else(|| vc.get("num_attention_heads").and_then(|v| v.as_u64()))
                .unwrap_or(16) as usize;
            let intermediate_size = vc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(4304) as usize;
            let spatial_merge_size = vc.get("spatial_merge_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out_hidden_size = vc.get("out_hidden_size").and_then(|v| v.as_u64()).unwrap_or(config.hidden_size as u64) as usize;
            let in_channels = vc.get("in_channels").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let hidden_act = vc.get("hidden_act").and_then(|v| v.as_str()).unwrap_or("gelu_pytorch_tanh").to_string();

            config.vision = Some(VisionConfig {
                patch_size,
                depth,
                hidden_size,
                num_heads,
                intermediate_size,
                spatial_merge_size,
                out_hidden_size,
                in_channels,
                fused_qkv: !is_gemma,  // Qwen uses fused QKV, Gemma uses separate
                hidden_act,
                weight_prefix: if is_gemma {
                    "vision_tower.vision_model.".to_string()
                } else {
                    "model.visual.".to_string()
                },
                projector_prefix: if is_gemma {
                    "multi_modal_projector.".to_string()
                } else {
                    "model.visual.merger.".to_string()
                },
                min_pixels: vc.get("min_pixels").and_then(|v| v.as_u64()).unwrap_or(3136) as usize,
                max_pixels: vc.get("max_pixels").and_then(|v| v.as_u64()).unwrap_or(401408) as usize,
                image_size: vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            });
            config.image_token_id = raw_image_token_id;
            config.vision_start_token_id = raw_vision_start;
            config.vision_end_token_id = raw_vision_end;
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
    /// When `quantize` is true (pre-quantized model), projection weights use
    /// Q4 sizes; everything else (embeddings, norms, biases, router gates)
    /// stays BF16.
    ///
    /// The `quantized_proj_bytes` closure returns the byte count for a quantised
    /// [m, k] projection weight.  Callers pass `|m, k| backend.quantized_weight_bytes(m, k)`
    /// so the estimate matches whatever quantisation format the backend uses —
    /// without config needing to know the format itself.
    pub fn estimate_weight_bytes(
        &self,
        quantize: bool,
        quantized_proj_bytes: impl Fn(usize, usize) -> usize,
    ) -> usize {
        let hidden = self.hidden_size;
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let inter = self.intermediate_size;

        // Helper: byte count for a [m, k] projection weight.
        // When quantising, delegates to the backend-provided closure so the
        // estimate matches the backend's actual quantisation format.
        let proj = |m: usize, k: usize| -> usize {
            if quantize {
                quantized_proj_bytes(m, k)
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
            layer += proj(q_dim, hidden); // q_proj
            layer += proj(kv_dim, hidden); // k_proj
            layer += proj(kv_dim, hidden); // v_proj
            layer += proj(hidden, q_dim); // o_proj

            // Norm weights (always BF16, small).
            // Gemma 3 uses 4 norms per layer (sandwich norms); all others use 2.
            layer += hidden * 2; // input_layernorm
            layer += hidden * 2; // post_attention_layernorm
            if matches!(self.arch(), Ok(ModelArch::Gemma3)) {
                layer += hidden * 2; // pre_feedforward_layernorm
                layer += hidden * 2; // post_feedforward_layernorm
            }

            // QKV bias (BF16, Qwen2 and GPT-OSS).
            if self.arch().map_or(false, |a| a.has_qkv_bias()) {
                layer += (q_dim + kv_dim + kv_dim) * 2;
            }
            // O-proj bias (BF16, GPT-OSS only).
            if self.arch().map_or(false, |a| a.has_o_proj_bias()) {
                layer += hidden * 2;
            }

            // QK-norm weights (BF16, only Qwen3 MoE).
            if self.arch().map_or(false, |a| a.has_qk_norm()) {
                layer += self.head_dim * 2 * 2; // q_norm + k_norm
            }

            // FFN weights.
            if self.is_moe() {
                // Router gate (always BF16).
                layer += self.num_experts * hidden * 2;
                // Router bias (GPT-OSS: [num_experts] bf16).
                if self.arch().map_or(false, |a| a.has_router_bias()) {
                    layer += self.num_experts * 2;
                }
                // Expert weights (Q4 or BF16).
                let moe_inter = self.moe_intermediate_size;
                let per_expert = proj(moe_inter, hidden) // gate_proj
                    + proj(moe_inter, hidden)             // up_proj
                    + proj(hidden, moe_inter); // down_proj
                layer += self.num_experts * per_expert;
                // Expert biases (GPT-OSS: gate_bias + up_bias + down_bias per expert, bf16).
                if self.arch().map_or(false, |a| a.has_expert_bias()) {
                    layer += self.num_experts * (2 * moe_inter + hidden) * 2;
                }
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
    pub fn recommended_kv_blocks(
        &self,
        gpu_budget: u64,
        quantize: bool,
        kv_quant: crate::model::turboquant::KvQuantMode,
        quantized_proj_bytes: impl Fn(usize, usize) -> usize,
    ) -> usize {
        let weight_bytes = self.estimate_weight_bytes(quantize, quantized_proj_bytes) as u64;
        // Reserve 512 MB for scratch buffers and Metal overhead.
        let scratch_overhead = 512 * 1024 * 1024u64;
        let available = gpu_budget.saturating_sub(weight_bytes + scratch_overhead);

        // Bytes per position per pool (K or V).
        let bytes_per_pos = crate::model::turboquant::bytes_per_kv_position(
            self.head_dim, self.num_key_value_heads, kv_quant,
        ) as u64;
        // bytes per block = 2 (K+V) × num_layers × BLOCK_SIZE × bytes_per_pos
        let bytes_per_block =
            2 * self.num_hidden_layers as u64 * kv_cache::BLOCK_SIZE as u64 * bytes_per_pos;

        if bytes_per_block == 0 {
            return 8192;
        }

        // Use 75% of available space for KV cache.
        let num_blocks = (available * 3 / 4 / bytes_per_block) as usize;
        num_blocks.clamp(256, 8192)
    }

    /// Whether this model uses the hybrid DeltaNet + GQA architecture.
    pub fn is_hybrid_deltanet(&self) -> bool {
        !self.layer_types.is_empty() && self.layer_types.iter().any(|t| t == "linear_attention")
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
    /// For Nemotron-H, only "attention" layers (the * in the pattern) need KV —
    /// Mamba-2 ("mamba2") and MoE ("moe") layers are stateless or use recurrent
    /// state, not KV cache.
    pub fn num_kv_layers(&self) -> usize {
        if self.layer_types.is_empty() {
            self.num_hidden_layers
        } else {
            self.layer_types
                .iter()
                .filter(|t| Self::layer_needs_kv(t))
                .count()
        }
    }

    /// Build a mapping from layer_idx → kv_pool_idx.
    ///
    /// Layers using softmax attention get a KV pool slot (Some(idx)).
    /// Layers using recurrent state (DeltaNet, Mamba-2) or stateless MoE
    /// return None.
    pub fn kv_layer_map(&self) -> Vec<Option<usize>> {
        if self.layer_types.is_empty() {
            (0..self.num_hidden_layers).map(Some).collect()
        } else {
            let mut idx = 0;
            self.layer_types
                .iter()
                .map(|t| {
                    if Self::layer_needs_kv(t) {
                        let r = Some(idx);
                        idx += 1;
                        r
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Whether a layer type string represents a layer that needs KV cache.
    /// "linear_attention" (DeltaNet) and "mamba2" use recurrent state.
    /// "moe" layers are stateless FFN blocks.
    /// Everything else (full_attention, sliding_attention, attention) needs KV.
    fn layer_needs_kv(layer_type: &str) -> bool {
        !matches!(layer_type, "linear_attention" | "mamba2" | "moe")
    }

    /// Whether this model uses the hybrid Nemotron-H (Mamba-2 + MoE + attention)
    /// architecture.
    #[allow(dead_code)] // Nemotron-H architecture support (future)
    pub fn is_hybrid_mamba2(&self) -> bool {
        !self.layer_types.is_empty() && self.layer_types.iter().any(|t| t == "mamba2")
    }

    /// Whether a given layer is a Mamba-2 SSM layer.
    #[allow(dead_code)] // Nemotron-H architecture support (future)
    pub fn is_mamba2_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.layer_types.len() && self.layer_types[layer_idx] == "mamba2"
    }

    /// Whether a given layer is a standalone MoE FFN layer (Nemotron-H).
    /// Distinct from `is_moe()` which checks if the model has any MoE layers.
    #[allow(dead_code)] // Nemotron-H architecture support (future)
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.layer_types.len() && self.layer_types[layer_idx] == "moe"
    }

    /// Whether a given layer is an attention layer in Nemotron-H.
    #[allow(dead_code)] // Nemotron-H architecture support (future)
    pub fn is_nemotron_attention_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.layer_types.len() && self.layer_types[layer_idx] == "attention"
    }

    /// Mamba-2 inner dimension: d_inner = mamba_num_heads × mamba_head_dim.
    pub fn mamba2_d_inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Mamba-2 in_proj output dimension:
    /// 2 × d_inner (z + x) + 2 × n_groups × state_size (B + C) + num_heads (dt).
    pub fn mamba2_in_proj_dim(&self) -> usize {
        let d_inner = self.mamba2_d_inner();
        2 * d_inner + 2 * self.mamba_n_groups * self.ssm_state_size + self.mamba_num_heads
    }

    /// Mamba-2 conv1d dimension: d_inner + 2 × n_groups × state_size.
    ///
    /// The conv1d operates on the concatenation of [x, B, C] — not just x.
    /// This is because the SSM's B and C parameters also benefit from the
    /// causal convolution (temporal smoothing before the state update).
    pub fn mamba2_conv_dim(&self) -> usize {
        self.mamba2_d_inner() + 2 * self.mamba_n_groups * self.ssm_state_size
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
            Some(
                ModelConfig::from_file(&path)
                    .expect(&format!("failed to parse {}", path.display())),
            )
        } else {
            None
        }
    }

    #[test]
    fn test_model_arch_from_model_type() {
        assert_eq!(
            ModelArch::from_model_type("llama").unwrap(),
            ModelArch::Llama
        );
        assert_eq!(
            ModelArch::from_model_type("mistral").unwrap(),
            ModelArch::Mistral
        );
        assert_eq!(
            ModelArch::from_model_type("qwen2").unwrap(),
            ModelArch::Qwen2
        );
        assert_eq!(
            ModelArch::from_model_type("qwen3_moe").unwrap(),
            ModelArch::Qwen3Moe
        );
        assert_eq!(
            ModelArch::from_model_type("qwen3_5").unwrap(),
            ModelArch::Qwen3_5
        );
        assert_eq!(
            ModelArch::from_model_type("qwen3_5_text").unwrap(),
            ModelArch::Qwen3_5
        );
        assert_eq!(
            ModelArch::from_model_type("qwen3_5_moe").unwrap(),
            ModelArch::Qwen3_5
        );
        assert_eq!(
            ModelArch::from_model_type("qwen3_5_moe_text").unwrap(),
            ModelArch::Qwen3_5
        );
        assert_eq!(ModelArch::from_model_type("phi3").unwrap(), ModelArch::Phi);
        assert_eq!(ModelArch::from_model_type("phi4").unwrap(), ModelArch::Phi);
        assert_eq!(
            ModelArch::from_model_type("gemma3_text").unwrap(),
            ModelArch::Gemma3
        );
        assert_eq!(
            ModelArch::from_model_type("gemma3").unwrap(),
            ModelArch::Gemma3
        );
        assert_eq!(
            ModelArch::from_model_type("nemotron_h").unwrap(),
            ModelArch::NemotronH
        );
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
        assert!(!ModelArch::NemotronH.has_qkv_bias());
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
        assert!(!ModelArch::NemotronH.has_qk_norm());
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
        let Some(config) = load_config("llama-3.2-1b") else {
            return;
        };
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
        let Some(config) = load_config("qwen-2.5-3b-instruct") else {
            return;
        };
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.arch().unwrap(), ModelArch::Qwen2);
        assert!(config.arch().unwrap().has_qkv_bias());
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
        // head_dim should be computed from hidden_size / num_attention_heads
        assert!(config.head_dim > 0);
        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads
        );
    }

    #[test]
    fn test_parse_phi_config() {
        let Some(config) = load_config("phi-4") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Phi);
        assert!(config.arch().unwrap().has_fused_qkv());
        assert!(!config.arch().unwrap().has_qkv_bias());
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_qwen3_moe_config() {
        let Some(config) = load_config("qwen3-coder-30b-a3b-instruct") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Qwen3Moe);
        assert!(config.is_moe());
        assert!(config.num_experts > 0);
        assert!(config.num_experts_per_tok > 0);
        assert!(config.moe_intermediate_size > 0);
        assert!(config.arch().unwrap().has_qk_norm());
    }

    #[test]
    fn test_parse_gemma3_4b_config() {
        let Some(config) = load_config("gemma-3-4b-it") else {
            return;
        };
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
        // Gemma 3 stores image token as `image_token_index` (not `image_token_id`),
        // and has `"image_token_id": null` — parser must handle both.
        assert_eq!(
            config.image_token_id,
            Some(262144),
            "should parse image_token_index for Gemma 3"
        );
        // Vision config: Gemma 3 uses 14×14 patches on 896×896 images → 64×64 grid.
        let vc = config.vision.as_ref().expect("Gemma 3 should have vision config");
        assert_eq!(vc.patch_size, 14);
        assert_eq!(vc.image_size, 896);
        assert_eq!(vc.spatial_merge_size, 0, "Gemma 3 has no spatial merge");
        assert!(!vc.fused_qkv, "Gemma 3 uses separate Q/K/V");
    }

    #[test]
    fn test_parse_gemma3_27b_config() {
        let Some(config) = load_config("gemma-3-27b-it") else {
            return;
        };
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
        assert_eq!(
            config.kv_layer_map(),
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
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
        config.layer_types = vec!["linear_attention".into(), "full_attention".into()];
        assert!(config.is_linear_attention_layer(0));
        assert!(!config.is_linear_attention_layer(1));
        assert!(!config.is_linear_attention_layer(99)); // out of bounds
    }

    #[test]
    fn test_is_sliding_attention_layer() {
        let mut config = minimal_config();
        config.layer_types = vec!["sliding_attention".into(), "full_attention".into()];
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
        let Some(config) = load_config("llama-3.2-1b") else {
            return;
        };
        let q4_proj = |m, k| crate::gpu::q4_byte_count(m, k);
        let bf16_bytes = config.estimate_weight_bytes(false, &q4_proj);
        let q4_bytes = config.estimate_weight_bytes(true, &q4_proj);
        // BF16 should be larger than Q4
        assert!(
            bf16_bytes > q4_bytes,
            "bf16={bf16_bytes} should be > q4={q4_bytes}"
        );
        // Sanity: 1B model should be roughly 2GB bf16
        assert!(
            bf16_bytes > 1_000_000_000,
            "bf16 bytes {bf16_bytes} too small for 1B model"
        );
        assert!(
            bf16_bytes < 5_000_000_000,
            "bf16 bytes {bf16_bytes} too large for 1B model"
        );
    }

    #[test]
    fn test_model_arch_gpt_oss() {
        assert_eq!(
            ModelArch::from_model_type("gpt_oss").unwrap(),
            ModelArch::GptOss
        );
        assert!(ModelArch::GptOss.has_qkv_bias());
        assert!(ModelArch::GptOss.has_o_proj_bias());
        assert!(ModelArch::GptOss.has_router_bias());
        assert!(ModelArch::GptOss.has_expert_bias());
        assert!(!ModelArch::GptOss.has_qk_norm());
        assert!(!ModelArch::GptOss.has_fused_qkv());
    }

    #[test]
    fn test_parse_gpt_oss_config() {
        let Some(config) = load_config("gpt-oss-20b") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::GptOss);
        assert_eq!(config.hidden_size, 2880);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_experts, 32);
        assert_eq!(config.num_experts_per_tok, 4);
        assert_eq!(config.moe_intermediate_size, 2880); // filled from intermediate_size
        assert_eq!(config.vocab_size, 201088);
        assert_eq!(config.sliding_window, 128);
        assert!((config.swiglu_limit - 7.0).abs() < 0.01);
        assert!(config.is_moe());
        // YaRN rope scaling.
        let rs = config
            .rope_scaling
            .as_ref()
            .expect("rope_scaling should be present");
        assert_eq!(rs.rope_type, "yarn");
        assert!((rs.factor - 32.0).abs() < 0.01);
        assert_eq!(rs.original_max_position_embeddings, 4096);
        assert!((rs.beta_fast - 32.0).abs() < 0.01);
        assert!((rs.beta_slow - 1.0).abs() < 0.01);
        // Alternating sliding/full attention layers.
        assert_eq!(config.layer_types.len(), 24);
        assert!(config.is_sliding_attention_layer(0));
        assert!(!config.is_sliding_attention_layer(1));
    }

    #[test]
    fn test_parse_llama_3_1_8b_config() {
        let Some(config) = load_config("llama-3.1-8b") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128); // computed: 4096 / 32
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.vocab_size, 128256);
        assert!(!config.tie_word_embeddings);
        assert!(!config.is_moe());
        assert!(!config.arch().unwrap().has_qkv_bias());
        let rs = config.rope_scaling.as_ref().expect("rope_scaling should be present");
        assert_eq!(rs.rope_type, "llama3");
        assert!((rs.factor - 8.0).abs() < 0.01);
        assert_eq!(rs.original_max_position_embeddings, 8192);
    }

    #[test]
    fn test_parse_llama_3_1_8b_instruct_config() {
        let Some(config) = load_config("llama-3.1-8b-instruct") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.vocab_size, 128256);
        assert!(!config.tie_word_embeddings);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_llama_3_2_1b_instruct_config() {
        let Some(config) = load_config("llama-3.2-1b-instruct") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_llama_3_2_3b_config() {
        let Some(config) = load_config("llama-3.2-3b") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_llama_3_2_3b_instruct_config() {
        let Some(config) = load_config("llama-3.2-3b-instruct") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Llama);
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_parse_mistral_7b_config() {
        let Some(config) = load_config("mistral-7b-instruct-v0.3") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Mistral);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128); // computed: 4096 / 32
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.vocab_size, 32768);
        assert!(!config.tie_word_embeddings);
        assert!(!config.is_moe());
        assert!(!config.arch().unwrap().has_qkv_bias());
    }

    #[test]
    fn test_parse_mixtral_8x7b_config() {
        let Some(config) = load_config("mixtral-8x7b-instruct-v0.1") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Mixtral);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128); // computed: 4096 / 32
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.vocab_size, 32000);
        assert!(!config.tie_word_embeddings);
        assert!(config.is_moe());
        assert_eq!(config.num_experts, 8); // from num_local_experts
        assert_eq!(config.num_experts_per_tok, 2);
        assert!(!config.arch().unwrap().has_qkv_bias());
    }

    #[test]
    fn test_parse_qwen2_5_7b_config() {
        let Some(config) = load_config("qwen2.5-7b-instruct") else {
            return;
        };
        assert_eq!(config.arch().unwrap(), ModelArch::Qwen2);
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 28);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128); // computed: 3584 / 28
        assert_eq!(config.intermediate_size, 18944);
        assert_eq!(config.vocab_size, 152064);
        assert!(!config.tie_word_embeddings);
        assert!(!config.is_moe());
        assert!(config.arch().unwrap().has_qkv_bias());
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
            experts_per_token: 0,
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
            swiglu_limit: 0.0,
            hidden_activation: String::new(),
            weight_prefix: "model.".to_string(),
            vision: None,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            mamba_num_heads: 0,
            mamba_head_dim: 0,
            ssm_state_size: 0,
            mamba_n_groups: 0,
            mamba_conv_kernel: 0,
            use_conv_bias: false,
            rescale_prenorm_residual: false,
            hybrid_override_pattern: String::new(),
            routed_scaling_factor: 0.0,
            norm_topk_prob: false,
        }
    }

    #[test]
    fn test_hybrid_override_pattern_parsing() {
        // Simulate what from_file() does for the Nemotron-H pattern.
        let pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
        let layer_types: Vec<String> = pattern
            .chars()
            .map(|c| match c {
                'M' => "mamba2".to_string(),
                'E' => "moe".to_string(),
                '*' => "attention".to_string(),
                _ => panic!("unknown char"),
            })
            .collect();

        assert_eq!(layer_types.len(), 52);

        // Count each layer type.
        let mamba_count = layer_types.iter().filter(|t| t.as_str() == "mamba2").count();
        let moe_count = layer_types.iter().filter(|t| t.as_str() == "moe").count();
        let attn_count = layer_types.iter().filter(|t| t.as_str() == "attention").count();
        assert_eq!(mamba_count, 23);
        assert_eq!(moe_count, 23);
        assert_eq!(attn_count, 6);

        // Verify specific positions.
        assert_eq!(layer_types[0], "mamba2");   // M at position 0
        assert_eq!(layer_types[1], "moe");      // E at position 1
        assert_eq!(layer_types[5], "attention"); // * at position 5

        // Check that only attention layers need KV cache.
        let kv_count = layer_types.iter().filter(|t| ModelConfig::layer_needs_kv(t)).count();
        assert_eq!(kv_count, 6);
    }

    #[test]
    fn test_nemotron_h_config_helpers() {
        let mut config = minimal_config();
        config.model_type = "nemotron_h".to_string();
        config.mamba_num_heads = 64;
        config.mamba_head_dim = 64;
        config.ssm_state_size = 128;
        config.mamba_n_groups = 8;
        config.layer_types = vec![
            "mamba2".to_string(), "moe".to_string(), "attention".to_string(),
        ];

        assert!(config.is_hybrid_mamba2());
        assert!(config.is_mamba2_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(config.is_nemotron_attention_layer(2));
        assert!(!config.is_mamba2_layer(1));
        assert_eq!(config.mamba2_d_inner(), 4096);
        assert_eq!(config.mamba2_in_proj_dim(), 2 * 4096 + 2 * 8 * 128 + 64);
        // Only the attention layer needs KV.
        assert_eq!(config.num_kv_layers(), 1);
    }

    #[test]
    fn test_parse_nemotron_h_config() {
        let Some(config) = load_config("nemotron-3-30b") else {
            return;
        };
        assert_eq!(config.model_type, "nemotron_h");
        assert_eq!(config.arch().unwrap(), ModelArch::NemotronH);
        assert_eq!(config.hidden_size, 2688);
        assert_eq!(config.num_hidden_layers, 52);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 131072);
        assert_eq!(config.weight_prefix, "backbone.");
        // Mamba-2 fields.
        assert_eq!(config.mamba_num_heads, 64);
        assert_eq!(config.mamba_head_dim, 64);
        assert_eq!(config.ssm_state_size, 128);
        assert_eq!(config.mamba_n_groups, 8);
        assert_eq!(config.mamba_conv_kernel, 4);
        assert!(config.use_conv_bias);
        assert!(config.rescale_prenorm_residual);
        // MoE fields.
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.moe_intermediate_size, 1856);
        assert_eq!(config.shared_expert_intermediate_size, 3712);
        assert!(config.is_moe());
        assert!(config.has_shared_expert());
        // Hybrid layer pattern.
        assert_eq!(config.layer_types.len(), 52);
        assert!(config.is_hybrid_mamba2());
        assert!(config.is_mamba2_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(config.is_nemotron_attention_layer(5));
        // Only 6 attention layers need KV cache.
        assert_eq!(config.num_kv_layers(), 6);
    }
}
