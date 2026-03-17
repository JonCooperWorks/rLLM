// ===========================================================================
// Model weight loading from safetensors format.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Loads the learned weight tensors from disk into GPU memory.  This happens
//   once at startup — after that, all weights live on the GPU.
//
// Safetensors format:
//   HuggingFace's safetensors is a simple, safe tensor serialization format:
//     - Fixed-size JSON header containing tensor names, shapes, dtypes, and
//       byte offsets into the data section
//     - Raw tensor data, tightly packed, zero-copy accessible via mmap
//
//   Unlike pickle (PyTorch's default), safetensors cannot execute arbitrary
//   code — it's just a flat file of numbers with a metadata header.
//
// Single vs. sharded models:
//   Small models (1B) have a single `model.safetensors` file.  Larger models
//   (3B, 8B, 70B) split weights across multiple shard files:
//     model-00001-of-00004.safetensors
//     model-00002-of-00004.safetensors
//     ...
//   An index file (`model.safetensors.index.json`) maps each tensor name to
//   its shard file.  Our `TensorStore` abstraction handles both cases
//   transparently.
//
// Memory-mapping (mmap):
//   Instead of reading multi-GB files into heap allocations, we use `mmap`
//   to map them into the process's virtual address space.  The OS loads pages
//   on demand.  This means near-instant startup and no memory duplication
//   (shared with the OS page cache).
//
// Weight naming convention (shared by all dense transformer architectures):
//   model.embed_tokens.weight                          → [vocab_size, hidden_size]
//   model.layers.{i}.input_layernorm.weight            → [hidden_size]
//   model.layers.{i}.self_attn.q_proj.weight           → [q_dim, hidden_size]
//   model.layers.{i}.self_attn.k_proj.weight           → [kv_dim, hidden_size]
//   model.layers.{i}.self_attn.v_proj.weight           → [kv_dim, hidden_size]
//   model.layers.{i}.self_attn.o_proj.weight           → [hidden_size, q_dim]
//   model.layers.{i}.post_attention_layernorm.weight   → [hidden_size]
//   model.layers.{i}.mlp.gate_proj.weight              → [inter_size, hidden_size]
//   model.layers.{i}.mlp.up_proj.weight                → [inter_size, hidden_size]
//   model.layers.{i}.mlp.down_proj.weight              → [hidden_size, inter_size]
//   model.norm.weight                                  → [hidden_size]
//
// QKV bias (Qwen 2.5 and Qwen 3.5 only — Llama, Phi, Gemma, Mistral have none):
//   model.layers.{i}.self_attn.q_proj.bias             → [q_dim]
//   model.layers.{i}.self_attn.k_proj.bias             → [kv_dim]
//   model.layers.{i}.self_attn.v_proj.bias             → [kv_dim]
//   (O projection and FFN projections have NO bias in either family.)
//
// Tied embeddings:
//   Llama 3.2 1B has `tie_word_embeddings=true`, meaning there is no separate
//   `lm_head.weight` tensor.  Qwen 2.5 always has `tie_word_embeddings=false`.
//   The final output projection reuses the embedding table when tied.
//
// Q4 quantisation (on-load):
//   When `quantize=true`, linear projection weights are converted from bf16 to
//   block-wise 4-bit quantisation during loading.  This reduces memory ~3.2×
//   and speeds up matmul by reducing memory bandwidth.  Norm weights and the
//   embedding table stay in bf16.
// ===========================================================================

use std::collections::HashMap;
use std::path::Path;

use half::bf16;
use memmap2::Mmap;
use safetensors::SafeTensors;

use super::config::{ModelArch, ModelConfig};
use super::tokenizer::Tokenizer;
use crate::gpu::{GpuCore, TensorDtype};

// ---------------------------------------------------------------------------
// TensorStore — abstracts single-file vs sharded safetensors.
//
// For single-file models, there's one shard and no weight map.
// For sharded models, the index JSON maps tensor names to shard indices.
// The caller just calls `store.tensor("name")` and gets back the data
// regardless of which shard file it lives in.
// ---------------------------------------------------------------------------

struct TensorStore<'a> {
    shards: Vec<SafeTensors<'a>>,
    /// Maps tensor names to shard indices.  Empty for single-file models
    /// (all tensors are in shards[0]).
    weight_map: HashMap<String, usize>,
}

impl<'a> TensorStore<'a> {
    fn tensor(&self, name: &str) -> anyhow::Result<safetensors::tensor::TensorView<'a>> {
        if let Some(&idx) = self.weight_map.get(name) {
            self.shards[idx]
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("tensor '{name}' not in shard {idx}: {e}"))
        } else {
            // Single-file: try shard 0.
            self.shards[0]
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("tensor '{name}' not found: {e}"))
        }
    }
}

/// Load safetensors files from a model directory.
///
/// Returns the mmaps (kept alive for the SafeTensors references) and a weight
/// map for sharded models.
fn load_safetensors_files(model_dir: &Path) -> anyhow::Result<(Vec<Mmap>, HashMap<String, usize>)> {
    // Case 1: single model.safetensors file.
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        eprintln!("loading from {}", single.display());
        let file = std::fs::File::open(&single)?;
        let mmap = unsafe { Mmap::map(&file)? };
        return Ok((vec![mmap], HashMap::new()));
    }

    // Case 2: sharded model with index file.
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        anyhow::bail!(
            "no safetensors file found in {} (expected model.safetensors or model.safetensors.index.json)",
            model_dir.display()
        );
    }

    let index_str = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_str)?;
    let wm = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid index: missing weight_map"))?;

    // Collect unique shard filenames (preserving order).
    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    for filename in wm.values() {
        let f = filename.as_str().unwrap().to_string();
        if !file_to_idx.contains_key(&f) {
            file_to_idx.insert(f.clone(), shard_files.len());
            shard_files.push(f);
        }
    }

    // Build weight_map: tensor_name → shard_index.
    let mut weight_map = HashMap::new();
    for (tensor_name, filename) in wm {
        let f = filename.as_str().unwrap();
        weight_map.insert(tensor_name.clone(), file_to_idx[f]);
    }

    // Memory-map each shard file.
    eprintln!(
        "loading from {} shard files in {}",
        shard_files.len(),
        model_dir.display()
    );
    let mmaps: Vec<Mmap> = shard_files
        .iter()
        .map(|f| {
            let path = model_dir.join(f);
            let file = std::fs::File::open(&path)
                .map_err(|e| anyhow::anyhow!("failed to open {f}: {e}"))?;
            unsafe { Mmap::map(&file) }.map_err(|e| anyhow::anyhow!("failed to mmap {f}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;

    Ok((mmaps, weight_map))
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/// All model weights, organised by layer.
/// Generic over `B: GpuCore` — tensors are GPU-resident (Metal buffers,
/// CUDA device pointers, etc.) but this struct doesn't know or care which.
pub(crate) struct ModelWeights<B: GpuCore> {
    /// Token embedding table [vocab_size, hidden_size].
    /// Also used as the LM head weight matrix (tied embeddings).
    pub embed_tokens: B::Tensor,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights<B>>,
    /// Final RMSNorm weight [hidden_size].
    pub norm_weight: B::Tensor,
    /// Separate LM head weight, if not tied to embed_tokens.
    pub lm_head: Option<B::Tensor>,
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) weight structures.
//
// Learning note: in a dense model, each layer has one FFN with gate/up/down
// projections of size [intermediate_size, hidden_size].  In MoE, each layer
// has MANY smaller expert FFNs (e.g. 128 experts of size [768, 2048]).
// A learned router picks the top-k experts per token.
//
// The weight structure reflects this: per layer, there's one router gate
// matrix [num_experts, hidden_size] that produces routing logits, plus
// num_experts copies of the gate/up/down projections at a smaller size.
//
// Memory: despite having 128 experts, only 8 are used per token, so the
// active compute cost is similar to a small dense model.  But ALL expert
// weights must be stored in GPU memory — that's where the 30B total params
// come from (vs. 3B active params).
// ---------------------------------------------------------------------------

/// Weights for a single expert FFN sub-network.
pub(crate) struct ExpertWeights<B: GpuCore> {
    pub gate_proj: B::Tensor, // [moe_inter, hidden_size]
    pub up_proj: B::Tensor,   // [moe_inter, hidden_size]
    pub down_proj: B::Tensor, // [hidden_size, moe_inter]

    // --- Expert biases (GPT-OSS only) ---
    //
    // GPT-OSS expert weights have per-expert bias vectors stored as fused tensors
    // across all experts in the safetensors file.  During loading, the fused
    // gate_up_bias [2*moe_inter] is split into separate gate and up biases.
    pub gate_bias: Option<B::Tensor>, // [moe_inter], or None
    pub up_bias: Option<B::Tensor>,   // [moe_inter], or None
    pub down_bias: Option<B::Tensor>, // [hidden_size], or None
}

/// Weights for a single transformer layer.
pub(crate) struct LayerWeights<B: GpuCore> {
    // --- Attention sub-block ---
    pub input_layernorm: B::Tensor, // RMSNorm weight [hidden_size]
    pub q_proj: B::Tensor,          // Query projection [q_dim, hidden_size]
    pub k_proj: B::Tensor,          // Key projection [kv_dim, hidden_size]
    pub v_proj: B::Tensor,          // Value projection [kv_dim, hidden_size]
    pub o_proj: B::Tensor,          // Output projection [hidden_size, q_dim]

    // --- QKV bias (Qwen 2.5 / Qwen 3.5 only, None for Llama/Phi/Gemma/Mistral/Qwen3Moe) ---
    //
    // Learning note: bias in a linear layer means output = W @ x + b.
    // After computing Q = W_q @ hidden, Qwen adds: Q = Q + b_q.
    //
    // For single-token inference, the bias vector has the same length as
    // the matmul output — so bias-add is just an element-wise vector add.
    // No new GPU kernel needed: reuses the existing `backend.add()`.
    //
    // Bias tensors are always bf16 (1D, small) and never quantised.
    // O projection has NO bias in any supported architecture.
    pub q_bias: Option<B::Tensor>, // [hidden_size], or None for Llama
    pub k_bias: Option<B::Tensor>, // [kv_dim], or None for Llama
    pub v_bias: Option<B::Tensor>, // [kv_dim], or None for Llama

    // --- O-proj bias (GPT-OSS only) ---
    //
    // GPT-OSS has bias on the output projection: attn_out = O @ attn + o_bias.
    // Other architectures have NO bias on O-proj.
    pub o_proj_bias: Option<B::Tensor>, // [hidden_size], or None

    // --- QK-norm (Qwen 3 MoE only) ---
    //
    // Learning note: QK-norm applies per-head RMSNorm to Q and K projections
    // after the linear projection but before RoPE.  The norm weight is
    // [head_dim] and is shared across all heads (applied independently to
    // each head's vector).  This stabilises attention by preventing Q·K
    // dot products from growing too large in deep networks.
    pub q_norm: Option<B::Tensor>, // [head_dim], or None for non-QK-norm models
    pub k_norm: Option<B::Tensor>, // [head_dim], or None for non-QK-norm models

    // --- FFN sub-block (dense models) ---
    pub post_attention_layernorm: B::Tensor, // RMSNorm weight [hidden_size]
    pub gate_proj: B::Tensor,                // Gate projection [inter_size, hidden_size]
    pub up_proj: B::Tensor,                  // Up projection [inter_size, hidden_size]
    pub down_proj: B::Tensor,                // Down projection [hidden_size, inter_size]

    // --- MoE FFN sub-block (MoE models only) ---
    //
    // When present, the dense gate/up/down fields above are unused dummy
    // tensors.  The forward pass dispatches to MoE routing instead.
    pub router_gate: Option<B::Tensor>,            // [num_experts, hidden_size]
    pub router_bias: Option<B::Tensor>,            // [num_experts], GPT-OSS only
    pub experts: Option<Vec<ExpertWeights<B>>>,     // num_experts expert FFNs

    // --- DeltaNet attention (Qwen 3.5 linear attention layers only) ---
    //
    // DeltaNet layers use a fused QKV projection and separate gate projections
    // instead of the standard q/k/v_proj.  When these fields are Some, the
    // q_proj/k_proj/v_proj/o_proj fields above are dummy tensors.
    //
    // Learning note: DeltaNet has different head counts for QK (16) and V (48),
    // with 3 V-heads sharing each QK-head's state matrix.  The fused QKV
    // projection outputs [qk_dim, qk_dim, v_dim] = [2048, 2048, 6144] = [10240].
    pub in_proj_qkv: Option<B::Tensor>,    // [qk_dim*2 + v_dim, hidden_size]
    pub in_proj_a: Option<B::Tensor>,      // [num_v_heads, hidden_size] — decay gate
    pub in_proj_b: Option<B::Tensor>,      // [num_v_heads, hidden_size] — update gate
    pub in_proj_z: Option<B::Tensor>,      // [v_dim, hidden_size] — output gate
    pub conv1d_weight: Option<B::Tensor>,  // [dim, 1, kernel_size] — depthwise Conv1D
    pub linear_out_proj: Option<B::Tensor>, // [hidden_size, v_dim] — DeltaNet output projection
    pub a_log: Option<B::Tensor>,          // [num_v_heads] f32 — log decay rates
    pub dt_bias: Option<B::Tensor>,        // [num_v_heads] f32 — dt bias
    pub linear_norm: Option<B::Tensor>,    // [value_head_dim] bf16 — output norm weight

    // --- Gemma 3 sandwich norms (extra post-norms applied before residual add) ---
    //
    // Learning note: "sandwich norms" wrap each sub-layer in TWO norms:
    //   residual = residual + post_norm(sublayer(pre_norm(residual)))
    // This controls the magnitude of sub-layer outputs before they enter the
    // residual stream, preventing unbounded growth in deep networks.
    // Most models use only pre-norms; Gemma 3 adds post-norms for stability.
    pub pre_feedforward_layernorm: Option<B::Tensor>,   // [hidden_size] — pre-FFN norm (Gemma 3)
    pub post_feedforward_layernorm: Option<B::Tensor>,  // [hidden_size] — post-FFN norm (Gemma 3)

    // --- GQA output gate (Qwen 3.5 full-attention layers with attn_output_gate) ---
    pub attn_z_proj: Option<B::Tensor>,    // [q_dim, hidden_size] — output gate projection

    // --- Shared expert (Qwen 3.5 MoE models with always-active expert) ---
    //
    // The shared expert is a standard SwiGLU FFN that runs alongside the routed
    // experts on every token.  Its output is gated by a learned scalar (sigmoid
    // of a linear projection) before being added to the routed expert output.
    pub shared_expert_gate_proj: Option<B::Tensor>,  // [shared_inter, hidden_size]
    pub shared_expert_up_proj: Option<B::Tensor>,    // [shared_inter, hidden_size]
    pub shared_expert_down_proj: Option<B::Tensor>,  // [hidden_size, shared_inter]
    pub shared_expert_gate: Option<B::Tensor>,       // [1, hidden_size] — scalar gate weight
}

// ---------------------------------------------------------------------------
// MXFP4 dequantization — converts microscaling FP4 packed format to bf16.
//
// MXFP4 (Microscaling FP4) stores weights using 4-bit FP (E2M1 format):
//   - 1 sign bit, 2 exponent bits, 1 mantissa bit → 16 distinct values
//   - Two values packed per byte (low nibble = even index, high nibble = odd)
//   - Block scaling: every `block_size` elements share one scale factor
//
// Layout on disk (for a weight of shape [rows, cols]):
//   blocks: [rows, cols/2] bytes (packed pairs)
//   scales: [rows, cols/block_size] bf16 scale factors
//   bias:   [rows] bf16 per-row bias (optional, added after dequant)
//
// FP4 E2M1 encoding:
//   nibble  value       nibble  value
//   0b0000   0.0        0b1000  -0.0
//   0b0001   0.5        0b1001  -0.5
//   0b0010   1.0        0b1010  -1.0
//   0b0011   1.5        0b1011  -1.5
//   0b0100   2.0        0b1100  -2.0
//   0b0101   3.0        0b1101  -3.0
//   0b0110   4.0        0b1110  -4.0
//   0b0111   6.0        0b1111  -6.0
// ---------------------------------------------------------------------------

/// Lookup table: FP4 E2M1 nibble → f32 value.
const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    // positive (sign=0)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative (sign=1)
];

/// Transpose a row-major bf16 matrix from [rows, cols] to [cols, rows].
///
/// MXFP4 weights are stored transposed relative to HuggingFace convention:
/// on-disk `[in_features, out_features]` but the model expects `[out_features, in_features]`.
fn transpose_bf16(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let src: &[half::bf16] = bytemuck::cast_slice(data);
    assert_eq!(src.len(), rows * cols);
    let mut dst = vec![half::bf16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    bytemuck::cast_slice(&dst).to_vec()
}

/// Decode an E8M0 scale byte to f32.
///
/// E8M0 is an 8-bit exponent-only format used by MXFP4 for per-block scales:
///   value = 2^(byte - 127)
/// Special cases: 0 → 0.0, 255 → NaN.
#[inline]
fn e8m0_to_f32(byte: u8) -> f32 {
    match byte {
        0 => 0.0,
        255 => f32::NAN,
        e => f32::from_bits((e as u32) << 23), // 2^(e-127) via IEEE 754
    }
}

/// Dequantize MXFP4 packed blocks to bf16 using per-block E8M0 scaling.
///
/// Arguments:
///   - `blocks`: packed fp4 data, 2 values per byte, shape [rows, cols/2]
///   - `scales`: E8M0 block scales (1 byte each), shape [rows, num_scale_blocks]
///   - `rows`, `cols`: logical weight shape
///   - `block_size`: number of elements per scale block (typically 32)
///
/// Returns: bf16 bytes for [rows, cols] weight tensor.
fn dequantize_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block_size: usize,
) -> Vec<u8> {
    let num_scale_blocks = (cols + block_size - 1) / block_size;
    let mut out = vec![half::bf16::ZERO; rows * cols];

    for r in 0..rows {
        let row_block_offset = r * (cols / 2);
        let row_scale_offset = r * num_scale_blocks;
        for c in 0..cols {
            let byte_idx = row_block_offset + c / 2;
            let nibble = if c % 2 == 0 {
                blocks[byte_idx] & 0x0F // low nibble
            } else {
                (blocks[byte_idx] >> 4) & 0x0F // high nibble
            };
            let fp4_val = FP4_E2M1_LUT[nibble as usize];
            let scale_idx = row_scale_offset + c / block_size;
            let scale = e8m0_to_f32(scales[scale_idx]);
            out[r * cols + c] = half::bf16::from_f32(fp4_val * scale);
        }
    }

    bytemuck::cast_slice(&out).to_vec()
}

/// Load all model weights from safetensors file(s) into GPU memory.
///
/// When `quantize` is true, linear projection weights (Q/K/V/O/gate/up/down)
/// are quantised from bf16 to Q4 on the CPU during loading.  This reduces
/// memory ~3.2x and speeds up matmul ~1.5-2x.  Norm weights and the embedding
/// table stay in bf16 (they're small and used for lookup/norm, not matmul).
pub(crate) fn load_weights<B: GpuCore>(
    backend: &B,
    model_dir: &Path,
    config: &ModelConfig,
    quantize: bool,
    sharding: Option<&crate::gpu::parallel::ShardingPlan>,
) -> anyhow::Result<ModelWeights<B>> {
    // Sharding support: when provided, each weight is sliced to this rank's
    // portion before uploading to GPU.  For world_size=1 the plan has no
    // entries, so all weights pass through unmodified.  The actual byte-level
    // slicing is performed inside upload_tensor / upload_maybe_quantized
    // when sharding is fully wired up in a future change.  For now, the
    // parameter is threaded through so call sites can start passing plans.
    let _ = sharding;
    // Load safetensors file(s) — handles both single-file and sharded models.
    let (mmaps, weight_map) = load_safetensors_files(model_dir)?;

    // Parse each mmap as a SafeTensors container.
    let shards: Vec<SafeTensors> = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("failed to parse safetensors: {e}"))?;

    let store = TensorStore { shards, weight_map };

    let hidden = config.hidden_size;
    // Q dimension = num_attention_heads × head_dim.  For most models this equals
    // hidden_size, but Qwen3 MoE has hidden=2048 with 32 heads × 128 head_dim = 4096.
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let inter = config.intermediate_size;
    let wp = &config.weight_prefix; // "model." or "model.language_model."

    // Load the embedding table.
    let embed_tokens = upload_tensor(
        &store,
        backend,
        &format!("{wp}embed_tokens.weight"),
        &[config.vocab_size, hidden],
    )?;

    // Check for separate lm_head (untied embeddings, e.g. Llama 3.1 8B+).
    // Note: lm_head.weight is always at the top level (no weight prefix),
    // even for Qwen 3.5 where other weights are under model.language_model.
    let lm_head = if !config.tie_word_embeddings {
        Some(upload_tensor(
            &store,
            backend,
            "lm_head.weight",
            &[config.vocab_size, hidden],
        )?)
    } else {
        None
    };

    // Load per-layer weights.  Projection weights are optionally quantised to Q4.
    let arch = config.arch()?;
    let has_qkv_bias = arch.has_qkv_bias();
    let has_qk_norm = arch.has_qk_norm();
    let is_moe = config.is_moe();
    let moe_inter = config.moe_intermediate_size;
    let num_experts = config.num_experts;
    let head_dim = config.head_dim;
    let is_hybrid = config.is_hybrid_deltanet();
    let has_fused_qkv = arch.has_fused_qkv();
    // Qwen 3.5 and Gemma 3 store RMSNorm weights as residual offsets
    // (effective = 1 + stored_weight).  Both initialise norm weights to zero,
    // so the effective scale starts at 1.0 and learns from there.
    let residual_norm = matches!(arch, ModelArch::Qwen3_5 | ModelArch::Gemma3);
    let is_gemma3 = matches!(arch, ModelArch::Gemma3);
    let is_gpt_oss = matches!(arch, ModelArch::GptOss);
    let has_o_proj_bias = arch.has_o_proj_bias();
    let has_router_bias = arch.has_router_bias();
    let _has_expert_bias = arch.has_expert_bias();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("{wp}layers.{i}");
        let is_deltanet_layer = is_hybrid && config.is_linear_attention_layer(i);

        // Load QKV bias vectors if the architecture has them (Qwen 2.5).
        // Bias tensors are always bf16, never quantised — they're 1D and tiny.
        let (q_bias, k_bias, v_bias) = if has_qkv_bias {
            (
                Some(upload_tensor(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.q_proj.bias"),
                    &[q_dim],
                )?),
                Some(upload_tensor(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.k_proj.bias"),
                    &[kv_dim],
                )?),
                Some(upload_tensor(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.v_proj.bias"),
                    &[kv_dim],
                )?),
            )
        } else {
            (None, None, None)
        };

        // Load QK-norm weights if the architecture has them (Qwen 3 MoE, Qwen 3.5 GQA).
        // These are per-head RMSNorm weights [head_dim], applied to Q and K
        // after projection but before RoPE.
        // For hybrid models, only GQA (full_attention) layers have QK-norm —
        // DeltaNet layers use L2 normalization instead (no learned weights).
        let layer_has_qk_norm = has_qk_norm && !is_deltanet_layer;
        let (q_norm, k_norm) = if layer_has_qk_norm {
            // Qwen 3.5 uses Qwen3_5RMSNorm for QK-norm, which has residual weights:
            // effective_weight = 1.0 + stored_weight (stored ≈ 0, effective ≈ 1).
            // Other models (Qwen3-MoE) use standard RMSNorm (weight used directly).
            let upload_qk_norm = if residual_norm { upload_norm_residual } else { upload_tensor };
            (
                Some(upload_qk_norm(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[head_dim],
                )?),
                Some(upload_qk_norm(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[head_dim],
                )?),
            )
        } else {
            (None, None)
        };

        // Load FFN weights: either dense (gate/up/down) or MoE (router + experts).
        //
        // For MoE models, the dense FFN fields get dummy zero-element tensors
        // (never used — the forward pass dispatches to MoE routing instead).
        // This avoids making the fields Optional and cascading changes everywhere.
        //
        // Two MoE weight formats exist:
        //   1. Per-expert (Qwen3-MoE): mlp.experts.{j}.gate_proj.weight [moe_inter, hidden]
        //   2. Fused (Qwen3.5): mlp.experts.gate_up_proj [num_experts, 2*moe_inter, hidden]
        //      and mlp.experts.down_proj [num_experts, hidden, moe_inter]
        //   Format 2 is detected by the presence of the fused tensor name.
        let has_shared_expert = config.has_shared_expert();
        let shared_inter = config.shared_expert_intermediate_size;
        let (gate_proj, up_proj, down_proj, router_gate, router_bias_weight, experts,
         shared_expert_gate_proj, shared_expert_up_proj, shared_expert_down_proj,
         shared_expert_gate) = if is_moe {
            // Dummy tensors for the dense FFN fields (never accessed by MoE forward pass).
            let dummy = backend.alloc_tensor(&[1], TensorDtype::BF16);
            let dummy2 = backend.alloc_tensor(&[1], TensorDtype::BF16);
            let dummy3 = backend.alloc_tensor(&[1], TensorDtype::BF16);

            // Router gate: [num_experts, hidden_size].  Stays bf16 — routing
            // accuracy matters more than speed here (it's a single small matmul).
            //
            // Router naming varies by architecture:
            //   Mixtral:  block_sparse_moe.gate.weight
            //   GPT-OSS: mlp.router.weight (+ mlp.router.bias)
            //   Others:   mlp.gate.weight
            let is_mixtral = matches!(arch, ModelArch::Mixtral);
            let router_name = if is_mixtral {
                format!("{prefix}.block_sparse_moe.gate.weight")
            } else if is_gpt_oss {
                format!("{prefix}.mlp.router.weight")
            } else {
                format!("{prefix}.mlp.gate.weight")
            };
            let router = upload_tensor(
                &store,
                backend,
                &router_name,
                &[num_experts, hidden],
            )?;

            // Router bias (GPT-OSS only).
            let router_bias_tensor = if has_router_bias {
                Some(upload_tensor(
                    &store,
                    backend,
                    &format!("{prefix}.mlp.router.bias"),
                    &[num_experts],
                )?)
            } else {
                None
            };

            // Detect MXFP4 format (GPT-OSS) by looking for packed blocks tensor.
            let mxfp4_name = format!("{prefix}.mlp.experts.gate_up_proj_blocks");
            let is_mxfp4 = store.tensor(&mxfp4_name).is_ok();

            // Detect fused vs per-expert format.
            let fused_name = format!("{prefix}.mlp.experts.gate_up_proj");
            let is_fused = !is_mxfp4 && store.tensor(&fused_name).is_ok();

            let expert_vec = if is_mxfp4 {
                // -----------------------------------------------------------
                // MXFP4 format (GPT-OSS): weights stored as packed fp4 blocks
                // with per-block bf16 scales and optional per-expert biases.
                //
                // Tensors:
                //   gate_up_proj_blocks: [num_experts, 2*moe_inter, hidden/2]  (packed nibbles)
                //   gate_up_proj_scales: [num_experts, 2*moe_inter, num_blocks] (bf16)
                //   gate_up_proj_bias:   [num_experts, 2*moe_inter]  (bf16, optional)
                //   down_proj_blocks:    [num_experts, hidden, moe_inter/2]
                //   down_proj_scales:    [num_experts, hidden, num_blocks]
                //   down_proj_bias:      [num_experts, hidden]  (bf16, optional)
                //
                // We dequant each expert's slice to bf16, then optionally Q4.
                // -----------------------------------------------------------
                let block_size = 32usize; // MXFP4 standard block size

                let gu_blocks_view = store.tensor(&mxfp4_name)?;
                let gu_scales_view = store.tensor(
                    &format!("{prefix}.mlp.experts.gate_up_proj_scales"),
                )?;
                let gu_blocks_data = gu_blocks_view.data();
                let gu_scales_data = gu_scales_view.data();

                let down_blocks_view = store.tensor(
                    &format!("{prefix}.mlp.experts.down_proj_blocks"),
                )?;
                let down_scales_view = store.tensor(
                    &format!("{prefix}.mlp.experts.down_proj_scales"),
                )?;
                let down_blocks_data = down_blocks_view.data();
                let down_scales_data = down_scales_view.data();

                // Optional biases.
                let gu_bias_name = format!("{prefix}.mlp.experts.gate_up_proj_bias");
                let gu_bias_data = store.tensor(&gu_bias_name).ok().map(|v| v.data().to_vec());
                let down_bias_name = format!("{prefix}.mlp.experts.down_proj_bias");
                let down_bias_data = store.tensor(&down_bias_name).ok().map(|v| v.data().to_vec());

                // Per-expert byte sizes for slicing fused tensors.
                let gu_rows = 2 * moe_inter;
                let gu_blocks_per_expert = gu_rows * (hidden / 2);
                let gu_num_scale_blocks = (hidden + block_size - 1) / block_size;
                let gu_scales_per_expert = gu_rows * gu_num_scale_blocks; // E8M0: 1 byte each

                let down_rows = hidden;
                let down_blocks_per_expert = down_rows * (moe_inter / 2);
                let down_num_scale_blocks = (moe_inter + block_size - 1) / block_size;
                let down_scales_per_expert = down_rows * down_num_scale_blocks; // E8M0: 1 byte each

                let gu_bias_per_expert = gu_rows * 2; // bf16
                let down_bias_per_expert = down_rows * 2; // bf16

                let mut experts = Vec::with_capacity(num_experts);
                for j in 0..num_experts {
                    // Dequant gate_up: on-disk [2*moe_inter, hidden] after unpacking MXFP4.
                    //
                    // MXFP4 stores weights transposed relative to HuggingFace convention:
                    //   On-disk blocks shape: [experts, 2*moe_inter, blocks, bytes] → dequant → [2*moe_inter, hidden]
                    //   HF model sees: [experts, hidden, 2*moe_inter] (via x @ W, not W @ x)
                    //
                    // Our matmul convention: y = W @ x with W = [out, in].
                    // So we need gate = [moe_inter, hidden] and up = [moe_inter, hidden].
                    //
                    // The on-disk [2*moe_inter, hidden] is exactly [out, in] for our convention.
                    // (HF transposes it to [in, out] for torch's x @ W.T convention.)
                    // Split first half = gate [moe_inter, hidden], second half = up [moe_inter, hidden].
                    let gu_b_off = j * gu_blocks_per_expert;
                    let gu_s_off = j * gu_scales_per_expert;
                    let gu_bf16 = dequantize_mxfp4(
                        &gu_blocks_data[gu_b_off..gu_b_off + gu_blocks_per_expert],
                        &gu_scales_data[gu_s_off..gu_s_off + gu_scales_per_expert],
                        gu_rows, hidden, block_size,
                    );

                    // Split gate [moe_inter, hidden] and up [moe_inter, hidden].
                    let gate_bytes = moe_inter * hidden * 2;
                    let gate_raw = &gu_bf16[..gate_bytes];
                    let up_raw = &gu_bf16[gate_bytes..2 * gate_bytes];
                    let gate_t = upload_raw_maybe_quantized(backend, gate_raw, &[moe_inter, hidden], quantize);
                    let up_t = upload_raw_maybe_quantized(backend, up_raw, &[moe_inter, hidden], quantize);

                    // Dequant down: on-disk [hidden, moe_inter] → our convention [hidden, moe_inter] = [out, in].
                    let d_b_off = j * down_blocks_per_expert;
                    let d_s_off = j * down_scales_per_expert;
                    let down_bf16 = dequantize_mxfp4(
                        &down_blocks_data[d_b_off..d_b_off + down_blocks_per_expert],
                        &down_scales_data[d_s_off..d_s_off + down_scales_per_expert],
                        down_rows, moe_inter, block_size,
                    );
                    let down_t = upload_raw_maybe_quantized(backend, &down_bf16, &[hidden, moe_inter], quantize);

                    // Expert biases — split fused gate_up_bias into separate gate and up.
                    let (gate_bias, up_bias) = if let Some(ref bias) = gu_bias_data {
                        let off = j * gu_bias_per_expert;
                        let bias_slice = &bias[off..off + gu_bias_per_expert];
                        // Each element is bf16 (2 bytes), so split at moe_inter * 2 bytes.
                        let gate_bytes = moe_inter * 2;
                        let gate_part = &bias_slice[..gate_bytes];
                        let up_part = &bias_slice[gate_bytes..];
                        (
                            Some(backend.upload_tensor(gate_part, &[moe_inter], TensorDtype::BF16)),
                            Some(backend.upload_tensor(up_part, &[moe_inter], TensorDtype::BF16)),
                        )
                    } else {
                        (None, None)
                    };
                    let down_bias = if let Some(ref bias) = down_bias_data {
                        let off = j * down_bias_per_expert;
                        let bias_slice = &bias[off..off + down_bias_per_expert];
                        Some(backend.upload_tensor(bias_slice, &[down_rows], TensorDtype::BF16))
                    } else {
                        None
                    };

                    experts.push(ExpertWeights {
                        gate_proj: gate_t,
                        up_proj: up_t,
                        down_proj: down_t,
                        gate_bias,
                        up_bias,
                        down_bias,
                    });
                }
                experts
            } else if is_fused {
                // Fused format (Qwen3.5): gate_up_proj [num_experts, 2*moe_inter, hidden]
                // and down_proj [num_experts, hidden, moe_inter].
                // Split into per-expert tensors during loading.
                let gate_up_view = store.tensor(&fused_name)?;
                anyhow::ensure!(
                    gate_up_view.shape() == [num_experts, moe_inter * 2, hidden],
                    "fused gate_up_proj shape mismatch: expected [{}, {}, {}], got {:?}",
                    num_experts, moe_inter * 2, hidden, gate_up_view.shape()
                );
                let down_view = store.tensor(
                    &format!("{prefix}.mlp.experts.down_proj"),
                )?;
                anyhow::ensure!(
                    down_view.shape() == [num_experts, hidden, moe_inter],
                    "fused down_proj shape mismatch: expected [{}, {}, {}], got {:?}",
                    num_experts, hidden, moe_inter, down_view.shape()
                );

                let gate_up_data = gate_up_view.data();
                let down_data = down_view.data();
                let gate_up_expert_bytes = moe_inter * 2 * hidden * 2; // bf16
                let gate_bytes = moe_inter * hidden * 2;
                let down_expert_bytes = hidden * moe_inter * 2;

                let mut experts = Vec::with_capacity(num_experts);
                for j in 0..num_experts {
                    let gu_offset = j * gate_up_expert_bytes;
                    let gate_raw = &gate_up_data[gu_offset..gu_offset + gate_bytes];
                    let up_raw = &gate_up_data[gu_offset + gate_bytes..gu_offset + gate_up_expert_bytes];
                    let d_offset = j * down_expert_bytes;
                    let down_raw = &down_data[d_offset..d_offset + down_expert_bytes];

                    let gate_t = upload_raw_maybe_quantized(backend, gate_raw, &[moe_inter, hidden], quantize);
                    let up_t = upload_raw_maybe_quantized(backend, up_raw, &[moe_inter, hidden], quantize);
                    let down_t = upload_raw_maybe_quantized(backend, down_raw, &[hidden, moe_inter], quantize);

                    experts.push(ExpertWeights {
                        gate_proj: gate_t,
                        up_proj: up_t,
                        down_proj: down_t,
                        gate_bias: None,
                        up_bias: None,
                        down_bias: None,
                    });
                }
                experts
            } else {
                // Per-expert format: separate tensors per expert.
                //
                // Two naming conventions exist:
                //   Qwen3-MoE:  mlp.experts.{j}.gate_proj / up_proj / down_proj
                //   Mixtral:    block_sparse_moe.experts.{j}.w1 / w3 / w2
                //               (w1=gate, w3=up, w2=down — Mixtral convention)
                let mut experts = Vec::with_capacity(num_experts);
                for j in 0..num_experts {
                    let (gate_name, up_name, down_name) = if is_mixtral {
                        let ep = format!("{prefix}.block_sparse_moe.experts.{j}");
                        (
                            format!("{ep}.w1.weight"),
                            format!("{ep}.w3.weight"),
                            format!("{ep}.w2.weight"),
                        )
                    } else {
                        let ep = format!("{prefix}.mlp.experts.{j}");
                        (
                            format!("{ep}.gate_proj.weight"),
                            format!("{ep}.up_proj.weight"),
                            format!("{ep}.down_proj.weight"),
                        )
                    };
                    experts.push(ExpertWeights {
                        gate_proj: upload_maybe_quantized(
                            &store, backend,
                            &gate_name,
                            &[moe_inter, hidden], quantize,
                        )?,
                        up_proj: upload_maybe_quantized(
                            &store, backend,
                            &up_name,
                            &[moe_inter, hidden], quantize,
                        )?,
                        down_proj: upload_maybe_quantized(
                            &store, backend,
                            &down_name,
                            &[hidden, moe_inter], quantize,
                        )?,
                        gate_bias: None,
                        up_bias: None,
                        down_bias: None,
                    });
                }
                experts
            };
            if i == 0 {
                eprintln!(
                    "  loading {} experts per layer (moe_inter={}){}",
                    num_experts, moe_inter,
                    if is_mxfp4 { " [MXFP4 format]" } else if is_fused { " [fused format]" } else { "" },
                );
            }

            // Load shared expert weights if present.
            let (se_gate_proj, se_up_proj, se_down_proj, se_gate) = if has_shared_expert {
                let gp = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    &[shared_inter, hidden], quantize,
                )?;
                let up = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    &[shared_inter, hidden], quantize,
                )?;
                let dp = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    &[hidden, shared_inter], quantize,
                )?;
                let sg = upload_tensor(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert_gate.weight"),
                    &[1, hidden],
                )?;
                if i == 0 {
                    eprintln!("  shared expert: inter={}", shared_inter);
                }
                (Some(gp), Some(up), Some(dp), Some(sg))
            } else {
                (None, None, None, None)
            };

            (dummy, dummy2, dummy3, Some(router), router_bias_tensor, Some(expert_vec),
             se_gate_proj, se_up_proj, se_down_proj, se_gate)
        } else if has_fused_qkv {
            // -----------------------------------------------------------------
            // Phi FFN: fused gate_up_proj → split into separate gate and up.
            //
            // Phi stores gate and up projections as a single tensor:
            //   gate_up_proj shape = [2 * inter_size, hidden_size]
            //
            // Layout (row-major): first inter_size rows are gate, next inter_size
            // rows are up.  We slice the raw bytes and upload separately so the
            // forward pass can call gate and up as individual matvecs.
            //
            // Why fused in the first place?  Microsoft's original implementation
            // does `gate_up = gate_up_proj(x)` then splits the output, saving one
            // kernel launch vs two separate matmuls.  We split on load instead so
            // our forward pass stays architecture-generic.
            // -----------------------------------------------------------------
            let view = store.tensor(&format!("{prefix}.mlp.gate_up_proj.weight"))?;
            let raw = view.data();
            anyhow::ensure!(
                view.shape() == [2 * inter, hidden],
                "gate_up_proj shape mismatch: expected [{}, {}], got {:?}",
                2 * inter, hidden, view.shape()
            );
            let row_bytes = hidden * 2; // bf16
            let half = inter * row_bytes;
            let gate_raw = &raw[..half];
            let up_raw = &raw[half..2 * half];

            let gate = upload_raw_maybe_quantized(backend, gate_raw, &[inter, hidden], quantize);
            let up = upload_raw_maybe_quantized(backend, up_raw, &[inter, hidden], quantize);
            let down = upload_maybe_quantized(
                &store,
                backend,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, inter],
                quantize,
            )?;
            (gate, up, down, None, None, None, None, None, None, None)
        } else {
            // Dense FFN: standard gate/up/down projections.
            let gate = upload_maybe_quantized(
                &store,
                backend,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[inter, hidden],
                quantize,
            )?;
            let up = upload_maybe_quantized(
                &store,
                backend,
                &format!("{prefix}.mlp.up_proj.weight"),
                &[inter, hidden],
                quantize,
            )?;
            let down = upload_maybe_quantized(
                &store,
                backend,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, inter],
                quantize,
            )?;
            (gate, up, down, None, None, None, None, None, None, None)
        };

        // Load attention weights: DeltaNet layers use linear_attn namespace,
        // GQA layers use self_attn namespace.
        let (q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, conv1d_weight, linear_out_proj, a_log, dt_bias, linear_norm, attn_z_proj) =
            if is_deltanet_layer {
                let dummy = backend.alloc_tensor(&[1], TensorDtype::BF16);
                let dummy2 = backend.alloc_tensor(&[1], TensorDtype::BF16);
                let dummy3 = backend.alloc_tensor(&[1], TensorDtype::BF16);
                let dummy4 = backend.alloc_tensor(&[1], TensorDtype::BF16);

                let qk_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let fused_dim = qk_dim * 2 + v_dim; // Q + K + V fused
                let conv_dim = fused_dim; // Conv1D applied to concatenated QKV
                let kernel_size = config.linear_conv_kernel_dim;

                let qkv = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    &[fused_dim, hidden], quantize,
                )?;
                let a = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    &[config.linear_num_value_heads, hidden], quantize,
                )?;
                let b = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    &[config.linear_num_value_heads, hidden], quantize,
                )?;
                let z = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_z.weight"),
                    &[v_dim, hidden], quantize,
                )?;
                // Conv1D: depthwise, shape [channels, 1, kernel_size] in safetensors.
                let conv = upload_tensor(
                    &store, backend,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    &[conv_dim, 1, kernel_size],
                )?;
                let out = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.linear_attn.out_proj.weight"),
                    &[hidden, v_dim], quantize,
                )?;

                // DeltaNet-specific non-projection weights:
                //   A_log [num_v_heads] — log decay rates (stored f32, kept f32)
                //   dt_bias [num_v_heads] — timestep bias (stored bf16, converted to f32)
                //   norm.weight [value_head_dim] — output norm weight (stored f32, converted to bf16)
                let a_log_tensor = upload_tensor(
                    &store, backend,
                    &format!("{prefix}.linear_attn.A_log"),
                    &[config.linear_num_value_heads],
                )?;
                // dt_bias: convert bf16 → f32 for precision in decay gate computation.
                let dt_bias_view = store.tensor(
                    &format!("{prefix}.linear_attn.dt_bias"),
                )?;
                let dt_bias_f32: Vec<f32> = bytemuck::cast_slice::<u8, half::bf16>(dt_bias_view.data())
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let dt_bias_tensor = backend.upload_tensor(
                    bytemuck::cast_slice(&dt_bias_f32),
                    &[config.linear_num_value_heads],
                    TensorDtype::F32,
                );
                // norm.weight: convert f32 → bf16 for compatibility with rms_norm_batch.
                // Note: linear_attn.norm uses Qwen3_5MoeRMSNormGated which does NOT
                // use residual form — weights are initialized to ones, not zeros.
                // Only the layer norms use (1 + weight) residual form.
                let norm_view = store.tensor(
                    &format!("{prefix}.linear_attn.norm.weight"),
                )?;
                let norm_bf16: Vec<half::bf16> = bytemuck::cast_slice::<u8, f32>(norm_view.data())
                    .iter()
                    .map(|v| half::bf16::from_f32(*v))
                    .collect();
                let norm_weight = backend.upload_tensor(
                    bytemuck::cast_slice(&norm_bf16),
                    &[config.linear_value_head_dim],
                    TensorDtype::BF16,
                );

                if i == 0 {
                    eprintln!(
                        "  DeltaNet layers: qk_dim={}, v_dim={}, kernel_size={}",
                        qk_dim, v_dim, kernel_size
                    );
                }

                (dummy, dummy2, dummy3, dummy4,
                 Some(qkv), Some(a), Some(b), Some(z), Some(conv), Some(out),
                 Some(a_log_tensor), Some(dt_bias_tensor), Some(norm_weight), None)
            } else if has_fused_qkv {
                // -----------------------------------------------------------------
                // Phi attention: fused qkv_proj → split into separate Q, K, V.
                //
                // Phi stores all three projections as a single tensor:
                //   qkv_proj shape = [q_dim + 2*kv_dim, hidden_size]
                //
                // For Phi-4 (40Q/10KV heads, head_dim=128):
                //   q_dim  = 40 * 128 = 5120
                //   kv_dim = 10 * 128 = 1280
                //   fused  = 5120 + 1280 + 1280 = 7680 rows
                //
                // Layout (row-major): first q_dim rows = Q, next kv_dim = K,
                // next kv_dim = V.  We slice the raw bytes at the correct
                // offsets and upload as three separate GPU tensors.
                // -----------------------------------------------------------------
                let fused_dim = q_dim + 2 * kv_dim;
                let view = store.tensor(&format!("{prefix}.self_attn.qkv_proj.weight"))?;
                let raw = view.data();
                anyhow::ensure!(
                    view.shape() == [fused_dim, hidden],
                    "qkv_proj shape mismatch: expected [{}, {}], got {:?}",
                    fused_dim, hidden, view.shape()
                );
                let row_bytes = hidden * 2; // bf16
                let q_bytes = q_dim * row_bytes;
                let kv_bytes = kv_dim * row_bytes;
                let q_raw = &raw[..q_bytes];
                let k_raw = &raw[q_bytes..q_bytes + kv_bytes];
                let v_raw = &raw[q_bytes + kv_bytes..q_bytes + 2 * kv_bytes];

                let qp = upload_raw_maybe_quantized(backend, q_raw, &[q_dim, hidden], quantize);
                let kp = upload_raw_maybe_quantized(backend, k_raw, &[kv_dim, hidden], quantize);
                let vp = upload_raw_maybe_quantized(backend, v_raw, &[kv_dim, hidden], quantize);
                let op = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &[hidden, q_dim], quantize,
                )?;
                (qp, kp, vp, op, None, None, None, None, None, None,
                 None, None, None, None)
            } else {
                // Standard GQA attention.
                //
                // When attn_output_gate is true (Qwen 3.5), q_proj is fused with an
                // output gate projection: [2*q_dim, hidden].
                //
                // HF does: view(bsz, q_len, num_heads, head_dim*2).chunk(2, dim=-1)
                // This means the weight rows are interleaved per head:
                //   [head0_Q(head_dim rows), head0_gate(head_dim rows),
                //    head1_Q(head_dim rows), head1_gate(head_dim rows), ...]
                // We deinterleave into separate Q [q_dim, hidden] and Z [q_dim, hidden].
                let attn_output_gate = config.attn_output_gate;
                let (qp, z_proj) = if attn_output_gate {
                    let view = store.tensor(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                    )?;
                    let raw = view.data();
                    let fused_q_dim = q_dim * 2;
                    anyhow::ensure!(
                        view.shape() == [fused_q_dim, hidden],
                        "q_proj fused shape mismatch: expected [{}, {}], got {:?}",
                        fused_q_dim, hidden, view.shape()
                    );
                    let row_bytes = hidden * 2; // bf16
                    let hd_bytes = head_dim * row_bytes;
                    let num_heads = config.num_attention_heads;

                    // Deinterleave: for each head, first head_dim rows are Q,
                    // next head_dim rows are gate.
                    let mut q_raw = Vec::with_capacity(q_dim * row_bytes);
                    let mut z_raw = Vec::with_capacity(q_dim * row_bytes);
                    for h in 0..num_heads {
                        let base = h * 2 * hd_bytes;
                        q_raw.extend_from_slice(&raw[base..base + hd_bytes]);
                        z_raw.extend_from_slice(&raw[base + hd_bytes..base + 2 * hd_bytes]);
                    }

                    let q_tensor = upload_raw_maybe_quantized(backend, &q_raw, &[q_dim, hidden], quantize);
                    let z_tensor = upload_raw_maybe_quantized(backend, &z_raw, &[q_dim, hidden], quantize);
                    (q_tensor, Some(z_tensor))
                } else {
                    let qp = upload_maybe_quantized(
                        &store, backend,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        &[q_dim, hidden], quantize,
                    )?;
                    (qp, None)
                };
                let kp = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &[kv_dim, hidden], quantize,
                )?;
                let vp = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden], quantize,
                )?;
                let op = upload_maybe_quantized(
                    &store, backend,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &[hidden, q_dim], quantize,
                )?;
                (qp, kp, vp, op, None, None, None, None, None, None,
                 None, None, None, z_proj)
            };

        // Upload norm weight, adding 1.0 for Qwen 3.5 residual normalization.
        let upload_norm = |name: &str, shape: &[usize]| -> anyhow::Result<B::Tensor> {
            if residual_norm {
                upload_norm_residual(&store, backend, name, shape)
            } else {
                upload_tensor(&store, backend, name, shape)
            }
        };

        // Gemma 3 sandwich norms: extra pre/post norms around the FFN sub-block.
        // These are loaded with residual form (1 + stored) just like the other norms.
        let (pre_ffn_norm, post_ffn_norm) = if is_gemma3 {
            (
                Some(upload_norm(
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                    &[hidden],
                )?),
                Some(upload_norm(
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                    &[hidden],
                )?),
            )
        } else {
            (None, None)
        };

        // O-proj bias (GPT-OSS only).
        let o_proj_bias = if has_o_proj_bias {
            Some(upload_tensor(
                &store,
                backend,
                &format!("{prefix}.self_attn.o_proj.bias"),
                &[hidden],
            )?)
        } else {
            None
        };

        layers.push(LayerWeights {
            input_layernorm: upload_norm(
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            o_proj_bias,
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            post_attention_layernorm: upload_norm(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            gate_proj,
            up_proj,
            down_proj,
            router_gate,
            router_bias: router_bias_weight,
            experts,
            pre_feedforward_layernorm: pre_ffn_norm,
            post_feedforward_layernorm: post_ffn_norm,
            in_proj_qkv,
            in_proj_a,
            in_proj_b,
            in_proj_z,
            conv1d_weight,
            linear_out_proj,
            a_log,
            dt_bias,
            linear_norm,
            attn_z_proj,
            shared_expert_gate_proj,
            shared_expert_up_proj,
            shared_expert_down_proj,
            shared_expert_gate,
        });
    }

    // Final RMSNorm weight (applied after all layers, before lm_head).
    let norm_weight = if residual_norm {
        upload_norm_residual(&store, backend, &format!("{wp}norm.weight"), &[hidden])?
    } else {
        upload_tensor(&store, backend, &format!("{wp}norm.weight"), &[hidden])?
    };

    eprintln!("loaded {} layers", layers.len());
    Ok(ModelWeights {
        embed_tokens,
        layers,
        norm_weight,
        lm_head,
    })
}

// ---------------------------------------------------------------------------
// Tensor upload helpers.
// ---------------------------------------------------------------------------

/// Upload a single tensor from the store to GPU memory (bf16 or f32).
fn upload_tensor<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
) -> anyhow::Result<B::Tensor> {
    let view = store.tensor(name)?;

    let shape = view.shape();
    anyhow::ensure!(
        shape == expected_shape,
        "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {shape:?}"
    );

    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => TensorDtype::BF16,
        safetensors::Dtype::F32 => TensorDtype::F32,
        other => anyhow::bail!("unsupported dtype {:?} for tensor '{name}'", other),
    };

    Ok(backend.upload_tensor(view.data(), shape, dtype))
}

/// Upload a norm weight with residual form: effective_weight = 1.0 + stored_weight.
///
/// Qwen 3.5 stores RMSNorm weights as offsets from 1.0 (initialized to zeros, so
/// effective weight starts at 1.0).  This differs from Llama/Qwen2 which store
/// direct scale factors (initialized to ones).  We add 1.0 during loading so the
/// existing rms_norm kernel works unchanged.
fn upload_norm_residual<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
) -> anyhow::Result<B::Tensor> {
    let view = store.tensor(name)?;
    let shape = view.shape();
    anyhow::ensure!(
        shape == expected_shape,
        "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {shape:?}"
    );

    let bf16_out: Vec<bf16> = match view.dtype() {
        safetensors::Dtype::BF16 => {
            bytemuck::cast_slice::<u8, bf16>(view.data())
                .iter()
                .map(|v| bf16::from_f32(v.to_f32() + 1.0))
                .collect()
        }
        safetensors::Dtype::F32 => {
            bytemuck::cast_slice::<u8, f32>(view.data())
                .iter()
                .map(|v| bf16::from_f32(v + 1.0))
                .collect()
        }
        other => anyhow::bail!("unsupported dtype {:?} for tensor '{name}'", other),
    };

    Ok(backend.upload_tensor(bytemuck::cast_slice(&bf16_out), shape, TensorDtype::BF16))
}

/// Upload a tensor, optionally quantising via the backend's quantisation format.
///
/// When `quantize` is true, delegates to `backend.quantize_upload()` which
/// each backend can override to use its own format (e.g. Q4 for Metal,
/// INT4 for CUDA).
fn upload_maybe_quantized<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
    quantize: bool,
) -> anyhow::Result<B::Tensor> {
    if !quantize {
        return upload_tensor(store, backend, name, expected_shape);
    }

    let view = store.tensor(name)?;

    let shape = view.shape();
    anyhow::ensure!(
        shape == expected_shape,
        "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {shape:?}"
    );

    Ok(backend.quantize_upload(view.data(), shape))
}

/// Upload raw bf16 bytes, optionally quantising via the backend's format.
///
/// Like `upload_maybe_quantized` but for pre-sliced byte buffers (e.g. when
/// splitting a fused QKV or MoE expert weight from a larger tensor).
fn upload_raw_maybe_quantized<B: GpuCore>(
    backend: &B,
    bf16_data: &[u8],
    shape: &[usize],
    quantize: bool,
) -> B::Tensor {
    if quantize {
        backend.quantize_upload(bf16_data, shape)
    } else {
        backend.upload_tensor(bf16_data, shape, TensorDtype::BF16)
    }
}

// ---------------------------------------------------------------------------
// Shared model loading — eliminates duplication across run, batch, and serve.
//
// Every inference path needs the same four things: model config, architecture
// tag, tokenizer, and GPU-resident weights.  This function loads all four.
//
// The backend isn't included because Model<'a, B> borrows it — the caller
// must own the backend's lifetime.
// ---------------------------------------------------------------------------

/// Everything needed to run inference, loaded from a model directory.
pub(crate) struct LoadedModel<B: GpuCore> {
    pub config: ModelConfig,
    pub arch: ModelArch,
    pub tokenizer: Tokenizer,
    pub weights: ModelWeights<B>,
}

/// Load config, tokenizer, and weights from a model directory.
///
/// Logs progress to stderr so the user sees what's happening.
pub(crate) fn load_model<B: GpuCore>(
    backend: &B,
    model_dir: &Path,
    quantize: bool,
) -> anyhow::Result<LoadedModel<B>> {
    let config = ModelConfig::from_file(&model_dir.join("config.json"))?;
    let arch = config.arch()?;
    eprintln!(
        "loaded config: {:?}, {} layers, {} heads, hidden_size={}",
        arch, config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"), arch)?;
    eprintln!("tokenizer loaded");

    let weights = load_weights(backend, model_dir, &config, quantize, None)?;
    eprintln!(
        "weights loaded{}",
        if quantize { " (Q4 quantised)" } else { "" }
    );

    Ok(LoadedModel {
        config,
        arch,
        tokenizer,
        weights,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp4_e2m1_lut() {
        // Verify the FP4 E2M1 lookup table values.
        assert_eq!(FP4_E2M1_LUT[0b0000], 0.0);
        assert_eq!(FP4_E2M1_LUT[0b0001], 0.5);
        assert_eq!(FP4_E2M1_LUT[0b0010], 1.0);
        assert_eq!(FP4_E2M1_LUT[0b0011], 1.5);
        assert_eq!(FP4_E2M1_LUT[0b0100], 2.0);
        assert_eq!(FP4_E2M1_LUT[0b0101], 3.0);
        assert_eq!(FP4_E2M1_LUT[0b0110], 4.0);
        assert_eq!(FP4_E2M1_LUT[0b0111], 6.0);
        assert_eq!(FP4_E2M1_LUT[0b1000], -0.0);
        assert_eq!(FP4_E2M1_LUT[0b1001], -0.5);
        assert_eq!(FP4_E2M1_LUT[0b1010], -1.0);
        assert_eq!(FP4_E2M1_LUT[0b1111], -6.0);
    }

    #[test]
    fn test_e8m0_to_f32() {
        assert_eq!(e8m0_to_f32(0), 0.0);
        assert_eq!(e8m0_to_f32(127), 1.0);   // 2^(127-127) = 2^0
        assert_eq!(e8m0_to_f32(128), 2.0);   // 2^(128-127) = 2^1
        assert_eq!(e8m0_to_f32(126), 0.5);   // 2^(126-127) = 2^-1
        assert_eq!(e8m0_to_f32(129), 4.0);   // 2^(129-127) = 2^2
        assert_eq!(e8m0_to_f32(130), 8.0);   // 2^(130-127) = 2^3
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn test_dequantize_mxfp4_basic() {
        // 1 row, 4 columns, block_size=4 (1 E8M0 scale per row).
        // Packed: 2 bytes for 4 values (low + high nibble per byte).
        //   byte 0: nibble 0b0010 (1.0), nibble 0b0100 (2.0) → packed 0x42
        //   byte 1: nibble 0b0101 (3.0), nibble 0b0000 (0.0) → packed 0x05
        let blocks: Vec<u8> = vec![0x42, 0x05];
        // E8M0 scale = 2.0 → exponent byte = 128.
        let scales: Vec<u8> = vec![128];

        let result = dequantize_mxfp4(&blocks, &scales, 1, 4, 4);
        let bf16_values: &[half::bf16] = bytemuck::cast_slice(&result);
        let f32_values: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

        // Expected: [1.0 * 2.0, 2.0 * 2.0, 3.0 * 2.0, 0.0 * 2.0] = [2.0, 4.0, 6.0, 0.0]
        assert_eq!(f32_values.len(), 4);
        assert!((f32_values[0] - 2.0).abs() < 0.1, "val[0]={}", f32_values[0]);
        assert!((f32_values[1] - 4.0).abs() < 0.1, "val[1]={}", f32_values[1]);
        assert!((f32_values[2] - 6.0).abs() < 0.1, "val[2]={}", f32_values[2]);
        assert!((f32_values[3] - 0.0).abs() < 0.1, "val[3]={}", f32_values[3]);
    }

    #[test]
    fn test_dequantize_mxfp4_negative_values() {
        // 1 row, 2 columns, block_size=2.
        // byte 0: nibble 0b1001 (-0.5), nibble 0b1010 (-1.0) → packed 0xA9
        let blocks: Vec<u8> = vec![0xA9];
        // E8M0 scale = 1.0 → exponent byte = 127.
        let scales: Vec<u8> = vec![127];

        let result = dequantize_mxfp4(&blocks, &scales, 1, 2, 2);
        let bf16_values: &[half::bf16] = bytemuck::cast_slice(&result);
        let f32_values: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

        assert!((f32_values[0] - (-0.5)).abs() < 0.1, "val[0]={}", f32_values[0]);
        assert!((f32_values[1] - (-1.0)).abs() < 0.1, "val[1]={}", f32_values[1]);
    }

    #[test]
    fn test_dequantize_mxfp4_multi_row() {
        // 2 rows, 4 columns, block_size=4.
        // Row 0: all 0b0010 (1.0) → packed 0x22, 0x22; scale 1.0 → output [1,1,1,1]
        // Row 1: all 0b0110 (4.0) → packed 0x66, 0x66; scale 0.5 → output [2,2,2,2]
        let blocks: Vec<u8> = vec![0x22, 0x22, 0x66, 0x66];
        // E8M0: scale=1.0 → 127, scale=0.5 → 126
        let scales: Vec<u8> = vec![127, 126];

        let result = dequantize_mxfp4(&blocks, &scales, 2, 4, 4);
        let bf16_values: &[half::bf16] = bytemuck::cast_slice(&result);
        let f32_values: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

        assert_eq!(f32_values.len(), 8);
        for i in 0..4 {
            assert!((f32_values[i] - 1.0).abs() < 0.1, "row0[{i}]={}", f32_values[i]);
        }
        for i in 4..8 {
            assert!((f32_values[i] - 2.0).abs() < 0.1, "row1[{i}]={}", f32_values[i]);
        }
    }

    /// Regression test: MXFP4 scale slicing must use E8M0 (1 byte/scale),
    /// not bf16 (2 bytes/scale).  With bf16 sizing, this panics because the
    /// computed slice length is 2× the actual data.
    #[test]
    fn test_mxfp4_scale_slice_sizes_e8m0() {
        // Simulate GPT-OSS dimensions for one expert:
        //   gate_up: [5760, 2880] → rows=5760, cols=2880, block_size=32
        //   num_scale_blocks = 2880/32 = 90
        //   scales_per_expert = 5760 * 90 = 518_400 bytes (E8M0)
        //   blocks_per_expert = 5760 * (2880/2) = 8_294_400 bytes (packed nibbles)
        let rows = 5760usize;
        let cols = 2880usize;
        let block_size = 32usize;
        let num_experts = 2; // use 2 to test slicing, not 32 (too much memory)

        let num_scale_blocks = (cols + block_size - 1) / block_size; // 90
        let scales_per_expert = rows * num_scale_blocks;             // 518_400
        let blocks_per_expert = rows * (cols / 2);                   // 8_294_400

        // Allocate fused tensors for all experts.
        let all_blocks = vec![0x22u8; num_experts * blocks_per_expert];
        let all_scales = vec![127u8; num_experts * scales_per_expert]; // E8M0 = 1.0

        // This should NOT panic — the old bf16 code computed scales_per_expert * 2,
        // which would be 1_036_800 and exceed the buffer.
        for j in 0..num_experts {
            let b_off = j * blocks_per_expert;
            let s_off = j * scales_per_expert;
            let _result = dequantize_mxfp4(
                &all_blocks[b_off..b_off + blocks_per_expert],
                &all_scales[s_off..s_off + scales_per_expert],
                rows, cols, block_size,
            );
        }
    }
}
