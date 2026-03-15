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
// Weight naming convention (shared by Llama and Qwen):
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
// QKV bias (Qwen 2.5 only — Llama has no biases):
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
use crate::gpu::{self, GpuCore, TensorDtype};

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
}

/// Weights for a single transformer layer.
pub(crate) struct LayerWeights<B: GpuCore> {
    // --- Attention sub-block ---
    pub input_layernorm: B::Tensor, // RMSNorm weight [hidden_size]
    pub q_proj: B::Tensor,          // Query projection [q_dim, hidden_size]
    pub k_proj: B::Tensor,          // Key projection [kv_dim, hidden_size]
    pub v_proj: B::Tensor,          // Value projection [kv_dim, hidden_size]
    pub o_proj: B::Tensor,          // Output projection [hidden_size, q_dim]

    // --- QKV bias (Qwen 2.5 only, None for Llama and Qwen3Moe) ---
    //
    // Learning note: bias in a linear layer means output = W @ x + b.
    // After computing Q = W_q @ hidden, Qwen adds: Q = Q + b_q.
    //
    // For single-token inference, the bias vector has the same length as
    // the matmul output — so bias-add is just an element-wise vector add.
    // No new GPU kernel needed: reuses the existing `backend.add()`.
    //
    // Bias tensors are always bf16 (1D, small) and never quantised.
    // O projection has NO bias in either Llama or Qwen.
    pub q_bias: Option<B::Tensor>, // [hidden_size], or None for Llama
    pub k_bias: Option<B::Tensor>, // [kv_dim], or None for Llama
    pub v_bias: Option<B::Tensor>, // [kv_dim], or None for Llama

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
) -> anyhow::Result<ModelWeights<B>> {
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
        let (gate_proj, up_proj, down_proj, router_gate, experts,
         shared_expert_gate_proj, shared_expert_up_proj, shared_expert_down_proj,
         shared_expert_gate) = if is_moe {
            // Dummy tensors for the dense FFN fields (never accessed by MoE forward pass).
            let dummy = backend.alloc_tensor(&[1], TensorDtype::BF16);
            let dummy2 = backend.alloc_tensor(&[1], TensorDtype::BF16);
            let dummy3 = backend.alloc_tensor(&[1], TensorDtype::BF16);

            // Router gate: [num_experts, hidden_size].  Stays bf16 — routing
            // accuracy matters more than speed here (it's a single small matmul).
            let router = upload_tensor(
                &store,
                backend,
                &format!("{prefix}.mlp.gate.weight"),
                &[num_experts, hidden],
            )?;

            // Detect fused vs per-expert format.
            let fused_name = format!("{prefix}.mlp.experts.gate_up_proj");
            let is_fused = store.tensor(&fused_name).is_ok();

            let expert_vec = if is_fused {
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

                    let gate_t = if quantize {
                        let q4 = quantize_bf16_to_q4(gate_raw, moe_inter, hidden);
                        backend.upload_tensor(&q4, &[moe_inter, hidden], TensorDtype::Q4)
                    } else {
                        backend.upload_tensor(gate_raw, &[moe_inter, hidden], TensorDtype::BF16)
                    };
                    let up_t = if quantize {
                        let q4 = quantize_bf16_to_q4(up_raw, moe_inter, hidden);
                        backend.upload_tensor(&q4, &[moe_inter, hidden], TensorDtype::Q4)
                    } else {
                        backend.upload_tensor(up_raw, &[moe_inter, hidden], TensorDtype::BF16)
                    };
                    let down_t = if quantize {
                        let q4 = quantize_bf16_to_q4(down_raw, hidden, moe_inter);
                        backend.upload_tensor(&q4, &[hidden, moe_inter], TensorDtype::Q4)
                    } else {
                        backend.upload_tensor(down_raw, &[hidden, moe_inter], TensorDtype::BF16)
                    };

                    experts.push(ExpertWeights {
                        gate_proj: gate_t,
                        up_proj: up_t,
                        down_proj: down_t,
                    });
                }
                experts
            } else {
                // Per-expert format (Qwen3-MoE): separate tensors per expert.
                let mut experts = Vec::with_capacity(num_experts);
                for j in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{j}");
                    experts.push(ExpertWeights {
                        gate_proj: upload_maybe_q4(
                            &store, backend,
                            &format!("{ep}.gate_proj.weight"),
                            &[moe_inter, hidden], quantize,
                        )?,
                        up_proj: upload_maybe_q4(
                            &store, backend,
                            &format!("{ep}.up_proj.weight"),
                            &[moe_inter, hidden], quantize,
                        )?,
                        down_proj: upload_maybe_q4(
                            &store, backend,
                            &format!("{ep}.down_proj.weight"),
                            &[hidden, moe_inter], quantize,
                        )?,
                    });
                }
                experts
            };
            if i == 0 {
                eprintln!(
                    "  loading {} experts per layer (moe_inter={}){}",
                    num_experts, moe_inter,
                    if is_fused { " [fused format]" } else { "" },
                );
            }

            // Load shared expert weights if present.
            let (se_gate_proj, se_up_proj, se_down_proj, se_gate) = if has_shared_expert {
                let gp = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    &[shared_inter, hidden], quantize,
                )?;
                let up = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    &[shared_inter, hidden], quantize,
                )?;
                let dp = upload_maybe_q4(
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

            (dummy, dummy2, dummy3, Some(router), Some(expert_vec),
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

            let gate = if quantize {
                let q4_data = quantize_bf16_to_q4(gate_raw, inter, hidden);
                backend.upload_tensor(&q4_data, &[inter, hidden], TensorDtype::Q4)
            } else {
                backend.upload_tensor(gate_raw, &[inter, hidden], TensorDtype::BF16)
            };
            let up = if quantize {
                let q4_data = quantize_bf16_to_q4(up_raw, inter, hidden);
                backend.upload_tensor(&q4_data, &[inter, hidden], TensorDtype::Q4)
            } else {
                backend.upload_tensor(up_raw, &[inter, hidden], TensorDtype::BF16)
            };
            let down = upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, inter],
                quantize,
            )?;
            (gate, up, down, None, None, None, None, None, None)
        } else {
            // Dense FFN: standard gate/up/down projections.
            let gate = upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[inter, hidden],
                quantize,
            )?;
            let up = upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.mlp.up_proj.weight"),
                &[inter, hidden],
                quantize,
            )?;
            let down = upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, inter],
                quantize,
            )?;
            (gate, up, down, None, None, None, None, None, None)
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

                let qkv = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    &[fused_dim, hidden], quantize,
                )?;
                let a = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    &[config.linear_num_value_heads, hidden], quantize,
                )?;
                let b = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    &[config.linear_num_value_heads, hidden], quantize,
                )?;
                let z = upload_maybe_q4(
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
                let out = upload_maybe_q4(
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

                let qp = if quantize {
                    let q4 = quantize_bf16_to_q4(q_raw, q_dim, hidden);
                    backend.upload_tensor(&q4, &[q_dim, hidden], TensorDtype::Q4)
                } else {
                    backend.upload_tensor(q_raw, &[q_dim, hidden], TensorDtype::BF16)
                };
                let kp = if quantize {
                    let q4 = quantize_bf16_to_q4(k_raw, kv_dim, hidden);
                    backend.upload_tensor(&q4, &[kv_dim, hidden], TensorDtype::Q4)
                } else {
                    backend.upload_tensor(k_raw, &[kv_dim, hidden], TensorDtype::BF16)
                };
                let vp = if quantize {
                    let q4 = quantize_bf16_to_q4(v_raw, kv_dim, hidden);
                    backend.upload_tensor(&q4, &[kv_dim, hidden], TensorDtype::Q4)
                } else {
                    backend.upload_tensor(v_raw, &[kv_dim, hidden], TensorDtype::BF16)
                };
                let op = upload_maybe_q4(
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

                    let q_tensor = if quantize {
                        let q4_data = quantize_bf16_to_q4(&q_raw, q_dim, hidden);
                        backend.upload_tensor(&q4_data, &[q_dim, hidden], TensorDtype::Q4)
                    } else {
                        backend.upload_tensor(&q_raw, &[q_dim, hidden], TensorDtype::BF16)
                    };
                    let z_tensor = if quantize {
                        let q4_data = quantize_bf16_to_q4(&z_raw, q_dim, hidden);
                        backend.upload_tensor(&q4_data, &[q_dim, hidden], TensorDtype::Q4)
                    } else {
                        backend.upload_tensor(&z_raw, &[q_dim, hidden], TensorDtype::BF16)
                    };
                    (q_tensor, Some(z_tensor))
                } else {
                    let qp = upload_maybe_q4(
                        &store, backend,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        &[q_dim, hidden], quantize,
                    )?;
                    (qp, None)
                };
                let kp = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &[kv_dim, hidden], quantize,
                )?;
                let vp = upload_maybe_q4(
                    &store, backend,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden], quantize,
                )?;
                let op = upload_maybe_q4(
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

        layers.push(LayerWeights {
            input_layernorm: upload_norm(
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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

/// Upload a tensor, optionally quantising to Q4.
fn upload_maybe_q4<B: GpuCore>(
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
    anyhow::ensure!(
        shape.len() == 2 && shape[1] % 32 == 0,
        "Q4 quantisation requires 2D shape with K divisible by 32, got {shape:?}"
    );

    let q4_data = quantize_bf16_to_q4(view.data(), shape[0], shape[1]);
    Ok(backend.upload_tensor(&q4_data, shape, TensorDtype::Q4))
}

// ---------------------------------------------------------------------------
// Q4 quantisation.
// ---------------------------------------------------------------------------

/// Quantise a bf16 weight matrix to block-wise Q4.
///
/// Block layout (symmetric quantisation, block_size=32):
///   For each block of 32 consecutive weights:
///     scale = max(|w_i|) / 7.0           (maps [-max, max] to [-7, 7])
///     q_i = clamp(round(w_i / scale), -8, 7)  (4-bit signed, range [-8, 7])
///     stored as unsigned: u_i = q_i + 8  (range [0, 15], fits in 4 bits)
///
///   Output per block (20 bytes):
///     [0..4]:   f32 scale (little-endian)
///     [4..20]:  16 bytes, 2 packed nibbles each
///               byte[i] = u[2i] | (u[2i+1] << 4)
fn quantize_bf16_to_q4(bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(bf16_data.len(), m * k * 2);

    // Try zero-copy cast first; fall back to a copy if the mmap slice
    // isn't 2-byte aligned (can happen with some safetensors packing).
    let owned_buf: Vec<bf16>;
    let values: &[bf16] = match bytemuck::try_cast_slice(bf16_data) {
        Ok(v) => v,
        Err(_) => {
            owned_buf = bf16_data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                .collect();
            &owned_buf
        }
    };
    assert_eq!(values.len(), m * k);

    let blocks_per_row = k / 32;
    let mut out = vec![0u8; gpu::q4_byte_count(m, k)];

    for row in 0..m {
        for block in 0..blocks_per_row {
            let src_offset = row * k + block * 32;
            let dst_offset = (row * blocks_per_row + block) * 20;

            // Find max absolute value in the block for scale computation.
            let mut max_abs: f32 = 0.0;
            for i in 0..32 {
                let v = values[src_offset + i].to_f32().abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let inv_scale = 1.0 / scale;

            // Write scale.
            out[dst_offset..dst_offset + 4].copy_from_slice(&scale.to_le_bytes());

            // Quantise and pack pairs of weights into bytes.
            for i in 0..16 {
                let v0 = values[src_offset + i * 2].to_f32();
                let v1 = values[src_offset + i * 2 + 1].to_f32();

                let q0 = ((v0 * inv_scale).round() as i32).clamp(-8, 7) + 8;
                let q1 = ((v1 * inv_scale).round() as i32).clamp(-8, 7) + 8;

                out[dst_offset + 4 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }

    out
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

    let weights = load_weights(backend, model_dir, &config, quantize)?;
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
