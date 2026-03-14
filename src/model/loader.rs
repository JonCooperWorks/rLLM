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
use crate::gpu::{self, GpuBackend, TensorDtype};

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
/// Generic over `B: GpuBackend` — tensors are GPU-resident (Metal buffers,
/// CUDA device pointers, etc.) but this struct doesn't know or care which.
pub(crate) struct ModelWeights<B: GpuBackend> {
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
pub(crate) struct ExpertWeights<B: GpuBackend> {
    pub gate_proj: B::Tensor, // [moe_inter, hidden_size]
    pub up_proj: B::Tensor,   // [moe_inter, hidden_size]
    pub down_proj: B::Tensor, // [hidden_size, moe_inter]
}

/// Weights for a single transformer layer.
pub(crate) struct LayerWeights<B: GpuBackend> {
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
}

/// Load all model weights from safetensors file(s) into GPU memory.
///
/// When `quantize` is true, linear projection weights (Q/K/V/O/gate/up/down)
/// are quantised from bf16 to Q4 on the CPU during loading.  This reduces
/// memory ~3.2x and speeds up matmul ~1.5-2x.  Norm weights and the embedding
/// table stay in bf16 (they're small and used for lookup/norm, not matmul).
pub(crate) fn load_weights<B: GpuBackend>(
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

    // Load the embedding table.
    let embed_tokens = upload_tensor(
        &store,
        backend,
        "model.embed_tokens.weight",
        &[config.vocab_size, hidden],
    )?;

    // Check for separate lm_head (untied embeddings, e.g. Llama 3.1 8B+).
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
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");

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

        // Load QK-norm weights if the architecture has them (Qwen 3 MoE).
        // These are per-head RMSNorm weights [head_dim], applied to Q and K
        // after projection but before RoPE.
        let (q_norm, k_norm) = if has_qk_norm {
            (
                Some(upload_tensor(
                    &store,
                    backend,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[head_dim],
                )?),
                Some(upload_tensor(
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
        let (gate_proj, up_proj, down_proj, router_gate, experts) = if is_moe {
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

            // Load all expert weights.  Each expert has gate/up/down projections
            // at the smaller moe_intermediate_size.
            let mut expert_vec = Vec::with_capacity(num_experts);
            for j in 0..num_experts {
                let ep = format!("{prefix}.mlp.experts.{j}");
                expert_vec.push(ExpertWeights {
                    gate_proj: upload_maybe_q4(
                        &store,
                        backend,
                        &format!("{ep}.gate_proj.weight"),
                        &[moe_inter, hidden],
                        quantize,
                    )?,
                    up_proj: upload_maybe_q4(
                        &store,
                        backend,
                        &format!("{ep}.up_proj.weight"),
                        &[moe_inter, hidden],
                        quantize,
                    )?,
                    down_proj: upload_maybe_q4(
                        &store,
                        backend,
                        &format!("{ep}.down_proj.weight"),
                        &[hidden, moe_inter],
                        quantize,
                    )?,
                });
            }
            if i == 0 {
                eprintln!("  loading {} experts per layer (moe_inter={})", num_experts, moe_inter);
            }

            (dummy, dummy2, dummy3, Some(router), Some(expert_vec))
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
            (gate, up, down, None, None)
        };

        layers.push(LayerWeights {
            // Norm weights stay bf16 (1D, tiny, used for RMSNorm not matmul).
            input_layernorm: upload_tensor(
                &store,
                backend,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            q_proj: upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &[q_dim, hidden],
                quantize,
            )?,
            k_proj: upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &[kv_dim, hidden],
                quantize,
            )?,
            v_proj: upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_dim, hidden],
                quantize,
            )?,
            o_proj: upload_maybe_q4(
                &store,
                backend,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[hidden, q_dim],
                quantize,
            )?,
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            post_attention_layernorm: upload_tensor(
                &store,
                backend,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            gate_proj,
            up_proj,
            down_proj,
            router_gate,
            experts,
        });
    }

    // Final RMSNorm weight (applied after all layers, before lm_head).
    let norm_weight = upload_tensor(&store, backend, "model.norm.weight", &[hidden])?;

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
fn upload_tensor<B: GpuBackend>(
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

/// Upload a tensor, optionally quantising to Q4.
fn upload_maybe_q4<B: GpuBackend>(
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
pub(crate) struct LoadedModel<B: GpuBackend> {
    pub config: ModelConfig,
    pub arch: ModelArch,
    pub tokenizer: Tokenizer,
    pub weights: ModelWeights<B>,
}

/// Load config, tokenizer, and weights from a model directory.
///
/// Logs progress to stderr so the user sees what's happening.
pub(crate) fn load_model<B: GpuBackend>(
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
