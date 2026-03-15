// ===========================================================================
// Shared transformer primitives — building blocks that model families compose.
//
// These functions implement the common operations found in most transformer
// architectures (Llama, Qwen, DeepSeek, etc.).  Each model family's forward
// pass calls these primitives in its own specific order with its own
// configuration (e.g., with or without QKV bias).
//
// This is the "model primitives" layer that sits between the GPU backend
// (raw kernel dispatch) and the model family forward passes (architecture-
// specific orchestration).  Similar to vLLM's model registry concept:
//   GPU backend → primitives → model families → registry dispatch
// ===========================================================================

use crate::gpu::GpuBackend;
use crate::model::config::ModelConfig;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::loader::{LayerWeights, ModelWeights};
use crate::model::PrefillBuffers;

// ===========================================================================
// Dims — pre-computed dimension constants extracted from ModelConfig.
//
// LEARNING OVERVIEW
//
// Why this exists:
//   Every forward pass function starts with the same 8-10 lines extracting
//   hidden_size, num_heads, head_dim, etc. from the config and casting to u32.
//   This struct computes them once and passes them around, eliminating the
//   boilerplate without hiding what the values mean.
//
// Why u32?
//   GPU kernels use u32 for dimension parameters (Metal/CUDA thread indices
//   are unsigned).  The config stores usize for Rust idiom, but every kernel
//   call needs the cast.  Doing it once here avoids `as u32` scattered
//   throughout every forward pass.
//
// Why not just store u32 in ModelConfig?
//   ModelConfig maps 1:1 to config.json (usize is natural for Rust structs).
//   Dims is a GPU-side concern — it belongs in the forward-pass layer, not
//   the deserialization layer.
// ===========================================================================

pub(crate) struct Dims {
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub inter_size: u32,
    /// Q projection dimension (num_heads × head_dim).
    /// Usually equals hidden_size, but some models differ (e.g. Qwen3 MoE:
    /// hidden=2048, q_dim=32×128=4096).
    pub q_dim: u32,
    /// KV projection dimension (num_kv_heads × head_dim).
    pub kv_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
}

impl Dims {
    /// Extract dimensions from a model config.
    pub fn from_config(config: &ModelConfig) -> Self {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        Self {
            hidden_size: config.hidden_size as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            inter_size: config.intermediate_size as u32,
            q_dim: (num_heads * head_dim) as u32,
            kv_dim: (num_kv_heads * head_dim) as u32,
            eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta as f32,
        }
    }
}

// ===========================================================================
// Embedding + final projection primitives.
// ===========================================================================

/// Look up token embedding and write to hidden buffer.
pub(crate) fn embed_token<B: GpuBackend>(
    backend: &B,
    weights: &ModelWeights<B>,
    token_id: u32,
    hidden: &B::Tensor,
    hidden_size: u32,
) {
    backend.embed_lookup(&weights.embed_tokens, token_id, hidden, hidden_size);
}

/// Final RMSNorm + LM head projection → logits.
pub(crate) fn final_norm_and_lm_head<B: GpuBackend>(
    backend: &B,
    weights: &ModelWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    logits_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    vocab_size: u32,
) {
    backend.rms_norm(hidden, &weights.norm_weight, eps, norm_buf);
    let lm_head_weight = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
    backend.matmul(lm_head_weight, norm_buf, logits_buf, vocab_size, hidden_size);
}

// ===========================================================================
// Single-token attention block primitives.
// ===========================================================================

/// QKV projection: matmul Q, K, V from the normalized hidden state.
pub(crate) fn qkv_projection<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    norm_buf: &B::Tensor,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    hidden_size: u32,
    kv_dim: u32,
) {
    // Q projection: [q_dim, hidden_size] × [hidden_size] → [q_dim].
    // For most models q_dim == hidden_size, so callers pass hidden_size for both.
    // Models with q_dim ≠ hidden_size (e.g. Qwen3 MoE) should use
    // qkv_projection_qdim() instead.
    backend.matmul(&layer.q_proj, norm_buf, q_buf, hidden_size, hidden_size);
    backend.matmul(&layer.k_proj, norm_buf, k_buf, kv_dim, hidden_size);
    backend.matmul(&layer.v_proj, norm_buf, v_buf, kv_dim, hidden_size);
}

/// Like qkv_projection but with explicit q_dim for models where
/// num_heads × head_dim ≠ hidden_size (e.g. Qwen3 MoE: 4096 vs 2048).
pub(crate) fn qkv_projection_qdim<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    norm_buf: &B::Tensor,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    q_dim: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul(&layer.q_proj, norm_buf, q_buf, q_dim, hidden_size);
    backend.matmul(&layer.k_proj, norm_buf, k_buf, kv_dim, hidden_size);
    backend.matmul(&layer.v_proj, norm_buf, v_buf, kv_dim, hidden_size);
}

/// Apply QKV bias (for architectures that have it, e.g. Qwen).
pub(crate) fn apply_qkv_bias<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    hidden_size: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.add(q_buf, q_bias, q_buf, hidden_size);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.add(k_buf, k_bias, k_buf, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.add(v_buf, v_bias, v_buf, kv_dim);
    }
}

/// RoPE on Q and K buffers.
pub(crate) fn apply_rope<B: GpuBackend>(
    backend: &B,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) {
    backend.rope(q_buf, k_buf, pos, rope_theta, num_heads, num_kv_heads, head_dim);
}

/// Write K/V to paged cache and compute attention.
///
/// `window_size`: sliding window size (0 = full context, attend to all positions).
/// `attn_scale`: custom attention scale (0.0 = default 1/sqrt(head_dim)).
/// Most models pass 0/0.0 for standard full-context attention with default scaling.
pub(crate) fn paged_kv_and_attention<B: GpuBackend>(
    backend: &B,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    q_buf: &B::Tensor,
    attn_out: &B::Tensor,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    layer_idx: usize,
    pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
) {
    backend.copy_to_paged_kv_cache(
        k_buf, &pool.k_pool[layer_idx], &seq_state.block_table_gpu,
        pos, num_kv_heads, head_dim,
    );
    backend.copy_to_paged_kv_cache(
        v_buf, &pool.v_pool[layer_idx], &seq_state.block_table_gpu,
        pos, num_kv_heads, head_dim,
    );
    backend.paged_attention(
        q_buf, &pool.k_pool[layer_idx], &pool.v_pool[layer_idx],
        &seq_state.block_table_gpu, attn_out,
        pos + 1, num_heads, num_kv_heads, head_dim,
        window_size, attn_scale,
    );
}

/// O projection + residual add.
///
/// For models where q_dim == hidden_size, pass hidden_size for both args.
/// For models where q_dim ≠ hidden_size (Qwen3 MoE), use o_proj_residual_qdim.
pub(crate) fn o_proj_residual<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, hidden_size);
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

/// O projection + residual for models where q_dim ≠ hidden_size.
/// o_proj weight is [hidden_size, q_dim], input attn_out is [q_dim].
pub(crate) fn o_proj_residual_qdim<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    q_dim: u32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, q_dim);
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

// ===========================================================================
// SwiGLU FFN block.
// ===========================================================================

/// Full FFN sub-block: RMSNorm → gate/up projections → SwiGLU → down → residual.
pub(crate) fn ffn_block<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    gate_buf: &B::Tensor,
    up_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    inter_size: u32,
) {
    backend.rms_norm(hidden, &layer.post_attention_layernorm, eps, norm_buf);
    backend.matmul(&layer.gate_proj, norm_buf, gate_buf, inter_size, hidden_size);
    backend.matmul(&layer.up_proj, norm_buf, up_buf, inter_size, hidden_size);
    backend.silu_mul(gate_buf, up_buf, gate_buf, inter_size);
    backend.matmul(&layer.down_proj, gate_buf, norm_buf, hidden_size, inter_size);
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

// ===========================================================================
// Batched prefill primitives.
// ===========================================================================

/// Upload token IDs and positions for batched prefill.
pub(crate) fn upload_prefill_inputs<B: GpuBackend>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    tokens: &[u32],
    start_pos: u32,
    bs: u32,
) {
    let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
    backend.copy_to_tensor(&bufs.token_ids, token_bytes);

    let positions: Vec<u32> = (start_pos..start_pos + bs).collect();
    let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
    backend.copy_to_tensor(&bufs.positions, pos_bytes);
}

/// Batched embedding lookup.
pub(crate) fn embed_batch<B: GpuBackend>(
    backend: &B,
    weights: &ModelWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
) {
    backend.embed_lookup_batch(
        &weights.embed_tokens, &bufs.token_ids, &bufs.hidden, bs, hidden_size,
    );
}

/// Batched QKV projection (GEMM).
pub(crate) fn qkv_projection_batch<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul_batch(&layer.q_proj, &bufs.norm_buf, &bufs.q_buf, bs, hidden_size, hidden_size);
    backend.matmul_batch(&layer.k_proj, &bufs.norm_buf, &bufs.k_buf, bs, kv_dim, hidden_size);
    backend.matmul_batch(&layer.v_proj, &bufs.norm_buf, &bufs.v_buf, bs, kv_dim, hidden_size);
}

/// Batched QKV projection with explicit q_dim (for q_dim ≠ hidden_size).
pub(crate) fn qkv_projection_batch_qdim<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    q_dim: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul_batch(&layer.q_proj, &bufs.norm_buf, &bufs.q_buf, bs, q_dim, hidden_size);
    backend.matmul_batch(&layer.k_proj, &bufs.norm_buf, &bufs.k_buf, bs, kv_dim, hidden_size);
    backend.matmul_batch(&layer.v_proj, &bufs.norm_buf, &bufs.v_buf, bs, kv_dim, hidden_size);
}

/// Apply QKV bias in batched mode (broadcast-add).
pub(crate) fn apply_qkv_bias_batch<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.bias_add_batch(&bufs.q_buf, q_bias, &bufs.q_buf, bs, hidden_size);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.bias_add_batch(&bufs.k_buf, k_bias, &bufs.k_buf, bs, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.bias_add_batch(&bufs.v_buf, v_bias, &bufs.v_buf, bs, kv_dim);
    }
}

/// Batched RoPE.
pub(crate) fn apply_rope_batch<B: GpuBackend>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    rope_theta: f32,
    bs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) {
    backend.rope_batch(
        &bufs.q_buf, &bufs.k_buf, &bufs.positions,
        rope_theta, bs, num_heads, num_kv_heads, head_dim,
    );
}

/// Write K/V to paged cache (batched) and compute prefill attention.
///
/// `window_size`: sliding window size (0 = full context).
/// `attn_scale`: custom attention scale (0.0 = default 1/sqrt(head_dim)).
pub(crate) fn paged_kv_and_prefill_attention<B: GpuBackend>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    layer_idx: usize,
    bs: u32,
    start_pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
) {
    backend.copy_to_paged_kv_cache_batch(
        &bufs.k_buf, &pool.k_pool[layer_idx], &seq_state.block_table_gpu,
        &bufs.positions, bs, num_kv_heads, head_dim,
    );
    backend.copy_to_paged_kv_cache_batch(
        &bufs.v_buf, &pool.v_pool[layer_idx], &seq_state.block_table_gpu,
        &bufs.positions, bs, num_kv_heads, head_dim,
    );
    backend.prefill_attention(
        &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
        bs, start_pos, num_heads, num_kv_heads, head_dim,
        window_size, attn_scale,
    );
}

/// Batched O projection + residual.
pub(crate) fn o_proj_residual_batch<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
) {
    backend.matmul_batch(&layer.o_proj, &bufs.attn_out, &bufs.norm_buf, bs, hidden_size, hidden_size);
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Batched O projection + residual with explicit q_dim.
pub(crate) fn o_proj_residual_batch_qdim<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    q_dim: u32,
) {
    backend.matmul_batch(&layer.o_proj, &bufs.attn_out, &bufs.norm_buf, bs, hidden_size, q_dim);
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Batched FFN sub-block.
pub(crate) fn ffn_block_batch<B: GpuBackend>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    eps: f32,
    bs: u32,
    hidden_size: u32,
    inter_size: u32,
) {
    backend.rms_norm_batch(&bufs.hidden, &layer.post_attention_layernorm, eps, &bufs.norm_buf, bs);
    backend.matmul_batch(&layer.gate_proj, &bufs.norm_buf, &bufs.gate_buf, bs, inter_size, hidden_size);
    backend.matmul_batch(&layer.up_proj, &bufs.norm_buf, &bufs.up_buf, bs, inter_size, hidden_size);
    backend.silu_mul(&bufs.gate_buf, &bufs.up_buf, &bufs.gate_buf, bs * inter_size);
    backend.matmul_batch(&layer.down_proj, &bufs.gate_buf, &bufs.norm_buf, bs, hidden_size, inter_size);
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Final norm + LM head for batched prefill (extracts last token's hidden state).
pub(crate) fn final_norm_and_lm_head_prefill<B: GpuBackend>(
    backend: &B,
    weights: &ModelWeights<B>,
    bufs: &PrefillBuffers<B>,
    norm_buf: &B::Tensor,
    logits_buf: &B::Tensor,
    eps: f32,
    bs: u32,
    hidden_size: usize,
    vocab_size: u32,
) {
    backend.rms_norm_batch(&bufs.hidden, &weights.norm_weight, eps, &bufs.norm_buf, bs);

    // Extract last token's hidden state to single-token norm_buf.
    let hidden_byte_size = hidden_size * 2; // bf16
    let full_tensor_bytes = backend.tensor_byte_count(&bufs.norm_buf);
    let mut host_buf = vec![0u8; full_tensor_bytes];
    backend.copy_to_host(&bufs.norm_buf, &mut host_buf);
    let chunk_size = bs as usize;
    let last_row_start = (chunk_size - 1) * hidden_byte_size;
    backend.copy_to_tensor(norm_buf, &host_buf[last_row_start..last_row_start + hidden_byte_size]);

    let lm_head_weight = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
    backend.matmul(lm_head_weight, norm_buf, logits_buf, vocab_size, hidden_size as u32);
}
