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
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::loader::{LayerWeights, ModelWeights};
use crate::model::PrefillBuffers;

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
    backend.matmul(&layer.q_proj, norm_buf, q_buf, hidden_size, hidden_size);
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
    );
}

/// O projection + residual add.
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
