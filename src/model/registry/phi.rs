// ===========================================================================
// Phi model family (Microsoft Phi-3, Phi-4).
//
// Phi-3/4 (14B):
//   - RMSNorm + GQA attention + SwiGLU FFN + RoPE
//   - NO bias on any projection (Q/K/V/O/FFN)
//   - Chat template: <|im_start|>/<|im_sep|>/<|im_end|> markers
//   - RoPE theta: 250000
//
// LEARNING OVERVIEW
//
// Why a separate file when Phi is architecturally identical to Llama?
//   The FORWARD PASS is identical — same primitives in the same order.
//   The difference is in weight LOADING: Phi uses fused projections
//   (qkv_proj, gate_up_proj) that must be split on-load in loader.rs.
//   Once the weights are loaded as separate q/k/v and gate/up tensors,
//   the forward pass is exactly Llama.
//
//   Having a separate registry file keeps the dispatch clean and makes it
//   easy to add Phi-specific behaviour later (e.g. if future Phi models
//   diverge from Llama).
// ===========================================================================

use crate::gpu::GpuBackend;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives;
use crate::model::profile::{self, Component};
use crate::model::{KvMode, Model, PrefillBuffers};

/// Single-token forward pass using an external paged KV cache.
pub(crate) fn forward_single_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let inter_size = m.config.intermediate_size as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;
    let pos = seq_state.seq_len as u32;

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // Attention sub-block.
        let t = profile::begin(m.backend);
        m.backend.rms_norm(&m.hidden, &layer.input_layernorm, eps, &m.norm_buf);
        primitives::qkv_projection(m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf, hidden_size, kv_dim);

        // No QKV bias for Phi.

        primitives::apply_rope(m.backend, &m.q_buf, &m.k_buf, pos, rope_theta, num_heads, num_kv_heads, head_dim);
        primitives::paged_kv_and_attention(
            m.backend, &m.k_buf, &m.v_buf, &m.q_buf, &m.attn_out,
            pool, seq_state, layer_idx, pos, num_heads, num_kv_heads, head_dim,
        );
        primitives::o_proj_residual(m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden, hidden_size);
        profile::record(m.backend, t, Component::Attention);

        // FFN sub-block.
        let t = profile::begin(m.backend);
        primitives::ffn_block(m.backend, layer, &m.hidden, &m.norm_buf, &m.gate_buf, &m.up_buf, eps, hidden_size, inter_size);
        profile::record(m.backend, t, Component::Ffn);
    }

    let t = profile::begin(m.backend);
    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        eps, hidden_size, m.config.vocab_size as u32,
    );
    profile::record(m.backend, t, Component::Other);

    Ok(())
}

/// Single-token forward pass with flat KV cache (legacy path).
pub(crate) fn forward<B: GpuBackend>(m: &mut Model<'_, B>, token_id: u32) -> anyhow::Result<()> {
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let inter_size = m.config.intermediate_size as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;

    let pos = match &mut m.kv_mode {
        KvMode::Flat { pos, .. } => *pos as u32,
        KvMode::Paged { pool, seq_state } => {
            seq_state.ensure_slot(pool)?;
            seq_state.sync_block_table(m.backend);
            seq_state.seq_len as u32
        }
    };

    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        m.backend.rms_norm(&m.hidden, &layer.input_layernorm, eps, &m.norm_buf);
        primitives::qkv_projection(m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf, hidden_size, kv_dim);

        // No QKV bias for Phi.

        primitives::apply_rope(m.backend, &m.q_buf, &m.k_buf, pos, rope_theta, num_heads, num_kv_heads, head_dim);

        match &m.kv_mode {
            KvMode::Flat { k_cache, v_cache, .. } => {
                m.backend.copy_to_kv_cache(&m.k_buf, &k_cache[layer_idx], pos, num_kv_heads, head_dim);
                m.backend.copy_to_kv_cache(&m.v_buf, &v_cache[layer_idx], pos, num_kv_heads, head_dim);
                m.backend.attention(
                    &m.q_buf, &k_cache[layer_idx], &v_cache[layer_idx], &m.attn_out,
                    pos + 1, num_heads, num_kv_heads, head_dim,
                );
            }
            KvMode::Paged { pool, seq_state } => {
                primitives::paged_kv_and_attention(
                    m.backend, &m.k_buf, &m.v_buf, &m.q_buf, &m.attn_out,
                    pool, seq_state, layer_idx, pos, num_heads, num_kv_heads, head_dim,
                );
            }
        }

        primitives::o_proj_residual(m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden, hidden_size);

        // FFN sub-block.
        primitives::ffn_block(m.backend, layer, &m.hidden, &m.norm_buf, &m.gate_buf, &m.up_buf, eps, hidden_size, inter_size);
    }

    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        eps, hidden_size, m.config.vocab_size as u32,
    );

    match &mut m.kv_mode {
        KvMode::Flat { pos, .. } => *pos += 1,
        KvMode::Paged { seq_state, .. } => seq_state.advance(),
    }
    Ok(())
}

/// Batched prefill: process entire prompt in one GEMM-based forward pass.
pub(crate) fn forward_prefill_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let bs = tokens.len() as u32;
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let inter_size = m.config.intermediate_size as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;
    let start_pos = seq_state.seq_len as u32;

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        m.backend.rms_norm_batch(&bufs.hidden, &layer.input_layernorm, eps, &bufs.norm_buf, bs);
        primitives::qkv_projection_batch(m.backend, layer, bufs, bs, hidden_size, kv_dim);

        // No QKV bias for Phi.

        primitives::apply_rope_batch(m.backend, bufs, rope_theta, bs, num_heads, num_kv_heads, head_dim);
        primitives::paged_kv_and_prefill_attention(
            m.backend, bufs, pool, seq_state, layer_idx,
            bs, start_pos, num_heads, num_kv_heads, head_dim,
        );
        primitives::o_proj_residual_batch(m.backend, layer, bufs, bs, hidden_size);

        // FFN sub-block.
        primitives::ffn_block_batch(m.backend, layer, bufs, eps, bs, hidden_size, inter_size);

        // Submit this layer's work so the GPU starts executing while we encode the next layer.
        m.backend.submit();
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend, &m.weights, bufs, &m.norm_buf, &m.logits_buf,
        eps, bs, m.config.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}
