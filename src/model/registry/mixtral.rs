// ===========================================================================
// Mixtral 8x7B model family (Mistral AI, sparse MoE).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for Mixtral-8x7B-Instruct, a sparse Mixture
//   of Experts model.  Attention is identical to Llama/Mistral (GQA, RoPE,
//   SwiGLU, no bias, no QK-norm).  The FFN is replaced by 8 expert FFNs
//   with top-2 routing — only 2 of 8 experts activate per token.
//
// Architecture (Mixtral-8x7B-Instruct):
//   - 32 transformer layers
//   - 32 query heads, 8 KV heads (GQA ratio 4:1)
//   - Hidden size: 4096, head dim: 128
//   - 8 expert FFNs per layer (intermediate_size=14336 each), top-2 routing
//   - ~46.7B total params, ~12.9B active per token
//   - RoPE theta: 1,000,000
//   - Chat template: [INST]/[/INST] markers (same as Mistral)
//   - Vocab: 32000 tokens (SentencePiece BPE)
//
// Why this file can't reuse llama.rs:
//   Llama.rs uses a dense FFN (gate/up/down projections).  Mixtral replaces
//   that with MoE routing + per-expert FFNs.  The attention sub-block is
//   identical though — we reuse the same primitives.
//
// Compared to qwen3_moe.rs, Mixtral is simpler:
//   - No QK-norm (no per-head RMSNorm on Q/K)
//   - q_dim == hidden_size (no separate attention dimension)
//   - Standard qkv_projection / o_proj_residual (not the _qdim variants)
//
// MoE routing:
//   Each layer has a router gate weight [8, 4096].  The router computes
//   logits = gate @ hidden_norm, selects top-2 experts by logit value,
//   and sums their outputs weighted by softmax(top_2_logits).
//   See primitives::moe_ffn_block for the shared implementation.
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// MoE FFN block: delegates to primitives::moe_ffn_block.
//
// The core MoE dispatch (router → top-k → expert FFNs → weighted sum →
// residual) is shared across all MoE model families.  See primitives.rs.
// ---------------------------------------------------------------------------

/// Run the MoE FFN block for a single token (delegates to shared primitive).
fn moe_ffn_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &Dims,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    let layer = &m.weights.layers[layer_idx];
    primitives::moe_ffn_block(
        m.backend,
        &layer.post_attention_layernorm,
        layer.router_gate.as_ref().unwrap(),
        layer.experts.as_ref().unwrap(),
        &m.hidden,
        &m.norm_buf,
        m.moe_gate_buf.as_ref().unwrap(),
        m.moe_up_buf.as_ref().unwrap(),
        m.moe_output.as_ref().unwrap(),
        m.routing_output.as_ref().unwrap(),
        &m.gate_buf,
        d.eps,
        d.hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
    );
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
pub(crate) fn forward_single_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed + GpuAllReduce>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let d = Dims::from_config(&m.config);
    let pos = seq_state.seq_len as u32;
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Attention sub-block (identical to Llama: no bias, no QK-norm) ---
        let t = profile::begin(m.backend);
        m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);
        primitives::qkv_projection(
            m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
            d.hidden_size, d.kv_dim,
        );

        // No QKV bias, no QK-norm.

        primitives::apply_rope(
            m.backend, &m.q_buf, &m.k_buf, pos, d.rope_theta,
            d.num_heads, d.num_kv_heads, d.head_dim,
        );
        primitives::paged_kv_and_attention(
            m.backend, &m.k_buf, &m.v_buf, &m.q_buf, &m.attn_out,
            pool, seq_state, layer_idx, pos,
            d.num_heads, d.num_kv_heads, d.head_dim,
            0, 0.0,
        );
        primitives::o_proj_residual(
            m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden, d.hidden_size,
        );
        profile::record(m.backend, t, Component::Attention);

        // --- MoE FFN sub-block ---
        let t = profile::begin(m.backend);
        moe_ffn_block(m, layer_idx, &d, moe_inter, num_experts, num_experts_per_tok);
        profile::record(m.backend, t, Component::Ffn);
    }

    let t = profile::begin(m.backend);
    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        d.eps, d.hidden_size, m.config.vocab_size as u32,
    );
    profile::record(m.backend, t, Component::Other);

    Ok(())
}

// ===========================================================================
// Batched prefill.
//
// Learning note: for MoE models, batched prefill is more complex than dense
// models because each token in the batch may route to different experts.
// A fully optimised implementation would group tokens by expert and batch
// the expert matmuls.  For simplicity, we use batched attention (GEMM) but
// process the MoE FFN token-by-token within each layer.
//
// This means the FFN part of prefill is O(N) sequential expert dispatches
// instead of one big GEMM.  For short prompts (typical benchmarking), this
// is fast enough.  For long-context prefill, a batched MoE implementation
// would be needed.
// ===========================================================================

/// Batched prefill: process entire prompt, MoE FFN is token-by-token per layer.
pub(crate) fn forward_prefill_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed + GpuAllReduce>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let d = Dims::from_config(&m.config);
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, d.hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Batched attention (GEMM-based, identical to Llama) ---
        m.backend.rms_norm_batch(
            &bufs.hidden, &layer.input_layernorm, d.eps, &bufs.norm_buf, bs,
        );
        primitives::qkv_projection_batch(
            m.backend, layer, bufs, bs, d.hidden_size, d.kv_dim,
        );

        // No QKV bias, no QK-norm.

        primitives::apply_rope_batch(
            m.backend, bufs, d.rope_theta, bs,
            d.num_heads, d.num_kv_heads, d.head_dim,
        );
        primitives::paged_kv_and_prefill_attention(
            m.backend, bufs, pool, seq_state, layer_idx,
            bs, start_pos, d.num_heads, d.num_kv_heads, d.head_dim,
            0, 0.0,
        );
        primitives::o_proj_residual_batch(m.backend, layer, bufs, bs, d.hidden_size);

        // --- MoE FFN: process each token independently ---
        //
        // Extract each token's hidden state from the batched buffer,
        // run MoE routing + expert FFNs, and write back.
        let hidden_byte_size = m.config.hidden_size * crate::gpu::TensorDtype::BF16.byte_size();
        let full_bytes = m.backend.tensor_byte_count(&bufs.hidden);
        let mut host_hidden = vec![0u8; full_bytes];
        m.backend.copy_to_host(&bufs.hidden, &mut host_hidden);

        for t in 0..tokens.len() {
            let offset = t * hidden_byte_size;
            m.backend.copy_to_tensor(
                &m.hidden,
                &host_hidden[offset..offset + hidden_byte_size],
            );

            moe_ffn_block(m, layer_idx, &d, moe_inter, num_experts, num_experts_per_tok);

            let mut token_hidden = vec![0u8; hidden_byte_size];
            m.backend.copy_to_host(&m.hidden, &mut token_hidden);
            host_hidden[offset..offset + hidden_byte_size]
                .copy_from_slice(&token_hidden);
        }

        m.backend.copy_to_tensor(&bufs.hidden, &host_hidden);
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend, &m.weights, bufs, &m.norm_buf, &m.logits_buf,
        d.eps, bs, m.config.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}
