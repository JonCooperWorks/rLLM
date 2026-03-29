// ===========================================================================
// Qwen 3 Mixture-of-Experts model family.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for Qwen 3 MoE models (e.g. Qwen3-Coder-30B-
//   A3B-Instruct).  These models have many small expert FFN sub-networks per
//   layer but only activate a subset (top-k) for each token.
//
// Why MoE is efficient:
//   A 30B-parameter MoE model with 128 experts and top-8 routing activates
//   only ~3B parameters per token.  The memory bandwidth per token is similar
//   to a 3B dense model, but the model has 10x more total knowledge stored
//   in the expert weights.  This is why MoE achieves strong quality at low
//   inference cost.
//
// Architecture (Qwen3-Coder-30B-A3B):
//   - 48 transformer layers
//   - Attention: identical to Llama (no QKV bias), with QK-norm added
//   - FFN: 128 expert sub-networks per layer, top-8 routing
//   - Each expert: gate_proj [768, 2048], up_proj [768, 2048], down_proj [2048, 768]
//   - Chat template: ChatML (same as Qwen 2.5)
//
// Why this file can't reuse llama.rs:
//   Two differences from the standard dense pipeline:
//     1. QK-norm: after QKV projection, RMSNorm is applied per-head to Q and K
//        (with learned weights).  This happens BEFORE RoPE.
//     2. q_dim ≠ hidden_size: Qwen3 MoE has hidden=2048 but 32×128=4096
//        attention dimension, requiring the _qdim projection variants.
//     3. MoE FFN: router → top-k experts → weighted sum replaces dense FFN.
//
// QK-norm (new in Qwen 3):
//   After computing Q = W_q @ hidden and K = W_k @ hidden, RMSNorm is applied
//   to each head's Q and K vector independently (using per-head learned weights
//   of dimension [head_dim]).  This happens BEFORE RoPE.  It stabilises the
//   attention dot products and prevents them from growing unboundedly in deep
//   networks.
//
// MoE routing:
//   Each layer has a router gate weight [num_experts, hidden_size].  The
//   router computes logits = gate @ hidden_norm, then selects the top-k
//   experts by logit value.  The selected experts' outputs are weighted by
//   softmax(top_k_logits) and summed.
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuBackend, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul,
    GpuMoe, GpuNorm, GpuRope, GpuTurboQuant,
};
use crate::model::forward::{MoeBuffers, ModelForward};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ===========================================================================
// Qwen3MoeForward — ModelForward trait implementation.
// ===========================================================================

pub(crate) struct Qwen3MoeForward<B: GpuCore> {
    pub moe: MoeBuffers<B>,
}

impl<B: GpuBackend> ModelForward<B> for Qwen3MoeForward<B> {
    fn forward_decode(
        &self,
        m: &Model<'_, B>,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        forward_single_paged(m, token_id, pool, seq_state, &self.moe)
    }

    fn forward_prefill(
        &self,
        m: &Model<'_, B>,
        tokens: &[u32],
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
        bufs: &PrefillBuffers<B>,
    ) -> anyhow::Result<()> {
        forward_prefill_paged(m, tokens, pool, seq_state, bufs, &self.moe)
    }
}

// ---------------------------------------------------------------------------
// MoE FFN block: delegates to primitives::moe_ffn_block.
//
// The core MoE dispatch (router → top-k → expert FFNs → weighted sum →
// residual) is shared across all MoE model families.  See primitives.rs.
// ---------------------------------------------------------------------------

/// Run the MoE FFN block for a single token (delegates to shared primitive).
fn moe_ffn_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuMoe + GpuAllReduce>(
    m: &Model<'_, B>,
    moe: &MoeBuffers<B>,
    layer_idx: usize,
    d: &Dims,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    let layer = &m.weights.layers[layer_idx];
    if let Some(ref streamer) = moe.expert_streamer {
        primitives::moe_ffn_block_streamed(
            m.backend,
            streamer,
            layer_idx,
            &layer.post_attention_layernorm,
            layer.router_gate.as_ref().unwrap(),
            &m.hidden,
            &m.norm_buf,
            &moe.moe_gate_buf,
            &moe.moe_output,
            &moe.routing_output,
            &m.gate_buf,
            d.eps,
            d.hidden_size,
            moe_inter,
            num_experts,
            num_experts_per_tok,
        );
    } else {
        primitives::moe_ffn_block(
            m.backend,
            &layer.post_attention_layernorm,
            layer.router_gate.as_ref().unwrap(),
            layer.experts.as_ref().unwrap(),
            &m.hidden,
            &m.norm_buf,
            &moe.moe_gate_buf,
            &moe.moe_up_buf,
            &moe.moe_output,
            &moe.routing_output,
            &m.gate_buf,
            d.eps,
            d.hidden_size,
            moe_inter,
            num_experts,
            num_experts_per_tok,
            moe.local_expert_start,
            moe.local_expert_count,
        );
    }
}

/// Pre-normed MoE FFN block — norm_buf already populated by fused residual+norm.
/// Inspired by rvLLM (m0at).
fn moe_ffn_block_pre_normed<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuMoe + GpuAllReduce>(
    m: &Model<'_, B>,
    moe: &MoeBuffers<B>,
    layer_idx: usize,
    d: &Dims,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    let layer = &m.weights.layers[layer_idx];
    if let Some(ref streamer) = moe.expert_streamer {
        primitives::moe_ffn_block_streamed_pre_normed(
            m.backend,
            streamer,
            layer_idx,
            layer.router_gate.as_ref().unwrap(),
            &m.hidden,
            &m.norm_buf,
            &moe.moe_gate_buf,
            &moe.moe_output,
            &moe.routing_output,
            &m.gate_buf,
            d.hidden_size,
            moe_inter,
            num_experts,
            num_experts_per_tok,
        );
    } else {
        primitives::moe_ffn_block_pre_normed(
            m.backend,
            layer.router_gate.as_ref().unwrap(),
            layer.experts.as_ref().unwrap(),
            &m.hidden,
            &m.norm_buf,
            &moe.moe_gate_buf,
            &moe.moe_up_buf,
            &moe.moe_output,
            &moe.routing_output,
            &m.gate_buf,
            d.hidden_size,
            moe_inter,
            num_experts,
            num_experts_per_tok,
            moe.local_expert_start,
            moe.local_expert_count,
        );
    }
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
pub(crate) fn forward_single_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuMoe
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    moe: &MoeBuffers<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let pos = seq_state.seq_len as u32;
    let moe_inter = if moe.local_expert_count < m.config.num_experts {
        m.config.moe_intermediate_size as u32
    } else {
        (m.config.moe_intermediate_size / m.world_size) as u32
    };
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Attention sub-block (same as Llama, plus QK-norm) ---
        let t = profile::begin(m.backend);
        m.backend
            .rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);
        primitives::qkv_projection_qdim(
            m.backend,
            layer,
            &m.norm_buf,
            &m.q_buf,
            &m.k_buf,
            &m.v_buf,
            d.q_dim,
            d.hidden_size,
            d.kv_dim,
        );

        // No QKV bias for Qwen 3 MoE.

        // QK-norm: apply per-head RMSNorm to Q and K before RoPE.
        //
        // Learning note: rms_norm_batch with batch_size=num_heads treats the
        // Q buffer [num_heads * head_dim] as [num_heads, head_dim] and
        // normalises each head independently.  The norm weight [head_dim]
        // is broadcast across all heads.  Same for K with num_kv_heads.
        if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
            m.backend
                .rms_norm_batch(&m.q_buf, q_norm, d.eps, &m.q_buf, d.num_heads);
            m.backend
                .rms_norm_batch(&m.k_buf, k_norm, d.eps, &m.k_buf, d.num_kv_heads);
        }

        primitives::apply_rope(
            m.backend,
            &m.q_buf,
            &m.k_buf,
            pos,
            d.rope_theta,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
        );
        primitives::paged_kv_and_attention_maybe_quantized(
            m.backend,
            &m.k_buf,
            &m.v_buf,
            &m.q_buf,
            &m.attn_out,
            pool,
            seq_state,
            layer_idx,
            pos,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
            0,
            0.0,
            None,
            m.turbo_ctx.as_ref(),
        );
        // Fused O-proj + residual-add + post-attention RMSNorm: saves one
        // full read of the hidden tensor by combining the residual add with
        // the FFN's initial norm.  Inspired by rvLLM (m0at).
        primitives::o_proj_fused_residual_norm_qdim(
            m.backend,
            layer,
            &m.attn_out,
            &m.norm_buf,
            &m.hidden,
            d.hidden_size,
            d.q_dim,
            d.eps,
        );
        profile::record(m.backend, t, Component::Attention);

        // --- MoE FFN sub-block (pre-normed: fused residual+norm already produced norm_buf) ---
        let t = profile::begin(m.backend);
        moe_ffn_block_pre_normed(
            m,
            moe,
            layer_idx,
            &d,
            moe_inter,
            num_experts,
            num_experts_per_tok,
        );
        profile::record(m.backend, t, Component::Ffn);
    }

    let t = profile::begin(m.backend);
    primitives::final_norm_and_lm_head(
        m.backend,
        &m.weights,
        &m.hidden,
        &m.norm_buf,
        &m.logits_buf,
        d.eps,
        d.hidden_size,
        m.config.vocab_size as u32,
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
pub(crate) fn forward_prefill_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuMoe
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
    moe: &MoeBuffers<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let moe_inter = if moe.local_expert_count < m.config.num_experts {
        m.config.moe_intermediate_size as u32
    } else {
        (m.config.moe_intermediate_size / m.world_size) as u32
    };
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Batched attention (GEMM-based, same as dense models) ---
        m.backend.rms_norm_batch(
            &bufs.hidden,
            &layer.input_layernorm,
            d.eps,
            &bufs.norm_buf,
            bs,
        );
        primitives::qkv_projection_batch_qdim(
            m.backend,
            layer,
            bufs,
            bs,
            d.q_dim,
            d.hidden_size,
            d.kv_dim,
        );

        // No QKV bias for Qwen 3 MoE.

        // QK-norm (batched): treat [batch_size * num_heads, head_dim] as batch.
        if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
            m.backend
                .rms_norm_batch(&bufs.q_buf, q_norm, d.eps, &bufs.q_buf, bs * d.num_heads);
            m.backend
                .rms_norm_batch(&bufs.k_buf, k_norm, d.eps, &bufs.k_buf, bs * d.num_kv_heads);
        }

        primitives::apply_rope_batch(
            m.backend,
            bufs,
            d.rope_theta,
            bs,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
        );
        primitives::paged_kv_and_prefill_attention_maybe_quantized(
            m.backend,
            bufs,
            pool,
            seq_state,
            layer_idx,
            bs,
            start_pos,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
            0,
            0.0,
            None,
            m.turbo_ctx.as_ref(),
        );
        primitives::o_proj_residual_batch_qdim(m.backend, layer, bufs, bs, d.hidden_size, d.q_dim);

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
            m.backend
                .copy_to_tensor(&m.hidden, &host_hidden[offset..offset + hidden_byte_size]);

            moe_ffn_block(
                m,
                moe,
                layer_idx,
                &d,
                moe_inter,
                num_experts,
                num_experts_per_tok,
            );

            let mut token_hidden = vec![0u8; hidden_byte_size];
            m.backend.copy_to_host(&m.hidden, &mut token_hidden);
            host_hidden[offset..offset + hidden_byte_size].copy_from_slice(&token_hidden);
        }

        m.backend.copy_to_tensor(&bufs.hidden, &host_hidden);
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend,
        &m.weights,
        bufs,
        &m.norm_buf,
        &m.logits_buf,
        d.eps,
        bs,
        m.config.hidden_size,
        m.config.vocab_size as u32,
    );

    Ok(())
}
