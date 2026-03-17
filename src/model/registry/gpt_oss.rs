// ===========================================================================
// GPT-OSS-20B model family (OpenAI, 20B sparse MoE).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for GPT-OSS-20B, a 20B-parameter Mixture-of-
//   Experts model with 3.6B active parameters per token (32 experts, top-4).
//
// Architecture:
//   - 24 layers, hidden_size=2880, head_dim=64
//   - 64 query heads, 8 KV heads → q_dim=4096 (≠ hidden_size)
//   - 32 experts per layer, top-4 routing with router bias
//   - GPT-OSS gated activation: (clamp(up,-7,7)+1) * clamp(gate,max=7) * σ(gate×1.702)
//   - QKV AND O-proj bias on all attention projections
//   - Expert biases: per-expert gate, up, and down biases
//   - Alternating sliding_attention (window=128) / full_attention layers
//   - YaRN RoPE for extended context (factor=32, original_max_pos=4096)
//   - MXFP4 expert weights dequantized to bf16 during loading
//
// Key differences from Qwen3 MoE (closest existing architecture):
//   1. All projections have bias (QKV + O-proj + router + expert FFN)
//   2. Non-standard gated activation (NOT SwiGLU — uses alpha=1.702 sigmoid)
//   3. Interleaved gate/up weights (even/odd rows, not first-half/second-half)
//   4. YaRN RoPE instead of standard RoPE
//   5. Sliding window attention on alternating layers
//   6. q_dim ≠ hidden_size (4096 vs 2880)
//
// Related files:
//   Config:      model/config.rs (GptOss variant)
//   Loader:      model/loader.rs (MXFP4 dequant, expert biases, de-interleaving)
//   Primitives:  model/primitives.rs (biased MoE dispatch, YaRN RoPE)
//   Dispatch:    model/mod.rs (forward_single/prefill_paged arms)
// ===========================================================================

use crate::gpu::{
    GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// MoE FFN block: biased dispatch with GPT-OSS gated activation.
// ---------------------------------------------------------------------------

fn moe_ffn_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &Dims,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    swiglu_limit: f32,
) {
    let layer = &m.weights.layers[layer_idx];
    primitives::moe_ffn_block_biased(
        m.backend,
        &layer.post_attention_layernorm,
        layer.router_gate.as_ref().unwrap(),
        layer.router_bias.as_ref(),
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
        swiglu_limit,
    );
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass for GPT-OSS-20B.
pub(crate) fn forward_single_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let pos = seq_state.seq_len as u32;
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;
    let swiglu_limit = m.config.swiglu_limit as f32;
    let rope_scaling = m.config.rope_scaling.as_ref();

    // YaRN attention scaling: HF multiplies cos/sin by attention_scaling, which
    // effectively scales Q·K by attention_scaling².  Instead of modifying the RoPE
    // kernel, we fold this into the attention scale factor.
    let attn_scale = rope_scaling
        .map(|rs| {
            let s = rs.attention_scaling() as f32;
            s * s / (d.head_dim as f32).sqrt()
        })
        .unwrap_or(0.0);

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Attention sub-block ---
        let t = profile::begin(m.backend);
        m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

        // QKV projection with explicit q_dim (4096 vs 2880 hidden).
        primitives::qkv_projection_qdim(
            m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
            d.q_dim, d.hidden_size, d.kv_dim,
        );

        // QKV bias (GPT-OSS has bias on all projections).
        primitives::apply_qkv_bias_qdim(
            m.backend, layer, &m.q_buf, &m.k_buf, &m.v_buf,
            d.q_dim, d.kv_dim,
        );

        // RoPE — use YaRN if rope_scaling is configured.
        if let Some(scaling) = rope_scaling {
            primitives::apply_rope_yarn(
                m.backend, &m.q_buf, &m.k_buf, pos, d.rope_theta,
                d.num_heads, d.num_kv_heads, d.head_dim, scaling,
            );
        } else {
            primitives::apply_rope(
                m.backend, &m.q_buf, &m.k_buf, pos, d.rope_theta,
                d.num_heads, d.num_kv_heads, d.head_dim,
            );
        }

        // Sliding window: use window_size for sliding_attention layers, 0 for full.
        let window_size = if m.config.is_sliding_attention_layer(layer_idx) {
            m.config.sliding_window as u32
        } else {
            0
        };

        primitives::paged_kv_and_attention(
            m.backend, &m.k_buf, &m.v_buf, &m.q_buf, &m.attn_out,
            pool, seq_state, layer_idx, pos,
            d.num_heads, d.num_kv_heads, d.head_dim,
            window_size, attn_scale,
            layer.sinks.as_ref(),
        );

        // O projection with bias + residual.
        primitives::o_proj_residual_qdim_biased(
            m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden,
            d.hidden_size, d.q_dim,
        );
        profile::record(m.backend, t, Component::Attention);

        // --- MoE FFN sub-block ---
        let t = profile::begin(m.backend);
        moe_ffn_block(m, layer_idx, &d, moe_inter, num_experts, num_experts_per_tok, swiglu_limit);
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
// Batched attention (GEMM) + per-token MoE FFN (same pattern as Qwen3 MoE).
// ===========================================================================

pub(crate) fn forward_prefill_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;
    let swiglu_limit = m.config.swiglu_limit as f32;
    let rope_scaling = m.config.rope_scaling.as_ref();

    // YaRN attention scaling (see forward_single_paged for explanation).
    let attn_scale = rope_scaling
        .map(|rs| {
            let s = rs.attention_scaling() as f32;
            s * s / (d.head_dim as f32).sqrt()
        })
        .unwrap_or(0.0);

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, d.hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Batched attention ---
        m.backend.rms_norm_batch(
            &bufs.hidden, &layer.input_layernorm, d.eps, &bufs.norm_buf, bs,
        );

        primitives::qkv_projection_batch_qdim(
            m.backend, layer, bufs, bs, d.q_dim, d.hidden_size, d.kv_dim,
        );

        // QKV bias (batched broadcast-add).
        primitives::apply_qkv_bias_batch_qdim(
            m.backend, layer, bufs, bs, d.q_dim, d.kv_dim,
        );

        // Batched RoPE — use YaRN if configured.
        if let Some(scaling) = rope_scaling {
            primitives::apply_rope_yarn_batch(
                m.backend, bufs, d.rope_theta, bs,
                d.num_heads, d.num_kv_heads, d.head_dim, scaling,
            );
        } else {
            primitives::apply_rope_batch(
                m.backend, bufs, d.rope_theta, bs,
                d.num_heads, d.num_kv_heads, d.head_dim,
            );
        }

        // Sliding window.
        let window_size = if m.config.is_sliding_attention_layer(layer_idx) {
            m.config.sliding_window as u32
        } else {
            0
        };

        primitives::paged_kv_and_prefill_attention(
            m.backend, bufs, pool, seq_state, layer_idx,
            bs, start_pos, d.num_heads, d.num_kv_heads, d.head_dim,
            window_size, attn_scale,
            layer.sinks.as_ref(),
        );

        // O projection with bias (batched) + residual.
        m.backend.matmul_batch(
            &layer.o_proj, &bufs.attn_out, &bufs.norm_buf, bs, d.hidden_size, d.q_dim,
        );
        if let Some(ref o_bias) = layer.o_proj_bias {
            m.backend.bias_add_batch(&bufs.norm_buf, o_bias, &bufs.norm_buf, bs, d.hidden_size);
        }
        m.backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * d.hidden_size);

        // --- MoE FFN: process each token independently ---
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

            moe_ffn_block(m, layer_idx, &d, moe_inter, num_experts, num_experts_per_tok, swiglu_limit);

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
