// ===========================================================================
// Standard dense transformer forward pass — shared by Llama, Phi, and Qwen.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the "vanilla" dense transformer pipeline that most model
//   families share:
//     embed → (norm → qkv → [bias?] → rope → attn → o_proj → ffn) × N → lm_head
//
//   Individual model files (llama.rs, phi.rs, qwen.rs) declare their
//   ArchFeatures and delegate here.  This eliminates ~400 lines of
//   copy-paste while keeping each model's file as a clear, self-documenting
//   "config" that shows exactly what makes it different.
//
// What ArchFeatures captures:
//   The knobs that vary across the standard models:
//     - has_qkv_bias: Qwen 2.5 adds bias after Q/K/V projections; Llama/Phi don't.
//
//   If a future standard-ish model needs a new knob (e.g. LayerNorm instead
//   of RMSNorm), add it to ArchFeatures rather than copy-pasting the whole
//   forward pass into a new file.
//
// What does NOT belong here:
//   Models with fundamentally different pipelines should keep their own files:
//     - Gemma 3: sandwich norms, GeGLU, sliding window, embed scaling
//     - Qwen 3 MoE: QK-norm, MoE expert routing, q_dim ≠ hidden_size
//     - Qwen 3.5: hybrid DeltaNet + GQA attention, MoE FFN
//
//   Trying to parameterise all of those into ArchFeatures would make this
//   file harder to read than the copy-paste it's replacing.
// ===========================================================================

use crate::gpu::GpuBackend;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ===========================================================================
// ArchFeatures — the knobs that distinguish standard model families.
//
// Each model file creates a `const FEATURES: ArchFeatures` that captures
// its architectural choices.  The forward pass reads these at zero cost
// (const propagation eliminates the branches).
// ===========================================================================

pub(crate) struct ArchFeatures {
    /// Whether Q/K/V projections have bias terms (output = W @ x + b).
    ///
    /// - Llama, Phi: false  (no bias on any projection)
    /// - Qwen 2.5:   true   (bias on Q/K/V, but NOT on O or FFN projections)
    pub has_qkv_bias: bool,
}

// ===========================================================================
// Single-token decode (mat-vec, paged KV cache).
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
///
/// This is the standard dense transformer pipeline:
///   1. Embed token
///   2. For each layer: norm → QKV → [bias] → RoPE → attention → O proj → FFN
///   3. Final norm → LM head
pub(crate) fn forward_single_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    features: &ArchFeatures,
) -> anyhow::Result<()> {
    let d = Dims::from_config(&m.config);
    let pos = seq_state.seq_len as u32;

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Attention sub-block ---
        let t = profile::begin(m.backend);
        m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);
        primitives::qkv_projection(
            m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
            d.hidden_size, d.kv_dim,
        );

        // QKV bias: the one line that separates Qwen from Llama/Phi.
        if features.has_qkv_bias {
            primitives::apply_qkv_bias(
                m.backend, layer, &m.q_buf, &m.k_buf, &m.v_buf,
                d.hidden_size, d.kv_dim,
            );
        }

        primitives::apply_rope(
            m.backend, &m.q_buf, &m.k_buf, pos, d.rope_theta,
            d.num_heads, d.num_kv_heads, d.head_dim,
        );
        primitives::paged_kv_and_attention(
            m.backend, &m.k_buf, &m.v_buf, &m.q_buf, &m.attn_out,
            pool, seq_state, layer_idx, pos,
            d.num_heads, d.num_kv_heads, d.head_dim,
            0, 0.0, // No sliding window, default attention scale.
        );
        primitives::o_proj_residual(
            m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden, d.hidden_size,
        );
        profile::record(m.backend, t, Component::Attention);

        // --- FFN sub-block ---
        let t = profile::begin(m.backend);
        primitives::ffn_block(
            m.backend, layer, &m.hidden, &m.norm_buf, &m.gate_buf, &m.up_buf,
            d.eps, d.hidden_size, d.inter_size,
        );
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
// Batched prefill (GEMM, paged KV cache).
// ===========================================================================

/// Batched prefill: process entire prompt in one GEMM-based forward pass.
pub(crate) fn forward_prefill_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
    features: &ArchFeatures,
) -> anyhow::Result<()> {
    let d = Dims::from_config(&m.config);
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, d.hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        m.backend.rms_norm_batch(
            &bufs.hidden, &layer.input_layernorm, d.eps, &bufs.norm_buf, bs,
        );
        primitives::qkv_projection_batch(
            m.backend, layer, bufs, bs, d.hidden_size, d.kv_dim,
        );

        // QKV bias (batched): broadcast-add [dim] bias across [batch_size, dim].
        if features.has_qkv_bias {
            primitives::apply_qkv_bias_batch(
                m.backend, layer, bufs, bs, d.hidden_size, d.kv_dim,
            );
        }

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

        // FFN sub-block.
        primitives::ffn_block_batch(
            m.backend, layer, bufs, d.eps, bs, d.hidden_size, d.inter_size,
        );

        // Submit this layer's work so the GPU starts executing while we
        // encode the next layer.
        m.backend.submit();
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend, &m.weights, bufs, &m.norm_buf, &m.logits_buf,
        d.eps, bs, m.config.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}
