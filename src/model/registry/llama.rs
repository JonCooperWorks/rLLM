// ===========================================================================
// Llama 3.x model family — the reference dense transformer.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the "vanilla" dense transformer pipeline:
//     embed → (norm → qkv → [bias?] → rope → attn → o_proj → ffn) × N → lm_head
//
//   Llama is the baseline — no QKV bias, no special features.  Variants
//   that only differ in small ways (Qwen: QKV bias, Phi: fused weight
//   loading) call into this file's implementation with their own ArchFeatures.
//
// Llama 3.x family (1B, 3B, 8B, 70B):
//   - RMSNorm + GQA attention + SwiGLU FFN + RoPE
//   - NO bias on any projection (Q/K/V/O/FFN)
//   - Chat template: <|start_header_id|> markers
//   - RoPE theta: 500000
//
// ArchFeatures — the knobs that vary across Llama-like models:
//   - has_qkv_bias: Qwen 2.5 adds bias after Q/K/V projections; Llama/Phi don't.
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
//
// Phi (Microsoft):
//   Phi's forward pass is identical to Llama's — the difference is in weight
//   loading (fused qkv_proj/gate_up_proj split on load in loader.rs).  Once
//   weights are loaded, inference is the same.  phi.rs re-exports from here.
//
// Qwen 2.5:
//   Differs by exactly one thing: QKV bias.  qwen.rs calls into the _impl
//   functions here with `has_qkv_bias: true`.
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
    GpuTurboQuant,
};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives;
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ===========================================================================
// ArchFeatures — the knobs that distinguish Llama-like model families.
//
// Each variant creates a `const FEATURES` that captures its architectural
// choices.  The forward pass reads these at zero cost (const propagation
// eliminates the branches).
// ===========================================================================

pub(crate) struct ArchFeatures {
    /// Whether Q/K/V projections have bias terms (output = W @ x + b).
    ///
    /// - Llama, Phi: false  (no bias on any projection)
    /// - Qwen 2.5:   true   (bias on Q/K/V, but NOT on O or FFN projections)
    pub has_qkv_bias: bool,
}

/// Llama features: the baseline.  No QKV bias, nothing extra.
const FEATURES: ArchFeatures = ArchFeatures {
    has_qkv_bias: false,
};

// ===========================================================================
// Public entry points — called from model/mod.rs dispatch.
// ===========================================================================

pub(crate) fn forward_single_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    forward_single_impl(m, token_id, pool, seq_state, &FEATURES)
}

pub(crate) fn forward_decode_batch_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    positions: &[u32],
    pool: &KvPool<B>,
    seq_states: &[&SeqKvState<B>],
    bufs: &PrefillBuffers<B>,
    logits_batch: &B::Tensor,
) -> anyhow::Result<()> {
    forward_decode_batch_impl(m, tokens, positions, pool, seq_states, bufs, logits_batch, &FEATURES)
}

pub(crate) fn forward_prefill_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    forward_prefill_impl(m, tokens, pool, seq_state, bufs, &FEATURES)
}

// ===========================================================================
// Implementation — shared by Llama, Phi, and Qwen via ArchFeatures.
//
// These are pub(crate) so that Qwen can call them with its own features.
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
///
/// This is the standard dense transformer pipeline:
///   1. Embed token
///   2. For each layer: norm → QKV → [bias] → RoPE → attention → O proj → FFN
///   3. Final norm → LM head
pub(crate) fn forward_single_impl<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    features: &ArchFeatures,
) -> anyhow::Result<()> {
    let d = m.dims();
    let pos = seq_state.seq_len as u32;

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Attention sub-block ---
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

        // QKV bias: the one line that separates Qwen from Llama/Phi.
        if features.has_qkv_bias {
            primitives::apply_qkv_bias_qdim(
                m.backend, layer, &m.q_buf, &m.k_buf, &m.v_buf, d.q_dim, d.kv_dim,
            );
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
            None, // No sliding window, default attention scale, no sinks.
            m.turbo_ctx.as_ref(),
        );
        primitives::o_proj_residual_qdim(
            m.backend,
            layer,
            &m.attn_out,
            &m.norm_buf,
            &m.hidden,
            d.hidden_size,
            d.q_dim,
        );
        profile::record(m.backend, t, Component::Attention);

        // --- FFN sub-block ---
        let t = profile::begin(m.backend);
        primitives::ffn_block(
            m.backend,
            layer,
            &m.hidden,
            &m.norm_buf,
            &m.gate_buf,
            &m.up_buf,
            d.eps,
            d.hidden_size,
            d.inter_size,
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

/// Batched prefill: process entire prompt in one GEMM-based forward pass.
pub(crate) fn forward_prefill_impl<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
    features: &ArchFeatures,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

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

        // QKV bias (batched): broadcast-add [dim] bias across [batch_size, dim].
        if features.has_qkv_bias {
            primitives::apply_qkv_bias_batch_qdim(m.backend, layer, bufs, bs, d.q_dim, d.kv_dim);
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
        primitives::paged_kv_and_prefill_attention(
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
        );
        primitives::o_proj_residual_batch_qdim(m.backend, layer, bufs, bs, d.hidden_size, d.q_dim);

        // FFN sub-block.
        primitives::ffn_block_batch(
            m.backend,
            layer,
            bufs,
            d.eps,
            bs,
            d.hidden_size,
            d.inter_size,
        );

        // Submit this layer's work so the GPU starts executing while we
        // encode the next layer.
        m.backend.submit();
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

/// Batched decode: process N tokens from N different sequences in one pass.
///
/// LEARNING OVERVIEW
///
/// This is the key throughput optimisation for serving concurrent requests.
/// Instead of N separate forward passes (each doing mat-vec through all
/// layers), we run ONE forward pass with GEMM on [N, dim] tensors.
///
/// The structure mirrors `forward_prefill_impl`, with two critical differences:
///
///   1. Positions are non-contiguous: each sequence is at a different point
///      in its generation (e.g., seq A at position 50, seq B at position 203).
///      Prefill has contiguous positions from one sequence.
///
///   2. Attention is per-sequence: each sequence has its own paged KV cache
///      with a different block table.  We extract individual Q/K/V rows from
///      the batched tensors, run paged_attention_fused per sequence, then
///      write the attention output back.  Everything else (projections, FFN,
///      AllReduce) is fully batched.
///
/// Performance: for N=8 sequences, projections become ~4-6x faster (GEMM
/// vs 8 mat-vecs).  NCCL AllReduce calls drop from 8×2×num_layers to
/// 2×num_layers.  Attention stays per-sequence (N kernel launches per layer)
/// but it's already the cheapest part of decode.
pub(crate) fn forward_decode_batch_impl<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    positions: &[u32],
    pool: &KvPool<B>,
    seq_states: &[&SeqKvState<B>],
    bufs: &PrefillBuffers<B>,
    logits_batch: &B::Tensor,
    features: &ArchFeatures,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    assert_eq!(positions.len(), tokens.len());
    assert_eq!(seq_states.len(), tokens.len());

    // Upload N token IDs and N non-contiguous positions to GPU.
    primitives::upload_decode_batch_inputs(m.backend, bufs, tokens, positions, bs);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];
        let kv_layer_idx = m.config.kv_layer_map()[layer_idx].unwrap();

        // --- Attention sub-block (batched projections, per-seq attention) ---

        // Batched RMSNorm: [N, hidden_size] → [N, hidden_size].
        m.backend.rms_norm_batch(
            &bufs.hidden,
            &layer.input_layernorm,
            d.eps,
            &bufs.norm_buf,
            bs,
        );

        // Batched QKV projection: 3 GEMMs instead of 3×N mat-vecs.
        primitives::qkv_projection_batch_qdim(
            m.backend, layer, bufs, bs, d.q_dim, d.hidden_size, d.kv_dim,
        );

        if features.has_qkv_bias {
            primitives::apply_qkv_bias_batch_qdim(m.backend, layer, bufs, bs, d.q_dim, d.kv_dim);
        }

        // Batched RoPE: each token gets its own position from the positions tensor.
        primitives::apply_rope_batch(
            m.backend, bufs, d.rope_theta, bs, d.num_heads, d.num_kv_heads, d.head_dim,
        );

        // Per-sequence attention: extract row, run paged attention, write back.
        // This is the one non-batched step — each sequence has a different block
        // table and seq_len.  The row copies are ~2 KB each (trivial overhead).
        for (i, &seq_state) in seq_states.iter().enumerate() {
            primitives::batched_decode_per_seq_attention(
                m.backend,
                bufs,
                &m.q_buf,
                &m.k_buf,
                &m.v_buf,
                &m.attn_out,
                pool,
                seq_state,
                kv_layer_idx,
                i,
                positions[i],
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
                0,    // window_size (0 = full context)
                0.0,  // attn_scale (0.0 = default 1/sqrt(head_dim))
                None, // sinks
            );
        }

        // Batched O projection + residual: 1 GEMM + 1 AllReduce for all N seqs.
        primitives::o_proj_residual_batch_qdim(
            m.backend, layer, bufs, bs, d.hidden_size, d.q_dim,
        );

        // --- FFN sub-block (fully batched) ---
        primitives::ffn_block_batch(
            m.backend, layer, bufs, d.eps, bs, d.hidden_size, d.inter_size,
        );

        // Submit this layer's work so the GPU starts while we encode the next.
        m.backend.submit();
    }

    // Final norm + LM head → [N, vocab_size] logits.
    primitives::final_norm_and_lm_head_decode_batch(
        m.backend,
        &m.weights,
        bufs,
        logits_batch,
        d.eps,
        bs,
        d.hidden_size as u32,
        m.config.vocab_size as u32,
    );

    Ok(())
}
