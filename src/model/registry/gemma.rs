// ===========================================================================
// Gemma 3 model family (Google, 1B, 4B, 12B, 27B).
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for Google's Gemma 3 text models.  Gemma 3
//   shares the Llama backbone (GQA attention + gated FFN + RoPE) but has
//   several distinctive architectural choices that set it apart.
//
// Why this file can't reuse llama.rs:
//   Gemma 3 differs from the standard dense pipeline in too many ways:
//
//   1. **Sandwich norms**: 4 RMSNorm layers per decoder block instead of 2.
//      Pre-norm AND post-norm around BOTH attention and FFN sub-blocks:
//        residual += post_attn_norm(attention(pre_attn_norm(residual)))
//        residual += post_ffn_norm(ffn(pre_ffn_norm(residual)))
//      The post-norms prevent sub-layer outputs from growing unboundedly.
//
//   2. **GeGLU activation**: gelu(gate) * up instead of silu(gate) * up.
//      GELU's smoother gradients help training stability in deeper networks.
//
//   3. **Embedding scaling**: embeddings multiplied by sqrt(hidden_size) after
//      lookup.  No learned embedding norm — just a fixed scalar.
//
//   4. **Custom attention scale**: uses query_pre_attn_scalar from config
//      instead of head_dim for computing 1/sqrt(scale).
//
//   5. **Sliding window attention**: layers alternate between "local" (sliding
//      window of ~512 tokens) and "global" (full context).  Pattern: 5 local
//      + 1 global, repeating.  Local layers use a smaller RoPE base (10000)
//      while global layers use a larger base (1000000) for better long-range
//      position discrimination.
//
//   6. **Offset RMSNorm**: weights stored as offsets from 1.0 (initialized
//      to zero, effective = 1 + stored).  Handled at load time.
//
// Forward pass pipeline (single token decode):
//   1. Embed token + scale by sqrt(hidden_size)
//   2. For each layer:
//      a. Pre-attention norm (RMSNorm)
//      b. Q/K/V projections
//      c. RoPE (local or global theta depending on layer type)
//      d. KV cache write + attention (with sliding window for local layers)
//      e. Post-attention norm (RMSNorm) ← sandwich norm
//      f. Residual add
//      g. Pre-FFN norm (RMSNorm)
//      h. Gate/up projections + GeGLU activation + down projection
//      i. Post-FFN norm (RMSNorm) ← sandwich norm
//      j. Residual add
//   3. Final norm + LM head projection
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
// Gemma 3 attention sub-block helpers.
//
// The attention is standard GQA but with two Gemma-specific twists:
//   1. Sliding window: local layers only attend to the last `sliding_window`
//      tokens.  Global layers attend to the full context (window_size=0).
//   2. Dual RoPE bases: local layers use rope_local_base_freq (10000),
//      global layers use rope_theta (1000000).
//   3. Custom scale: 1/sqrt(query_pre_attn_scalar) instead of 1/sqrt(head_dim).
// ===========================================================================

/// Get the RoPE theta for a given layer.
///
/// Local (sliding window) layers use the local base frequency for sharper
/// positional encoding of nearby tokens.  Global layers use the main
/// rope_theta for long-range position discrimination.
fn layer_rope_theta(config: &crate::model::config::ModelConfig, layer_idx: usize) -> f32 {
    if config.is_sliding_attention_layer(layer_idx) {
        if config.rope_local_base_freq > 0.0 {
            config.rope_local_base_freq as f32
        } else {
            config.rope_theta as f32
        }
    } else {
        config.rope_theta as f32
    }
}

/// Get the attention window size for a given layer.
///
/// Local layers use sliding_window; global layers use 0 (full context).
fn layer_window_size(config: &crate::model::config::ModelConfig, layer_idx: usize) -> u32 {
    if config.is_sliding_attention_layer(layer_idx) {
        config.sliding_window as u32
    } else {
        0 // Full context.
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
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let pos = seq_state.seq_len as u32;
    let uses_geglu = m.config.uses_geglu();

    // Gemma 3 attention scale: 1/sqrt(query_pre_attn_scalar) instead of 1/sqrt(head_dim).
    let attn_scale = if m.config.query_pre_attn_scalar > 0.0 {
        1.0 / (m.config.query_pre_attn_scalar as f32).sqrt()
    } else {
        0.0 // Let the kernel use the default 1/sqrt(head_dim).
    };

    // Embedding scaling factor: sqrt(hidden_size).
    let embed_scale = (m.config.hidden_size as f32).sqrt();

    let t = profile::begin(m.backend);
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);
    // Scale embeddings by sqrt(hidden_size) — Gemma's way of ensuring the
    // embedding vectors start at the right magnitude for the residual stream.
    m.backend
        .scalar_mul(&m.hidden, &m.hidden, embed_scale, d.hidden_size);
    profile::record(m.backend, t, Component::Embed);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];
        let rope_theta = layer_rope_theta(&m.config, layer_idx);
        let window_size = layer_window_size(&m.config, layer_idx);

        // --- Attention sub-block (with sandwich norms) ---
        let t = profile::begin(m.backend);

        // Pre-attention norm.
        m.backend
            .rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

        // Q/K/V projections (q_dim may differ from hidden_size, e.g. 4B: 2048 vs 2560).
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

        // QK-norm: per-head RMSNorm on Q and K after projection, before RoPE.
        if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
            m.backend
                .rms_norm_batch(&m.q_buf, q_norm, d.eps, &m.q_buf, d.num_heads);
            m.backend
                .rms_norm_batch(&m.k_buf, k_norm, d.eps, &m.k_buf, d.num_kv_heads);
        }

        // RoPE with layer-specific theta (local vs global).
        primitives::apply_rope(
            m.backend,
            &m.q_buf,
            &m.k_buf,
            pos,
            rope_theta,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
        );

        // Paged KV cache write + attention with sliding window + custom scale.
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
            window_size,
            attn_scale,
            None,
            m.turbo_ctx.as_ref(),
        );

        // O projection into norm_buf (reused as scratch).
        // O weight shape: [hidden_size, q_dim] — maps attention output back to residual stream.
        m.backend.matmul(
            &layer.o_proj,
            &m.attn_out,
            &m.norm_buf,
            d.hidden_size,
            d.q_dim,
        );
        // AllReduce: sum partial O-projection results across GPUs (tensor parallelism).
        m.backend.all_reduce_sum(&m.norm_buf, d.hidden_size);

        // Post-attention norm (sandwich norm): normalise before adding to residual.
        // This is the key Gemma 3 difference — Llama skips this step.
        m.backend.rms_norm(
            &m.norm_buf,
            &layer.post_attention_layernorm,
            d.eps,
            &m.norm_buf,
        );

        // Residual add.
        m.backend
            .add(&m.hidden, &m.norm_buf, &m.hidden, d.hidden_size);
        profile::record(m.backend, t, Component::Attention);

        // --- FFN sub-block (with sandwich norms + GeGLU) ---
        let t = profile::begin(m.backend);

        // Pre-FFN norm.
        let pre_ffn_norm = layer.pre_feedforward_layernorm.as_ref().unwrap();
        m.backend
            .rms_norm(&m.hidden, pre_ffn_norm, d.eps, &m.norm_buf);

        // Gate and up projections.
        m.backend.matmul(
            &layer.gate_proj,
            &m.norm_buf,
            &m.gate_buf,
            d.inter_size,
            d.hidden_size,
        );
        m.backend.matmul(
            &layer.up_proj,
            &m.norm_buf,
            &m.up_buf,
            d.inter_size,
            d.hidden_size,
        );

        // GeGLU or SwiGLU activation.
        if uses_geglu {
            m.backend
                .gelu_mul(&m.gate_buf, &m.up_buf, &m.gate_buf, d.inter_size);
        } else {
            m.backend
                .silu_mul(&m.gate_buf, &m.up_buf, &m.gate_buf, d.inter_size);
        }

        // Down projection.
        m.backend.matmul(
            &layer.down_proj,
            &m.gate_buf,
            &m.norm_buf,
            d.hidden_size,
            d.inter_size,
        );
        // AllReduce: sum partial down-projection results across GPUs (tensor parallelism).
        m.backend.all_reduce_sum(&m.norm_buf, d.hidden_size);

        // Post-FFN norm (sandwich norm).
        let post_ffn_norm = layer.post_feedforward_layernorm.as_ref().unwrap();
        m.backend
            .rms_norm(&m.norm_buf, post_ffn_norm, d.eps, &m.norm_buf);

        // Residual add.
        m.backend
            .add(&m.hidden, &m.norm_buf, &m.hidden, d.hidden_size);
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
// Learning note: Gemma 3's sliding window attention complicates batched
// prefill slightly — each layer needs to pass the correct window_size and
// attn_scale.  But since we compute these from config, it's just a matter
// of passing the right values to the prefill attention kernel.
//
// The dual RoPE bases (local vs global) mean we need per-layer rope_theta.
// Since rope_batch uses a single theta, we handle this by using the standard
// full-batch RoPE for each layer with the appropriate theta.
// ===========================================================================

/// Batched prefill: process entire prompt in one GEMM-based forward pass.
pub(crate) fn forward_prefill_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuAllReduce,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let uses_geglu = m.config.uses_geglu();

    let attn_scale = if m.config.query_pre_attn_scalar > 0.0 {
        1.0 / (m.config.query_pre_attn_scalar as f32).sqrt()
    } else {
        0.0
    };
    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];
        let rope_theta = layer_rope_theta(&m.config, layer_idx);
        let window_size = layer_window_size(&m.config, layer_idx);

        // Pre-attention norm (batched).
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

        // QK-norm: per-head RMSNorm on Q and K after projection, before RoPE.
        if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
            m.backend
                .rms_norm_batch(&bufs.q_buf, q_norm, d.eps, &bufs.q_buf, bs * d.num_heads);
            m.backend
                .rms_norm_batch(&bufs.k_buf, k_norm, d.eps, &bufs.k_buf, bs * d.num_kv_heads);
        }

        // RoPE with per-layer theta.
        primitives::apply_rope_batch(
            m.backend,
            bufs,
            rope_theta,
            bs,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
        );

        // KV cache write + prefill attention with sliding window.
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
            window_size,
            attn_scale,
            None,
        );

        // O projection (batched): O weight [hidden_size, q_dim].
        m.backend.matmul_batch(
            &layer.o_proj,
            &bufs.attn_out,
            &bufs.norm_buf,
            bs,
            d.hidden_size,
            d.q_dim,
        );
        // AllReduce: sum partial O-projection results across GPUs (tensor parallelism).
        m.backend.all_reduce_sum(&bufs.norm_buf, bs * d.hidden_size);

        // Post-attention norm (batched sandwich norm).
        m.backend.rms_norm_batch(
            &bufs.norm_buf,
            &layer.post_attention_layernorm,
            d.eps,
            &bufs.norm_buf,
            bs,
        );

        // Residual add.
        m.backend.add(
            &bufs.hidden,
            &bufs.norm_buf,
            &bufs.hidden,
            bs * d.hidden_size,
        );

        // --- FFN sub-block with sandwich norms + GeGLU ---

        // Pre-FFN norm (batched).
        let pre_ffn_norm = layer.pre_feedforward_layernorm.as_ref().unwrap();
        m.backend
            .rms_norm_batch(&bufs.hidden, pre_ffn_norm, d.eps, &bufs.norm_buf, bs);

        // Gate/up projections (GEMM).
        m.backend.matmul_batch(
            &layer.gate_proj,
            &bufs.norm_buf,
            &bufs.gate_buf,
            bs,
            d.inter_size,
            d.hidden_size,
        );
        m.backend.matmul_batch(
            &layer.up_proj,
            &bufs.norm_buf,
            &bufs.up_buf,
            bs,
            d.inter_size,
            d.hidden_size,
        );

        // GeGLU or SwiGLU activation.
        if uses_geglu {
            m.backend.gelu_mul(
                &bufs.gate_buf,
                &bufs.up_buf,
                &bufs.gate_buf,
                bs * d.inter_size,
            );
        } else {
            m.backend.silu_mul(
                &bufs.gate_buf,
                &bufs.up_buf,
                &bufs.gate_buf,
                bs * d.inter_size,
            );
        }

        // Down projection.
        m.backend.matmul_batch(
            &layer.down_proj,
            &bufs.gate_buf,
            &bufs.norm_buf,
            bs,
            d.hidden_size,
            d.inter_size,
        );
        // AllReduce: sum partial down-projection results across GPUs (tensor parallelism).
        m.backend.all_reduce_sum(&bufs.norm_buf, bs * d.hidden_size);

        // Post-FFN norm (batched sandwich norm).
        let post_ffn_norm = layer.post_feedforward_layernorm.as_ref().unwrap();
        m.backend
            .rms_norm_batch(&bufs.norm_buf, post_ffn_norm, d.eps, &bufs.norm_buf, bs);

        // Residual add.
        m.backend.add(
            &bufs.hidden,
            &bufs.norm_buf,
            &bufs.hidden,
            bs * d.hidden_size,
        );

        // Submit this layer's work so the GPU starts executing while we encode the next layer.
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
