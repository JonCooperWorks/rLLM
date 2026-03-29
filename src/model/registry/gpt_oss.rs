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
//   Loader:      model/loader/ (MXFP4 dequant, expert biases, de-interleaving)
//   Primitives:  model/primitives.rs (biased MoE dispatch, YaRN RoPE)
//   Forward:     model/forward.rs (ModelForward trait, MoeBuffers)
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuBackend, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul,
    GpuNorm, GpuRope, GpuTurboQuant, TensorDtype,
};
use crate::model::forward::{MoeBuffers, ModelForward};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::profile::{self, Component};
use crate::model::{Model, PrefillBuffers};

// ===========================================================================
// GptOssForward — ModelForward trait implementation.
// ===========================================================================

pub(crate) struct GptOssForward<B: GpuCore> {
    pub moe: MoeBuffers<B>,
}

impl<B: GpuBackend> ModelForward<B> for GptOssForward<B> {
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
    moe: &MoeBuffers<B>,
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
        swiglu_limit,
    );
}

/// Pre-normed biased MoE FFN block — norm_buf already populated by fused residual+norm.
/// Inspired by rvLLM (m0at).
fn moe_ffn_block_pre_normed<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &Dims,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    swiglu_limit: f32,
    moe: &MoeBuffers<B>,
) {
    let layer = &m.weights.layers[layer_idx];
    primitives::moe_ffn_block_biased_pre_normed(
        m.backend,
        layer.router_gate.as_ref().unwrap(),
        layer.router_bias.as_ref(),
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
        swiglu_limit,
    );
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass for GPT-OSS-20B.
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
    moe: &MoeBuffers<B>,
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
        m.backend
            .rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

        // QKV projection with explicit q_dim (4096 vs 2880 hidden).
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

        // QKV bias (GPT-OSS has bias on all projections).
        primitives::apply_qkv_bias_qdim(
            m.backend, layer, &m.q_buf, &m.k_buf, &m.v_buf, d.q_dim, d.kv_dim,
        );

        // RoPE — use YaRN if rope_scaling is configured.
        if let Some(scaling) = rope_scaling {
            primitives::apply_rope_yarn(
                m.backend,
                &m.q_buf,
                &m.k_buf,
                pos,
                d.rope_theta,
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
                scaling,
            );
        } else {
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
        }

        // Sliding window: use window_size for sliding_attention layers, 0 for full.
        let window_size = if m.config.is_sliding_attention_layer(layer_idx) {
            m.config.sliding_window as u32
        } else {
            0
        };

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
            layer.sinks.as_ref(),
            m.turbo_ctx.as_ref(),
        );

        // Fused O-proj with bias + residual-add + post-attention RMSNorm: saves one
        // full read of the hidden tensor.  Inspired by rvLLM (m0at).
        primitives::o_proj_fused_residual_norm_qdim_biased(
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
            layer_idx,
            &d,
            moe_inter,
            num_experts,
            num_experts_per_tok,
            swiglu_limit,
            moe,
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
// Batched attention (GEMM) + per-token MoE FFN (same pattern as Qwen3 MoE).
// ===========================================================================

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
    moe: &MoeBuffers<B>,
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

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        // --- Batched attention ---
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

        // QKV bias (batched broadcast-add).
        primitives::apply_qkv_bias_batch_qdim(m.backend, layer, bufs, bs, d.q_dim, d.kv_dim);

        // Batched RoPE — use YaRN if configured.
        if let Some(scaling) = rope_scaling {
            primitives::apply_rope_yarn_batch(
                m.backend,
                bufs,
                d.rope_theta,
                bs,
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
                scaling,
            );
        } else {
            primitives::apply_rope_batch(
                m.backend,
                bufs,
                d.rope_theta,
                bs,
                d.num_heads,
                d.num_kv_heads,
                d.head_dim,
            );
        }

        // Sliding window.
        let window_size = if m.config.is_sliding_attention_layer(layer_idx) {
            m.config.sliding_window as u32
        } else {
            0
        };

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
            window_size,
            attn_scale,
            layer.sinks.as_ref(),
            m.turbo_ctx.as_ref(),
        );

        // O projection with bias (batched) + residual.
        m.backend.matmul_batch(
            &layer.o_proj,
            &bufs.attn_out,
            &bufs.norm_buf,
            bs,
            d.hidden_size,
            d.q_dim,
        );
        if let Some(ref o_bias) = layer.o_proj_bias {
            m.backend
                .bias_add_batch(&bufs.norm_buf, o_bias, &bufs.norm_buf, bs, d.hidden_size);
        }
        // AllReduce: sum partial O-projection results across GPUs (tensor parallelism).
        m.backend
            .all_reduce_sum(&bufs.norm_buf, bs * d.hidden_size);
        m.backend.add(
            &bufs.hidden,
            &bufs.norm_buf,
            &bufs.hidden,
            bs * d.hidden_size,
        );

        // --- MoE FFN: process each token independently ---
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
                layer_idx,
                &d,
                moe_inter,
                num_experts,
                num_experts_per_tok,
                swiglu_limit,
                moe,
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

// ===========================================================================
// Weight loading — GPT-OSS-specific layer loading.
//
// GPT-OSS has several unique loading needs vs. standard models:
//   - O-projection bias on attention (other models have no o_proj bias)
//   - Attention sinks (per-head scalar logits)
//   - MXFP4 expert weight format (microscaling FP4)
//   - Router bias on MoE gate
//   - Per-expert biases (gate, up, down)
//   - Interleaved gate/up weights (even/odd rows, not first/second half)
//
// These are moved from loader.rs to keep GPT-OSS logic self-contained.
// Shared primitives (upload_tensor, load_layer_norms, etc.) are called
// from loader.rs.
// ===========================================================================

use crate::model::loader::{
    TensorStore, LayerWeights, LoaderHints, ExpertWeights, FfnLoaded,
    upload_tensor, upload_raw_bf16, dequantize_mxfp4,
    load_layer_norms, load_attention_weights, assemble_layer_weights,
};
use crate::model::config::ModelConfig;

/// Load weights for one GPT-OSS layer (attention + MoE FFN).
///
/// Delegates to shared helpers for norms and standard attention projections,
/// then adds GPT-OSS-specific fields (o_proj_bias, sinks, MXFP4 MoE).
pub(crate) fn load_layer<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    prefix: &str,
    config: &ModelConfig,
    hints: &LoaderHints,
    layer_idx: usize,
    sharding: Option<&crate::gpu::parallel::ShardingPlan>,
    skip_experts: bool,
) -> anyhow::Result<LayerWeights<B>> {
    // Norms: standard (no residual, no sandwich).
    let norms = load_layer_norms(store, backend, prefix, config, hints)?;

    // Attention: standard GQA via shared helper.
    // The shared helper loads QKV bias (via hints.has_qkv_bias), O-proj bias
    // (via hints.has_o_proj_bias), and sinks (via hints.is_gpt_oss).
    let attn = load_attention_weights(
        store, backend, prefix, config, hints, layer_idx,
        false, // not deltanet
        sharding,
    )?;

    // MoE FFN: router + MXFP4 experts + router bias.
    let ffn = load_gpt_oss_moe(
        store, backend, prefix, config, layer_idx, sharding, skip_experts,
    )?;

    Ok(assemble_layer_weights(norms, attn, ffn))
}

/// Load GPT-OSS MoE FFN weights: router with bias + MXFP4 experts.
fn load_gpt_oss_moe<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    prefix: &str,
    config: &ModelConfig,
    layer_idx: usize,
    _sharding: Option<&crate::gpu::parallel::ShardingPlan>,
    skip_experts: bool,
) -> anyhow::Result<FfnLoaded<B>> {
    let hidden = config.hidden_size;
    let moe_inter = config.moe_intermediate_size;
    let num_experts = config.num_experts;

    // Dummy tensors for dense FFN fields (unused by MoE forward pass).
    let dummy = backend.alloc_tensor(&[1], TensorDtype::BF16);
    let dummy2 = backend.alloc_tensor(&[1], TensorDtype::BF16);
    let dummy3 = backend.alloc_tensor(&[1], TensorDtype::BF16);

    // Router gate: GPT-OSS uses mlp.router.weight naming.
    let router = upload_tensor(
        store, backend,
        &format!("{prefix}.mlp.router.weight"),
        &[num_experts, hidden],
    )?;

    // Router bias (GPT-OSS only).
    let router_bias = Some(upload_tensor(
        store, backend,
        &format!("{prefix}.mlp.router.bias"),
        &[num_experts],
    )?);

    // Detect MXFP4 format by looking for packed blocks tensor.
    let mxfp4_name = format!("{prefix}.mlp.experts.gate_up_proj_blocks");
    let is_mxfp4 = store.tensor(&mxfp4_name).is_ok();

    let expert_vec = if skip_experts {
        if layer_idx == 0 {
            eprintln!("  skipping {} experts per layer (streaming from SSD)", num_experts);
        }
        Vec::new()
    } else if is_mxfp4 {
        load_mxfp4_experts_local(store, backend, prefix, hidden, moe_inter, num_experts)?
    } else {
        // Fallback: per-expert format (shouldn't happen for GPT-OSS, but be safe).
        let mut experts = Vec::with_capacity(num_experts);
        for j in 0..num_experts {
            let ep = format!("{prefix}.mlp.experts.{j}");
            experts.push(ExpertWeights {
                gate_proj: upload_tensor(store, backend, &format!("{ep}.gate_proj.weight"), &[moe_inter, hidden])?,
                up_proj: upload_tensor(store, backend, &format!("{ep}.up_proj.weight"), &[moe_inter, hidden])?,
                down_proj: upload_tensor(store, backend, &format!("{ep}.down_proj.weight"), &[hidden, moe_inter])?,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
            });
        }
        experts
    };

    if layer_idx == 0 {
        eprintln!(
            "  loading {} experts per layer (moe_inter={}){}",
            num_experts, moe_inter,
            if is_mxfp4 { " [MXFP4 format]" } else { "" },
        );
    }

    Ok(FfnLoaded {
        gate_proj: dummy,
        up_proj: dummy2,
        down_proj: dummy3,
        router_gate: Some(router),
        router_bias,
        experts: if skip_experts { None } else { Some(expert_vec) },
        shared_expert_gate_proj: None,
        shared_expert_up_proj: None,
        shared_expert_down_proj: None,
        shared_expert_gate: None,
    })
}

/// Load MXFP4-format expert weights (GPT-OSS).
///
/// MXFP4 stores weights as packed fp4 blocks with per-block E8M0 scales
/// and optional per-expert biases.  We dequant to bf16 on the CPU,
/// de-interleave gate/up rows, then upload.
fn load_mxfp4_experts_local<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    prefix: &str,
    hidden: usize,
    moe_inter: usize,
    num_experts: usize,
) -> anyhow::Result<Vec<ExpertWeights<B>>> {
    let block_size = 32usize;

    let gu_blocks_view = store.tensor(&format!("{prefix}.mlp.experts.gate_up_proj_blocks"))?;
    let gu_scales_view = store.tensor(&format!("{prefix}.mlp.experts.gate_up_proj_scales"))?;
    let gu_blocks_data = gu_blocks_view.data();
    let gu_scales_data = gu_scales_view.data();

    let down_blocks_view = store.tensor(&format!("{prefix}.mlp.experts.down_proj_blocks"))?;
    let down_scales_view = store.tensor(&format!("{prefix}.mlp.experts.down_proj_scales"))?;
    let down_blocks_data = down_blocks_view.data();
    let down_scales_data = down_scales_view.data();

    // Optional biases.
    let gu_bias_name = format!("{prefix}.mlp.experts.gate_up_proj_bias");
    let gu_bias_data = store.tensor(&gu_bias_name).ok().map(|v| v.data().to_vec());
    let down_bias_name = format!("{prefix}.mlp.experts.down_proj_bias");
    let down_bias_data = store.tensor(&down_bias_name).ok().map(|v| v.data().to_vec());

    // Per-expert byte sizes for slicing fused tensors.
    let gu_rows = 2 * moe_inter;
    let gu_blocks_per_expert = gu_rows * (hidden / 2);
    let gu_num_scale_blocks = (hidden + block_size - 1) / block_size;
    let gu_scales_per_expert = gu_rows * gu_num_scale_blocks;

    let down_rows = hidden;
    let down_blocks_per_expert = down_rows * (moe_inter / 2);
    let down_num_scale_blocks = (moe_inter + block_size - 1) / block_size;
    let down_scales_per_expert = down_rows * down_num_scale_blocks;

    let gu_bias_per_expert = gu_rows * 2; // bf16
    let down_bias_per_expert = down_rows * 2;

    let mut experts = Vec::with_capacity(num_experts);
    for j in 0..num_experts {
        // Dequant gate_up.
        let gu_b_off = j * gu_blocks_per_expert;
        let gu_s_off = j * gu_scales_per_expert;
        let gu_bf16 = dequantize_mxfp4(
            &gu_blocks_data[gu_b_off..gu_b_off + gu_blocks_per_expert],
            &gu_scales_data[gu_s_off..gu_s_off + gu_scales_per_expert],
            gu_rows, hidden, block_size,
        );

        // De-interleave gate and up from fused gate_up_proj.
        // GPT-OSS: interleaved rows (even=gate, odd=up).
        let row_bytes = hidden * 2;
        let mut gate_raw = vec![0u8; moe_inter * row_bytes];
        let mut up_raw = vec![0u8; moe_inter * row_bytes];
        for r in 0..moe_inter {
            let even_start = (2 * r) * row_bytes;
            let odd_start = (2 * r + 1) * row_bytes;
            gate_raw[r * row_bytes..(r + 1) * row_bytes]
                .copy_from_slice(&gu_bf16[even_start..even_start + row_bytes]);
            up_raw[r * row_bytes..(r + 1) * row_bytes]
                .copy_from_slice(&gu_bf16[odd_start..odd_start + row_bytes]);
        }
        let gate_t = upload_raw_bf16(backend, &gate_raw, &[moe_inter, hidden]);
        let up_t = upload_raw_bf16(backend, &up_raw, &[moe_inter, hidden]);

        // Dequant down.
        let d_b_off = j * down_blocks_per_expert;
        let d_s_off = j * down_scales_per_expert;
        let down_bf16 = dequantize_mxfp4(
            &down_blocks_data[d_b_off..d_b_off + down_blocks_per_expert],
            &down_scales_data[d_s_off..d_s_off + down_scales_per_expert],
            down_rows, moe_inter, block_size,
        );
        let down_t = upload_raw_bf16(backend, &down_bf16, &[hidden, moe_inter]);

        // Expert biases — de-interleave fused gate_up_bias.
        let (gate_bias, up_bias) = if let Some(ref bias) = gu_bias_data {
            let off = j * gu_bias_per_expert;
            let bias_slice = &bias[off..off + gu_bias_per_expert];
            let bias_bf16: &[u16] = bytemuck::cast_slice(bias_slice);
            let gate_vals: Vec<u16> = (0..moe_inter).map(|i| bias_bf16[2 * i]).collect();
            let up_vals: Vec<u16> = (0..moe_inter).map(|i| bias_bf16[2 * i + 1]).collect();
            let gate_bytes: &[u8] = bytemuck::cast_slice(&gate_vals);
            let up_bytes: &[u8] = bytemuck::cast_slice(&up_vals);
            (
                Some(backend.upload_tensor(gate_bytes, &[moe_inter], TensorDtype::BF16)),
                Some(backend.upload_tensor(up_bytes, &[moe_inter], TensorDtype::BF16)),
            )
        } else {
            (None, None)
        };
        let down_bias = if let Some(ref bias) = down_bias_data {
            let off = j * down_bias_per_expert;
            let bias_slice = &bias[off..off + down_bias_per_expert];
            Some(backend.upload_tensor(bias_slice, &[down_rows], TensorDtype::BF16))
        } else {
            None
        };

        experts.push(ExpertWeights {
            gate_proj: gate_t,
            up_proj: up_t,
            down_proj: down_t,
            gate_bias,
            up_bias,
            down_bias,
        });
    }
    Ok(experts)
}
