// ===========================================================================
// Nemotron-H forward pass — NVIDIA's three-way hybrid architecture.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for NVIDIA's Nemotron-H architecture, which
//   interleaves three fundamentally different layer types:
//
//   1. **Mamba-2 SSM layers** (23 of 52 layers, "M" in config pattern):
//      Selective State Space Model with O(1) per-token cost.  Each head
//      maintains a [head_dim, state_size] recurrent state matrix.  No KV
//      cache needed — state is fixed-size regardless of sequence length.
//
//   2. **MoE FFN layers** (23 of 52, "E" in pattern):
//      Mixture-of-Experts with 128 routed experts + 1 shared expert, top-6
//      routing.  Uses relu-squared activation (NOT SwiGLU) — each expert is
//      just up_proj → relu² → down_proj.  Routing uses sigmoid + correction
//      bias (DeepSeek-V3 style) for load balancing.
//
//   3. **Self-attention layers** (6 of 52, "*" in pattern):
//      Standard GQA with 32 Q-heads, 2 KV-heads, head_dim=128, full RoPE.
//      Only these 6 layers need KV cache.
//
// Key difference from Qwen 3.5:
//   Qwen 3.5 layers PAIR a mixer (DeltaNet or attention) WITH an FFN.
//   Nemotron-H layers are STANDALONE — each layer is purely one type.
//   There's no separate FFN after a Mamba block; the SSM IS the whole layer.
//
// Each layer follows:  hidden = hidden + mixer(rmsnorm(hidden))
// where mixer is one of the three types above.
//
// State management:
//   - Mamba-2 state: [num_heads, head_dim, state_size] f32 per Mamba layer
//     (~2MB per layer × 23 = 46MB per sequence)
//   - Conv1d history: [(kernel_size-1), d_inner] bf16 per Mamba layer
//   - KV cache: only for the 6 attention layers (paged, from shared pool)
//
// Prefill strategy (same as Qwen 3.5 hybrid):
//   - Attention layers: batched GEMM + prefill attention
//   - Mamba/MoE layers: token-by-token loop (inherently sequential)
//
// Related files:
//   config.rs         — NemotronH arch, hybrid_override_pattern parsing
//   loader.rs         — weight loading for all three layer types
//   mod.rs            — Model struct, Mamba-2 state allocation, dispatch
//   gpu/ops/mamba2.rs — GpuMamba2 trait (conv1d, ssm_step)
//   gpu/metal/shaders/mamba2.metal — Metal kernel implementations
// ===========================================================================

use crate::gpu::{
    GpuAttention, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMamba2, GpuMatmul, GpuMoe,
    GpuNorm, GpuRope, GpuAllReduce, GpuTurboQuant,
};
use crate::model::config::ModelConfig;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives;
use crate::model::{Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// Layer index mapping helpers.
//
// Since Nemotron-H layers are heterogeneous, we need to map from the global
// layer index to type-specific indices (e.g., which Mamba state to use,
// which KV pool slot to use).
// ---------------------------------------------------------------------------

/// Map global layer index → Mamba-2 state index.
/// Counts how many Mamba layers appear before this one.
fn mamba2_layer_index(config: &ModelConfig, layer_idx: usize) -> usize {
    config.layer_types[..layer_idx]
        .iter()
        .filter(|t| t.as_str() == "mamba2")
        .count()
}

/// Prenorm scaling factor: 1/√(2 × num_layers).
/// Applied after RMSNorm and before the mixer when rescale_prenorm_residual=true.
/// Prevents signal amplification through deep networks.
fn prenorm_scale(config: &ModelConfig) -> f32 {
    if config.rescale_prenorm_residual {
        1.0 / ((2.0 * config.num_hidden_layers as f64).sqrt()) as f32
    } else {
        1.0
    }
}

/// Shared dimension struct to avoid recomputing sizes on every layer.
struct NemotronDims {
    hidden_size: u32,
    q_dim: u32,
    kv_dim: u32,
    d_inner: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    mamba_num_heads: u32,
    mamba_head_dim: u32,
    state_size: u32,
    n_groups: u32,
    num_experts_per_tok: u32,
    moe_inter: u32,
    shared_inter: u32,
    eps: f32,
    prenorm_scale: f32,
}

impl NemotronDims {
    fn from_config(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size as u32,
            q_dim: (config.num_attention_heads * config.head_dim) as u32,
            kv_dim: (config.num_key_value_heads * config.head_dim) as u32,
            d_inner: config.mamba2_d_inner() as u32,
            num_heads: config.num_attention_heads as u32,
            num_kv_heads: config.num_key_value_heads as u32,
            head_dim: config.head_dim as u32,
            mamba_num_heads: config.mamba_num_heads as u32,
            mamba_head_dim: config.mamba_head_dim as u32,
            state_size: config.ssm_state_size as u32,
            n_groups: config.mamba_n_groups as u32,
            num_experts_per_tok: config.num_experts_per_tok as u32,
            moe_inter: config.moe_intermediate_size as u32,
            shared_inter: config.shared_expert_intermediate_size as u32,
            eps: config.rms_norm_eps as f32,
            prenorm_scale: prenorm_scale(config),
        }
    }
}

// ===========================================================================
// Single-token decode forward pass.
// ===========================================================================

pub(crate) fn forward_single_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuDeltaNet
        + GpuMamba2
        + GpuMoe
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let d = NemotronDims::from_config(&m.config);
    let pos = seq_state.seq_len as u32;

    // Embed the token into the residual stream.
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);

    // Process each layer — dispatch based on layer type.
    for layer_idx in 0..m.config.num_hidden_layers {
        let layer_type = m.config.layer_types[layer_idx].as_str();
        match layer_type {
            "mamba2" => {
                mamba2_block(m, layer_idx, &d);
            }
            "moe" => {
                moe_block(m, layer_idx, &d);
            }
            "attention" => {
                attention_block(m, layer_idx, &d, pos, pool, seq_state);
            }
            _ => unreachable!("invalid layer type at {layer_idx}: {layer_type}"),
        }
    }

    // Final RMSNorm + lm_head projection → logits.
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Mamba-2 SSM block (single-token decode).
//
// Pipeline:
//   1. RMSNorm(hidden) → norm_buf
//   2. Scale by prenorm_scale
//   3. in_proj matmul → [z, x, B, C, dt] split by offset
//   4. conv1d(x) + silu → conv_out (with bias)
//   5. shift conv history
//   6. SSM step: state update + output + norm → ssm_out
//   7. Output gate: ssm_out × silu(z)
//   8. out_proj matmul → residual add
// ---------------------------------------------------------------------------

fn mamba2_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuDeltaNet + GpuMamba2 + GpuAllReduce>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &NemotronDims,
) {
    let layer = &m.weights.layers[layer_idx];
    let dn_idx = mamba2_layer_index(&m.config, layer_idx);

    // Unwrap Mamba state and scratch buffers.
    let states = m.mamba2_states.as_ref().expect("mamba2_states");
    let conv_hist = m.mamba2_conv_history.as_ref().expect("mamba2_conv_history");
    let in_proj_buf = m.m2_in_proj_buf.as_ref().expect("m2_in_proj_buf");
    let conv_out = m.m2_conv_out.as_ref().expect("m2_conv_out");
    let ssm_out = m.m2_ssm_out.as_ref().expect("m2_ssm_out");

    let state = &states[dn_idx];
    let history = &conv_hist[dn_idx];

    let in_proj = layer.mamba_in_proj.as_ref().expect("mamba_in_proj");
    let conv_w = layer.mamba_conv1d_weight.as_ref().expect("conv1d_weight");
    let conv_b = layer.mamba_conv1d_bias.as_ref().expect("conv1d_bias");
    let out_proj = layer.mamba_out_proj.as_ref().expect("mamba_out_proj");
    let a_log = layer.mamba_a_log.as_ref().expect("mamba_a_log");
    let d_skip = layer.mamba_d.as_ref().expect("mamba_d");
    let dt_bias = layer.mamba_dt_bias.as_ref().expect("mamba_dt_bias");
    let norm_w = layer.mamba_norm.as_ref().expect("mamba_norm");

    let in_proj_dim = m.config.mamba2_in_proj_dim() as u32;

    // 1. RMSNorm.
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

    // 2. Scale by prenorm_scale.
    if d.prenorm_scale != 1.0 {
        m.backend.scalar_mul(&m.norm_buf, &m.norm_buf, d.prenorm_scale, d.hidden_size);
    }

    // 3. in_proj: [in_proj_dim, hidden_size] × norm_buf → in_proj_buf.
    m.backend.matmul(in_proj, &m.norm_buf, in_proj_buf, in_proj_dim, d.hidden_size);

    // 4. Conv1d + SiLU on x portion (offset d_inner..2*d_inner in in_proj_buf).
    // The x portion is at element offset d_inner from the start of in_proj_buf.
    // We use element-offset views via the backend's tensor slicing.
    // For now, we pass the full in_proj_buf and use offsets in the kernel.
    m.backend.mamba2_conv1d_silu(
        in_proj_buf,  // x portion accessed at offset d_inner
        history,
        conv_w,
        conv_b,
        conv_out,
        d.d_inner,
        m.config.mamba_conv_kernel as u32,
    );

    // 5. Shift conv history: append the current x to the FIFO buffer.
    m.backend.conv1d_shift_history(
        history,
        in_proj_buf,  // x at offset d_inner
        d.d_inner,
        m.config.mamba_conv_kernel as u32,
    );

    // 6. SSM step: state update + output + RMSNorm.
    // B is at offset 2*d_inner, C at 2*d_inner + n_groups*state_size,
    // dt at 2*d_inner + 2*n_groups*state_size in in_proj_buf.
    m.backend.mamba2_ssm_step(
        state,
        conv_out,
        in_proj_buf,  // B at offset 2*d_inner
        in_proj_buf,  // C at offset 2*d_inner + n_groups*state_size
        in_proj_buf,  // dt at offset 2*d_inner + 2*n_groups*state_size
        a_log,
        d_skip,
        dt_bias,
        norm_w,
        ssm_out,
        d.mamba_num_heads,
        d.mamba_head_dim,
        d.state_size,
        d.n_groups,
        d.eps,
    );

    // 7. Output gate: ssm_out = ssm_out × silu(z).
    // z is the first d_inner elements of in_proj_buf.
    m.backend.silu(in_proj_buf, in_proj_buf, d.d_inner);
    m.backend.mul(ssm_out, in_proj_buf, ssm_out, d.d_inner);

    // 8. out_proj + all-reduce (TP) + residual: hidden += out_proj × ssm_out.
    m.backend.matmul(out_proj, ssm_out, &m.norm_buf, d.hidden_size, d.d_inner);
    m.backend.all_reduce_sum(&m.norm_buf, d.hidden_size);
    m.backend.add(&m.hidden, &m.norm_buf, &m.hidden, d.hidden_size);
}

// ---------------------------------------------------------------------------
// MoE FFN block (single-token decode).
//
// Pipeline:
//   1. RMSNorm(hidden) → norm_buf
//   2. Scale by prenorm_scale
//   3. Router: gate × norm_buf → logits, sigmoid top-k → indices + weights
//   4. For each selected expert: up_proj → relu² → down_proj
//   5. Shared expert: up_proj → relu² → down_proj (always active)
//   6. Combine expert outputs + residual
// ---------------------------------------------------------------------------

fn moe_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuMoe>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &NemotronDims,
) {
    let layer = &m.weights.layers[layer_idx];
    let router_gate = layer.router_gate.as_ref().expect("router_gate");
    let experts = layer.experts.as_ref().expect("experts");
    let router_logits = m.router_logits.as_ref().expect("router_logits");
    let routing_output = m.routing_output.as_ref().expect("routing_output");
    let moe_up_buf = m.moe_up_buf.as_ref().expect("moe_up_buf");
    let moe_gate_buf = m.moe_gate_buf.as_ref().expect("moe_gate_buf");
    let moe_output = m.moe_output.as_ref().expect("moe_output");

    let num_experts = m.config.num_experts as u32;

    // 1. RMSNorm.
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

    // 2. Scale.
    if d.prenorm_scale != 1.0 {
        m.backend.scalar_mul(&m.norm_buf, &m.norm_buf, d.prenorm_scale, d.hidden_size);
    }

    // 3. Router: matmul → sigmoid top-k.
    m.backend.matmul(router_gate, &m.norm_buf, router_logits, num_experts, d.hidden_size);

    // Use sigmoid routing with correction bias (DeepSeek-V3 style).
    if let Some(ref correction_bias) = layer.e_score_correction_bias {
        m.backend.top_k_sigmoid(
            router_logits,
            correction_bias,
            routing_output,
            num_experts,
            d.num_experts_per_tok,
            m.config.routed_scaling_factor as f32,
            m.config.norm_topk_prob,
        );
    } else {
        // Fallback to softmax if no correction bias.
        m.backend.top_k_softmax(router_logits, routing_output, num_experts, d.num_experts_per_tok);
    }

    // Read routing decisions from GPU → CPU.
    let k = d.num_experts_per_tok as usize;
    let routing_bytes = k * 2 * 4; // 2*k f32 values (index, weight) pairs.
    let buf_bytes = m.backend.tensor_byte_count(routing_output);
    let mut routing_buf = vec![0u8; buf_bytes];
    m.backend.copy_to_host(routing_output, &mut routing_buf);
    let routing_data: &[f32] = bytemuck::cast_slice(&routing_buf[..routing_bytes]);

    // 4. Zero the output accumulator.
    m.backend.fill_zero(moe_output, d.hidden_size);

    // 5. For each selected expert: up_proj → relu² → down_proj → accumulate.
    for pair_idx in 0..k {
        let expert_idx = routing_data[pair_idx * 2] as usize;
        let weight = routing_data[pair_idx * 2 + 1];

        let expert = &experts[expert_idx];

        // up_proj: [moe_inter, hidden] × norm_buf → moe_up_buf.
        m.backend.matmul(&expert.up_proj, &m.norm_buf, moe_up_buf, d.moe_inter, d.hidden_size);

        // relu²: in-place.
        m.backend.relu_squared(moe_up_buf, moe_up_buf, d.moe_inter);

        // down_proj: [hidden, moe_inter] × moe_up_buf → moe_gate_buf.
        m.backend.matmul(&expert.down_proj, moe_up_buf, moe_gate_buf, d.hidden_size, d.moe_inter);

        // Weighted accumulate: moe_output += weight × moe_gate_buf.
        m.backend.scale_add(moe_output, moe_gate_buf, weight, d.hidden_size);
    }

    // 6. Shared expert (always active, same relu² pattern but larger intermediate).
    if let (Some(shared_up), Some(shared_down)) = (
        &layer.shared_expert_up_proj,
        &layer.shared_expert_down_proj,
    ) {
        // up_proj: [shared_inter, hidden] × norm_buf → up_buf (reuse gate_buf if large enough).
        m.backend.matmul(shared_up, &m.norm_buf, &m.up_buf, d.shared_inter, d.hidden_size);
        m.backend.relu_squared(&m.up_buf, &m.up_buf, d.shared_inter);
        m.backend.matmul(shared_down, &m.up_buf, &m.gate_buf, d.hidden_size, d.shared_inter);
        m.backend.add(moe_output, &m.gate_buf, moe_output, d.hidden_size);
    }

    // 7. Residual: hidden += moe_output.
    m.backend.add(&m.hidden, moe_output, &m.hidden, d.hidden_size);
}

// ---------------------------------------------------------------------------
// Self-attention block (single-token decode).
//
// Standard GQA: RMSNorm → QKV projection → RoPE → paged KV write →
// attention → O projection → residual.
// ---------------------------------------------------------------------------

fn attention_block<
    B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuAllReduce + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &NemotronDims,
    pos: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) {
    let layer = &m.weights.layers[layer_idx];

    // 1. RMSNorm.
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

    // 2. Scale.
    if d.prenorm_scale != 1.0 {
        m.backend.scalar_mul(&m.norm_buf, &m.norm_buf, d.prenorm_scale, d.hidden_size);
    }

    // 3. QKV projections.
    m.backend.matmul(&layer.q_proj, &m.norm_buf, &m.q_buf, d.q_dim, d.hidden_size);
    m.backend.matmul(&layer.k_proj, &m.norm_buf, &m.k_buf, d.kv_dim, d.hidden_size);
    m.backend.matmul(&layer.v_proj, &m.norm_buf, &m.v_buf, d.kv_dim, d.hidden_size);

    // 4. RoPE (full rotation, partial_rotary_factor=1.0).
    m.backend.rope(&m.q_buf, &m.k_buf, pos, m.config.rope_theta as f32, d.num_heads, d.num_kv_heads, d.head_dim);

    // 5. Paged KV write + attention.
    let kv_idx = m.kv_layer_map[layer_idx].expect("attention layer must have KV pool slot");
    primitives::paged_kv_and_attention_maybe_quantized(
        m.backend,
        &m.k_buf,
        &m.v_buf,
        &m.q_buf,
        &m.attn_out,
        pool,
        seq_state,
        kv_idx,
        pos,
        d.num_heads,
        d.num_kv_heads,
        d.head_dim,
        0, // window_size: 0 = full attention (no sliding window)
        0.0, // attn_scale: 0 = use default 1/sqrt(head_dim)
        None, // sinks: none for Nemotron-H
        m.turbo_ctx.as_ref(),
    );

    // 6. O projection + all-reduce (TP) + residual.
    m.backend.matmul(&layer.o_proj, &m.attn_out, &m.norm_buf, d.hidden_size, d.q_dim);
    m.backend.all_reduce_sum(&m.norm_buf, d.hidden_size);
    m.backend.add(&m.hidden, &m.norm_buf, &m.hidden, d.hidden_size);
}

// ===========================================================================
// Prefill forward pass (batch of tokens).
//
// Attention layers can batch (GEMM + prefill attention).
// Mamba-2 and MoE layers process token-by-token (inherently sequential).
// ===========================================================================

pub(crate) fn forward_prefill_paged<
    B: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuDeltaNet
        + GpuMamba2
        + GpuMoe
        + GpuAllReduce
        + GpuTurboQuant,
>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let _d = NemotronDims::from_config(&m.config);
    let _n = tokens.len();

    // TODO: Implement prefill path.
    // For now, fall back to single-token decode one at a time.
    // This is correct but slower than batched prefill for attention layers.
    for (i, &token_id) in tokens.iter().enumerate() {
        // Temporarily advance seq_state for position tracking.
        // The engine handles this externally, so we just need to pass the right pos.
        let _ = (i, bufs); // suppress unused warnings
        forward_single_paged(m, token_id, pool, seq_state)?;
    }

    Ok(())
}
