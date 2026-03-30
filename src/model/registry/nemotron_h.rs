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

use tracing::debug;

use crate::gpu::{
    GpuAttention, GpuBackend, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMamba2,
    GpuMatmul, GpuMoe, GpuNorm, GpuRope, GpuAllReduce, GpuTurboQuant, TensorDtype,
};
use crate::model::config::ModelConfig;
use crate::model::forward::{Mamba2Buffers, MoeBuffers, ModelForward};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives;
use crate::model::{Model, PrefillBuffers};

// ===========================================================================
// NemotronForward — ModelForward trait implementation.
//
// Holds MoE buffers (for MoE FFN layers) and Mamba-2 buffers (for SSM layers).
// ===========================================================================

pub(crate) struct NemotronForward<B: GpuCore> {
    pub moe: MoeBuffers<B>,
    pub mamba2: Mamba2Buffers<B>,
}

impl<B: GpuBackend> ModelForward<B> for NemotronForward<B> {
    fn forward_decode(
        &self,
        m: &Model<'_, B>,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        forward_single_paged(m, token_id, pool, seq_state, &self.moe, &self.mamba2)
    }

    fn forward_prefill(
        &self,
        m: &Model<'_, B>,
        tokens: &[u32],
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
        bufs: &PrefillBuffers<B>,
    ) -> anyhow::Result<()> {
        forward_prefill_paged(m, tokens, pool, seq_state, bufs, &self.moe, &self.mamba2)
    }
}

// ---------------------------------------------------------------------------
// Layer index mapping helpers.
//
// Since Nemotron-H layers are heterogeneous, we need to map from the global
// layer index to type-specific indices (e.g., which Mamba state to use,
// which KV pool slot to use).
// ---------------------------------------------------------------------------

/// Shared dimension struct to avoid recomputing sizes on every layer.
///
/// Note: rescale_prenorm_residual is a training-time initialization flag, not
/// a runtime scaling.  The out_proj weights were already trained with the
/// 1/sqrt(2*N) scaling baked in, so no runtime adjustment is needed.
struct NemotronDims {
    hidden_size: u32,
    q_dim: u32,
    kv_dim: u32,
    d_inner: u32,
    /// Conv1d operates on [x, B, C] concatenated, not just x.
    /// conv_dim = d_inner + 2 × n_groups × state_size.
    conv_dim: u32,
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
    /// Precomputed mapping: global layer_idx → Mamba-2 state index.
    /// None for non-Mamba layers.  Avoids O(n) string scan per layer call.
    mamba2_state_map: Vec<Option<usize>>,
}

impl NemotronDims {
    fn from_config(config: &ModelConfig, world_size: usize) -> Self {
        // Precompute Mamba-2 state indices.
        let mut m2_idx = 0;
        let mamba2_state_map: Vec<Option<usize>> = config.layer_types.iter().map(|t| {
            if t == "mamba2" {
                let idx = m2_idx;
                m2_idx += 1;
                Some(idx)
            } else {
                None
            }
        }).collect();

        // TP divides heads (and therefore q_dim/kv_dim) and Mamba-2 heads.
        // hidden_size, head_dim, moe_inter stay full (EP handles MoE splitting).
        let ws = world_size;
        Self {
            hidden_size: config.hidden_size as u32,
            q_dim: (config.num_attention_heads / ws * config.head_dim) as u32,
            kv_dim: (config.num_key_value_heads / ws * config.head_dim) as u32,
            d_inner: (config.mamba2_d_inner() / ws) as u32,
            conv_dim: (config.mamba2_conv_dim() / ws) as u32,
            num_heads: (config.num_attention_heads / ws) as u32,
            num_kv_heads: (config.num_key_value_heads / ws) as u32,
            head_dim: config.head_dim as u32,
            mamba_num_heads: (config.mamba_num_heads / ws) as u32,
            mamba_head_dim: config.mamba_head_dim as u32,
            state_size: config.ssm_state_size as u32,
            n_groups: config.mamba_n_groups as u32,
            num_experts_per_tok: config.num_experts_per_tok as u32,
            moe_inter: config.moe_intermediate_size as u32,
            shared_inter: config.shared_expert_intermediate_size as u32,
            eps: config.rms_norm_eps as f32,
            mamba2_state_map,
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
    moe: &MoeBuffers<B>,
    mamba2: &Mamba2Buffers<B>,
) -> anyhow::Result<()> {
    let d = NemotronDims::from_config(&m.config, m.world_size);
    let pos = seq_state.seq_len as u32;

    // Embed the token into the residual stream.
    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);

    // Process each layer — dispatch based on layer type.
    for layer_idx in 0..m.config.num_hidden_layers {
        let layer_type = m.config.layer_types[layer_idx].as_str();
        match layer_type {
            "mamba2" => {
                mamba2_block(m, layer_idx, &d, mamba2);
            }
            "moe" => {
                moe_block(m, layer_idx, &d, moe);
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
//   2. in_proj matmul → [z, x, B, C, dt] split by offset
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
    mamba2: &Mamba2Buffers<B>,
) {
    let layer = &m.weights.layers[layer_idx];
    let dn_idx = d.mamba2_state_map[layer_idx].expect("mamba2 layer must have state index");

    // Unwrap Mamba state and scratch buffers.
    let states = &mamba2.states;
    let conv_hist = &mamba2.conv_history;
    let in_proj_buf = &mamba2.in_proj_buf;
    let conv_out = &mamba2.conv_out;
    let ssm_out = &mamba2.ssm_out;

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

    // 2. in_proj: [in_proj_dim, hidden_size] × norm_buf → in_proj_buf.
    m.backend.matmul(in_proj, &m.norm_buf, in_proj_buf, in_proj_dim, d.hidden_size);

    // 4. Conv1d + SiLU on the [x, B, C] portion of in_proj_buf.
    //
    // in_proj output layout: [z(d_inner), xBC(conv_dim), dt(num_heads)]
    // The conv1d operates on ALL of xBC (not just x) — this is how Mamba-2
    // applies causal convolution to the SSM's B/C parameters too.
    // After conv, conv_out contains [x(d_inner), B(ngs), C(ngs)] with SiLU applied.
    m.backend.mamba2_conv1d_silu(
        in_proj_buf,
        history,
        conv_w,
        conv_b,
        conv_out,
        d.conv_dim,
        m.config.mamba_conv_kernel as u32,
        d.d_inner, // input_offset: xBC starts at element d_inner in in_proj_buf
    );

    // 5. Shift conv history: append the current xBC to the FIFO.
    m.backend.conv1d_shift_history(
        history,
        in_proj_buf,
        d.conv_dim,
        m.config.mamba_conv_kernel as u32,
        d.d_inner, // input_offset: xBC starts at element d_inner
    );

    // 6. SSM step: state update + output + RMSNorm.
    //
    // After conv1d, conv_out contains [x(d_inner), B(ngs), C(ngs)].
    // dt is in in_proj_buf at offset d_inner + conv_dim (not convolved).
    // The SSM kernel reads x from conv_out[0..d_inner], B/C from
    // conv_out[d_inner..], and dt from in_proj_buf[d_inner+conv_dim..].
    let ngs = d.n_groups * d.state_size;
    m.backend.mamba2_ssm_step(
        state,
        conv_out,                  // x at offset 0
        conv_out,                  // bc_buf: B/C also in conv_out
        in_proj_buf,               // dt_buf: dt lives in in_proj_buf
        a_log,
        d_skip,
        dt_bias,
        norm_w,
        ssm_out,
        d.mamba_num_heads,
        d.mamba_head_dim,
        d.state_size,
        d.n_groups,
        d.d_inner,                 // b_offset: B starts after x in conv_out
        d.d_inner + ngs,           // c_offset: C starts after B in conv_out
        d.d_inner + d.conv_dim,    // dt_offset: dt in in_proj_buf, after z + xBC
        d.eps,
    );

    // 7. Gated grouped RMSNorm: out = rmsnorm(ssm_out × silu(z)) × weight.
    //
    // The Mamba-2 architecture applies the gate (silu(z)) BEFORE normalization,
    // then normalizes in groups of (d_inner / n_groups) elements.  This is
    // the MambaRMSNormGated with norm_before_gate=False.
    //
    // z is the first d_inner elements of in_proj_buf.
    let group_size = d.d_inner / d.n_groups;
    m.backend.mamba2_gated_rms_norm(
        ssm_out,        // y: raw SSM output
        in_proj_buf,    // z_buf: z is at offset 0
        norm_w,         // weight
        ssm_out,        // out (can reuse since kernel reads y first)
        d.d_inner,
        group_size,
        0,              // z_offset: z starts at beginning of in_proj_buf
        d.eps,
    );

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
//   2. Router: gate × norm_buf → logits, sigmoid top-k → indices + weights
//   4. For each selected expert: up_proj → relu² → down_proj
//   5. Shared expert: up_proj → relu² → down_proj (always active)
//   6. Combine expert outputs + residual
// ---------------------------------------------------------------------------

fn moe_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuMoe + GpuAllReduce>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &NemotronDims,
    moe: &MoeBuffers<B>,
) {
    let layer = &m.weights.layers[layer_idx];
    let router_gate = layer.router_gate.as_ref().expect("router_gate");
    let router_logits = &moe.router_logits;
    let routing_output = &moe.routing_output;
    let moe_up_buf = &moe.moe_up_buf;
    let _moe_gate_buf = &moe.moe_gate_buf;
    let moe_output = &moe.moe_output;
    let use_ep = moe.local_expert_count < m.config.num_experts;
    let local_expert_end = moe.local_expert_start + moe.local_expert_count;

    let num_experts = m.config.num_experts as u32;

    // 1. RMSNorm.
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

    // 2. Router: matmul → sigmoid top-k.
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
    if let Some(ref streamer) = moe.expert_streamer {
        // SSD streaming path: load selected experts from disk on demand.
        let selected: Vec<(usize, f32)> = (0..k)
            .map(|i| (routing_data[i * 2] as usize, routing_data[i * 2 + 1]))
            .collect();
        streamer.load_experts(m.backend, layer_idx, &selected);

        for (slot_idx, &(_expert_idx, weight)) in selected.iter().enumerate() {
            let slot = streamer.active_slot(slot_idx);
            m.backend.matmul(&slot.up_proj, &m.norm_buf, moe_up_buf, d.moe_inter, d.hidden_size);
            m.backend.relu_squared(moe_up_buf, moe_up_buf, d.moe_inter);
            m.backend.matmul(&slot.down_proj, moe_up_buf, &m.gate_buf, d.hidden_size, d.moe_inter);
            m.backend.scale_add(moe_output, &m.gate_buf, weight, d.hidden_size);
        }
    } else {
        // GPU-resident path: all experts loaded in memory.
        let experts = layer.experts.as_ref().expect("experts");
        // EP: skip experts not owned by this rank, remap global → local index.
        for pair_idx in 0..k {
            let expert_idx = routing_data[pair_idx * 2] as usize;
            let weight = routing_data[pair_idx * 2 + 1];

            if use_ep && (expert_idx < moe.local_expert_start || expert_idx >= local_expert_end) {
                continue;
            }
            let local_idx = expert_idx - moe.local_expert_start;
            let expert = &experts[local_idx];

            m.backend.matmul(&expert.up_proj, &m.norm_buf, moe_up_buf, d.moe_inter, d.hidden_size);
            m.backend.relu_squared(moe_up_buf, moe_up_buf, d.moe_inter);
            m.backend.matmul(&expert.down_proj, moe_up_buf, &m.gate_buf, d.hidden_size, d.moe_inter);
            m.backend.scale_add(moe_output, &m.gate_buf, weight, d.hidden_size);
        }
    }

    // EP: combine partial expert outputs across ranks before shared expert.
    if use_ep {
        m.backend.all_reduce_sum(moe_output, d.hidden_size);
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

    // 2. QKV projections.
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
    moe: &MoeBuffers<B>,
    mamba2: &Mamba2Buffers<B>,
) -> anyhow::Result<()> {
    let d = NemotronDims::from_config(&m.config, m.world_size);
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let hidden_byte_size = m.config.hidden_size * crate::gpu::TensorDtype::BF16.byte_size();

    // Pre-allocate host buffers for the token-by-token Mamba/MoE loop.
    // These are reused across all 46 sequential layers, avoiding ~94K
    // allocations during a 1000-token prefill.
    let mut host_hidden_buf = Vec::new();
    let token_buf_size = m.backend.tensor_byte_count(&m.hidden);
    let mut token_hidden_buf = vec![0u8; token_buf_size];

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];
        let layer_type = m.config.layer_types[layer_idx].as_str();

        match layer_type {
            // -----------------------------------------------------------------
            // Mamba-2 and MoE layers: token-by-token through host memory.
            //
            // These layers are inherently sequential — Mamba-2 because each
            // token's state depends on the previous, MoE because per-token
            // routing selects different experts.  We round-trip through host
            // memory: copy batch → extract one token → run single-token forward
            // → write back → repeat.
            //
            // This is the same pattern Qwen 3.5 uses for DeltaNet layers and
            // MoE FFN during prefill.
            // -----------------------------------------------------------------
            "mamba2" | "moe" => {
                let full_bytes = m.backend.tensor_byte_count(&bufs.hidden);
                // Pre-allocate host buffers outside the token loop to avoid
                // per-token allocation overhead (~46 layers × N tokens).
                host_hidden_buf.resize(full_bytes, 0u8);
                m.backend.copy_to_host(&bufs.hidden, &mut host_hidden_buf);

                for t in 0..tokens.len() {
                    let offset = t * hidden_byte_size;
                    m.backend.copy_to_tensor(
                        &m.hidden,
                        &host_hidden_buf[offset..offset + hidden_byte_size],
                    );

                    if layer_type == "mamba2" {
                        mamba2_block(m, layer_idx, &d, mamba2);
                    } else {
                        moe_block(m, layer_idx, &d, moe);
                    }

                    m.backend.copy_to_host(&m.hidden, &mut token_hidden_buf);
                    host_hidden_buf[offset..offset + hidden_byte_size]
                        .copy_from_slice(&token_hidden_buf[..hidden_byte_size]);
                }

                m.backend.copy_to_tensor(&bufs.hidden, &host_hidden_buf);
            }

            // -----------------------------------------------------------------
            // Attention layers: fully batched (GEMM + prefill attention).
            //
            // Only 6 of 52 layers are attention, but batching them is important
            // for prefill throughput — GEMM is much faster than N mat-vecs.
            // -----------------------------------------------------------------
            "attention" => {
                // Batched RMSNorm.
                m.backend.rms_norm_batch(
                    &bufs.hidden,
                    &layer.input_layernorm,
                    d.eps,
                    &bufs.norm_buf,
                    bs,
                );

                // Batched QKV projections.
                m.backend.matmul_batch(
                    &layer.q_proj, &bufs.norm_buf, &bufs.q_buf,
                    bs, d.q_dim, d.hidden_size,
                );
                m.backend.matmul_batch(
                    &layer.k_proj, &bufs.norm_buf, &bufs.k_buf,
                    bs, d.kv_dim, d.hidden_size,
                );
                m.backend.matmul_batch(
                    &layer.v_proj, &bufs.norm_buf, &bufs.v_buf,
                    bs, d.kv_dim, d.hidden_size,
                );

                // RoPE: per-token positions, done token-by-token via host round-trip.
                // (Nemotron-H uses full rotation, no partial RoPE.)
                {
                    let q_row_bytes = d.q_dim as usize * 2; // bf16
                    let k_row_bytes = d.kv_dim as usize * 2;
                    let q_bytes = m.backend.tensor_byte_count(&bufs.q_buf);
                    let k_bytes = m.backend.tensor_byte_count(&bufs.k_buf);
                    let mut host_q = vec![0u8; q_bytes];
                    let mut host_k = vec![0u8; k_bytes];
                    m.backend.copy_to_host(&bufs.q_buf, &mut host_q);
                    m.backend.copy_to_host(&bufs.k_buf, &mut host_k);

                    let q_tensor_bytes = m.backend.tensor_byte_count(&m.q_buf);
                    let k_tensor_bytes = m.backend.tensor_byte_count(&m.k_buf);
                    // Pre-allocate upload/download buffers outside the loop.
                    let mut q_upload = vec![0u8; q_tensor_bytes];
                    let mut k_upload = vec![0u8; k_tensor_bytes];

                    for t in 0..tokens.len() {
                        let token_pos = start_pos + t as u32;
                        q_upload[..q_row_bytes].copy_from_slice(
                            &host_q[t * q_row_bytes..(t + 1) * q_row_bytes],
                        );
                        k_upload[..k_row_bytes].copy_from_slice(
                            &host_k[t * k_row_bytes..(t + 1) * k_row_bytes],
                        );
                        m.backend.copy_to_tensor(&m.q_buf, &q_upload);
                        m.backend.copy_to_tensor(&m.k_buf, &k_upload);

                        m.backend.rope(
                            &m.q_buf, &m.k_buf,
                            token_pos,
                            m.config.rope_theta as f32,
                            d.num_heads, d.num_kv_heads, d.head_dim,
                        );

                        // Reuse the same buffers for readback.
                        m.backend.copy_to_host(&m.q_buf, &mut q_upload);
                        m.backend.copy_to_host(&m.k_buf, &mut k_upload);
                        host_q[t * q_row_bytes..(t + 1) * q_row_bytes]
                            .copy_from_slice(&q_upload[..q_row_bytes]);
                        host_k[t * k_row_bytes..(t + 1) * k_row_bytes]
                            .copy_from_slice(&k_upload[..k_row_bytes]);
                    }

                    m.backend.copy_to_tensor(&bufs.q_buf, &host_q);
                    m.backend.copy_to_tensor(&bufs.k_buf, &host_k);
                }

                // Paged KV cache write (batched).
                let kv_idx = m.kv_layer_map[layer_idx].unwrap();
                if let Some(tc) = &m.turbo_ctx {
                    m.backend.turbo_quantize_to_paged_batch(
                        &bufs.k_buf, &pool.k_pool[kv_idx],
                        &seq_state.block_table_gpu, &bufs.positions,
                        &tc.pi, &tc.centroids,
                        bs, d.num_kv_heads, d.head_dim,
                        tc.config.bits, tc.config.bytes_per_head_pos as u32,
                    );
                    m.backend.turbo_quantize_to_paged_batch(
                        &bufs.v_buf, &pool.v_pool[kv_idx],
                        &seq_state.block_table_gpu, &bufs.positions,
                        &tc.pi, &tc.centroids,
                        bs, d.num_kv_heads, d.head_dim,
                        tc.config.bits, tc.config.bytes_per_head_pos as u32,
                    );
                } else {
                    m.backend.copy_to_paged_kv_cache_batch(
                        &bufs.k_buf, &pool.k_pool[kv_idx],
                        &seq_state.block_table_gpu, &bufs.positions,
                        bs, d.num_kv_heads, d.head_dim,
                    );
                    m.backend.copy_to_paged_kv_cache_batch(
                        &bufs.v_buf, &pool.v_pool[kv_idx],
                        &seq_state.block_table_gpu, &bufs.positions,
                        bs, d.num_kv_heads, d.head_dim,
                    );
                }

                // Prefill attention (causal, full context — no sliding window).
                m.backend.prefill_attention(
                    &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
                    bs, start_pos,
                    d.num_heads, d.num_kv_heads, d.head_dim,
                    0, 0.0, None, true,
                );

                // O projection + all-reduce + residual (batched).
                m.backend.matmul_batch(
                    &layer.o_proj, &bufs.attn_out, &bufs.norm_buf,
                    bs, d.hidden_size, d.q_dim,
                );
                m.backend.all_reduce_sum(&bufs.norm_buf, bs * d.hidden_size);
                m.backend.add(
                    &bufs.hidden, &bufs.norm_buf, &bufs.hidden,
                    bs * d.hidden_size,
                );
            }

            _ => unreachable!("invalid layer type at {layer_idx}: {layer_type}"),
        }
    }

    // Final norm + lm_head (only last token's logits needed).
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
// Weight loading — Nemotron-H three-way hybrid layer loading.
//
// Each layer is purely one type (Mamba-2, MoE, or attention), unlike
// standard transformers where every layer has both attention + FFN.
// Moved here from loader.rs to keep architecture-specific logic co-located
// with the forward pass.
//
// Weight naming convention (different from standard models):
//   backbone.layers.{i}.norm.weight           — pre-layer RMSNorm (all types)
//   backbone.layers.{i}.mixer.in_proj.weight  — Mamba-2 input projection
//   backbone.layers.{i}.mixer.q_proj.weight   — attention Q projection
//   backbone.layers.{i}.mixer.experts.{j}.*   — MoE expert weights
//   backbone.layers.{i}.mixer.gate.weight     — MoE router
// ===========================================================================

use crate::model::loader::{
    TensorStore, LayerWeights, ExpertWeights, upload_tensor,
};

/// Convert a safetensor view to Vec<f32>, handling both bf16 and f32 storage.
fn to_f32_vec(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    match view.dtype() {
        safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
        _ => bytemuck::cast_slice::<u8, half::bf16>(view.data())
            .iter().map(|v| v.to_f32()).collect(),
    }
}

/// Load weights for one Nemotron-H layer (Mamba-2, MoE, or attention).
pub(crate) fn load_layer<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    prefix: &str,
    config: &ModelConfig,
    layer_idx: usize,
    _sharding: Option<&crate::gpu::parallel::ShardingPlan>,
    skip_experts: bool,
) -> anyhow::Result<LayerWeights<B>> {
    let hidden = config.hidden_size;

    // All Nemotron-H layers have a pre-layer RMSNorm.
    let input_layernorm = upload_tensor(store, backend, &format!("{prefix}.norm.weight"), &[hidden])?;

    // NOTE: rescale_prenorm_residual is an INITIALIZATION-time optimization
    // (GPT-2 style), not a runtime one.  The model's out_proj weights were
    // already scaled by 1/sqrt(2*num_layers) during training.  We do NOT
    // apply any additional scaling during inference.

    // Nemotron-H has only one norm per layer (no post-attention norm).
    let dummy_norm = backend.alloc_tensor(&[1], TensorDtype::BF16);

    let layer_type = config.layer_types.get(layer_idx).map(|s| s.as_str()).unwrap_or("");
    let dummy = || backend.alloc_tensor(&[1], TensorDtype::BF16);

    let mut lw = LayerWeights {
        input_layernorm,
        post_attention_layernorm: dummy_norm,
        pre_feedforward_layernorm: None,
        post_feedforward_layernorm: None,
        q_proj: dummy(), k_proj: dummy(), v_proj: dummy(), o_proj: dummy(),
        q_bias: None, k_bias: None, v_bias: None, o_proj_bias: None,
        sinks: None, q_norm: None, k_norm: None, attn_z_proj: None,
        in_proj_qkv: None, in_proj_a: None, in_proj_b: None, in_proj_z: None,
        conv1d_weight: None, linear_out_proj: None, a_log: None, dt_bias: None,
        linear_norm: None,
        gate_proj: dummy(), up_proj: dummy(), down_proj: dummy(),
        router_gate: None, router_bias: None, experts: None,
        shared_expert_gate_proj: None, shared_expert_up_proj: None,
        shared_expert_down_proj: None, shared_expert_gate: None,
        mamba_in_proj: None, mamba_conv1d_weight: None, mamba_conv1d_bias: None,
        mamba_out_proj: None, mamba_a_log: None, mamba_d: None,
        mamba_dt_bias: None, mamba_norm: None,
        e_score_correction_bias: None,
    };

    match layer_type {
        "mamba2" => {
            let d_inner = config.mamba2_d_inner();
            let conv_dim = config.mamba2_conv_dim(); // d_inner + 2*n_groups*state_size
            let in_proj_dim = config.mamba2_in_proj_dim();
            let ks = config.mamba_conv_kernel;
            let n_heads = config.mamba_num_heads;

            lw.mamba_in_proj = Some(upload_tensor(
                store, backend, &format!("{prefix}.mixer.in_proj.weight"), &[in_proj_dim, hidden],
            )?);

            // conv1d weight is stored as [conv_dim, 1, kernel_size] (PyTorch Conv1d format).
            // We reshape to [conv_dim, kernel_size] — same contiguous data, just dropping
            // the middle dimension.  upload_tensor does strict shape checks, so we load
            // the raw tensor and upload with our desired 2D shape.
            let conv_w_view = store.tensor(&format!("{prefix}.mixer.conv1d.weight"))?;
            lw.mamba_conv1d_weight = Some(backend.upload_tensor(
                conv_w_view.data(), &[conv_dim, ks], TensorDtype::BF16,
            ));
            // conv1d bias is bf16 in the checkpoint — upload as bf16.
            lw.mamba_conv1d_bias = Some(upload_tensor(
                store, backend, &format!("{prefix}.mixer.conv1d.bias"), &[conv_dim],
            )?);

            lw.mamba_out_proj = Some(upload_tensor(
                store, backend, &format!("{prefix}.mixer.out_proj.weight"), &[hidden, d_inner],
            )?);

            // A_log, D, dt_bias may be stored as bf16 (original) or f32
            // (pre-quantized models preserve precision).  The SSM kernel needs
            // f32 for numerical precision (exp/softplus on small values), so
            // convert bf16→f32 if needed, or pass f32 through directly.
            let a_log_view = store.tensor(&format!("{prefix}.mixer.A_log"))?;
            let a_log_f32 = to_f32_vec(&a_log_view);
            lw.mamba_a_log = Some(backend.upload_tensor(
                bytemuck::cast_slice(&a_log_f32), &[n_heads], TensorDtype::F32,
            ));

            let d_view = store.tensor(&format!("{prefix}.mixer.D"))?;
            let d_f32 = to_f32_vec(&d_view);
            lw.mamba_d = Some(backend.upload_tensor(
                bytemuck::cast_slice(&d_f32), &[n_heads], TensorDtype::F32,
            ));

            let dt_bias_view = store.tensor(&format!("{prefix}.mixer.dt_bias"))?;
            let dt_bias_f32 = to_f32_vec(&dt_bias_view);
            lw.mamba_dt_bias = Some(backend.upload_tensor(
                bytemuck::cast_slice(&dt_bias_f32), &[n_heads], TensorDtype::F32,
            ));

            lw.mamba_norm = Some(upload_tensor(
                store, backend, &format!("{prefix}.mixer.norm.weight"), &[d_inner],
            )?);

            if layer_idx == 0 {
                debug!(
                    d_inner = d_inner, conv_dim = conv_dim, in_proj = in_proj_dim,
                    conv_k = ks, state = config.ssm_state_size,
                    groups = config.mamba_n_groups, heads = n_heads,
                    "mamba2 layer config"
                );
            }
        }
        "moe" => {
            let n_experts = config.num_experts;
            let moe_inter = config.moe_intermediate_size;
            let shared_inter = config.shared_expert_intermediate_size;

            lw.router_gate = Some(upload_tensor(
                store, backend, &format!("{prefix}.mixer.gate.weight"), &[n_experts, hidden],
            )?);

            let e_bias_name = format!("{prefix}.mixer.gate.e_score_correction_bias");
            if let Ok(view) = store.tensor(&e_bias_name) {
                lw.e_score_correction_bias = Some(
                    backend.upload_tensor(view.data(), &[n_experts], TensorDtype::F32),
                );
            }

            if !skip_experts {
                let mut expert_vec = Vec::with_capacity(n_experts);
                for j in 0..n_experts {
                    let ep = format!("{prefix}.mixer.experts.{j}");
                    let up = upload_tensor(store, backend, &format!("{ep}.up_proj.weight"), &[moe_inter, hidden])?;
                    let down = upload_tensor(store, backend, &format!("{ep}.down_proj.weight"), &[hidden, moe_inter])?;
                    expert_vec.push(ExpertWeights {
                        gate_proj: dummy(),
                        up_proj: up,
                        down_proj: down,
                        gate_bias: None,
                        up_bias: None,
                        down_bias: None,
                    });
                }
                lw.experts = Some(expert_vec);
            }

            if shared_inter > 0 {
                lw.shared_expert_up_proj = Some(upload_tensor(
                    store, backend,
                    &format!("{prefix}.mixer.shared_experts.up_proj.weight"),
                    &[shared_inter, hidden],
                )?);
                lw.shared_expert_down_proj = Some(upload_tensor(
                    store, backend,
                    &format!("{prefix}.mixer.shared_experts.down_proj.weight"),
                    &[hidden, shared_inter],
                )?);
            }

            if layer_idx <= 1 {
                debug!(
                    experts = n_experts, inter = moe_inter, shared = shared_inter,
                    top_k = config.num_experts_per_tok,
                    scale = format_args!("{:.1}", config.routed_scaling_factor),
                    "moe layer config"
                );
            }
        }
        "attention" => {
            let q_dim = config.num_attention_heads * config.head_dim;
            let kv_dim = config.num_key_value_heads * config.head_dim;

            lw.q_proj = upload_tensor(store, backend, &format!("{prefix}.mixer.q_proj.weight"), &[q_dim, hidden])?;
            lw.k_proj = upload_tensor(store, backend, &format!("{prefix}.mixer.k_proj.weight"), &[kv_dim, hidden])?;
            lw.v_proj = upload_tensor(store, backend, &format!("{prefix}.mixer.v_proj.weight"), &[kv_dim, hidden])?;
            lw.o_proj = upload_tensor(store, backend, &format!("{prefix}.mixer.o_proj.weight"), &[hidden, q_dim])?;

            if layer_idx <= 5 {
                debug!(
                    q_dim = q_dim, kv_dim = kv_dim,
                    heads = config.num_attention_heads,
                    kv_heads = config.num_key_value_heads,
                    "attention layer config"
                );
            }
        }
        other => anyhow::bail!("unknown Nemotron-H layer type '{other}' at layer {layer_idx}"),
    }

    Ok(lw)
}
