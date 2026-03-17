// ===========================================================================
// Qwen 3.5 model family — hybrid Gated DeltaNet + GQA attention.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the forward pass for Qwen 3.5 models (e.g. Qwen3.5-27B).
//   These models have a hybrid attention architecture where 75% of layers
//   use Gated DeltaNet (linear attention with a fixed-size recurrent state)
//   and 25% use standard Grouped Query Attention (GQA) with softmax.
//
// Why hybrid?
//   DeltaNet gives O(1) per-token cost and infinite context (no KV cache
//   growth), but may struggle with precise retrieval over very long
//   distances.  Standard attention excels at retrieval but is O(n) per
//   token.  The hybrid mixes both: DeltaNet for most layers (fast, memory-
//   efficient) and periodic GQA layers for retrieval "checkpoints".
//
// Why this file can't reuse llama.rs:
//   The hybrid architecture is fundamentally different from a standard dense
//   transformer.  Each layer is either DeltaNet (recurrent state + Conv1D +
//   gating) or GQA (with QK-norm + partial RoPE + output gate).  The per-
//   layer branching, DeltaNet state management, and MoE FFN make this too
//   complex to parameterise with ArchFeatures.
//
// Architecture (Qwen3.5-27B):
//   - 64 transformer layers: 48 DeltaNet + 16 GQA (every 4th layer)
//   - Hidden size: 5120
//   - DeltaNet: 16 QK-heads × 128 dim, 48 V-heads × 128 dim (3 V per QK)
//   - GQA: 24 Q-heads × 256 dim, 4 KV-heads × 256 dim
//   - Partial RoPE: only first 64 of 256 dims in GQA layers
//   - Chat template: ChatML (same as Qwen 2.5)
//
// DeltaNet layer pipeline (single token decode):
//   1. RMSNorm
//   2. Fused QKV matmul → [Q, K, V] concatenated
//   3. Alpha, Beta, Z projections (gates)
//   4. Causal depthwise Conv1D on concatenated QKV
//   5. L2 normalize Q and K (per head)
//   6. Sigmoid(alpha) → decay gate, Sigmoid(beta) → update gate
//   7. DeltaNet state update: S = g*S + k ⊗ β*(v - S^T@k)
//   8. Output: o = S^T @ q
//   9. Output gate: o = rmsnorm_no_weight(o) * silu(z)
//  10. Output projection + residual add
//
// GQA layer pipeline (single token decode):
//   1. RMSNorm
//   2. Separate Q/K/V projections
//   3. Partial RoPE (first 64 of 256 dims)
//   4. Paged KV cache write + softmax attention
//   5. O projection + residual add
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives::{self, Dims};
use crate::model::{Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// DeltaNet attention block — single token decode.
//
// This replaces the standard attention block for DeltaNet layers.
// Instead of KV cache + softmax attention, it uses a fixed-size recurrent
// state matrix that is updated with the delta rule.
//
// The Conv1D provides local positional context (since DeltaNet has no RoPE).
// The L2 norm on Q and K prevents the state matrix from growing unboundedly.
// ---------------------------------------------------------------------------

/// DeltaNet attention block index within the Vec of DeltaNet states/histories.
/// DeltaNet layers are numbered sequentially (0, 1, 2, ...) independent of
/// their position in the full layer stack.
fn dn_layer_index(config: &crate::model::config::ModelConfig, layer_idx: usize) -> usize {
    config.layer_types[..layer_idx]
        .iter()
        .filter(|t| t.as_str() == "linear_attention")
        .count()
}

/// Run the DeltaNet attention sub-block for a single token.
fn deltanet_attention_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuDeltaNet + GpuAllReduce>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &Dims,
) {
    let layer = &m.weights.layers[layer_idx];

    // DeltaNet dimensions (TP-aware: divide heads by world_size).
    let ws = m.world_size as u32;
    let num_qk_heads = m.config.linear_num_key_heads as u32 / ws;
    let num_v_heads = m.config.linear_num_value_heads as u32 / ws;
    let hd = m.config.linear_key_head_dim as u32;
    let qk_dim = num_qk_heads * hd;
    let v_dim = num_v_heads * m.config.linear_value_head_dim as u32;
    let conv_dim = qk_dim * 2 + v_dim;
    let kernel_size = m.config.linear_conv_kernel_dim as u32;

    // Unwrap DeltaNet-specific buffers and state.
    let dn_idx = dn_layer_index(&m.config, layer_idx);
    let states = m.deltanet_states.as_ref().unwrap();
    let conv_histories = m.deltanet_conv_history.as_ref().unwrap();
    let state = &states[dn_idx];
    let conv_history = &conv_histories[dn_idx];
    let qkv_buf = m.dn_qkv_buf.as_ref().unwrap();
    let alpha_buf = m.dn_alpha_buf.as_ref().unwrap();
    let beta_buf = m.dn_beta_buf.as_ref().unwrap();
    let z_buf = m.dn_z_buf.as_ref().unwrap();
    let conv_out = m.dn_conv_out.as_ref().unwrap();
    let attn_out = m.dn_attn_out.as_ref().unwrap();
    let norm_out = m.dn_norm_out.as_ref().unwrap();

    // Unwrap DeltaNet-specific weights.
    let in_proj_qkv = layer.in_proj_qkv.as_ref().unwrap();
    let in_proj_a = layer.in_proj_a.as_ref().unwrap();
    let in_proj_b = layer.in_proj_b.as_ref().unwrap();
    let in_proj_z = layer.in_proj_z.as_ref().unwrap();
    let conv1d_weight = layer.conv1d_weight.as_ref().unwrap();
    let out_proj = layer.linear_out_proj.as_ref().unwrap();
    let a_log = layer.a_log.as_ref().unwrap();
    let dt_bias = layer.dt_bias.as_ref().unwrap();
    let linear_norm = layer.linear_norm.as_ref().unwrap();

    // Step 1: RMSNorm → norm_buf.
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);

    // Step 2: Fused QKV matmul.
    m.backend.matmul(in_proj_qkv, &m.norm_buf, qkv_buf, conv_dim, d.hidden_size);

    // Step 3: Gate projections.
    m.backend.matmul(in_proj_a, &m.norm_buf, &m.q_buf, num_v_heads, d.hidden_size);
    m.backend.matmul(in_proj_b, &m.norm_buf, &m.k_buf, num_v_heads, d.hidden_size);
    m.backend.matmul(in_proj_z, &m.norm_buf, z_buf, v_dim, d.hidden_size);

    // Step 4: Causal depthwise Conv1D on concatenated QKV.
    m.backend.conv1d_depthwise_single(qkv_buf, conv_history, conv1d_weight, conv_out, conv_dim, kernel_size);

    // Update conv history: shift left and append current token's QKV.
    m.backend.conv1d_shift_history(conv_history, qkv_buf, conv_dim, kernel_size);

    // Step 5: L2 normalize Q and K (per head, in-place on conv_out).
    m.backend.l2_normalize_heads(conv_out, num_qk_heads, hd, 0);
    m.backend.l2_normalize_heads(conv_out, num_qk_heads, hd, qk_dim);

    // Step 6: Compute decay and update gates.
    m.backend.deltanet_decay_gate(&m.q_buf, dt_bias, a_log, alpha_buf, num_v_heads);
    m.backend.sigmoid(&m.k_buf, beta_buf, num_v_heads);

    // Step 7-8: DeltaNet state update + output computation.
    m.backend.deltanet_step(
        state, conv_out, conv_out, conv_out,
        alpha_buf, beta_buf, attn_out,
        num_qk_heads, num_v_heads, hd,
        0,       // Q offset
        qk_dim,  // K offset
        qk_dim * 2,  // V offset
    );

    // Step 9: Output gate — rmsnorm(attn_out, learned_norm) * silu(z).
    m.backend.rms_norm_batch(attn_out, linear_norm, d.eps, norm_out, num_v_heads);
    m.backend.silu(z_buf, z_buf, v_dim);
    m.backend.mul(norm_out, z_buf, attn_out, v_dim);

    // Step 10: Output projection + AllReduce (row-split) + residual add.
    m.backend.matmul(out_proj, attn_out, &m.norm_buf, d.hidden_size, v_dim);
    m.backend.all_reduce_sum(&m.norm_buf, d.hidden_size); // no-op when world_size=1
    m.backend.add(&m.hidden, &m.norm_buf, &m.hidden, d.hidden_size);
}

// ---------------------------------------------------------------------------
// MoE FFN block with shared expert — replaces dense FFN for Qwen 3.5.
//
// The core MoE routing (router → top-k → expert FFNs → weighted sum)
// delegates to primitives::moe_expert_dispatch.  The shared expert is
// Qwen3.5-specific: an always-active FFN gated by sigmoid(linear(hidden)).
//
// Flow: hidden += routed_experts(hidden) + gate * shared_expert(hidden)
// ---------------------------------------------------------------------------

fn moe_ffn_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise>(
    m: &Model<'_, B>,
    layer_idx: usize,
    d: &Dims,
) {
    let layer = &m.weights.layers[layer_idx];
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;
    let moe_gate_buf = m.moe_gate_buf.as_ref().unwrap();
    let moe_up_buf = m.moe_up_buf.as_ref().unwrap();
    let moe_output = m.moe_output.as_ref().unwrap();

    // Step 1: RMSNorm → norm_buf.
    m.backend.rms_norm(&m.hidden, &layer.post_attention_layernorm, d.eps, &m.norm_buf);

    // Step 2: Core MoE expert dispatch → moe_output.
    // Uses the shared primitive (same as Qwen3Moe, Mixtral, etc.).
    primitives::moe_expert_dispatch(
        m.backend,
        layer.router_gate.as_ref().unwrap(),
        layer.experts.as_ref().unwrap(),
        &m.norm_buf,
        moe_gate_buf,
        moe_up_buf,
        moe_output,
        m.routing_output.as_ref().unwrap(),
        &m.gate_buf,
        d.hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
    );

    // Step 3: Shared expert (Qwen3.5-specific) — always-active, gated by sigmoid scalar.
    // Uses the same norm_buf from step 1 (still valid, expert dispatch doesn't modify it).
    if let (Some(se_gate_proj), Some(se_up_proj), Some(se_down_proj), Some(se_gate)) = (
        layer.shared_expert_gate_proj.as_ref(),
        layer.shared_expert_up_proj.as_ref(),
        layer.shared_expert_down_proj.as_ref(),
        layer.shared_expert_gate.as_ref(),
    ) {
        let se_inter = m.config.shared_expert_intermediate_size as u32;

        // Compute scalar gate: sigmoid(se_gate @ norm_buf) → single f32.
        m.backend.matmul(se_gate, &m.norm_buf, &m.up_buf, 1, d.hidden_size);
        let mut gate_bytes = vec![0u8; m.backend.tensor_byte_count(&m.up_buf)];
        m.backend.copy_to_host(&m.up_buf, &mut gate_bytes);
        let gate_val: f32 = 1.0 / (1.0 + (-bytemuck::cast_slice::<u8, half::bf16>(&gate_bytes)[0].to_f32()).exp());

        // Shared expert SwiGLU FFN.
        m.backend.matmul(se_gate_proj, &m.norm_buf, moe_gate_buf, se_inter, d.hidden_size);
        m.backend.matmul(se_up_proj, &m.norm_buf, moe_up_buf, se_inter, d.hidden_size);
        m.backend.silu_mul(moe_gate_buf, moe_up_buf, moe_gate_buf, se_inter);
        m.backend.matmul(se_down_proj, moe_gate_buf, &m.gate_buf, d.hidden_size, se_inter);

        // Add gated shared expert output to MoE output.
        m.backend.scale_add(moe_output, &m.gate_buf, gate_val, d.hidden_size);
    }

    // Step 4: Residual add: hidden += moe_output.
    m.backend.add(&m.hidden, moe_output, &m.hidden, d.hidden_size);
}

// ---------------------------------------------------------------------------
// FFN dispatch — MoE or dense SwiGLU depending on model config.
//
// The Qwen 3.5 family has two variants:
//   - MoE models (e.g. 35B-A3B): router → top-k experts + shared expert
//   - Dense models (e.g. 27B): standard SwiGLU FFN (gate/up/down projections)
//
// Both share the same hybrid DeltaNet+GQA attention architecture, so this
// single forward pass file handles both.  The only difference is the FFN:
// MoE models have router_gate/expert weights, dense models have the usual
// gate_proj/up_proj/down_proj per layer.
// ---------------------------------------------------------------------------

fn ffn_block<B: GpuCore + GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(m: &Model<'_, B>, layer_idx: usize, d: &Dims) {
    if m.config.is_moe() {
        moe_ffn_block(m, layer_idx, d);
    } else {
        let layer = &m.weights.layers[layer_idx];
        primitives::ffn_block(
            m.backend, layer, &m.hidden, &m.norm_buf, &m.gate_buf, &m.up_buf,
            d.eps, d.hidden_size, d.inter_size,
        );
    }
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
pub(crate) fn forward_single_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed + GpuDeltaNet + GpuAllReduce>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let pos = seq_state.seq_len as u32;
    let rotary_dim = m.config.rotary_dim() as u32;
    let has_output_gate = m.config.attn_output_gate;

    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, d.hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        if m.config.is_linear_attention_layer(layer_idx) {
            deltanet_attention_block(m, layer_idx, &d);
        } else {
            let layer = &m.weights.layers[layer_idx];

            m.backend.rms_norm(&m.hidden, &layer.input_layernorm, d.eps, &m.norm_buf);
            primitives::qkv_projection_qdim(
                m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
                d.q_dim, d.hidden_size, d.kv_dim,
            );

            if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                m.backend.rms_norm_batch(&m.q_buf, q_norm, d.eps, &m.q_buf, d.num_heads);
                m.backend.rms_norm_batch(&m.k_buf, k_norm, d.eps, &m.k_buf, d.num_kv_heads);
            }

            m.backend.rope_partial(
                &m.q_buf, &m.k_buf, pos, d.rope_theta,
                d.num_heads, d.num_kv_heads, d.head_dim, rotary_dim,
            );

            let z_buf = m.dn_z_buf.as_ref();
            if has_output_gate {
                let attn_z_proj = layer.attn_z_proj.as_ref().unwrap();
                m.backend.matmul(attn_z_proj, &m.norm_buf, z_buf.unwrap(), d.q_dim, d.hidden_size);
            }

            let kv_idx = m.kv_layer_map[layer_idx].unwrap();
            m.backend.copy_to_paged_kv_cache(
                &m.k_buf, &pool.k_pool[kv_idx], &seq_state.block_table_gpu,
                pos, d.num_kv_heads, d.head_dim,
            );
            m.backend.copy_to_paged_kv_cache(
                &m.v_buf, &pool.v_pool[kv_idx], &seq_state.block_table_gpu,
                pos, d.num_kv_heads, d.head_dim,
            );
            m.backend.paged_attention(
                &m.q_buf, &pool.k_pool[kv_idx], &pool.v_pool[kv_idx],
                &seq_state.block_table_gpu, &m.attn_out,
                pos + 1, d.num_heads, d.num_kv_heads, d.head_dim,
                0, 0.0, None,
            );

            if has_output_gate {
                let z = z_buf.unwrap();
                m.backend.sigmoid_bf16(z, z, d.q_dim);
                m.backend.mul(&m.attn_out, z, &m.attn_out, d.q_dim);
            }

            primitives::o_proj_residual_qdim(
                m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden,
                d.hidden_size, d.q_dim,
            );
        }

        ffn_block(m, layer_idx, &d);
    }

    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        d.eps, d.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}

// ===========================================================================
// Batched prefill.
//
// Learning note: DeltaNet layers must process tokens sequentially because
// the recurrent state depends on the previous token's state.  GQA layers
// can use full batched prefill (GEMM-based attention).
//
// Strategy:
//   - GQA layers: batched projections (GEMM) + prefill attention (normal)
//   - DeltaNet layers: batch the QKV/gate projections (GEMM), then process
//     the state update token-by-token (like MoE in qwen3_moe.rs)
//
// This means DeltaNet prefill is O(seq_len) per DeltaNet layer (inherent
// to recurrent models), while GQA prefill is O(1) matmuls (batched).
// ===========================================================================

/// Batched prefill for Qwen 3.5 hybrid model.
pub(crate) fn forward_prefill_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed + GpuDeltaNet + GpuAllReduce>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let d = m.dims();
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;
    let rotary_dim = m.config.rotary_dim() as u32;

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, d.hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        if m.config.is_linear_attention_layer(layer_idx) {
            // --- DeltaNet layer: batch projections, sequential state update ---
            //
            // Extract each token's hidden state, run DeltaNet single-token
            // forward, and write results back.  Similar to MoE prefill in
            // qwen3_moe.rs — inherently sequential for recurrent layers.
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

                // Run DeltaNet attention + FFN on this single token.
                deltanet_attention_block(m, layer_idx, &d);
                ffn_block(m, layer_idx, &d);

                // Copy updated hidden back.
                let mut token_hidden = vec![0u8; hidden_byte_size];
                m.backend.copy_to_host(&m.hidden, &mut token_hidden);
                host_hidden[offset..offset + hidden_byte_size]
                    .copy_from_slice(&token_hidden);
            }

            // Upload modified hidden states back to batch buffer.
            m.backend.copy_to_tensor(&bufs.hidden, &host_hidden);

        } else {
            // --- GQA layer: fully batched (GEMM + prefill attention) ---
            let has_output_gate = m.config.attn_output_gate;

            m.backend.rms_norm_batch(
                &bufs.hidden, &layer.input_layernorm, d.eps, &bufs.norm_buf, bs,
            );

            primitives::qkv_projection_batch_qdim(
                m.backend, layer, bufs, bs, d.q_dim, d.hidden_size, d.kv_dim,
            );

            // QK-norm (batched): treat [batch_size * num_heads, head_dim] as batch.
            if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                m.backend.rms_norm_batch(
                    &bufs.q_buf, q_norm, d.eps, &bufs.q_buf, bs * d.num_heads,
                );
                m.backend.rms_norm_batch(
                    &bufs.k_buf, k_norm, d.eps, &bufs.k_buf, bs * d.num_kv_heads,
                );
            }

            // Output gate Z projection (batched) — compute before RoPE/attention.
            if has_output_gate {
                let attn_z_proj = layer.attn_z_proj.as_ref().unwrap();
                m.backend.matmul_batch(
                    attn_z_proj, &bufs.norm_buf, &bufs.gate_buf,
                    bs, d.q_dim, d.hidden_size,
                );
            }

            // Partial RoPE (batched).
            // TODO: add a rope_partial_batch kernel for efficiency.
            {
                let q_bytes = m.backend.tensor_byte_count(&bufs.q_buf);
                let k_bytes = m.backend.tensor_byte_count(&bufs.k_buf);
                let q_row_bytes = d.q_dim as usize * 2; // bf16
                let k_row_bytes = d.kv_dim as usize * 2;
                let mut host_q = vec![0u8; q_bytes];
                let mut host_k = vec![0u8; k_bytes];
                m.backend.copy_to_host(&bufs.q_buf, &mut host_q);
                m.backend.copy_to_host(&bufs.k_buf, &mut host_k);

                let q_tensor_bytes = m.backend.tensor_byte_count(&m.q_buf);
                let k_tensor_bytes = m.backend.tensor_byte_count(&m.k_buf);

                for t in 0..tokens.len() {
                    let token_pos = start_pos + t as u32;
                    let mut q_upload = vec![0u8; q_tensor_bytes];
                    let mut k_upload = vec![0u8; k_tensor_bytes];
                    q_upload[..q_row_bytes].copy_from_slice(
                        &host_q[t * q_row_bytes..(t + 1) * q_row_bytes],
                    );
                    k_upload[..k_row_bytes].copy_from_slice(
                        &host_k[t * k_row_bytes..(t + 1) * k_row_bytes],
                    );
                    m.backend.copy_to_tensor(&m.q_buf, &q_upload);
                    m.backend.copy_to_tensor(&m.k_buf, &k_upload);

                    m.backend.rope_partial(
                        &m.q_buf, &m.k_buf, token_pos, d.rope_theta,
                        d.num_heads, d.num_kv_heads, d.head_dim, rotary_dim,
                    );

                    let mut q_out = vec![0u8; q_tensor_bytes];
                    let mut k_out = vec![0u8; k_tensor_bytes];
                    m.backend.copy_to_host(&m.q_buf, &mut q_out);
                    m.backend.copy_to_host(&m.k_buf, &mut k_out);
                    host_q[t * q_row_bytes..(t + 1) * q_row_bytes]
                        .copy_from_slice(&q_out[..q_row_bytes]);
                    host_k[t * k_row_bytes..(t + 1) * k_row_bytes]
                        .copy_from_slice(&k_out[..k_row_bytes]);
                }

                m.backend.copy_to_tensor(&bufs.q_buf, &host_q);
                m.backend.copy_to_tensor(&bufs.k_buf, &host_k);
            }
            let kv_idx = m.kv_layer_map[layer_idx].unwrap();
            m.backend.copy_to_paged_kv_cache_batch(
                &bufs.k_buf, &pool.k_pool[kv_idx], &seq_state.block_table_gpu,
                &bufs.positions, bs, d.num_kv_heads, d.head_dim,
            );
            m.backend.copy_to_paged_kv_cache_batch(
                &bufs.v_buf, &pool.v_pool[kv_idx], &seq_state.block_table_gpu,
                &bufs.positions, bs, d.num_kv_heads, d.head_dim,
            );
            m.backend.prefill_attention(
                &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
                bs, start_pos, d.num_heads, d.num_kv_heads, d.head_dim,
                0, 0.0, None,
            );
            // GQA output gate (batched): attn_out = attn_out * sigmoid(z).
            if has_output_gate {
                let total_elems = bs * d.q_dim;
                m.backend.sigmoid_bf16(&bufs.gate_buf, &bufs.gate_buf, total_elems);
                m.backend.mul(&bufs.attn_out, &bufs.gate_buf, &bufs.attn_out, total_elems);
            }

            primitives::o_proj_residual_batch_qdim(
                m.backend, layer, bufs, bs, d.hidden_size, d.q_dim,
            );

            // FFN sub-block: MoE requires token-by-token routing (each token
            // picks different experts), so we round-trip through host memory.
            // Dense models can use batched GEMM for the entire prefill chunk.
            if m.config.is_moe() {
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
                    moe_ffn_block(m, layer_idx, &d);
                    let mut token_hidden = vec![0u8; hidden_byte_size];
                    m.backend.copy_to_host(&m.hidden, &mut token_hidden);
                    host_hidden[offset..offset + hidden_byte_size]
                        .copy_from_slice(&token_hidden);
                }

                m.backend.copy_to_tensor(&bufs.hidden, &host_hidden);
            } else {
                primitives::ffn_block_batch(
                    m.backend, layer, bufs, d.eps, bs, d.hidden_size, d.inter_size,
                );
            }
        }
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend, &m.weights, bufs, &m.norm_buf, &m.logits_buf,
        d.eps, bs, m.config.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}
