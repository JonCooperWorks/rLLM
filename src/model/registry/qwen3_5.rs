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

use crate::gpu::GpuBackend;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::primitives;
use crate::model::{KvMode, Model, PrefillBuffers};

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
fn deltanet_attention_block<B: GpuBackend>(
    m: &Model<'_, B>,
    layer_idx: usize,
    hidden_size: u32,
) {
    let layer = &m.weights.layers[layer_idx];
    let eps = m.config.rms_norm_eps as f32;

    // DeltaNet dimensions.
    let num_qk_heads = m.config.linear_num_key_heads as u32;
    let num_v_heads = m.config.linear_num_value_heads as u32;
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
    m.backend.rms_norm(&m.hidden, &layer.input_layernorm, eps, &m.norm_buf);

    // Step 2: Fused QKV matmul.
    m.backend.matmul(in_proj_qkv, &m.norm_buf, qkv_buf, conv_dim, hidden_size);

    // Step 3: Gate projections.
    m.backend.matmul(in_proj_a, &m.norm_buf, &m.q_buf, num_v_heads, hidden_size);
    m.backend.matmul(in_proj_b, &m.norm_buf, &m.k_buf, num_v_heads, hidden_size);
    m.backend.matmul(in_proj_z, &m.norm_buf, z_buf, v_dim, hidden_size);

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
    m.backend.rms_norm_batch(attn_out, linear_norm, eps, norm_out, num_v_heads);
    m.backend.silu(z_buf, z_buf, v_dim);
    m.backend.mul(norm_out, z_buf, attn_out, v_dim);

    // Step 10: Output projection + residual add.
    m.backend.matmul(out_proj, attn_out, &m.norm_buf, hidden_size, v_dim);
    m.backend.add(&m.hidden, &m.norm_buf, &m.hidden, hidden_size);
}

// ---------------------------------------------------------------------------
// MoE FFN block with shared expert — replaces dense FFN for Qwen 3.5.
//
// MoE: router → top-k experts → weighted sum
// Shared expert: always-active FFN, gated by sigmoid(linear(hidden))
// Final: hidden += moe_output + gate * shared_expert_output
// ---------------------------------------------------------------------------

fn moe_ffn_block<B: GpuBackend>(
    m: &Model<'_, B>,
    layer_idx: usize,
    hidden_size: u32,
) {
    let layer = &m.weights.layers[layer_idx];
    let eps = m.config.rms_norm_eps as f32;
    let moe_inter = m.config.moe_intermediate_size as u32;
    let num_experts = m.config.num_experts;
    let num_experts_per_tok = m.config.num_experts_per_tok;

    let router_gate = layer.router_gate.as_ref().unwrap();
    let experts = layer.experts.as_ref().unwrap();
    let moe_gate_buf = m.moe_gate_buf.as_ref().unwrap();
    let moe_up_buf = m.moe_up_buf.as_ref().unwrap();
    let moe_output = m.moe_output.as_ref().unwrap();
    let routing_output = m.routing_output.as_ref().unwrap();

    // Step 1: RMSNorm → norm_buf.
    m.backend.rms_norm(&m.hidden, &layer.post_attention_layernorm, eps, &m.norm_buf);

    // Step 2: Router matmul → per-expert scores.
    m.backend.matmul(router_gate, &m.norm_buf, moe_gate_buf, num_experts as u32, hidden_size);

    // Step 3: GPU top-k + softmax.
    m.backend.top_k_softmax(moe_gate_buf, routing_output, num_experts as u32, num_experts_per_tok as u32);

    // Step 4: Read routing results to CPU.
    let k = num_experts_per_tok;
    let routing_bytes = k * 2 * 4;
    let buf_bytes = m.backend.tensor_byte_count(routing_output);
    let mut routing_buf = vec![0u8; buf_bytes];
    m.backend.copy_to_host(routing_output, &mut routing_buf);
    let routing_data: &[f32] = bytemuck::cast_slice(&routing_buf[..routing_bytes]);
    let selected: Vec<(usize, f32)> = (0..k)
        .map(|i| (routing_data[2 * i] as usize, routing_data[2 * i + 1]))
        .collect();

    // Step 5: Run selected experts.
    m.backend.fill_zero(moe_output, hidden_size);
    for &(expert_idx, routing_weight) in &selected {
        let expert = &experts[expert_idx];
        m.backend.matmul(&expert.gate_proj, &m.norm_buf, moe_gate_buf, moe_inter, hidden_size);
        m.backend.matmul(&expert.up_proj, &m.norm_buf, moe_up_buf, moe_inter, hidden_size);
        m.backend.silu_mul(moe_gate_buf, moe_up_buf, moe_gate_buf, moe_inter);
        m.backend.matmul(&expert.down_proj, moe_gate_buf, &m.gate_buf, hidden_size, moe_inter);
        m.backend.scale_add(moe_output, &m.gate_buf, routing_weight, hidden_size);
    }

    // Step 6: Shared expert (always-active, gated by sigmoid scalar).
    if let (Some(se_gate_proj), Some(se_up_proj), Some(se_down_proj), Some(se_gate)) = (
        layer.shared_expert_gate_proj.as_ref(),
        layer.shared_expert_up_proj.as_ref(),
        layer.shared_expert_down_proj.as_ref(),
        layer.shared_expert_gate.as_ref(),
    ) {
        let se_inter = m.config.shared_expert_intermediate_size as u32;

        // Compute scalar gate: sigmoid(se_gate @ norm_buf) → single f32.
        // Reuse up_buf (bf16) for the 1-element matmul result — up_buf isn't used yet.
        m.backend.matmul(se_gate, &m.norm_buf, &m.up_buf, 1, hidden_size);
        let mut gate_bytes = vec![0u8; m.backend.tensor_byte_count(&m.up_buf)];
        m.backend.copy_to_host(&m.up_buf, &mut gate_bytes);
        let gate_val: f32 = 1.0 / (1.0 + (-bytemuck::cast_slice::<u8, half::bf16>(&gate_bytes)[0].to_f32()).exp());

        // Shared expert SwiGLU FFN.
        m.backend.matmul(se_gate_proj, &m.norm_buf, moe_gate_buf, se_inter, hidden_size);
        m.backend.matmul(se_up_proj, &m.norm_buf, moe_up_buf, se_inter, hidden_size);
        m.backend.silu_mul(moe_gate_buf, moe_up_buf, moe_gate_buf, se_inter);
        m.backend.matmul(se_down_proj, moe_gate_buf, &m.gate_buf, hidden_size, se_inter);

        // Add gated shared expert output to MoE output.
        m.backend.scale_add(moe_output, &m.gate_buf, gate_val, hidden_size);
    }

    // Step 7: Residual add.
    m.backend.add(&m.hidden, moe_output, &m.hidden, hidden_size);
}

// ---------------------------------------------------------------------------
// FFN dispatch — MoE or dense SwiGLU depending on model config.
// ---------------------------------------------------------------------------

fn ffn_block<B: GpuBackend>(m: &Model<'_, B>, layer_idx: usize, hidden_size: u32) {
    if m.config.is_moe() {
        moe_ffn_block(m, layer_idx, hidden_size);
    } else {
        let layer = &m.weights.layers[layer_idx];
        let eps = m.config.rms_norm_eps as f32;
        let inter_size = m.config.intermediate_size as u32;
        primitives::ffn_block(
            m.backend, layer, &m.hidden, &m.norm_buf, &m.gate_buf, &m.up_buf,
            eps, hidden_size, inter_size,
        );
    }
}

// ===========================================================================
// Forward pass implementations.
// ===========================================================================

/// Single-token forward pass using an external paged KV cache.
pub(crate) fn forward_single_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let q_dim = (m.config.num_attention_heads * m.config.head_dim) as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;
    let pos = seq_state.seq_len as u32;
    let rotary_dim = m.config.rotary_dim() as u32;
    let has_output_gate = m.config.attn_output_gate;

    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        if m.config.is_linear_attention_layer(layer_idx) {
            deltanet_attention_block(m, layer_idx, hidden_size);
        } else {
            let layer = &m.weights.layers[layer_idx];

            m.backend.rms_norm(&m.hidden, &layer.input_layernorm, eps, &m.norm_buf);
            primitives::qkv_projection_qdim(
                m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
                q_dim, hidden_size, kv_dim,
            );

            if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                m.backend.rms_norm_batch(&m.q_buf, q_norm, eps, &m.q_buf, num_heads);
                m.backend.rms_norm_batch(&m.k_buf, k_norm, eps, &m.k_buf, num_kv_heads);
            }

            m.backend.rope_partial(
                &m.q_buf, &m.k_buf, pos, rope_theta,
                num_heads, num_kv_heads, head_dim, rotary_dim,
            );

            let z_buf = m.dn_z_buf.as_ref();
            if has_output_gate {
                let attn_z_proj = layer.attn_z_proj.as_ref().unwrap();
                m.backend.matmul(attn_z_proj, &m.norm_buf, z_buf.unwrap(), q_dim, hidden_size);
            }

            let kv_idx = m.kv_layer_map[layer_idx].unwrap();
            m.backend.copy_to_paged_kv_cache(
                &m.k_buf, &pool.k_pool[kv_idx], &seq_state.block_table_gpu,
                pos, num_kv_heads, head_dim,
            );
            m.backend.copy_to_paged_kv_cache(
                &m.v_buf, &pool.v_pool[kv_idx], &seq_state.block_table_gpu,
                pos, num_kv_heads, head_dim,
            );
            m.backend.paged_attention(
                &m.q_buf, &pool.k_pool[kv_idx], &pool.v_pool[kv_idx],
                &seq_state.block_table_gpu, &m.attn_out,
                pos + 1, num_heads, num_kv_heads, head_dim,
            );

            if has_output_gate {
                let z = z_buf.unwrap();
                m.backend.sigmoid_bf16(z, z, q_dim);
                m.backend.mul(&m.attn_out, z, &m.attn_out, q_dim);
            }

            primitives::o_proj_residual_qdim(
                m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden,
                hidden_size, q_dim,
            );
        }

        ffn_block(m, layer_idx, hidden_size);
    }

    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        eps, hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}

/// Single-token forward pass with flat/paged KV cache (legacy path).
pub(crate) fn forward<B: GpuBackend>(m: &mut Model<'_, B>, token_id: u32) -> anyhow::Result<()> {
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let q_dim = (m.config.num_attention_heads * m.config.head_dim) as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;
    let rotary_dim = m.config.rotary_dim() as u32;
    let has_output_gate = m.config.attn_output_gate;

    let pos = match &mut m.kv_mode {
        KvMode::Flat { pos, .. } => *pos as u32,
        KvMode::Paged { pool, seq_state } => {
            seq_state.ensure_slot(pool)?;
            seq_state.sync_block_table(m.backend);
            seq_state.seq_len as u32
        }
    };

    primitives::embed_token(m.backend, &m.weights, token_id, &m.hidden, hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        if m.config.is_linear_attention_layer(layer_idx) {
            // --- DeltaNet attention sub-block ---
            deltanet_attention_block(m, layer_idx, hidden_size);
        } else {
            // --- GQA attention sub-block ---
            let layer = &m.weights.layers[layer_idx];

            m.backend.rms_norm(&m.hidden, &layer.input_layernorm, eps, &m.norm_buf);
            primitives::qkv_projection_qdim(
                m.backend, layer, &m.norm_buf, &m.q_buf, &m.k_buf, &m.v_buf,
                q_dim, hidden_size, kv_dim,
            );

            // QK-norm before RoPE.
            if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                m.backend.rms_norm_batch(&m.q_buf, q_norm, eps, &m.q_buf, num_heads);
                m.backend.rms_norm_batch(&m.k_buf, k_norm, eps, &m.k_buf, num_kv_heads);
            }

            m.backend.rope_partial(
                &m.q_buf, &m.k_buf, pos, rope_theta,
                num_heads, num_kv_heads, head_dim, rotary_dim,
            );

            // Output gate Z projection.
            let z_buf = m.dn_z_buf.as_ref();
            if has_output_gate {
                let attn_z_proj = layer.attn_z_proj.as_ref().unwrap();
                m.backend.matmul(attn_z_proj, &m.norm_buf, z_buf.unwrap(), q_dim, hidden_size);
            }

            let kv_idx = m.kv_layer_map[layer_idx].unwrap();
            match &m.kv_mode {
                KvMode::Flat { k_cache, v_cache, .. } => {
                    m.backend.copy_to_kv_cache(
                        &m.k_buf, &k_cache[kv_idx], pos, num_kv_heads, head_dim,
                    );
                    m.backend.copy_to_kv_cache(
                        &m.v_buf, &v_cache[kv_idx], pos, num_kv_heads, head_dim,
                    );
                    m.backend.attention(
                        &m.q_buf, &k_cache[kv_idx], &v_cache[kv_idx], &m.attn_out,
                        pos + 1, num_heads, num_kv_heads, head_dim,
                    );
                }
                KvMode::Paged { pool, seq_state } => {
                    m.backend.copy_to_paged_kv_cache(
                        &m.k_buf, &pool.k_pool[kv_idx], &seq_state.block_table_gpu,
                        pos, num_kv_heads, head_dim,
                    );
                    m.backend.copy_to_paged_kv_cache(
                        &m.v_buf, &pool.v_pool[kv_idx], &seq_state.block_table_gpu,
                        pos, num_kv_heads, head_dim,
                    );
                    m.backend.paged_attention(
                        &m.q_buf, &pool.k_pool[kv_idx], &pool.v_pool[kv_idx],
                        &seq_state.block_table_gpu, &m.attn_out,
                        pos + 1, num_heads, num_kv_heads, head_dim,
                    );
                }
            }

            if has_output_gate {
                let z = z_buf.unwrap();
                m.backend.sigmoid_bf16(z, z, q_dim);
                m.backend.mul(&m.attn_out, z, &m.attn_out, q_dim);
            }

            primitives::o_proj_residual_qdim(
                m.backend, layer, &m.attn_out, &m.norm_buf, &m.hidden,
                hidden_size, q_dim,
            );
        }

        // FFN sub-block (MoE with shared expert, or dense SwiGLU).
        ffn_block(m, layer_idx, hidden_size);
    }

    primitives::final_norm_and_lm_head(
        m.backend, &m.weights, &m.hidden, &m.norm_buf, &m.logits_buf,
        eps, hidden_size, m.config.vocab_size as u32,
    );

    match &mut m.kv_mode {
        KvMode::Flat { pos, .. } => *pos += 1,
        KvMode::Paged { seq_state, .. } => seq_state.advance(),
    }
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
pub(crate) fn forward_prefill_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    let bs = tokens.len() as u32;
    let hidden_size = m.config.hidden_size as u32;
    let num_heads = m.config.num_attention_heads as u32;
    let num_kv_heads = m.config.num_key_value_heads as u32;
    let head_dim = m.config.head_dim as u32;
    let q_dim = (m.config.num_attention_heads * m.config.head_dim) as u32;
    let kv_dim = (m.config.num_key_value_heads * m.config.head_dim) as u32;
    let eps = m.config.rms_norm_eps as f32;
    let rope_theta = m.config.rope_theta as f32;
    let start_pos = seq_state.seq_len as u32;
    let rotary_dim = m.config.rotary_dim() as u32;

    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(m.backend, &m.weights, bufs, bs, hidden_size);

    for layer_idx in 0..m.config.num_hidden_layers {
        let layer = &m.weights.layers[layer_idx];

        if m.config.is_linear_attention_layer(layer_idx) {
            // --- DeltaNet layer: batch projections, sequential state update ---
            //
            // Extract each token's hidden state, run DeltaNet single-token
            // forward, and write results back.  Similar to MoE prefill in
            // qwen3_moe.rs — inherently sequential for recurrent layers.
            let hidden_byte_size = m.config.hidden_size * 2; // bf16
            let full_bytes = m.backend.tensor_byte_count(&bufs.hidden);
            let mut host_hidden = vec![0u8; full_bytes];

            // RMSNorm in batch (for the FFN later we still need per-layer norm,
            // but DeltaNet attention needs per-token processing).
            // Actually, we need to process token by token for DeltaNet attention.
            m.backend.copy_to_host(&bufs.hidden, &mut host_hidden);

            for t in 0..tokens.len() {
                let offset = t * hidden_byte_size;
                m.backend.copy_to_tensor(
                    &m.hidden,
                    &host_hidden[offset..offset + hidden_byte_size],
                );

                // Run DeltaNet attention + FFN on this single token.
                deltanet_attention_block(m, layer_idx, hidden_size);
                ffn_block(m, layer_idx, hidden_size);

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
                &bufs.hidden, &layer.input_layernorm, eps, &bufs.norm_buf, bs,
            );

            primitives::qkv_projection_batch_qdim(
                m.backend, layer, bufs, bs, q_dim, hidden_size, kv_dim,
            );

            // QK-norm (batched): treat [batch_size * num_heads, head_dim] as batch.
            if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                m.backend.rms_norm_batch(
                    &bufs.q_buf, q_norm, eps, &bufs.q_buf, bs * num_heads,
                );
                m.backend.rms_norm_batch(
                    &bufs.k_buf, k_norm, eps, &bufs.k_buf, bs * num_kv_heads,
                );
            }

            // Output gate Z projection (batched) — compute before RoPE/attention.
            if has_output_gate {
                let attn_z_proj = layer.attn_z_proj.as_ref().unwrap();
                m.backend.matmul_batch(
                    attn_z_proj, &bufs.norm_buf, &bufs.gate_buf,
                    bs, q_dim, hidden_size,
                );
            }

            // Partial RoPE (batched).
            // The existing rope_batch applies full RoPE.  We need partial RoPE.
            // For now, use the single-token partial RoPE per token in the batch.
            // TODO: add a rope_partial_batch kernel for efficiency.
            {
                let q_bytes = m.backend.tensor_byte_count(&bufs.q_buf);
                let k_bytes = m.backend.tensor_byte_count(&bufs.k_buf);
                let q_row_bytes = q_dim as usize * 2; // bf16
                let k_row_bytes = kv_dim as usize * 2;
                let mut host_q = vec![0u8; q_bytes];
                let mut host_k = vec![0u8; k_bytes];
                m.backend.copy_to_host(&bufs.q_buf, &mut host_q);
                m.backend.copy_to_host(&bufs.k_buf, &mut host_k);

                // Single-token buffers may be larger than kv_dim (shared with DeltaNet).
                // Allocate full tensor-sized host buffers for copy_to_host.
                let q_tensor_bytes = m.backend.tensor_byte_count(&m.q_buf);
                let k_tensor_bytes = m.backend.tensor_byte_count(&m.k_buf);

                for t in 0..tokens.len() {
                    let token_pos = start_pos + t as u32;
                    // Upload this token's Q/K into the (possibly oversized) single-token bufs.
                    // We zero-pad then write the actual data at the start.
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
                        &m.q_buf, &m.k_buf, token_pos, rope_theta,
                        num_heads, num_kv_heads, head_dim, rotary_dim,
                    );

                    // Copy back, extracting just the relevant portion.
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
                &bufs.positions, bs, num_kv_heads, head_dim,
            );
            m.backend.copy_to_paged_kv_cache_batch(
                &bufs.v_buf, &pool.v_pool[kv_idx], &seq_state.block_table_gpu,
                &bufs.positions, bs, num_kv_heads, head_dim,
            );
            m.backend.prefill_attention(
                &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
                bs, start_pos, num_heads, num_kv_heads, head_dim,
            );
            // GQA output gate (batched): attn_out = attn_out * sigmoid(z).
            if has_output_gate {
                let total_elems = bs * q_dim;
                m.backend.sigmoid_bf16(&bufs.gate_buf, &bufs.gate_buf, total_elems);
                m.backend.mul(&bufs.attn_out, &bufs.gate_buf, &bufs.attn_out, total_elems);
            }

            primitives::o_proj_residual_batch_qdim(
                m.backend, layer, bufs, bs, hidden_size, q_dim,
            );

            // FFN sub-block: MoE (token-by-token) or dense (batched).
            if m.config.is_moe() {
                let hidden_byte_size = m.config.hidden_size * 2; // bf16
                let full_bytes = m.backend.tensor_byte_count(&bufs.hidden);
                let mut host_hidden = vec![0u8; full_bytes];
                m.backend.copy_to_host(&bufs.hidden, &mut host_hidden);

                for t in 0..tokens.len() {
                    let offset = t * hidden_byte_size;
                    m.backend.copy_to_tensor(
                        &m.hidden,
                        &host_hidden[offset..offset + hidden_byte_size],
                    );
                    moe_ffn_block(m, layer_idx, hidden_size);
                    let mut token_hidden = vec![0u8; hidden_byte_size];
                    m.backend.copy_to_host(&m.hidden, &mut token_hidden);
                    host_hidden[offset..offset + hidden_byte_size]
                        .copy_from_slice(&token_hidden);
                }

                m.backend.copy_to_tensor(&bufs.hidden, &host_hidden);
            } else {
                let inter_size = m.config.intermediate_size as u32;
                primitives::ffn_block_batch(m.backend, layer, bufs, eps, bs, hidden_size, inter_size);
            }
        }
    }

    primitives::final_norm_and_lm_head_prefill(
        m.backend, &m.weights, bufs, &m.norm_buf, &m.logits_buf,
        eps, bs, m.config.hidden_size, m.config.vocab_size as u32,
    );

    Ok(())
}
