// ===========================================================================
// Shared transformer primitives — building blocks that model families compose.
//
// These functions implement the common operations found in most transformer
// architectures (Llama, Qwen, Gemma, etc.).  Each model family's forward
// pass calls these primitives in its own specific order with its own
// configuration (e.g., with or without QKV bias).
//
// This is the "model primitives" layer that sits between the GPU backend
// (raw kernel dispatch) and the model family forward passes (architecture-
// specific orchestration).  Similar to vLLM's model registry concept:
//   GPU backend → primitives → model families → registry dispatch
//
// Each function declares the minimal set of Gpu* sub-traits it needs,
// so callers only pay for the GPU capabilities they actually use.
// ===========================================================================

use crate::gpu::{
    GpuAllReduce, GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};
use crate::model::PrefillBuffers;
use crate::model::config::ModelConfig;
use crate::model::config::RopeScaling;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::loader::{ExpertWeights, LayerWeights, ModelWeights};

// ===========================================================================
// Dims — pre-computed dimension constants extracted from ModelConfig.
//
// LEARNING OVERVIEW
//
// Why this exists:
//   Every forward pass function starts with the same 8-10 lines extracting
//   hidden_size, num_heads, head_dim, etc. from the config and casting to u32.
//   This struct computes them once and passes them around, eliminating the
//   boilerplate without hiding what the values mean.
//
// Why u32?
//   GPU kernels use u32 for dimension parameters (Metal/CUDA thread indices
//   are unsigned).  The config stores usize for Rust idiom, but every kernel
//   call needs the cast.  Doing it once here avoids `as u32` scattered
//   throughout every forward pass.
//
// Why not just store u32 in ModelConfig?
//   ModelConfig maps 1:1 to config.json (usize is natural for Rust structs).
//   Dims is a GPU-side concern — it belongs in the forward-pass layer, not
//   the deserialization layer.
// ===========================================================================

pub(crate) struct Dims {
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub inter_size: u32,
    /// Q projection dimension (num_heads × head_dim).
    /// Usually equals hidden_size, but some models differ (e.g. Qwen3 MoE:
    /// hidden=2048, q_dim=32×128=4096).
    pub q_dim: u32,
    /// KV projection dimension (num_kv_heads × head_dim).
    pub kv_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
}

impl Dims {
    /// Extract dimensions from a model config.
    pub fn from_config(config: &ModelConfig) -> Self {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        Self {
            hidden_size: config.hidden_size as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            inter_size: config.intermediate_size as u32,
            q_dim: (num_heads * head_dim) as u32,
            kv_dim: (num_kv_heads * head_dim) as u32,
            eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta as f32,
        }
    }

    /// Extract TP-aware dimensions: heads and intermediate size are divided
    /// by `world_size`, but hidden_size and head_dim remain full.
    pub fn from_config_tp(config: &ModelConfig, world_size: usize) -> Self {
        let num_heads = config.num_attention_heads / world_size;
        let num_kv_heads = config.num_key_value_heads / world_size;
        let head_dim = config.head_dim; // NOT divided

        Self {
            hidden_size: config.hidden_size as u32, // NOT divided
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            inter_size: (config.intermediate_size / world_size) as u32,
            q_dim: (num_heads * head_dim) as u32,
            kv_dim: (num_kv_heads * head_dim) as u32,
            eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta as f32,
        }
    }
}

// ===========================================================================
// Embedding + final projection primitives.
// ===========================================================================

/// Look up token embedding and write to hidden buffer.
pub(crate) fn embed_token<B: GpuEmbed>(
    backend: &B,
    weights: &ModelWeights<B>,
    token_id: u32,
    hidden: &B::Tensor,
    hidden_size: u32,
) {
    backend.embed_lookup(&weights.embed_tokens, token_id, hidden, hidden_size);
}

/// Final RMSNorm + LM head projection → logits.
pub(crate) fn final_norm_and_lm_head<B: GpuNorm + GpuMatmul>(
    backend: &B,
    weights: &ModelWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    logits_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    vocab_size: u32,
) {
    backend.rms_norm(hidden, &weights.norm_weight, eps, norm_buf);
    let lm_head_weight = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
    backend.matmul(
        lm_head_weight,
        norm_buf,
        logits_buf,
        vocab_size,
        hidden_size,
    );
}

// ===========================================================================
// Single-token attention block primitives.
// ===========================================================================

/// QKV projection: matmul Q, K, V from the normalized hidden state.
pub(crate) fn qkv_projection<B: GpuMatmul>(
    backend: &B,
    layer: &LayerWeights<B>,
    norm_buf: &B::Tensor,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    hidden_size: u32,
    kv_dim: u32,
) {
    // Q projection: [q_dim, hidden_size] × [hidden_size] → [q_dim].
    // For most models q_dim == hidden_size, so callers pass hidden_size for both.
    // Models with q_dim ≠ hidden_size (e.g. Qwen3 MoE) should use
    // qkv_projection_qdim() instead.
    backend.matmul(&layer.q_proj, norm_buf, q_buf, hidden_size, hidden_size);
    backend.matmul(&layer.k_proj, norm_buf, k_buf, kv_dim, hidden_size);
    backend.matmul(&layer.v_proj, norm_buf, v_buf, kv_dim, hidden_size);
}

/// Like qkv_projection but with explicit q_dim for models where
/// num_heads × head_dim ≠ hidden_size (e.g. Qwen3 MoE: 4096 vs 2048).
pub(crate) fn qkv_projection_qdim<B: GpuMatmul>(
    backend: &B,
    layer: &LayerWeights<B>,
    norm_buf: &B::Tensor,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    q_dim: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul(&layer.q_proj, norm_buf, q_buf, q_dim, hidden_size);
    backend.matmul(&layer.k_proj, norm_buf, k_buf, kv_dim, hidden_size);
    backend.matmul(&layer.v_proj, norm_buf, v_buf, kv_dim, hidden_size);
}

/// Apply QKV bias (for architectures that have it, e.g. Qwen).
pub(crate) fn apply_qkv_bias<B: GpuElementwise>(
    backend: &B,
    layer: &LayerWeights<B>,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    hidden_size: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.add(q_buf, q_bias, q_buf, hidden_size);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.add(k_buf, k_bias, k_buf, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.add(v_buf, v_bias, v_buf, kv_dim);
    }
}

/// RoPE on Q and K buffers.
pub(crate) fn apply_rope<B: GpuRope>(
    backend: &B,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) {
    backend.rope(
        q_buf,
        k_buf,
        pos,
        rope_theta,
        num_heads,
        num_kv_heads,
        head_dim,
    );
}

/// Write K/V to paged cache and compute attention.
///
/// Delegates to `paged_attention_fused`, which backends can override with a
/// single fused kernel.  The default implementation calls the 3 separate
/// methods (copy K, copy V, attention) — correct everywhere, but backends
/// that fuse the operation avoid re-reading the block table and keep the
/// current token's K/V in fast memory.
///
/// `window_size`: sliding window size (0 = full context, attend to all positions).
/// `attn_scale`: custom attention scale (0.0 = default 1/sqrt(head_dim)).
/// Most models pass 0/0.0 for standard full-context attention with default scaling.
pub(crate) fn paged_kv_and_attention<B: GpuAttention>(
    backend: &B,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    q_buf: &B::Tensor,
    attn_out: &B::Tensor,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    layer_idx: usize,
    pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
    sinks: Option<&B::Tensor>,
) {
    backend.paged_attention_fused(
        q_buf,
        k_buf,
        v_buf,
        &pool.k_pool[layer_idx],
        &pool.v_pool[layer_idx],
        &seq_state.block_table_gpu,
        attn_out,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        attn_scale,
        sinks,
    );
}

/// O projection + residual add.
///
/// For models where q_dim == hidden_size, pass hidden_size for both args.
/// For models where q_dim ≠ hidden_size (Qwen3 MoE), use o_proj_residual_qdim.
pub(crate) fn o_proj_residual<B: GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, hidden_size);
    backend.all_reduce_sum(norm_buf, hidden_size); // no-op when world_size=1
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

/// O projection + residual for models where q_dim ≠ hidden_size.
/// o_proj weight is [hidden_size, q_dim], input attn_out is [q_dim].
pub(crate) fn o_proj_residual_qdim<B: GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    q_dim: u32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, q_dim);
    backend.all_reduce_sum(norm_buf, hidden_size); // no-op when world_size=1
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

// ===========================================================================
// SwiGLU FFN block.
// ===========================================================================

/// Full FFN sub-block: RMSNorm → gate/up projections → SwiGLU → down → residual.
pub(crate) fn ffn_block<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    gate_buf: &B::Tensor,
    up_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    inter_size: u32,
) {
    backend.rms_norm(hidden, &layer.post_attention_layernorm, eps, norm_buf);
    backend.matmul(
        &layer.gate_proj,
        norm_buf,
        gate_buf,
        inter_size,
        hidden_size,
    );
    backend.matmul(&layer.up_proj, norm_buf, up_buf, inter_size, hidden_size);
    backend.silu_mul(gate_buf, up_buf, gate_buf, inter_size);
    backend.matmul(
        &layer.down_proj,
        gate_buf,
        norm_buf,
        hidden_size,
        inter_size,
    );
    backend.all_reduce_sum(norm_buf, hidden_size); // no-op when world_size=1
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

// ===========================================================================
// MoE FFN block — shared by all Mixture-of-Experts model families.
//
// Replaces the dense FFN block for MoE layers.  The core algorithm:
//   1. RMSNorm on the hidden state
//   2. Router matmul → per-expert scores
//   3. GPU top-k + softmax
//   4. CPU readback of routing decisions (only k index-weight pairs)
//   5. For each selected expert: gate/up → SwiGLU → down → scale_add
//   6. Residual add
//
// Models with a shared expert (Qwen3.5) call this for the routed experts
// and handle the shared expert separately at the call site.
// ===========================================================================

/// MoE FFN block for a single token: norm → route → dispatch → accumulate → residual.
///
/// Performs the full MoE FFN sub-block including RMSNorm and residual add.
/// Models with a shared expert (Qwen3.5) should use `moe_expert_dispatch`
/// instead for finer control over the residual add.
///
/// Buffer requirements:
///   - `norm_buf`: [hidden_size] — receives RMSNorm output
///   - `moe_gate_buf`: [max(num_experts, moe_inter)] — scratch for router + expert gate
///   - `moe_up_buf`: [moe_inter] — scratch for expert up projection
///   - `moe_output`: [hidden_size] — expert output accumulator (zeroed internally)
///   - `routing_output`: [2 * num_experts_per_tok] — GPU top-k output
///   - `down_buf`: [hidden_size] — scratch for expert down projection
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block<B: GpuNorm + GpuMatmul + GpuElementwise>(
    backend: &B,
    // Weights
    post_attn_norm: &B::Tensor,
    router_gate: &B::Tensor,
    experts: &[ExpertWeights<B>],
    // Buffers
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_up_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    // Dimensions
    eps: f32,
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    // RMSNorm → norm_buf.
    backend.rms_norm(hidden, post_attn_norm, eps, norm_buf);

    // Core expert dispatch → moe_output.
    moe_expert_dispatch(
        backend,
        router_gate,
        experts,
        norm_buf,
        moe_gate_buf,
        moe_up_buf,
        moe_output,
        routing_output,
        down_buf,
        hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
    );

    // Residual add: hidden += moe_output.
    backend.add(hidden, moe_output, hidden, hidden_size);
}

/// Core MoE expert dispatch: route → select top-k → run expert FFNs → accumulate.
///
/// This is the inner loop shared by all MoE models.  It does NOT include
/// RMSNorm or the residual add — callers handle those.  This allows models
/// with a shared expert (Qwen3.5) to accumulate shared expert output into
/// `moe_output` before the residual add.
///
/// On return, `moe_output` contains the weighted sum of selected expert outputs.
/// The `norm_buf` (normalized hidden state) is consumed but not modified.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_expert_dispatch<B: GpuMatmul + GpuElementwise>(
    backend: &B,
    router_gate: &B::Tensor,
    experts: &[ExpertWeights<B>],
    // Buffers
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_up_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    // Dimensions
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    // Router matmul — compute per-expert scores.
    backend.matmul(
        router_gate,
        norm_buf,
        moe_gate_buf,
        num_experts as u32,
        hidden_size,
    );

    // GPU-side top-k + softmax.
    backend.top_k_softmax(
        moe_gate_buf,
        routing_output,
        num_experts as u32,
        num_experts_per_tok as u32,
    );

    // Read routing results to CPU.
    let k = num_experts_per_tok;
    let routing_bytes = k * 2 * 4; // 2*k f32 values (index, weight) pairs.
    let buf_bytes = backend.tensor_byte_count(routing_output);
    let mut routing_buf = vec![0u8; buf_bytes];
    backend.copy_to_host(routing_output, &mut routing_buf);
    let routing_data: &[f32] = bytemuck::cast_slice(&routing_buf[..routing_bytes]);

    let selected: Vec<(usize, f32)> = (0..k)
        .map(|i| (routing_data[2 * i] as usize, routing_data[2 * i + 1]))
        .collect();

    // Zero the accumulator, then run each selected expert's SwiGLU FFN.
    backend.fill_zero(moe_output, hidden_size);

    for &(expert_idx, routing_weight) in &selected {
        let expert = &experts[expert_idx];
        backend.matmul(
            &expert.gate_proj,
            norm_buf,
            moe_gate_buf,
            moe_inter,
            hidden_size,
        );
        backend.matmul(
            &expert.up_proj,
            norm_buf,
            moe_up_buf,
            moe_inter,
            hidden_size,
        );
        backend.silu_mul(moe_gate_buf, moe_up_buf, moe_gate_buf, moe_inter);
        backend.matmul(
            &expert.down_proj,
            moe_gate_buf,
            down_buf,
            hidden_size,
            moe_inter,
        );
        backend.scale_add(moe_output, down_buf, routing_weight, hidden_size);
    }
}

// ===========================================================================
// Batched prefill primitives.
// ===========================================================================

/// Upload token IDs and positions for batched prefill.
pub(crate) fn upload_prefill_inputs<B: GpuCore>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    tokens: &[u32],
    start_pos: u32,
    bs: u32,
) {
    let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
    backend.copy_to_tensor(&bufs.token_ids, token_bytes);

    let positions: Vec<u32> = (start_pos..start_pos + bs).collect();
    let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
    backend.copy_to_tensor(&bufs.positions, pos_bytes);
}

/// Batched embedding lookup.
pub(crate) fn embed_batch<B: GpuEmbed>(
    backend: &B,
    weights: &ModelWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
) {
    backend.embed_lookup_batch(
        &weights.embed_tokens,
        &bufs.token_ids,
        &bufs.hidden,
        bs,
        hidden_size,
    );
}

/// Batched QKV projection (GEMM).
pub(crate) fn qkv_projection_batch<B: GpuMatmul>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul_batch(
        &layer.q_proj,
        &bufs.norm_buf,
        &bufs.q_buf,
        bs,
        hidden_size,
        hidden_size,
    );
    backend.matmul_batch(
        &layer.k_proj,
        &bufs.norm_buf,
        &bufs.k_buf,
        bs,
        kv_dim,
        hidden_size,
    );
    backend.matmul_batch(
        &layer.v_proj,
        &bufs.norm_buf,
        &bufs.v_buf,
        bs,
        kv_dim,
        hidden_size,
    );
}

/// Batched QKV projection with explicit q_dim (for q_dim ≠ hidden_size).
pub(crate) fn qkv_projection_batch_qdim<B: GpuMatmul>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    q_dim: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    backend.matmul_batch(
        &layer.q_proj,
        &bufs.norm_buf,
        &bufs.q_buf,
        bs,
        q_dim,
        hidden_size,
    );
    backend.matmul_batch(
        &layer.k_proj,
        &bufs.norm_buf,
        &bufs.k_buf,
        bs,
        kv_dim,
        hidden_size,
    );
    backend.matmul_batch(
        &layer.v_proj,
        &bufs.norm_buf,
        &bufs.v_buf,
        bs,
        kv_dim,
        hidden_size,
    );
}

/// Apply QKV bias in batched mode (broadcast-add).
pub(crate) fn apply_qkv_bias_batch<B: GpuElementwise>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.bias_add_batch(&bufs.q_buf, q_bias, &bufs.q_buf, bs, hidden_size);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.bias_add_batch(&bufs.k_buf, k_bias, &bufs.k_buf, bs, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.bias_add_batch(&bufs.v_buf, v_bias, &bufs.v_buf, bs, kv_dim);
    }
}

/// Batched RoPE.
pub(crate) fn apply_rope_batch<B: GpuRope>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    rope_theta: f32,
    bs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) {
    backend.rope_batch(
        &bufs.q_buf,
        &bufs.k_buf,
        &bufs.positions,
        rope_theta,
        bs,
        num_heads,
        num_kv_heads,
        head_dim,
    );
}

/// Write K/V to paged cache (batched) and compute prefill attention.
///
/// `window_size`: sliding window size (0 = full context).
/// `attn_scale`: custom attention scale (0.0 = default 1/sqrt(head_dim)).
pub(crate) fn paged_kv_and_prefill_attention<B: GpuAttention>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    layer_idx: usize,
    bs: u32,
    start_pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
    sinks: Option<&B::Tensor>,
) {
    backend.copy_to_paged_kv_cache_batch(
        &bufs.k_buf,
        &pool.k_pool[layer_idx],
        &seq_state.block_table_gpu,
        &bufs.positions,
        bs,
        num_kv_heads,
        head_dim,
    );
    backend.copy_to_paged_kv_cache_batch(
        &bufs.v_buf,
        &pool.v_pool[layer_idx],
        &seq_state.block_table_gpu,
        &bufs.positions,
        bs,
        num_kv_heads,
        head_dim,
    );
    backend.prefill_attention(
        &bufs.q_buf,
        &bufs.k_buf,
        &bufs.v_buf,
        &bufs.attn_out,
        bs,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        attn_scale,
        sinks,
    );
}

/// Batched O projection + residual.
pub(crate) fn o_proj_residual_batch<B: GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
) {
    backend.matmul_batch(
        &layer.o_proj,
        &bufs.attn_out,
        &bufs.norm_buf,
        bs,
        hidden_size,
        hidden_size,
    );
    backend.all_reduce_sum(&bufs.norm_buf, bs * hidden_size); // no-op when world_size=1
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Batched O projection + residual with explicit q_dim.
pub(crate) fn o_proj_residual_batch_qdim<B: GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    hidden_size: u32,
    q_dim: u32,
) {
    backend.matmul_batch(
        &layer.o_proj,
        &bufs.attn_out,
        &bufs.norm_buf,
        bs,
        hidden_size,
        q_dim,
    );
    backend.all_reduce_sum(&bufs.norm_buf, bs * hidden_size); // no-op when world_size=1
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Batched FFN sub-block.
pub(crate) fn ffn_block_batch<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    eps: f32,
    bs: u32,
    hidden_size: u32,
    inter_size: u32,
) {
    backend.rms_norm_batch(
        &bufs.hidden,
        &layer.post_attention_layernorm,
        eps,
        &bufs.norm_buf,
        bs,
    );
    backend.matmul_batch(
        &layer.gate_proj,
        &bufs.norm_buf,
        &bufs.gate_buf,
        bs,
        inter_size,
        hidden_size,
    );
    backend.matmul_batch(
        &layer.up_proj,
        &bufs.norm_buf,
        &bufs.up_buf,
        bs,
        inter_size,
        hidden_size,
    );
    backend.silu_mul(
        &bufs.gate_buf,
        &bufs.up_buf,
        &bufs.gate_buf,
        bs * inter_size,
    );
    backend.matmul_batch(
        &layer.down_proj,
        &bufs.gate_buf,
        &bufs.norm_buf,
        bs,
        hidden_size,
        inter_size,
    );
    backend.all_reduce_sum(&bufs.norm_buf, bs * hidden_size); // no-op when world_size=1
    backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
}

/// Final norm + LM head for batched prefill (extracts last token's hidden state).
pub(crate) fn final_norm_and_lm_head_prefill<B: GpuCore + GpuNorm + GpuMatmul>(
    backend: &B,
    weights: &ModelWeights<B>,
    bufs: &PrefillBuffers<B>,
    norm_buf: &B::Tensor,
    logits_buf: &B::Tensor,
    eps: f32,
    bs: u32,
    hidden_size: usize,
    vocab_size: u32,
) {
    backend.rms_norm_batch(&bufs.hidden, &weights.norm_weight, eps, &bufs.norm_buf, bs);

    // Extract last token's hidden state to single-token norm_buf.
    // Hidden state tensors are always bf16 (the compute dtype for activations),
    // but we use TensorDtype::BF16.byte_size() instead of hardcoding `* 2`
    // so the assumption is explicit and grep-able, not a magic number.
    let hidden_byte_size = hidden_size * crate::gpu::TensorDtype::BF16.byte_size();
    let full_tensor_bytes = backend.tensor_byte_count(&bufs.norm_buf);
    let mut host_buf = vec![0u8; full_tensor_bytes];
    backend.copy_to_host(&bufs.norm_buf, &mut host_buf);
    let chunk_size = bs as usize;
    let last_row_start = (chunk_size - 1) * hidden_byte_size;
    backend.copy_to_tensor(
        norm_buf,
        &host_buf[last_row_start..last_row_start + hidden_byte_size],
    );

    let lm_head_weight = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
    backend.matmul(
        lm_head_weight,
        norm_buf,
        logits_buf,
        vocab_size,
        hidden_size as u32,
    );
}

// ===========================================================================
// GPT-OSS biased primitives.
//
// GPT-OSS-20B has bias on all projections (QKV, O-proj, router, expert FFN).
// These primitives extend the standard ones with bias-add steps.
// ===========================================================================

/// O projection + O-proj bias + residual add.
/// For models with q_dim ≠ hidden_size and O-proj bias (GPT-OSS).
#[allow(clippy::too_many_arguments)]
pub(crate) fn o_proj_residual_qdim_biased<B: GpuMatmul + GpuElementwise>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    q_dim: u32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, q_dim);
    if let Some(ref o_bias) = layer.o_proj_bias {
        backend.add(norm_buf, o_bias, norm_buf, hidden_size);
    }
    backend.add(hidden, norm_buf, hidden, hidden_size);
}

/// MoE expert dispatch with router bias, expert biases, and GPT-OSS gated activation.
///
/// Extends the standard `moe_expert_dispatch` with:
///   1. Router bias added after router matmul (before top-k)
///   2. Expert gate/up biases added after gate/up matmuls
///   3. GPT-OSS activation: (clamp(up,-lim,lim)+1) * clamp(gate,max=lim) * sigmoid(gate*alpha)
///   4. Expert down_bias added after down matmul
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_expert_dispatch_biased<B: GpuMatmul + GpuElementwise>(
    backend: &B,
    router_gate: &B::Tensor,
    router_bias: Option<&B::Tensor>,
    experts: &[ExpertWeights<B>],
    // Buffers
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_up_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    // Dimensions
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    swiglu_limit: f32,
) {
    // Router matmul → per-expert scores.
    backend.matmul(
        router_gate,
        norm_buf,
        moe_gate_buf,
        num_experts as u32,
        hidden_size,
    );

    // Router bias (GPT-OSS).
    if let Some(bias) = router_bias {
        backend.add(moe_gate_buf, bias, moe_gate_buf, num_experts as u32);
    }

    // GPU-side top-k + softmax.
    backend.top_k_softmax(
        moe_gate_buf,
        routing_output,
        num_experts as u32,
        num_experts_per_tok as u32,
    );

    // Read routing results to CPU.
    let k = num_experts_per_tok;
    let routing_bytes = k * 2 * 4;
    let buf_bytes = backend.tensor_byte_count(routing_output);
    let mut routing_buf = vec![0u8; buf_bytes];
    backend.copy_to_host(routing_output, &mut routing_buf);
    let routing_data: &[f32] = bytemuck::cast_slice(&routing_buf[..routing_bytes]);

    let selected: Vec<(usize, f32)> = (0..k)
        .map(|i| (routing_data[2 * i] as usize, routing_data[2 * i + 1]))
        .collect();

    // Zero the accumulator, then run each selected expert's FFN.
    backend.fill_zero(moe_output, hidden_size);

    for &(expert_idx, routing_weight) in &selected {
        let expert = &experts[expert_idx];

        // Gate and up projections.
        backend.matmul(
            &expert.gate_proj,
            norm_buf,
            moe_gate_buf,
            moe_inter,
            hidden_size,
        );
        backend.matmul(
            &expert.up_proj,
            norm_buf,
            moe_up_buf,
            moe_inter,
            hidden_size,
        );

        // Expert gate and up biases (split during loading).
        if let Some(ref g_bias) = expert.gate_bias {
            backend.add(moe_gate_buf, g_bias, moe_gate_buf, moe_inter);
        }
        if let Some(ref u_bias) = expert.up_bias {
            backend.add(moe_up_buf, u_bias, moe_up_buf, moe_inter);
        }

        // GPT-OSS gated activation: NOT standard SwiGLU.
        // Uses alpha=1.702 (fixed by architecture), asymmetric clamping, and (up+1) offset.
        if swiglu_limit > 0.0 {
            backend.gpt_oss_gated_act(
                moe_gate_buf,
                moe_up_buf,
                moe_gate_buf,
                moe_inter,
                1.702,
                swiglu_limit,
            );
        } else {
            backend.silu_mul(moe_gate_buf, moe_up_buf, moe_gate_buf, moe_inter);
        }

        // Down projection.
        backend.matmul(
            &expert.down_proj,
            moe_gate_buf,
            down_buf,
            hidden_size,
            moe_inter,
        );

        // Expert down bias.
        if let Some(ref d_bias) = expert.down_bias {
            backend.add(down_buf, d_bias, down_buf, hidden_size);
        }

        backend.scale_add(moe_output, down_buf, routing_weight, hidden_size);
    }
}

/// MoE FFN block with biased dispatch + GPT-OSS activation: norm → route → dispatch → residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block_biased<B: GpuNorm + GpuMatmul + GpuElementwise>(
    backend: &B,
    post_attn_norm: &B::Tensor,
    router_gate: &B::Tensor,
    router_bias: Option<&B::Tensor>,
    experts: &[ExpertWeights<B>],
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_up_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    swiglu_limit: f32,
) {
    backend.rms_norm(hidden, post_attn_norm, eps, norm_buf);
    moe_expert_dispatch_biased(
        backend,
        router_gate,
        router_bias,
        experts,
        norm_buf,
        moe_gate_buf,
        moe_up_buf,
        moe_output,
        routing_output,
        down_buf,
        hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
        swiglu_limit,
    );
    backend.add(hidden, moe_output, hidden, hidden_size);
}

// ===========================================================================
// YaRN RoPE primitives.
// ===========================================================================

/// Apply YaRN RoPE to Q and K buffers.
pub(crate) fn apply_rope_yarn<B: GpuRope>(
    backend: &B,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    pos: u32,
    rope_theta: f32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scaling: &RopeScaling,
) {
    backend.rope_yarn(
        q_buf,
        k_buf,
        pos,
        rope_theta,
        num_heads,
        num_kv_heads,
        head_dim,
        scaling.factor as f32,
        scaling.beta_fast as f32,
        scaling.beta_slow as f32,
        scaling.original_max_position_embeddings as u32,
    );
}

/// Batched YaRN RoPE for prefill.
pub(crate) fn apply_rope_yarn_batch<B: GpuRope>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    rope_theta: f32,
    bs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scaling: &RopeScaling,
) {
    backend.rope_yarn_batch(
        &bufs.q_buf,
        &bufs.k_buf,
        &bufs.positions,
        rope_theta,
        bs,
        num_heads,
        num_kv_heads,
        head_dim,
        scaling.factor as f32,
        scaling.beta_fast as f32,
        scaling.beta_slow as f32,
        scaling.original_max_position_embeddings as u32,
    );
}

/// Apply QKV bias for models where q_dim ≠ hidden_size (e.g. GPT-OSS: q_dim=4096, hidden=2880).
pub(crate) fn apply_qkv_bias_qdim<B: GpuElementwise>(
    backend: &B,
    layer: &LayerWeights<B>,
    q_buf: &B::Tensor,
    k_buf: &B::Tensor,
    v_buf: &B::Tensor,
    q_dim: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.add(q_buf, q_bias, q_buf, q_dim);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.add(k_buf, k_bias, k_buf, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.add(v_buf, v_bias, v_buf, kv_dim);
    }
}

/// Batched QKV bias for models where q_dim ≠ hidden_size.
pub(crate) fn apply_qkv_bias_batch_qdim<B: GpuElementwise>(
    backend: &B,
    layer: &LayerWeights<B>,
    bufs: &PrefillBuffers<B>,
    bs: u32,
    q_dim: u32,
    kv_dim: u32,
) {
    if let Some(ref q_bias) = layer.q_bias {
        backend.bias_add_batch(&bufs.q_buf, q_bias, &bufs.q_buf, bs, q_dim);
    }
    if let Some(ref k_bias) = layer.k_bias {
        backend.bias_add_batch(&bufs.k_buf, k_bias, &bufs.k_buf, bs, kv_dim);
    }
    if let Some(ref v_bias) = layer.v_bias {
        backend.bias_add_batch(&bufs.v_buf, v_bias, &bufs.v_buf, bs, kv_dim);
    }
}
