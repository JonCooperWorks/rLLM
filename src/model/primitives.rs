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
    GpuAllReduce, GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuMoe, GpuNorm,
    GpuRope, GpuTurboQuant,
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
#[allow(dead_code)] // building block; registry models use apply_qkv_bias_qdim variant
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

/// Write K/V to paged cache and compute attention, with optional TurboQuant.
///
/// When `turbo` is Some, uses TurboQuant quantized KV cache path: rotates Q,
/// quantizes K/V into packed codes, and runs quantized paged attention with
/// inline dequantization.  When None, falls through to BF16 paged attention.
///
/// This is a drop-in replacement for `paged_kv_and_attention` that all model
/// families should use — it transparently handles the quantization mode.
pub(crate) fn paged_kv_and_attention_maybe_quantized<B: GpuAttention + GpuTurboQuant>(
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
    turbo: Option<&crate::model::turboquant::TurboContext<B>>,
) {
    if let Some(tc) = turbo {
        backend.turbo_paged_attention_fused(
            q_buf,
            k_buf,
            v_buf,
            &pool.k_pool[layer_idx],
            &pool.v_pool[layer_idx],
            &seq_state.block_table_gpu,
            &tc.pi,
            &tc.pi_t,
            &tc.centroids,
            &tc.q_rot_buf,
            attn_out,
            pos,
            num_heads,
            num_kv_heads,
            head_dim,
            tc.config.bits,
            tc.config.bytes_per_head_pos as u32,
            window_size,
            attn_scale,
            sinks,
        );
    } else {
        paged_kv_and_attention(
            backend, k_buf, v_buf, q_buf, attn_out, pool, seq_state,
            layer_idx, pos, num_heads, num_kv_heads, head_dim,
            window_size, attn_scale, sinks,
        );
    }
}

/// O projection + residual add.
///
/// For models where q_dim == hidden_size, pass hidden_size for both args.
/// For models where q_dim ≠ hidden_size (Qwen3 MoE), use o_proj_residual_qdim.
#[allow(dead_code)] // building block; registry models use _qdim or _batch variants
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
#[allow(dead_code)] // building block; registry models use batched variant
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

/// O projection + fused residual-add + RMSNorm for the post-attention norm.
///
/// Combines `o_proj_residual` and the beginning of `ffn_block` into one call:
///   matmul(o_proj) → allreduce → fused(hidden += norm_buf; rms_norm → norm_buf)
///
/// This saves one full read of the hidden tensor per layer by fusing the
/// residual add with the post-attention RMSNorm.
///
/// Inspired by rvLLM (Andy Norris / m0at): fusing residual + norm eliminates
/// redundant memory traffic.  See: https://github.com/m0at/rvllm
/// O projection + fused residual-add + RMSNorm for the post-attention norm.
///
/// Combines `o_proj_residual` and the FFN block's initial rms_norm into one call:
///   matmul(o_proj) → allreduce → fused(hidden += proj; rms_norm → norm_buf)
///
/// After this call, `hidden` contains the updated residual state and `norm_buf`
/// contains the RMSNorm'd hidden state ready for the FFN.
///
/// Inspired by rvLLM (Andy Norris / m0at): fusing residual + norm eliminates
/// redundant memory traffic.  See: https://github.com/m0at/rvllm
pub(crate) fn o_proj_fused_residual_norm_qdim<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    q_dim: u32,
    eps: f32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, q_dim);
    backend.all_reduce_sum(norm_buf, hidden_size);
    backend.fused_residual_rms_norm(
        hidden, norm_buf, &layer.post_attention_layernorm, norm_buf, hidden_size, eps,
    );
}

// ===========================================================================
// SwiGLU FFN block.
// ===========================================================================

/// Full FFN sub-block: RMSNorm → gate/up projections → SwiGLU → down → residual.
///
/// Uses the fused gate+up+SwiGLU kernel (same as MoE experts) to combine
/// three separate operations (gate matmul, up matmul, silu_mul) into one
/// dispatch.  This reads the input vector once instead of twice and saves
/// two kernel launch overheads per layer per token (~64 fewer dispatches
/// across a 32-layer model).
#[allow(dead_code)] // building block; decode path now uses ffn_block_pre_normed
pub(crate) fn ffn_block<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce + GpuMoe>(
    backend: &B,
    layer: &LayerWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    gate_buf: &B::Tensor,
    _up_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    inter_size: u32,
) {
    backend.rms_norm(hidden, &layer.post_attention_layernorm, eps, norm_buf);
    // Fused gate+up+SwiGLU: one kernel reads norm_buf once, computes both
    // gate and up projections, applies SiLU activation, and multiplies.
    // Replaces: matmul(gate) + matmul(up) + silu_mul → 3 dispatches → 1.
    backend.fused_gate_up_swiglu(
        &layer.gate_proj,
        &layer.up_proj,
        norm_buf,
        gate_buf,
        inter_size,
        hidden_size,
    );
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

/// O projection + fused residual-add + RMSNorm (q_dim == hidden_size variant).
///
/// Same as `o_proj_fused_residual_norm_qdim` but for models where the attention
/// dimension equals hidden_size (e.g. Mixtral).  Inspired by rvLLM (m0at).
pub(crate) fn o_proj_fused_residual_norm<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    eps: f32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, hidden_size);
    backend.all_reduce_sum(norm_buf, hidden_size);
    backend.fused_residual_rms_norm(
        hidden, norm_buf, &layer.post_attention_layernorm, norm_buf, hidden_size, eps,
    );
}

/// O projection with bias + fused residual-add + RMSNorm for the post-attention norm.
///
/// For models with O-projection bias and q_dim ≠ hidden_size (e.g. GPT-OSS).
/// Inspired by rvLLM (m0at).
pub(crate) fn o_proj_fused_residual_norm_qdim_biased<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce>(
    backend: &B,
    layer: &LayerWeights<B>,
    attn_out: &B::Tensor,
    norm_buf: &B::Tensor,
    hidden: &B::Tensor,
    hidden_size: u32,
    q_dim: u32,
    eps: f32,
) {
    backend.matmul(&layer.o_proj, attn_out, norm_buf, hidden_size, q_dim);
    if let Some(ref o_bias) = layer.o_proj_bias {
        backend.add(norm_buf, o_bias, norm_buf, hidden_size);
    }
    backend.all_reduce_sum(norm_buf, hidden_size);
    backend.fused_residual_rms_norm(
        hidden, norm_buf, &layer.post_attention_layernorm, norm_buf, hidden_size, eps,
    );
}

/// FFN sub-block that takes a pre-normed input (norm already done by fused residual+norm).
///
/// Used when the preceding o_proj_fused_residual_norm already produced the
/// post-attention normalized state in norm_buf.  Skips the initial rms_norm.
pub(crate) fn ffn_block_pre_normed<B: GpuNorm + GpuMatmul + GpuElementwise + GpuAllReduce + GpuMoe>(
    backend: &B,
    layer: &LayerWeights<B>,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    gate_buf: &B::Tensor,
    _up_buf: &B::Tensor,
    hidden_size: u32,
    inter_size: u32,
) {
    // norm_buf already contains the RMSNorm'd hidden state.
    backend.fused_gate_up_swiglu(
        &layer.gate_proj,
        &layer.up_proj,
        norm_buf,
        gate_buf,
        inter_size,
        hidden_size,
    );
    backend.matmul(
        &layer.down_proj,
        gate_buf,
        norm_buf,
        hidden_size,
        inter_size,
    );
    backend.all_reduce_sum(norm_buf, hidden_size);
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
pub(crate) fn moe_ffn_block<B: GpuNorm + GpuMatmul + GpuElementwise + GpuMoe>(
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
///
/// Uses the fused gate+up+SwiGLU kernel (GpuMoe) to halve per-expert
/// dispatches: 2 kernels per expert (fused_gate_up_swiglu + down matmul)
/// instead of 4 (gate matmul + up matmul + silu_mul + down matmul).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_expert_dispatch<B: GpuMatmul + GpuElementwise + GpuMoe>(
    backend: &B,
    router_gate: &B::Tensor,
    experts: &[ExpertWeights<B>],
    // Buffers
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    _moe_up_buf: &B::Tensor,
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
    // Fused gate+up+SwiGLU: 1 dispatch instead of 3 (gate + up + silu_mul).
    backend.fill_zero(moe_output, hidden_size);

    for &(expert_idx, routing_weight) in &selected {
        let expert = &experts[expert_idx];

        // Fused: out = silu(gate_proj @ norm_buf) * (up_proj @ norm_buf).
        backend.fused_gate_up_swiglu(
            &expert.gate_proj,
            &expert.up_proj,
            norm_buf,
            moe_gate_buf,
            moe_inter,
            hidden_size,
        );

        // Down projection (unchanged).
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
// SSD-streamed MoE expert dispatch.
//
// Same algorithm as moe_expert_dispatch, but loads expert weights from disk
// on-demand instead of indexing into a resident Vec<ExpertWeights>.
// See expert_stream.rs for the streaming infrastructure.
// ===========================================================================

/// MoE expert dispatch with SSD streaming — loads experts from disk on demand.
///
/// Functionally identical to `moe_expert_dispatch` but uses an ExpertStreamer
/// to pread() selected expert weights from safetensors files instead of
/// keeping all experts in GPU memory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_expert_dispatch_streamed<B: GpuMatmul + GpuElementwise + GpuMoe>(
    backend: &B,
    streamer: &crate::model::expert_stream::ExpertStreamer<B>,
    layer_idx: usize,
    router_gate: &B::Tensor,
    // Buffers
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    // Dimensions
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    // Router matmul — compute per-expert scores (same as non-streamed).
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
    let routing_bytes = k * 2 * 4;
    let buf_bytes = backend.tensor_byte_count(routing_output);
    let mut routing_buf = vec![0u8; buf_bytes];
    backend.copy_to_host(routing_output, &mut routing_buf);
    let routing_data: &[f32] = bytemuck::cast_slice(&routing_buf[..routing_bytes]);

    let selected: Vec<(usize, f32)> = (0..k)
        .map(|i| (routing_data[2 * i] as usize, routing_data[2 * i + 1]))
        .collect();

    // Load selected experts from SSD into GPU buffer slots.
    streamer.load_experts(backend, layer_idx, &selected);

    // Run expert FFNs using the streamer's buffer slots.
    backend.fill_zero(moe_output, hidden_size);

    for (slot_idx, &(_expert_idx, routing_weight)) in selected.iter().enumerate() {
        let slot = streamer.active_slot(slot_idx);

        // Fused gate+up+SwiGLU (same as non-streamed path).
        backend.fused_gate_up_swiglu(
            &slot.gate_proj,
            &slot.up_proj,
            norm_buf,
            moe_gate_buf,
            moe_inter,
            hidden_size,
        );

        // Down projection.
        backend.matmul(
            &slot.down_proj,
            moe_gate_buf,
            down_buf,
            hidden_size,
            moe_inter,
        );
        backend.scale_add(moe_output, down_buf, routing_weight, hidden_size);
    }
}

/// MoE FFN block with streaming: norm → stream-dispatch → residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block_streamed<B: GpuNorm + GpuMatmul + GpuElementwise + GpuMoe>(
    backend: &B,
    streamer: &crate::model::expert_stream::ExpertStreamer<B>,
    layer_idx: usize,
    post_attn_norm: &B::Tensor,
    router_gate: &B::Tensor,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    eps: f32,
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    backend.rms_norm(hidden, post_attn_norm, eps, norm_buf);

    moe_expert_dispatch_streamed(
        backend,
        streamer,
        layer_idx,
        router_gate,
        norm_buf,
        moe_gate_buf,
        moe_output,
        routing_output,
        down_buf,
        hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
    );

    backend.add(hidden, moe_output, hidden, hidden_size);
}

/// MoE FFN block with pre-normed input (norm already done by fused residual+norm).
///
/// Same as `moe_ffn_block` but skips the initial RMSNorm since the fused
/// residual+norm kernel already produced the normalized state in `norm_buf`.
/// Inspired by rvLLM (m0at).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block_pre_normed<B: GpuNorm + GpuMatmul + GpuElementwise + GpuMoe>(
    backend: &B,
    router_gate: &B::Tensor,
    experts: &[ExpertWeights<B>],
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_up_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    // norm_buf already contains RMSNorm'd hidden from fused residual+norm kernel.
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

    backend.add(hidden, moe_output, hidden, hidden_size);
}

/// Streamed MoE FFN block with pre-normed input (norm already done by fused residual+norm).
///
/// Same as `moe_ffn_block_streamed` but skips the initial RMSNorm.
/// Inspired by rvLLM (m0at).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block_streamed_pre_normed<B: GpuNorm + GpuMatmul + GpuElementwise + GpuMoe>(
    backend: &B,
    streamer: &crate::model::expert_stream::ExpertStreamer<B>,
    layer_idx: usize,
    router_gate: &B::Tensor,
    hidden: &B::Tensor,
    norm_buf: &B::Tensor,
    moe_gate_buf: &B::Tensor,
    moe_output: &B::Tensor,
    routing_output: &B::Tensor,
    down_buf: &B::Tensor,
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
) {
    // norm_buf already contains RMSNorm'd hidden from fused residual+norm kernel.
    moe_expert_dispatch_streamed(
        backend,
        streamer,
        layer_idx,
        router_gate,
        norm_buf,
        moe_gate_buf,
        moe_output,
        routing_output,
        down_buf,
        hidden_size,
        moe_inter,
        num_experts,
        num_experts_per_tok,
    );

    backend.add(hidden, moe_output, hidden, hidden_size);
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
#[allow(dead_code)] // building block; registry models use apply_qkv_bias_batch_qdim variant
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
#[allow(dead_code)] // building block; callers use _maybe_quantized variant
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
        true, // causal — LLM text generation
    );
}

/// Write K/V to paged cache (batched) and compute prefill attention, with
/// optional TurboQuant.
///
/// When TurboQuant is enabled, K/V are rotated and quantized into the paged
/// pool (which is allocated for quantized format).  Prefill attention still
/// uses the BF16 Q/K/V directly (full precision for intra-chunk attention).
///
/// Without this function, the prefill path would write BF16 data into a
/// quantized-sized pool, corrupting the data layout and producing garbage
/// during subsequent decode steps.
pub(crate) fn paged_kv_and_prefill_attention_maybe_quantized<
    B: GpuAttention + GpuTurboQuant,
>(
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
    turbo: Option<&crate::model::turboquant::TurboContext<B>>,
) {
    if let Some(tc) = turbo {
        // TurboQuant: rotate + quantize K/V into paged pool.
        backend.turbo_quantize_to_paged_batch(
            &bufs.k_buf,
            &pool.k_pool[layer_idx],
            &seq_state.block_table_gpu,
            &bufs.positions,
            &tc.pi,
            &tc.centroids,
            bs,
            num_kv_heads,
            head_dim,
            tc.config.bits,
            tc.config.bytes_per_head_pos as u32,
        );
        backend.turbo_quantize_to_paged_batch(
            &bufs.v_buf,
            &pool.v_pool[layer_idx],
            &seq_state.block_table_gpu,
            &bufs.positions,
            &tc.pi,
            &tc.centroids,
            bs,
            num_kv_heads,
            head_dim,
            tc.config.bits,
            tc.config.bytes_per_head_pos as u32,
        );
    } else {
        // BF16: direct copy to paged pool.
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
    }
    // Prefill attention uses BF16 Q/K/V directly — no need for quantized
    // attention during prefill since we have all tokens in memory.
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
        true, // causal — LLM text generation
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
#[allow(dead_code)] // building block for GPT-OSS; decode path uses batched variant
pub(crate) fn o_proj_residual_qdim_biased<B: GpuMatmul + GpuElementwise + GpuAllReduce>(
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
    backend.all_reduce_sum(norm_buf, hidden_size); // no-op when world_size=1
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

/// Biased MoE FFN block with pre-normed input (norm already done by fused residual+norm).
///
/// Same as `moe_ffn_block_biased` but skips the initial RMSNorm.
/// For models with router bias and SwiGLU limit (GPT-OSS).
/// Inspired by rvLLM (m0at).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_block_biased_pre_normed<B: GpuNorm + GpuMatmul + GpuElementwise>(
    backend: &B,
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
    hidden_size: u32,
    moe_inter: u32,
    num_experts: usize,
    num_experts_per_tok: usize,
    swiglu_limit: f32,
) {
    // norm_buf already contains RMSNorm'd hidden from fused residual+norm kernel.
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

// ===========================================================================
// Batched decode primitives.
//
// LEARNING OVERVIEW
//
// What this section does:
//   Enables batched decode — processing N decoding sequences in a single
//   forward pass instead of N separate passes.  This turns N mat-vec ops
//   (bandwidth-bound, low GPU utilisation) into 1 GEMM (compute-bound,
//   high GPU utilisation).
//
// Why not just reuse the prefill primitives?
//   Most of them ARE reused directly (embed_batch, qkv_projection_batch,
//   apply_rope_batch, o_proj_residual_batch, ffn_block_batch).  The new
//   functions here handle the parts that DIFFER between prefill and
//   batched decode:
//
//   1. Position upload: prefill has contiguous positions [start..start+N],
//      but batched decode has non-contiguous positions (each sequence is
//      at a different point in its generation).
//
//   2. Attention: prefill uses dense causal attention on the batch (all
//      tokens are from the same sequence).  Batched decode has N tokens
//      from N DIFFERENT sequences, each needing its own block table and
//      seq_len for paged attention.  This requires per-sequence calls.
//
//   3. LM head: prefill only needs the last token's logits.  Batched
//      decode needs ALL N tokens' logits (each is a different sequence's
//      next-token prediction).
//
// The per-sequence attention inner loop is the one non-batched part.  It
// uses copy_tensor_region to extract/insert individual rows from the
// batched Q/K/V tensors into the model's single-token scratch buffers,
// then calls the existing paged_attention_fused.  This is bandwidth-bound
// and fast — the GEMM savings from batching projections far outweigh the
// row-copy overhead.
// ===========================================================================

/// Upload token IDs and non-contiguous positions for batched decode.
///
/// Unlike `upload_prefill_inputs` where positions are contiguous
/// (start..start+N), batched decode positions come from each sequence's
/// current seq_len — they can be any values (e.g., [15, 203, 42, 891]).
pub(crate) fn upload_decode_batch_inputs<B: GpuCore>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    tokens: &[u32],
    positions: &[u32],
    bs: u32,
) {
    assert_eq!(tokens.len(), bs as usize);
    assert_eq!(positions.len(), bs as usize);
    let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
    backend.copy_to_tensor(&bufs.token_ids, token_bytes);
    let pos_bytes: &[u8] = bytemuck::cast_slice(positions);
    backend.copy_to_tensor(&bufs.positions, pos_bytes);
}

/// Per-sequence attention within a batched decode forward pass.
///
/// Extracts row `seq_idx` from the batched Q/K/V tensors (written by
/// the preceding batched matmul), runs paged_attention_fused using that
/// sequence's block table, and writes the attention output back into
/// row `seq_idx` of the batched attention output tensor.
///
/// Why copy rows instead of a batched attention kernel?
///   Each sequence has a DIFFERENT block table (different physical KV block
///   allocations from the shared pool) and a different seq_len.  A batched
///   kernel would need per-sequence block table indirection — complex to
///   implement correctly.  Row copies are ~2 KB each (one [q_dim] vector),
///   taking microseconds on GPU, while attention itself is the cheap part
///   of decode (bandwidth-bound, short sequences).  Phase 2 can add a
///   batched paged attention kernel for further gains.
pub(crate) fn batched_decode_per_seq_attention<B: GpuAttention + GpuCore>(
    backend: &B,
    bufs: &PrefillBuffers<B>,
    // Model's single-token scratch buffers (used as temporaries)
    scratch_q: &B::Tensor,
    scratch_k: &B::Tensor,
    scratch_v: &B::Tensor,
    scratch_attn_out: &B::Tensor,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    layer_idx: usize,
    seq_idx: usize,
    pos: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    attn_scale: f32,
    sinks: Option<&B::Tensor>,
) {
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let bf16_size = 2; // bytes per bf16

    // Extract row seq_idx from batched buffers → model's single-token scratch.
    backend.copy_tensor_region(
        &bufs.q_buf, seq_idx * q_dim * bf16_size,
        scratch_q, 0,
        q_dim * bf16_size,
    );
    backend.copy_tensor_region(
        &bufs.k_buf, seq_idx * kv_dim * bf16_size,
        scratch_k, 0,
        kv_dim * bf16_size,
    );
    backend.copy_tensor_region(
        &bufs.v_buf, seq_idx * kv_dim * bf16_size,
        scratch_v, 0,
        kv_dim * bf16_size,
    );

    // Run fused KV cache write + paged attention for this one sequence.
    backend.paged_attention_fused(
        scratch_q,
        scratch_k,
        scratch_v,
        &pool.k_pool[layer_idx],
        &pool.v_pool[layer_idx],
        &seq_state.block_table_gpu,
        scratch_attn_out,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size,
        attn_scale,
        sinks,
    );

    // Write attention output back into row seq_idx of the batched tensor.
    backend.copy_tensor_region(
        scratch_attn_out, 0,
        &bufs.attn_out, seq_idx * q_dim * bf16_size,
        q_dim * bf16_size,
    );
}

/// Final norm + batched LM head for batched decode.
///
/// Unlike the prefill version which extracts only the last token's logits,
/// batched decode produces logits for ALL N tokens — each row is a different
/// sequence's next-token prediction.  The output is [N, vocab_size] in
/// `logits_batch`, ready for `sample_batch()`.
pub(crate) fn final_norm_and_lm_head_decode_batch<B: GpuCore + GpuNorm + GpuMatmul>(
    backend: &B,
    weights: &ModelWeights<B>,
    bufs: &PrefillBuffers<B>,
    logits_batch: &B::Tensor,
    eps: f32,
    bs: u32,
    hidden_size: u32,
    vocab_size: u32,
) {
    // RMSNorm each row of [N, hidden_size].
    backend.rms_norm_batch(&bufs.hidden, &weights.norm_weight, eps, &bufs.norm_buf, bs);
    // GEMM: [N, hidden_size] × [vocab_size, hidden_size]^T → [N, vocab_size].
    let lm_head_weight = weights.lm_head.as_ref().unwrap_or(&weights.embed_tokens);
    backend.matmul_batch(lm_head_weight, &bufs.norm_buf, logits_batch, bs, vocab_size, hidden_size);
}

// ===========================================================================
// Tests.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::cpu::CpuBackend;
    use crate::gpu::ops::GpuCore;
    use crate::gpu::TensorDtype;
    use crate::model::loader::ExpertWeights;

    /// Helper: create BF16 bytes from f32 values.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let bf16_values: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        bytemuck::cast_slice(&bf16_values).to_vec()
    }

    /// Regression test for multi-GPU MoE buffer sizing (Mixtral / Qwen3-MoE).
    ///
    /// Bug: `moe_inter` was set to the full `config.moe_intermediate_size`
    /// instead of dividing by `world_size` for tensor-parallel inference.
    /// The MoE scratch buffers (moe_gate_buf, moe_up_buf) are allocated at
    /// `moe_intermediate_size / world_size`, so passing the full size caused
    /// matmul to write beyond the buffer → CUDA_ERROR_ILLEGAL_ADDRESS.
    ///
    /// This test exercises the expert FFN matmul path with TP-reduced buffers
    /// to verify that moe_inter matches the buffer allocation.
    #[test]
    fn test_moe_tp_buffer_sizing() {
        let b = CpuBackend;

        // Miniature MoE dimensions (inspired by Mixtral but tiny).
        let hidden_size: u32 = 8;
        let full_moe_inter: u32 = 16;
        let world_size: usize = 2;
        let tp_moe_inter: u32 = full_moe_inter / world_size as u32;

        // Expert weights at TP-reduced intermediate size.
        let gate_data = vec![0.1f32; tp_moe_inter as usize * hidden_size as usize];
        let up_data = vec![0.1f32; tp_moe_inter as usize * hidden_size as usize];
        let down_data = vec![0.1f32; hidden_size as usize * tp_moe_inter as usize];

        let expert = ExpertWeights::<CpuBackend> {
            gate_proj: b.upload_tensor(
                &bf16_bytes(&gate_data),
                &[tp_moe_inter as usize, hidden_size as usize],
                TensorDtype::BF16,
            ),
            up_proj: b.upload_tensor(
                &bf16_bytes(&up_data),
                &[tp_moe_inter as usize, hidden_size as usize],
                TensorDtype::BF16,
            ),
            down_proj: b.upload_tensor(
                &bf16_bytes(&down_data),
                &[hidden_size as usize, tp_moe_inter as usize],
                TensorDtype::BF16,
            ),
            gate_bias: None,
            up_bias: None,
            down_bias: None,
        };

        // Input: normalized hidden state.
        let norm_data = vec![1.0f32; hidden_size as usize];
        let norm_buf = b.upload_tensor(
            &bf16_bytes(&norm_data),
            &[hidden_size as usize],
            TensorDtype::BF16,
        );

        // Scratch buffers at TP-reduced size (the fix).
        let moe_gate_buf = b.alloc_tensor(&[tp_moe_inter as usize], TensorDtype::BF16);
        let moe_up_buf = b.alloc_tensor(&[tp_moe_inter as usize], TensorDtype::BF16);
        let down_buf = b.alloc_tensor(&[hidden_size as usize], TensorDtype::BF16);

        // Run the expert FFN matmuls with TP-sized moe_inter.
        // Before the fix, moe_inter would be full_moe_inter=16 but buffers
        // are sized for tp_moe_inter=8 → out-of-bounds writes on CUDA.
        b.matmul(
            &expert.gate_proj,
            &norm_buf,
            &moe_gate_buf,
            tp_moe_inter,
            hidden_size,
        );
        b.matmul(
            &expert.up_proj,
            &norm_buf,
            &moe_up_buf,
            tp_moe_inter,
            hidden_size,
        );
        b.silu_mul(&moe_gate_buf, &moe_up_buf, &moe_gate_buf, tp_moe_inter);
        b.matmul(
            &expert.down_proj,
            &moe_gate_buf,
            &down_buf,
            hidden_size,
            tp_moe_inter,
        );

        // Verify output is non-zero (the expert FFN was executed correctly).
        let mut out_bytes = vec![0u8; hidden_size as usize * 2];
        b.copy_to_host(&down_buf, &mut out_bytes);
        let out_values: Vec<f32> = bytemuck::cast_slice::<u8, half::bf16>(&out_bytes)
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let sum: f32 = out_values.iter().map(|v| v.abs()).sum();
        assert!(
            sum > 0.0,
            "Expert FFN output should be non-zero with TP-sized buffers"
        );

        // Verify the buffer byte count matches what TP allocates.
        // This is the core invariant: buffer size == tp_moe_inter * sizeof(bf16).
        assert_eq!(
            b.tensor_byte_count(&moe_gate_buf),
            tp_moe_inter as usize * 2,
            "moe_gate_buf must be sized for tp_moe_inter, not full_moe_inter"
        );
    }

    /// Verify that Dims::from_config_tp divides heads and inter_size but not
    /// hidden_size.  MoE intermediate size is not in Dims — callers must
    /// divide `config.moe_intermediate_size` by `model.world_size` themselves.
    #[test]
    fn test_dims_tp_splits_heads_and_inter() {
        // Minimal Mixtral-like config JSON.
        let json = r#"{
            "model_type": "mixtral",
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "head_dim": 64,
            "num_hidden_layers": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        }"#;
        let config: crate::model::config::ModelConfig = serde_json::from_str(json).unwrap();

        let dims_ws1 = Dims::from_config(&config);
        let dims_ws2 = Dims::from_config_tp(&config, 2);

        // hidden_size is NOT divided.
        assert_eq!(dims_ws1.hidden_size, dims_ws2.hidden_size);
        // heads ARE divided.
        assert_eq!(dims_ws2.num_heads, dims_ws1.num_heads / 2);
        assert_eq!(dims_ws2.num_kv_heads, dims_ws1.num_kv_heads / 2);
        // inter_size IS divided.
        assert_eq!(dims_ws2.inter_size, dims_ws1.inter_size / 2);
    }

    // ===================================================================
    // Source-level invariant: all_reduce_sum after row-split matmuls.
    //
    // In tensor parallelism, o_proj and down_proj are row-split across
    // GPUs.  Each GPU computes a partial result that MUST be summed via
    // all_reduce_sum before feeding into the next operation (norm or
    // residual add).  Missing this call produces correct single-GPU
    // output but gibberish in multi-GPU mode.
    //
    // These tests scan source files to enforce the invariant statically,
    // catching the bug at compile/test time rather than at inference.
    // ===================================================================

    /// Scan a source file and verify that every inline matmul referencing
    /// `o_proj` or `down_proj` is followed by `all_reduce_sum` before the
    /// next `.add(` (residual connection).
    ///
    /// Returns a list of (line_number, description) for each violation.
    fn check_all_reduce_after_row_split_matmuls(source: &str) -> Vec<(usize, String)> {
        let mut violations = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        // Track: did we just see a matmul on o_proj/down_proj and haven't
        // yet seen all_reduce_sum?
        let mut pending_reduce: Option<(usize, String)> = None;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Detect inline o_proj/down_proj matmul calls.
            // These appear as `&layer.o_proj,` or `&layer.down_proj,` inside
            // a matmul/matmul_batch call.  We look for the weight reference
            // lines that appear inside .matmul( blocks.
            if (trimmed.contains("&layer.o_proj") || trimmed.contains("&layer.down_proj"))
                && !trimmed.starts_with("//")
            {
                // Check context: is this inside a matmul call?
                // Look up a few lines for `.matmul` or `.matmul_batch`.
                let start = i.saturating_sub(3);
                let context = &lines[start..=i];
                let in_matmul = context
                    .iter()
                    .any(|l| l.contains(".matmul(") || l.contains(".matmul_batch("));

                if in_matmul {
                    let proj_name = if trimmed.contains("o_proj") {
                        "o_proj"
                    } else {
                        "down_proj"
                    };
                    pending_reduce = Some((i + 1, proj_name.to_string()));
                }
            }

            // If we see all_reduce_sum, clear the pending flag.
            if trimmed.contains("all_reduce_sum") && !trimmed.starts_with("//") {
                pending_reduce = None;
            }

            // If we see a residual .add( while a reduce is pending, that's a violation.
            // Skip bias adds (they reference `bias` in the same line or nearby context).
            if trimmed.contains(".add(") && !trimmed.starts_with("//") {
                let is_bias_add = trimmed.contains("bias")
                    || lines.get(i.wrapping_sub(1)).map_or(false, |l| l.contains("bias"));
                if !is_bias_add {
                    if let Some((line_num, ref proj)) = pending_reduce {
                        violations.push((
                            line_num,
                            format!(
                                "{} matmul (line {}) has no all_reduce_sum before residual add (line {})",
                                proj,
                                line_num,
                                i + 1
                            ),
                        ));
                        pending_reduce = None;
                    }
                }
            }
        }

        violations
    }

    /// Verify all model registry files have all_reduce_sum after inline
    /// o_proj/down_proj matmuls.
    #[test]
    fn test_model_registry_all_reduce_after_row_split_matmuls() {
        let registry_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/registry");
        let mut all_violations = Vec::new();

        for entry in std::fs::read_dir(registry_dir).expect("can't read registry dir") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map_or(true, |e| e != "rs") {
                continue;
            }
            let filename = path.file_name().unwrap().to_string_lossy().to_string();
            if filename == "mod.rs" {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let violations = check_all_reduce_after_row_split_matmuls(&source);
            for (line, desc) in violations {
                all_violations.push(format!("  {}:{} — {}", filename, line, desc));
            }
        }

        assert!(
            all_violations.is_empty(),
            "Missing all_reduce_sum after row-split matmuls in model registry files:\n{}",
            all_violations.join("\n")
        );
    }

    /// Verify primitive helper functions have all_reduce_sum after o_proj/down_proj
    /// matmuls (catches bugs in shared helpers like o_proj_residual_qdim_biased).
    #[test]
    fn test_primitives_all_reduce_after_row_split_matmuls() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/primitives.rs");
        let source = std::fs::read_to_string(path).unwrap();
        let violations = check_all_reduce_after_row_split_matmuls(&source);

        let messages: Vec<String> = violations
            .iter()
            .map(|(line, desc)| format!("  primitives.rs:{} — {}", line, desc))
            .collect();
        assert!(
            messages.is_empty(),
            "Missing all_reduce_sum after row-split matmuls in primitives.rs:\n{}",
            messages.join("\n")
        );
    }
}
