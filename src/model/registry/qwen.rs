// ===========================================================================
// Qwen 2.5 model family.
//
// Qwen 2.5 family (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B):
//   - RMSNorm + GQA attention + SwiGLU FFN + RoPE
//   - QKV projections have BIAS (output = W @ x + b)
//   - O projection and FFN projections have NO bias
//   - Chat template: ChatML (<|im_start|>/<|im_end|>)
//   - RoPE theta: 10000
//
// LEARNING OVERVIEW
//
// What makes Qwen different from Llama?
//   Exactly one thing in the forward pass: bias-add after Q/K/V projections.
//
//   After computing Q = W_q @ hidden, Llama uses Q directly.  Qwen adds
//   a learned bias vector: Q = W_q @ hidden + b_q.  Same for K and V.
//   This is controlled by the `has_qkv_bias` flag in ArchFeatures.
//
//   Everything else — norm, RoPE, attention, FFN, residuals — is identical
//   to Llama.  The shared forward pass in llama.rs handles both.
// ===========================================================================

use crate::gpu::{
    GpuAttention, GpuCore, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::{Model, PrefillBuffers};

use super::llama::ArchFeatures;

/// Qwen features: QKV bias is the only difference from Llama.
const FEATURES: ArchFeatures = ArchFeatures {
    has_qkv_bias: true,
};

pub(crate) fn forward_single_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    super::llama::forward_single_impl(m, token_id, pool, seq_state, &FEATURES)
}

pub(crate) fn forward_prefill_paged<B: GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    super::llama::forward_prefill_impl(m, tokens, pool, seq_state, bufs, &FEATURES)
}
