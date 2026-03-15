// ===========================================================================
// Llama 3.x model family.
//
// Llama 3.x family (1B, 3B, 8B, 70B):
//   - RMSNorm + GQA attention + SwiGLU FFN + RoPE
//   - NO bias on any projection (Q/K/V/O/FFN)
//   - Chat template: <|start_header_id|> markers
//   - RoPE theta: 500000
//
// This is the "vanilla" dense transformer — no special features.
// The forward pass is fully handled by standard.rs with no extra flags.
// ===========================================================================

use crate::gpu::GpuBackend;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::{Model, PrefillBuffers};

use crate::model::standard::{self, ArchFeatures};

/// Llama features: the baseline.  No QKV bias, nothing extra.
const FEATURES: ArchFeatures = ArchFeatures {
    has_qkv_bias: false,
};

pub(crate) fn forward_single_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    token_id: u32,
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
) -> anyhow::Result<()> {
    standard::forward_single_paged(m, token_id, pool, seq_state, &FEATURES)
}

pub(crate) fn forward_prefill_paged<B: GpuBackend>(
    m: &Model<'_, B>,
    tokens: &[u32],
    pool: &KvPool<B>,
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
) -> anyhow::Result<()> {
    standard::forward_prefill_paged(m, tokens, pool, seq_state, bufs, &FEATURES)
}
