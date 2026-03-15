// ===========================================================================
// Mistral model family (Mistral AI, 7B).
//
// LEARNING OVERVIEW
//
// Why does this file exist when the forward pass is identical to Llama?
//   Mistral 7B is architecturally identical to Llama — same normalization
//   (RMSNorm), same attention (GQA), same FFN (SwiGLU), same positional
//   encoding (RoPE).  No QKV bias, no QK-norm, no fused weights.
//
//   We keep a separate registry entry so that:
//     1. `ModelArch::Mistral` has an obvious home in the registry
//     2. The dispatch table in model/mod.rs stays one-to-one with files
//     3. If future Mistral models diverge, we have a clear place for
//        Mistral-specific code
//
//   The implementation re-exports Llama's forward functions directly.
//
// Architecture (Mistral 7B Instruct):
//   - 32 transformer layers
//   - 32 query heads, 8 KV heads (GQA ratio 4:1)
//   - Hidden size: 4096, head dim: 128
//   - Intermediate size: 14336 (SwiGLU)
//   - RoPE theta: 1,000,000
//   - Chat template: [INST]/[/INST] markers
//   - Vocab: 32000 tokens (SentencePiece BPE)
// ===========================================================================

// Mistral's forward pass is identical to Llama's — re-export directly.
pub(crate) use super::llama::{forward_prefill_paged, forward_single_paged};
