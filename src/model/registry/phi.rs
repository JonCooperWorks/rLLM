// ===========================================================================
// Phi model family (Microsoft Phi-3, Phi-4).
//
// Phi-3/4 (14B):
//   - RMSNorm + GQA attention + SwiGLU FFN + RoPE
//   - NO bias on any projection (Q/K/V/O/FFN)
//   - Chat template: <|im_start|>/<|im_sep|>/<|im_end|> markers
//   - RoPE theta: 250000
//
// LEARNING OVERVIEW
//
// Why does this file exist when the forward pass is identical to Llama?
//   The difference is in weight LOADING, not inference.  Phi uses fused
//   weight matrices (qkv_proj, gate_up_proj) that must be split on-load
//   in loader.rs.  Once the weights are loaded as separate q/k/v and
//   gate/up tensors, the forward pass is exactly Llama — same primitives
//   in the same order.
//
//   We keep a separate file so that:
//     1. `ModelArch::Phi` has an obvious home in the registry
//     2. The dispatch table in model/mod.rs stays one-to-one with files
//     3. If future Phi models diverge, we can add Phi-specific features
//        without touching Llama
//
//   The implementation simply re-exports Llama's forward functions.
// ===========================================================================

// Phi's forward pass is identical to Llama's — re-export directly.
pub(crate) use super::llama::{forward_prefill_paged, forward_single_paged};
