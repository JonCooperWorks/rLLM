// ===========================================================================
// Mistral model family (Mistral AI, 7B).
//
// Mistral 7B is architecturally identical to Llama at inference time.  Same
// normalization (RMSNorm), attention (GQA), FFN (SwiGLU), and positional
// encoding (RoPE).  No QKV bias, no QK-norm, no fused weights.
//
// This file exists so ModelArch::Mistral has an obvious home in the registry.
// At runtime, engine/loader.rs constructs a LlamaForward with has_qkv_bias:
// false — no code in this file is needed.
//
// If future Mistral models diverge from Llama, add a MistralForward struct here.
// ===========================================================================
