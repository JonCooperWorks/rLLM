// ===========================================================================
// Phi model family (Microsoft Phi-3, Phi-4).
//
// Phi's forward pass is identical to Llama's — the difference is in weight
// LOADING (fused qkv_proj/gate_up_proj split on-load in loader.rs).  Once
// loaded, inference is the same.
//
// This file exists so ModelArch::Phi has an obvious home in the registry.
// At runtime, engine/loader.rs constructs a LlamaForward with has_qkv_bias:
// false — no code in this file is needed.
//
// If future Phi models diverge from Llama, add a PhiForward struct here.
// ===========================================================================
