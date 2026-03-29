// ===========================================================================
// Qwen 2.5 model family.
//
// Qwen 2.5 differs from Llama by exactly one thing: QKV bias.  After
// computing Q = W_q @ hidden, Qwen adds a learned bias: Q = W_q @ hidden + b_q.
// Same for K and V.  Everything else is identical to Llama.
//
// This file exists so ModelArch::Qwen2 has an obvious home in the registry.
// At runtime, engine/loader.rs constructs a LlamaForward with has_qkv_bias:
// true — no code in this file is needed.
//
// If future Qwen models diverge from Llama, add a QwenForward struct here.
// ===========================================================================
