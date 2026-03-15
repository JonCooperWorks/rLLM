// ===========================================================================
// Model registry — each model family gets its own file.
//
// Similar to vLLM's registry: model families compose shared primitives
// from model/primitives.rs in their architecture-specific order.
//
// Each model family gets its own file that wires up the forward pass.
// Standard dense transformers (Llama, Qwen, Phi) delegate to the shared
// pipeline in model/standard.rs via ArchFeatures.  Models with custom
// pipelines (Gemma, Qwen3.5, Qwen3 MoE) implement their own.
//
// Adding a new architecture:
//   1. Create a new file here (e.g. deepseek.rs)
//   2. Add a variant to ModelArch in config.rs
//   3. Add dispatch arms in model/mod.rs
//   4. If it's a standard dense transformer, delegate to standard.rs
//      with the right ArchFeatures.  Otherwise, write a custom forward pass.
// ===========================================================================

pub(crate) mod gemma;
pub(crate) mod llama;
pub(crate) mod mistral;
pub(crate) mod mixtral;
pub(crate) mod phi;
pub(crate) mod qwen;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_moe;
