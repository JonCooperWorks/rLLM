// ===========================================================================
// Model registry — each model family gets its own file.
//
// Similar to vLLM's registry: model families compose shared primitives
// from model/primitives.rs in their architecture-specific order.
//
// Each model family implements the `ModelForward` trait (model/forward.rs)
// in its own file.  Standard dense transformers (Llama, Phi, Qwen, Mistral)
// share a common implementation via `LlamaForward` parameterised by
// `ArchFeatures`.  Models with custom pipelines (Gemma, Qwen3.5, Qwen3 MoE)
// have their own `ModelForward` implementor struct.
//
// Adding a new architecture:
//   1. Create a new file here (e.g. deepseek.rs)
//   2. Add a variant to ModelArch in config.rs
//   3. Implement ModelForward trait in the new file
//   4. Add a construction match arm in engine/loader.rs
//   5. If it's a standard dense transformer, use LlamaForward with the
//      right ArchFeatures.  Otherwise, create a new struct.
// ===========================================================================

pub(crate) mod gemma;
pub(crate) mod gpt_oss;
pub(crate) mod llama;
pub(crate) mod mistral;
pub(crate) mod mixtral;
pub(crate) mod phi;
pub(crate) mod qwen;
pub(crate) mod nemotron_h;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_moe;
