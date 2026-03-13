// ===========================================================================
// Model registry — each model family gets its own file.
//
// Similar to vLLM's registry: model families compose shared primitives
// from model/primitives.rs in their architecture-specific order.
//
// Adding a new architecture:
//   1. Create a new file here (e.g. deepseek.rs)
//   2. Add a variant to ModelArch in config.rs
//   3. Add dispatch arms in model/mod.rs
// ===========================================================================

pub(crate) mod llama;
pub(crate) mod qwen;
