// ===========================================================================
// Metal backend — Apple GPU implementation via the metal/objc2 crates.
//
// Module structure:
//   backend.rs  — MetalBackend struct, pipeline compilation, async dispatch
//   tensor.rs   — MetalTensor (MTLBuffer + shape + dtype)
//   kernels/    — one file per ops/ sub-trait (impl GpuFoo for MetalBackend)
//   shaders/    — .metal shader sources (embedded via include_str!)
//
// Related files:
//   CUDA backend:  cuda/ (equivalent for Linux)
//   Trait defs:    gpu/ops/*.rs
//   Platform gate: gpu/mod.rs
// ===========================================================================

mod backend;
mod kernels;
mod tensor;

pub(crate) use backend::MetalBackend;
// Re-exported so gpu::Backend::Tensor resolves through the module boundary.
#[allow(unused_imports)]
pub(crate) use tensor::MetalTensor;
