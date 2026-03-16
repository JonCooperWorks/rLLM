// ===========================================================================
// CUDA backend — NVIDIA GPU implementation via cudarc.
//
// Module structure mirrors metal/:
//   backend.rs  — CudaBackend struct, NVRTC compilation, stream management
//   tensor.rs   — CudaTensor (device buffer + shape + dtype)
//   kernels/    — one file per ops/ sub-trait (impl GpuFoo for CudaBackend)
//   shaders/    — .cu shader sources (embedded via include_str!)
//
// Related files:
//   Metal backend:  metal/ (equivalent for macOS)
//   Trait defs:     gpu/ops/*.rs
//   Platform gate:  gpu/mod.rs
// ===========================================================================

mod backend;
mod kernels;
mod tensor;

pub(crate) use backend::CudaBackend;
// Re-exported so gpu::Backend::Tensor resolves through the module boundary.
#[allow(unused_imports)]
pub(crate) use tensor::CudaTensor;
