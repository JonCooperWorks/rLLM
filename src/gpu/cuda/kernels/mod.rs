// ---------------------------------------------------------------------------
// CUDA kernel implementations — one file per ops/ sub-trait.
//
// Each file here implements one Gpu* trait from `gpu/ops/` for CudaBackend.
// The pattern is always the same:
//
//   1. Define #[repr(C)] param structs that mirror the CUDA shader's
//      struct layout (must match byte-for-byte).
//   2. Mark them `unsafe impl DeviceRepr` so cudarc can pass them by value.
//   3. impl GpuFoo for CudaBackend { ... } — each method packs params,
//      computes grid/block dims, and calls `stream.launch_builder()`.
//
// Related files:
//   Trait definitions: gpu/ops/*.rs
//   CUDA shaders:      cuda/shaders/*.cu
//   Backend struct:     cuda/backend.rs
// ---------------------------------------------------------------------------

mod allreduce;
mod attention;
mod core;
mod deltanet;
mod elementwise;
mod embed;
mod matmul;
mod moe;
mod norm;
mod rope;
