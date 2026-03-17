// ---------------------------------------------------------------------------
// Metal kernel implementations — one file per ops/ sub-trait.
//
// Each file here implements one Gpu* trait from `gpu/ops/` for MetalBackend.
// The pattern is always the same:
//
//   1. Define #[repr(C)] param structs that mirror the Metal shader's
//      argument buffer layout (must match byte-for-byte).
//   2. impl GpuFoo for MetalBackend { ... } — each method packs params,
//      picks a pipeline, and calls `self.dispatch_async()`.
//
// Adding a new kernel family:
//   1. Create the trait in gpu/ops/new_family.rs
//   2. Create the shader in metal/shaders/new_family.metal
//   3. Add pipeline fields + compilation in metal/backend.rs
//   4. Create metal/kernels/new_family.rs implementing the trait
//   5. Add `mod new_family;` here
//   6. Add the trait to the GpuBackend supertrait bound in gpu/mod.rs
// ---------------------------------------------------------------------------

mod allreduce;
mod attention;
mod core;
mod deltanet;
mod elementwise;
mod embed;
mod matmul;
mod norm;
mod rope;
