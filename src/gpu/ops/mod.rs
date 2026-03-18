// ===========================================================================
// GPU operation traits — platform-agnostic kernel contracts.
//
// WHAT THIS DIRECTORY IS
//
//   Each file defines one sub-trait of the GpuBackend supertrait (see
//   gpu/mod.rs).  Together they specify *what* GPU operations exist without
//   any platform-specific implementation.
//
//   The traits are split by kernel family so that:
//     1. A new backend (e.g. CUDA) can be brought up incrementally —
//        implement GpuCore + GpuMatmul first, get a basic forward pass,
//        then add GpuAttention, GpuDeltaNet, etc. one at a time.
//     2. Each file is small enough to read in one sitting (~20-80 lines).
//     3. Model code *could* express fine-grained bounds (e.g.
//        `B: GpuCore + GpuMatmul`) but in practice everything uses the
//        `B: GpuBackend` supertrait today.
//
// HOW THE TRAITS COMPOSE
//
//   Every sub-trait extends GpuCore so it can reference `Self::Tensor`.
//   GpuBackend is a blanket supertrait — any type implementing all nine
//   sub-traits automatically implements GpuBackend (see gpu/mod.rs).
//
// WHERE THE IMPLEMENTATIONS LIVE
//
//   Metal:  gpu/metal/kernels/{core,norm,matmul,...}.rs
//   CUDA:   gpu/cuda/kernels/{core,norm,matmul,...}.rs
//
//   Each kernel impl file mirrors the ops file it implements, plus the
//   `#[repr(C)]` param structs that must match the corresponding .metal
//   shader structs byte-for-byte.
//
// ADDING A NEW KERNEL FAMILY
//
//   1. Create a new trait file here (e.g. ops/new_family.rs)
//   2. Add the trait to the GpuBackend supertrait bounds in gpu/mod.rs
//   3. Implement for MetalBackend in metal/kernels/new_family.rs
//   4. Stub for CudaBackend in cuda/mod.rs
// ===========================================================================

mod allreduce;
mod attention;
mod core;
mod deltanet;
mod elementwise;
mod embed;
mod matmul;
mod norm;
mod rope;

pub(crate) use allreduce::GpuAllReduce;
pub(crate) use attention::GpuAttention;
pub(crate) use core::GpuCore;
pub(crate) use deltanet::GpuDeltaNet;
pub(crate) use elementwise::GpuElementwise;
pub(crate) use embed::GpuEmbed;
pub(crate) use matmul::GpuMatmul;
pub(crate) use norm::GpuNorm;
pub(crate) use rope::GpuRope;
