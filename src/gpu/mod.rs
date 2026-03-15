// ===========================================================================
// GPU backend trait abstraction for LLM inference.
//
// LEARNING OVERVIEW
//
// This module defines the interface between the model layer (forward pass,
// KV cache, sampling) and the GPU compute layer (Metal kernels, CUDA
// kernels).  The model code is generic over `B: GpuBackend` and never
// touches platform-specific types — it calls `backend.matmul(...)`,
// `backend.attention(...)`, etc. and the trait implementation dispatches
// the right GPU kernel.
//
// TRAIT ORGANISATION
//
// The GpuBackend supertrait is composed from focused sub-traits, each
// covering one kernel family:
//
//   GpuCore         — device info, tensor memory management, flush/submit
//   GpuNorm         — RMSNorm variants
//   GpuMatmul       — mat-vec and GEMM
//   GpuRope         — Rotary Positional Embeddings
//   GpuAttention    — attention + KV cache operations
//   GpuElementwise  — point-wise activations, reductions, MoE routing
//   GpuEmbed        — embedding table lookups
//   GpuDeltaNet     — Gated DeltaNet linear attention (Qwen 3.5)
//
// This composability means a new backend can implement sub-traits
// incrementally and model code can express fine-grained bounds.
//
// PLATFORM SELECTION
//
// We use OS-conditional compilation (`#[cfg(target_os = "...")]`) rather
// than Cargo feature flags.  On macOS we compile the Metal backend; on
// Linux we compile the CUDA backend.  Only one backend exists in a given
// binary.  The `Backend` type alias resolves to whichever backend is
// active, so call sites just write `gpu::Backend` and `gpu::create_backend()`.
// ===========================================================================

// ---------------------------------------------------------------------------
// Sub-trait definitions, organised by kernel family.
// ---------------------------------------------------------------------------

pub(crate) mod ops;

pub(crate) use ops::{
    GpuAttention, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMatmul, GpuNorm, GpuRope,
};

// ---------------------------------------------------------------------------
// GpuBackend — the supertrait combining all kernel families.
//
// Callers write `B: GpuBackend` and get access to every operation.
// A blanket impl means any type implementing all sub-traits is
// automatically a GpuBackend.
// ---------------------------------------------------------------------------

pub(crate) trait GpuBackend:
    GpuCore + GpuNorm + GpuMatmul + GpuRope + GpuAttention + GpuElementwise + GpuEmbed + GpuDeltaNet
{
}

impl<T> GpuBackend for T where
    T: GpuCore
        + GpuNorm
        + GpuMatmul
        + GpuRope
        + GpuAttention
        + GpuElementwise
        + GpuEmbed
        + GpuDeltaNet
{
}

// ---------------------------------------------------------------------------
// Conditional module compilation gates.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) mod metal;

#[cfg(target_os = "linux")]
pub(crate) mod cuda;

// ---------------------------------------------------------------------------
// Platform type aliases.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) type Backend = self::metal::MetalBackend;

#[cfg(target_os = "linux")]
pub(crate) type Backend = self::cuda::CudaBackend;

// ---------------------------------------------------------------------------
// Tensor data types.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorDtype {
    BF16,
    F32,
    /// Block-wise 4-bit quantization: 32 weights per block, each block is
    /// 20 bytes (4-byte f32 scale + 16 bytes of packed nibbles).
    Q4,
}

impl TensorDtype {
    /// Bytes per element for fixed-size types.  Panics for Q4 (use q4_byte_count).
    pub fn byte_size(self) -> usize {
        match self {
            TensorDtype::BF16 => 2,
            TensorDtype::F32 => 4,
            TensorDtype::Q4 => panic!("Q4 has variable byte size; use q4_byte_count()"),
        }
    }
}

/// Compute total byte count for a Q4 weight tensor [m, k].
pub(crate) fn q4_byte_count(m: usize, k: usize) -> usize {
    let blocks_per_row = k / 32;
    m * blocks_per_row * 20
}

// ---------------------------------------------------------------------------
// Factory function.
// ---------------------------------------------------------------------------

pub(crate) fn create_backend() -> anyhow::Result<Backend> {
    Backend::new()
}
