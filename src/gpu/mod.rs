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

#[cfg(test)]
pub(crate) mod cpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dtype_byte_size_bf16() {
        assert_eq!(TensorDtype::BF16.byte_size(), 2);
    }

    #[test]
    fn test_tensor_dtype_byte_size_f32() {
        assert_eq!(TensorDtype::F32.byte_size(), 4);
    }

    #[test]
    #[should_panic(expected = "Q4 has variable byte size")]
    fn test_tensor_dtype_byte_size_q4_panics() {
        let _ = TensorDtype::Q4.byte_size();
    }

    #[test]
    fn test_q4_byte_count_single_block() {
        // 1 row, 32 elements = 1 block = 20 bytes
        assert_eq!(q4_byte_count(1, 32), 20);
    }

    #[test]
    fn test_q4_byte_count_multiple_blocks() {
        // 1 row, 64 elements = 2 blocks = 40 bytes
        assert_eq!(q4_byte_count(1, 64), 40);
    }

    #[test]
    fn test_q4_byte_count_multiple_rows() {
        // 4 rows, 64 elements each = 4 * 2 blocks * 20 = 160
        assert_eq!(q4_byte_count(4, 64), 160);
    }

    #[test]
    fn test_q4_byte_count_large() {
        // 2048 rows, 2048 cols = 2048 * 64 * 20 = 2621440
        assert_eq!(q4_byte_count(2048, 2048), 2048 * (2048 / 32) * 20);
    }
}
