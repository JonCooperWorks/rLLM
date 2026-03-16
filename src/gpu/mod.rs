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
// On macOS we compile the Metal backend unconditionally.  On Linux the
// CUDA backend is gated behind the `cuda` Cargo feature — this allows
// `cargo test` on any Linux machine without a CUDA toolkit.  Only one
// backend exists in a given binary.  The `Backend` type alias resolves
// to whichever backend is active, so call sites just write
// `gpu::Backend` and `gpu::create_backend()`.
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

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

// ---------------------------------------------------------------------------
// Platform type aliases.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) type Backend = self::metal::MetalBackend;

#[cfg(feature = "cuda")]
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
// Q4 quantisation — default format used by Metal and CPU backends.
//
// Backends that use a different quantisation format (e.g. CUDA with CUTLASS
// INT4) override `GpuCore::quantize_upload` and never call this function.
// ---------------------------------------------------------------------------

/// Quantise a bf16 weight matrix [m, k] to block-wise Q4.
///
/// Block layout (symmetric quantisation, block_size=32):
///   For each block of 32 consecutive weights:
///     scale = max(|w_i|) / 7.0           (maps [-max, max] to [-7, 7])
///     q_i = clamp(round(w_i / scale), -8, 7)  (4-bit signed, range [-8, 7])
///     stored as unsigned: u_i = q_i + 8  (range [0, 15], fits in 4 bits)
///
///   Output per block (20 bytes):
///     [0..4]:   f32 scale (little-endian)
///     [4..20]:  16 bytes, 2 packed nibbles each
///               byte[i] = u[2i] | (u[2i+1] << 4)
pub(crate) fn quantize_bf16_to_q4(bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
    use half::bf16;

    assert_eq!(bf16_data.len(), m * k * 2);

    // Try zero-copy cast first; fall back to a copy if the mmap slice
    // isn't 2-byte aligned (can happen with some safetensors packing).
    let owned_buf: Vec<bf16>;
    let values: &[bf16] = match bytemuck::try_cast_slice(bf16_data) {
        Ok(v) => v,
        Err(_) => {
            owned_buf = bf16_data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                .collect();
            &owned_buf
        }
    };
    assert_eq!(values.len(), m * k);

    let blocks_per_row = k / 32;
    let mut out = vec![0u8; q4_byte_count(m, k)];

    for row in 0..m {
        for block in 0..blocks_per_row {
            let src_offset = row * k + block * 32;
            let dst_offset = (row * blocks_per_row + block) * 20;

            // Find max absolute value in the block for scale computation.
            let mut max_abs: f32 = 0.0;
            for i in 0..32 {
                let v = values[src_offset + i].to_f32().abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let inv_scale = 1.0 / scale;

            // Write scale.
            out[dst_offset..dst_offset + 4].copy_from_slice(&scale.to_le_bytes());

            // Quantise and pack pairs of weights into bytes.
            for i in 0..16 {
                let v0 = values[src_offset + i * 2].to_f32();
                let v1 = values[src_offset + i * 2 + 1].to_f32();

                let q0 = ((v0 * inv_scale).round() as i32).clamp(-8, 7) + 8;
                let q1 = ((v1 * inv_scale).round() as i32).clamp(-8, 7) + 8;

                out[dst_offset + 4 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Factory function.
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", feature = "cuda"))]
pub(crate) fn create_backend() -> anyhow::Result<Backend> {
    Backend::new()
}

#[cfg(not(any(target_os = "macos", feature = "cuda")))]
pub(crate) fn create_backend() -> anyhow::Result<cpu::CpuBackend> {
    eprintln!("warning: no GPU backend available, using CPU (slow)");
    Ok(cpu::CpuBackend)
}

#[cfg(not(any(target_os = "macos", feature = "cuda")))]
pub(crate) type Backend = cpu::CpuBackend;


// CPU backend: always available in tests for correctness validation,
// and used as the runtime fallback when no GPU backend is compiled.
#[cfg(any(test, not(any(target_os = "macos", feature = "cuda"))))]
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
