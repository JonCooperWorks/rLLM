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
//   GpuAllReduce    — collective communication for tensor parallelism
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

pub(crate) mod multi_gpu;
pub(crate) mod ops;
#[allow(dead_code)] // CUDA multi-GPU tensor parallelism; unused on macOS
pub(crate) mod parallel;

pub(crate) use ops::{
    GpuAllReduce, GpuAttention, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMamba2, GpuMatmul,
    GpuMoe, GpuNorm, GpuRope, GpuTurboQuant, GpuVision,
};
pub(crate) use ops::quant::QuantFormat;

// ---------------------------------------------------------------------------
// GpuBackend — the supertrait combining all kernel families.
//
// Callers write `B: GpuBackend` and get access to every operation.
// A blanket impl means any type implementing all sub-traits is
// automatically a GpuBackend.
// ---------------------------------------------------------------------------

pub(crate) trait GpuBackend:
    GpuCore
    + GpuNorm
    + GpuMatmul
    + GpuRope
    + GpuAttention
    + GpuElementwise
    + GpuEmbed
    + GpuDeltaNet
    + GpuMamba2
    + GpuAllReduce
    + GpuMoe
    + GpuVision
    + GpuTurboQuant
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
        + GpuMamba2
        + GpuAllReduce
        + GpuMoe
        + GpuVision
        + GpuTurboQuant
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
    /// 18 bytes (2-byte bf16 scale + 16 bytes of packed nibbles).
    Q4,
    /// Block-wise 8-bit quantization: 32 weights per block, each block is
    /// 34 bytes (2-byte bf16 scale + 32 signed int8 values).
    Q8,
    /// FP8 E4M3 — IEEE 8-bit float (1 sign + 4 exponent + 3 mantissa).
    /// 1 byte per weight, no block structure, no per-block scale.
    /// Used on NVIDIA SM 89+ (Ada/Hopper) where hardware supports native FP8.
    /// Range ±448, precision ~0.125.  On Metal, Q8 blocks are used instead.
    FP8,
}

impl TensorDtype {
    /// Bytes per element for fixed-size types.  Panics for block formats (use q4/q8_byte_count).
    pub fn byte_size(self) -> usize {
        match self {
            TensorDtype::BF16 => 2,
            TensorDtype::F32 => 4,
            TensorDtype::FP8 => 1,
            TensorDtype::Q4 => panic!("Q4 has variable byte size; use q4_byte_count()"),
            TensorDtype::Q8 => panic!("Q8 has variable byte size; use q8_byte_count()"),
        }
    }
}

/// Compute total byte count for a Q4 weight tensor [m, k].
///
/// Panics on overflow instead of silently wrapping around, which would cause
/// undersized GPU buffer allocations and memory corruption.
pub(crate) fn q4_byte_count(m: usize, k: usize) -> usize {
    let blocks_per_row = k / 32;
    m.checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(18))
        .expect("q4_byte_count overflow: tensor dimensions too large")
}

/// Compute total byte count for a Q8 weight tensor [m, k].
///
/// Same block size as Q4 (32 weights) but 34 bytes per block:
/// 2-byte bf16 scale + 32 signed int8 values.
pub(crate) fn q8_byte_count(m: usize, k: usize) -> usize {
    let blocks_per_row = k / 32;
    m.checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(34))
        .expect("q8_byte_count overflow: tensor dimensions too large")
}

/// Compute total byte count for an FP8 weight tensor [m, k].
///
/// FP8 has no block structure — 1 byte per weight, so total is just m * k.
/// Helper kept for consistency with q4_byte_count / q8_byte_count.
pub(crate) fn fp8_byte_count(m: usize, k: usize) -> usize {
    m.checked_mul(k)
        .expect("fp8_byte_count overflow: tensor dimensions too large")
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
///   Output per block (18 bytes):
///     [0..2]:   bf16 scale (little-endian)
///     [2..18]:  16 bytes, 2 packed nibbles each
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
            let dst_offset = (row * blocks_per_row + block) * 18;

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

            // Write scale as bf16 (2 bytes instead of 4).
            // bf16 truncation: just take upper 16 bits of f32.
            let scale_bf16 = (scale.to_bits() >> 16) as u16;
            out[dst_offset..dst_offset + 2].copy_from_slice(&scale_bf16.to_le_bytes());

            // Quantise and pack pairs of weights into bytes.
            for i in 0..16 {
                let v0 = values[src_offset + i * 2].to_f32();
                let v1 = values[src_offset + i * 2 + 1].to_f32();

                let q0 = ((v0 * inv_scale).round() as i32).clamp(-8, 7) + 8;
                let q1 = ((v1 * inv_scale).round() as i32).clamp(-8, 7) + 8;

                out[dst_offset + 2 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Q8 quantisation — 8-bit symmetric, higher fidelity than Q4.
//
// Same block structure as Q4 (32 weights per block) but each weight gets a
// full signed byte instead of a nibble.  Block layout (34 bytes):
//   [0..2]:  bf16 scale (little-endian)
//   [2..34]: 32 × signed int8 values
//
// scale = max(|w_i|) / 127.0
// q_i = clamp(round(w_i / scale), -128, 127)
// GPU dequant: w_i = float(q_i) * scale  (signed char, no offset)
// ---------------------------------------------------------------------------

/// Quantise a bf16 weight matrix [m, k] to block-wise Q8.
pub(crate) fn quantize_bf16_to_q8(bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
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
    let mut out = vec![0u8; q8_byte_count(m, k)];

    for row in 0..m {
        for block in 0..blocks_per_row {
            let src_offset = row * k + block * 32;
            let dst_offset = (row * blocks_per_row + block) * 34;

            // Find max absolute value in the block for scale computation.
            let mut max_abs: f32 = 0.0;
            for i in 0..32 {
                let v = values[src_offset + i].to_f32().abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / scale;

            // Write scale as bf16 (2 bytes).
            let scale_bf16 = (scale.to_bits() >> 16) as u16;
            out[dst_offset..dst_offset + 2].copy_from_slice(&scale_bf16.to_le_bytes());

            // Quantise each weight to a signed byte.
            for i in 0..32 {
                let v = values[src_offset + i].to_f32();
                let q = ((v * inv_scale).round() as i32).clamp(-128, 127);
                out[dst_offset + 2 + i] = q as i8 as u8;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// FP8 E4M3 quantisation — NVIDIA Ada/Hopper (SM 89+).
//
// Unlike Q4/Q8 block formats, FP8 E4M3 stores one IEEE 8-bit float per
// weight with no block structure and no per-block scale.  The format is:
//   1 sign bit + 4 exponent bits (bias 7) + 3 mantissa bits
//   Range: ±448, min subnormal: 2^-9 ≈ 0.00195
//   Special: no infinity representation; NaN = 0x7F (S=0, E=1111, M=111)
//
// Conversion: bf16 → f32 → clamp to ±448 → repack as E4M3.
// Output: m * k bytes (one byte per weight).
//
// Related:
//   gpu/ops/quant.rs                — FP8Quantiser (WeightQuantiser impl)
//   gpu/cuda/shaders/matmul.cu      — matvec_fp8, gemm_fp8 CUDA kernels
//   gpu/cuda/shaders/matmul_tc.cu   — gemm_fp8_tc tensor-core kernel
//   gpu/metal/kernels/matmul.rs     — panic stub (FP8 not supported on Metal)
// ---------------------------------------------------------------------------

/// Convert a single f32 value to FP8 E4M3 format.
///
/// Algorithm:
///   1. Extract sign, exponent, mantissa from f32 (IEEE 754)
///   2. Handle special cases: zero, NaN/Inf, overflow (>448), underflow
///   3. Rebias exponent from 127 (f32) to 7 (E4M3)
///   4. Truncate mantissa from 23 bits to 3 bits with round-to-nearest-even
///   5. Pack as: [sign:1][exp:4][man:3]
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32; // biased exponent (bias 127)
    let man = bits & 0x7FFFFF; // 23-bit mantissa

    // Zero (positive or negative).
    if exp == 0 && man == 0 {
        return (sign << 7) as u8;
    }

    // NaN or Inf → FP8 NaN (0x7F).  E4M3 has no infinity representation.
    if exp == 255 {
        return 0x7F;
    }

    // Compute unbiased exponent.  f32 bias = 127, E4M3 bias = 7.
    let unbiased = if exp == 0 {
        // f32 subnormal: effective exponent is -126, value = 0.mantissa * 2^-126
        -126_i32
    } else {
        exp - 127
    };

    // Reconstruct the absolute value as a normalised mantissa for rebias.
    // For f32 normals: value = 1.mantissa * 2^unbiased
    // For f32 subnormals: value = 0.mantissa * 2^-126
    let abs_val = val.abs();

    // Overflow: |val| > 448 (max representable E4M3 value) → clamp to 448.
    // Max E4M3: S=0, E=1110, M=111 → (1 + 7/8) * 2^7 = 1.875 * 128 = 240
    // Wait: E4M3 max exponent is 1110 (14-7=7), max mantissa 111 (7/8).
    // max = (1 + 7/8) * 2^7 = 240.  But E=1111 with M<111 are also valid
    // normals (no inf in E4M3): E=1111,M=110 → (1+6/8)*2^8 = 448.
    // Actually: E=1111 (unbiased 8), M=110 → (1 + 6/8) * 2^8 = 1.75*256 = 448.
    // E=1111, M=111 is NaN.  So max normal = 448.
    if abs_val > 448.0 {
        // Max E4M3 value: sign | E=1111 | M=110 = 0x7E (positive)
        return ((sign << 7) | 0x7E) as u8;
    }

    // Underflow: too small for even the smallest E4M3 subnormal.
    // Smallest E4M3 subnormal: E=0000, M=001 → 0.125 * 2^(-6) = 2^-9 ≈ 0.00195
    let min_subnormal = f32::from_bits(0x3A00_0000); // 2^-9 * 0.5 (half for rounding)
    if abs_val < min_subnormal {
        return (sign << 7) as u8; // flush to signed zero
    }

    // General conversion: find the best E4M3 representation.
    // E4M3 biased exponent range: 1..15 (normal), 0 (subnormal).
    // E4M3 exponent bias = 7, so unbiased range: -6..8 (normal), -6 (subnormal with shift).
    let fp8_biased_exp = unbiased + 7; // rebias from f32 to E4M3

    if fp8_biased_exp >= 1 && fp8_biased_exp <= 15 {
        // Normal E4M3 value.
        // Round mantissa from 23 bits to 3 bits (round-to-nearest-even).
        let shift = 23 - 3; // 20 bits to drop
        let truncated = man >> shift;
        let remainder = man & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);

        let rounded = if remainder > halfway {
            truncated + 1
        } else if remainder == halfway {
            // Round to even.
            if truncated & 1 == 1 {
                truncated + 1
            } else {
                truncated
            }
        } else {
            truncated
        };

        // Mantissa overflow from rounding → increment exponent.
        let (final_exp, final_man) = if rounded >= 8 {
            (fp8_biased_exp + 1, 0u32)
        } else {
            (fp8_biased_exp, rounded)
        };

        // Check for overflow after rounding.
        if final_exp > 15 || (final_exp == 15 && final_man == 7) {
            // Overflow to max or NaN boundary → clamp to max.
            return ((sign << 7) | 0x7E) as u8;
        }

        ((sign << 7) | ((final_exp as u32) << 3) | final_man) as u8
    } else if fp8_biased_exp <= 0 {
        // Subnormal E4M3: E=0000, mantissa encodes 0.XXX * 2^(-6).
        // value = mantissa/8 * 2^(-6), so mantissa = round(value * 8 * 2^6)
        //       = round(value * 512)
        let subnormal_man = (abs_val * 512.0).round() as u32;
        let clamped = subnormal_man.min(7); // max subnormal mantissa
        if clamped == 0 {
            return (sign << 7) as u8; // too small, flush to zero
        }
        ((sign << 7) | clamped) as u8
    } else {
        // fp8_biased_exp > 15: overflow → max value.
        ((sign << 7) | 0x7E) as u8
    }
}

/// Quantise a bf16 weight matrix [m, k] to FP8 E4M3.
///
/// Unlike Q4/Q8, FP8 has no block structure: each weight is independently
/// converted to a 1-byte IEEE FP8 E4M3 value.  Output size = m * k bytes.
pub(crate) fn quantize_bf16_to_fp8(bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
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

    let mut out = vec![0u8; fp8_byte_count(m, k)];

    for i in 0..m * k {
        out[i] = f32_to_fp8_e4m3(values[i].to_f32());
    }

    out
}

// ---------------------------------------------------------------------------
// PinnedBuf — page-locked host memory for async DMA transfers.
//
// On CUDA, allocated via cuMemAllocHost — the physical pages are pinned
// so the DMA engine can transfer directly without staging through a
// bounce buffer.  This is required for true async HtoD transfers:
// unpinned memory silently falls back to synchronous copies.
//
// On Metal/CPU, pinned buffers are never allocated (alloc_pinned_buf
// returns None) because unified memory doesn't need them.
//
// Used by: model/expert_stream.rs (staging buffers for parallel pread)
// Allocated by: gpu/cuda/kernels/core.rs (GpuCore::alloc_pinned_buf)
// ---------------------------------------------------------------------------

pub(crate) struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
    /// Prevent CUDA context destruction before cuMemFreeHost.
    #[cfg(feature = "cuda")]
    _ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// Safety: the buffer is a raw allocation with no interior references.
// Ownership semantics are like Vec<u8> — the holder has exclusive access.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

#[allow(dead_code)] // CUDA-only: pinned host memory for async GPU transfers
impl PinnedBuf {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
        }
    }
}

// ---------------------------------------------------------------------------
// Factory function.
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", feature = "cuda"))]
pub(crate) fn create_backend() -> anyhow::Result<Backend> {
    Backend::new()
}

/// Create multiple GPU backends for tensor parallelism.
///
/// When `world_size == 1`, returns a single backend (no NCCL overhead).
/// When `world_size > 1`, initializes NCCL communicators and creates one
/// backend per device, each with its own CUDA context and stream.
#[cfg(feature = "cuda")]
pub(crate) fn create_backends(world_size: usize) -> anyhow::Result<Vec<Backend>> {
    if world_size == 1 {
        return Ok(vec![Backend::new()?]);
    }

    let comms = cuda::nccl::init_nccl_comms(world_size)?;

    (0..world_size)
        .map(|rank| Backend::new_with_device(rank, rank, world_size, Some(comms[rank].clone())))
        .collect()
}

#[cfg(not(any(target_os = "macos", feature = "cuda")))]
pub(crate) fn create_backend() -> anyhow::Result<cpu::CpuBackend> {
    eprintln!("warning: no GPU backend available, using CPU (slow)");
    Ok(cpu::CpuBackend)
}

/// Number of GPUs available on this system.
///
/// Returns the count without creating a full backend — used to resolve
/// `--tp auto` before spawning worker threads.  CUDA queries the driver
/// directly; Metal and CPU always return 1.
#[cfg(feature = "cuda")]
pub(crate) fn device_count() -> usize {
    cudarc::driver::CudaContext::device_count().unwrap_or(1) as usize
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn device_count() -> usize {
    1
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
        // 1 row, 32 elements = 1 block = 18 bytes
        assert_eq!(q4_byte_count(1, 32), 18);
    }

    #[test]
    fn test_q4_byte_count_multiple_blocks() {
        // 1 row, 64 elements = 2 blocks = 36 bytes
        assert_eq!(q4_byte_count(1, 64), 36);
    }

    #[test]
    fn test_q4_byte_count_multiple_rows() {
        // 4 rows, 64 elements each = 4 * 2 blocks * 18 = 144
        assert_eq!(q4_byte_count(4, 64), 144);
    }

    #[test]
    fn test_q4_byte_count_large() {
        // 2048 rows, 2048 cols = 2048 * 64 * 18 = 2359296
        assert_eq!(q4_byte_count(2048, 2048), 2048 * (2048 / 32) * 18);
    }

    #[test]
    #[should_panic(expected = "q4_byte_count overflow")]
    fn test_q4_byte_count_overflow_panics() {
        // Dimensions large enough to overflow usize — should panic with a
        // clear message instead of silently wrapping around.
        let _ = q4_byte_count(usize::MAX / 2, 64);
    }

    // Q8 byte count tests.

    #[test]
    #[should_panic(expected = "Q8 has variable byte size")]
    fn test_tensor_dtype_byte_size_q8_panics() {
        let _ = TensorDtype::Q8.byte_size();
    }

    #[test]
    fn test_q8_byte_count_single_block() {
        assert_eq!(q8_byte_count(1, 32), 34);
    }

    #[test]
    fn test_q8_byte_count_multiple_blocks() {
        assert_eq!(q8_byte_count(1, 64), 68);
    }

    #[test]
    fn test_q8_byte_count_multiple_rows() {
        assert_eq!(q8_byte_count(4, 64), 4 * 2 * 34);
    }

    #[test]
    fn test_q8_byte_count_large() {
        assert_eq!(q8_byte_count(2048, 2048), 2048 * (2048 / 32) * 34);
    }

    #[test]
    #[should_panic(expected = "q8_byte_count overflow")]
    fn test_q8_byte_count_overflow_panics() {
        let _ = q8_byte_count(usize::MAX / 2, 64);
    }

    // FP8 byte count and dtype tests.

    #[test]
    fn test_tensor_dtype_byte_size_fp8() {
        assert_eq!(TensorDtype::FP8.byte_size(), 1);
    }

    #[test]
    fn test_fp8_byte_count() {
        assert_eq!(fp8_byte_count(1, 32), 32);
        assert_eq!(fp8_byte_count(1, 64), 64);
        assert_eq!(fp8_byte_count(4, 64), 256);
        assert_eq!(fp8_byte_count(2048, 2048), 2048 * 2048);
    }

    #[test]
    #[should_panic(expected = "fp8_byte_count overflow")]
    fn test_fp8_byte_count_overflow_panics() {
        let _ = fp8_byte_count(usize::MAX, 2);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_zero() {
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_one() {
        // 1.0 = 1.000 * 2^0, E4M3: sign=0, exp=0+7=7=0111, man=000
        // Byte: 0_0111_000 = 0x38
        assert_eq!(f32_to_fp8_e4m3(1.0), 0x38);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_negative_one() {
        // -1.0: sign=1, exp=0111, man=000 → 1_0111_000 = 0xB8
        assert_eq!(f32_to_fp8_e4m3(-1.0), 0xB8);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_max_value() {
        // Max E4M3 = 448.0: sign=0, E=1111, M=110 → 0_1111_110 = 0x7E
        assert_eq!(f32_to_fp8_e4m3(448.0), 0x7E);
        // Values > 448 clamp to max.
        assert_eq!(f32_to_fp8_e4m3(500.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(1000.0), 0x7E);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_nan_inf() {
        assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x7F);
        assert_eq!(f32_to_fp8_e4m3(f32::INFINITY), 0x7F);
        assert_eq!(f32_to_fp8_e4m3(f32::NEG_INFINITY), 0x7F);
    }

    #[test]
    fn test_f32_to_fp8_e4m3_round_trip_accuracy() {
        // Test a range of values and verify round-trip accuracy.
        // FP8 E4M3 dequant: reconstruct f32 from the 8-bit pattern.
        fn fp8_to_f32(bits: u8) -> f32 {
            let sign = (bits >> 7) & 1;
            let exp = ((bits >> 3) & 0xF) as i32;
            let man = (bits & 0x7) as f32;
            if exp == 0xF && (bits & 0x7) == 0x7 {
                return f32::NAN; // NaN
            }
            let val = if exp == 0 {
                // Subnormal: 0.man * 2^(-6)
                (man / 8.0) * (2.0_f32).powi(-6)
            } else {
                // Normal: (1 + man/8) * 2^(exp - 7)
                (1.0 + man / 8.0) * (2.0_f32).powi(exp - 7)
            };
            if sign == 1 { -val } else { val }
        }

        // Values within FP8 E4M3 representable range (min subnormal ~0.00195).
        // Very small values like 0.001 are below the smallest subnormal and
        // cannot round-trip accurately.
        let test_values = [0.5, 1.0, 2.0, -3.0, 0.125, 100.0, -200.0, 0.25];
        for &v in &test_values {
            let fp8 = f32_to_fp8_e4m3(v);
            let reconstructed = fp8_to_f32(fp8);
            if reconstructed.is_nan() {
                continue;
            }
            // Relative error should be reasonable for 3-bit mantissa.
            let rel_err = ((v - reconstructed) / v).abs();
            assert!(
                rel_err < 0.2,
                "FP8 round-trip error too large for {v}: got {reconstructed}, rel_err={rel_err}"
            );
        }
    }
}
