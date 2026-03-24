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
pub(crate) mod parallel;

pub(crate) use ops::{
    GpuAllReduce, GpuAttention, GpuCore, GpuDeltaNet, GpuElementwise, GpuEmbed, GpuMatmul, GpuMoe,
    GpuNorm, GpuRope, GpuVision,
};

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
    + GpuAllReduce
    + GpuMoe
    + GpuVision
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
        + GpuAllReduce
        + GpuMoe
        + GpuVision
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
///
/// Panics on overflow instead of silently wrapping around, which would cause
/// undersized GPU buffer allocations and memory corruption.
pub(crate) fn q4_byte_count(m: usize, k: usize) -> usize {
    let blocks_per_row = k / 32;
    m.checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(18))
        .expect("q4_byte_count overflow: tensor dimensions too large")
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
}
