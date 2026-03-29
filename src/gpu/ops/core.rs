// ---------------------------------------------------------------------------
// GpuCore — device management, tensor memory, synchronisation, and
// weight quantisation.
//
// This is the foundation trait that all other Gpu* traits extend.  It owns
// the associated `Tensor` type and provides the basic infrastructure every
// backend needs: device queries, tensor allocation/upload/download, and
// the flush/submit sync primitives that control when the CPU blocks on
// GPU work.
//
// Quantisation:
//   `quantize_upload` converts bf16 weight data into the backend's preferred
//   quantised format and uploads it in one step.  The default implementation
//   uses the Q4 block format (see gpu/mod.rs::quantize_bf16_to_q4).  A CUDA
//   backend could override this to produce CUTLASS INT4 or any other layout
//   that its matmul kernels understand.  The loader calls this method instead
//   of doing format-specific quantisation itself — so weight loading is
//   backend-agnostic.
//
// Metal impl: gpu/metal/kernels/core.rs
// ---------------------------------------------------------------------------

use super::super::TensorDtype;

pub(crate) trait GpuCore: Send + Sync {
    /// Opaque tensor handle.  Each backend defines its own type wrapping
    /// the platform's buffer handle, tensor shape, and dtype metadata.
    ///
    /// Send + Sync: tensor handles are safe to share across threads (they're
    /// just IDs/pointers — the GPU manages the underlying memory).  Required
    /// for ModelForward structs that hold arch-specific tensor buffers.
    type Tensor: Send + Sync;

    /// Human-readable GPU device name (e.g. "Apple M4 Max").
    fn device_name(&self) -> &str;

    /// Maximum recommended GPU working set size in bytes.
    fn recommended_max_memory(&self) -> u64;

    /// Wait for all pending GPU work to complete.
    fn flush(&self);

    /// Submit pending GPU work without waiting for completion.
    fn submit(&self);

    // --- Memory management ---

    /// Allocate an uninitialised tensor on the GPU.
    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> Self::Tensor;

    /// Allocate a tensor and copy `data` (raw bytes) from the host into it.
    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> Self::Tensor;

    /// Copy tensor contents from GPU to a host byte buffer.
    fn copy_to_host(&self, tensor: &Self::Tensor, dst: &mut [u8]);

    /// Copy raw bytes from the host into an existing GPU tensor.
    fn copy_to_tensor(&self, tensor: &Self::Tensor, src: &[u8]);

    /// Copy host data into an existing GPU tensor, avoiding buffer allocation.
    /// The data length must match the tensor's byte count.
    fn copy_to_tensor_from_host(&self, src: &[u8], dst: &Self::Tensor);

    /// Return the total byte count of a tensor's data.
    fn tensor_byte_count(&self, tensor: &Self::Tensor) -> usize;

    /// Quantise bf16 weight data and upload to the GPU.
    ///
    /// Takes raw bf16 bytes for a 2D weight matrix [m, k] and returns a
    /// quantised tensor in whatever format this backend's matmul kernels
    /// expect.  The default implementation uses the Q4 block format
    /// (see `gpu::quantize_bf16_to_q4`).
    ///
    /// Backends with a different quantisation format (e.g. CUDA with
    /// CUTLASS INT4) override this to produce their own layout.
    #[allow(dead_code)] // used by CUDA backend; Metal uses pre-quantized weights
    fn quantize_upload(&self, bf16_data: &[u8], shape: &[usize]) -> Self::Tensor {
        assert!(
            shape.len() == 2 && shape[1] % 32 == 0,
            "quantize_upload requires 2D shape with K divisible by 32, got {shape:?}"
        );
        let q4_data = super::super::quantize_bf16_to_q4(bf16_data, shape[0], shape[1]);
        self.upload_tensor(&q4_data, shape, TensorDtype::Q4)
    }

    /// Quantise bf16 weight data to Q8 and upload to the GPU.
    ///
    /// Same as `quantize_upload` but produces Q8 (8-bit) blocks instead of Q4.
    /// Used when loading Q8-quantized models.
    #[allow(dead_code)] // used by CUDA backend; Metal uses pre-quantized weights
    fn quantize_upload_q8(&self, bf16_data: &[u8], shape: &[usize]) -> Self::Tensor {
        assert!(
            shape.len() == 2 && shape[1] % 32 == 0,
            "quantize_upload_q8 requires 2D shape with K divisible by 32, got {shape:?}"
        );
        let q8_data = super::super::quantize_bf16_to_q8(bf16_data, shape[0], shape[1]);
        self.upload_tensor(&q8_data, shape, TensorDtype::Q8)
    }

    /// Copy a contiguous region of bytes between two GPU tensors.
    ///
    /// This is the primitive that enables batched decode: when we have a
    /// [batch_size, dim] tensor holding multiple sequences' data, we need to
    /// extract individual rows into single-token scratch buffers for per-sequence
    /// attention, then write the attention output back into the batched tensor.
    ///
    /// Example: extracting row 3 of a [8, 4096] bf16 Q buffer:
    ///   src_byte_offset = 3 * 4096 * 2 = 24576
    ///   dst_byte_offset = 0
    ///   byte_count = 4096 * 2 = 8192
    ///
    /// Metal: MTLBlitCommandEncoder copy (GPU-side, async, zero CPU involvement).
    /// CUDA:  cuMemcpyDtoDAsync on the backend's stream.
    /// CPU:   simple slice copy.
    fn copy_tensor_region(
        &self,
        src: &Self::Tensor,
        src_byte_offset: usize,
        dst: &Self::Tensor,
        dst_byte_offset: usize,
        byte_count: usize,
    );

    /// Copy raw bytes from host into a GPU tensor on a dedicated transfer
    /// stream, returning immediately.  The transfer may still be in flight.
    ///
    /// On CUDA, this uses a separate stream so the DMA engine works in
    /// parallel with compute kernels.  On Metal/CPU, this falls back to
    /// the synchronous `copy_to_tensor`.
    ///
    /// Callers must call `sync_transfers()` before reading the tensor data
    /// from the compute stream.
    ///
    /// CUDA impl: gpu/cuda/kernels/core.rs
    fn copy_to_tensor_async(&self, tensor: &Self::Tensor, src: &[u8]) {
        self.copy_to_tensor(tensor, src);
    }

    /// Block the compute stream until all prior `copy_to_tensor_async`
    /// transfers have completed.
    ///
    /// On CUDA: records an event on the transfer stream, then makes the
    /// compute stream wait on that event.  Both operations are GPU-side
    /// only — the CPU does not block.
    ///
    /// On Metal/CPU: no-op (async transfers fall back to synchronous).
    ///
    /// CUDA impl: gpu/cuda/kernels/core.rs
    fn sync_transfers(&self) {
        // Default: transfers are synchronous, nothing to wait for.
    }

    /// Get a mutable byte pointer to tensor data for direct CPU writes.
    ///
    /// Returns `Some` for backends with CPU-accessible unified memory (Metal,
    /// CPU) where the GPU buffer can be written directly by the host.
    /// Returns `None` for backends requiring explicit DMA transfers (CUDA
    /// with device-private memory).
    ///
    /// Used by expert streaming to pread() directly into GPU buffer contents,
    /// eliminating the intermediate CPU buffer + memcpy overhead.
    ///
    /// Safety: the returned pointer is valid for `tensor_byte_count()` bytes.
    /// Caller must ensure no GPU work is reading the tensor concurrently.
    fn tensor_mut_ptr(&self, _tensor: &Self::Tensor) -> Option<*mut u8> {
        None
    }

    /// Allocate a pinned (page-locked) host buffer for async DMA.
    ///
    /// Returns `Some(PinnedBuf)` on backends with discrete memory (CUDA)
    /// where pinned buffers enable true async transfers via `cuMemAllocHost`.
    /// Returns `None` on unified memory backends (Metal, CPU) where pinned
    /// buffers aren't needed.
    ///
    /// Pinned allocation is expensive (~1ms per call), so callers should
    /// allocate once at init and reuse.
    ///
    /// Used by: model/expert_stream.rs (staging buffers for parallel pread)
    /// CUDA impl: gpu/cuda/kernels/core.rs
    fn alloc_pinned_buf(&self, _byte_count: usize) -> Option<super::super::PinnedBuf> {
        None
    }

    /// Estimate the byte count of a 2D weight [m, k] after quantisation.
    ///
    /// Used by config.rs to predict GPU memory usage before loading weights.
    /// Must match whatever format `quantize_upload` produces — the default
    /// returns the Q4 byte count.  Backends with a different format override
    /// both methods together.
    fn quantized_weight_bytes(&self, m: usize, k: usize) -> usize {
        super::super::q4_byte_count(m, k)
    }
}
