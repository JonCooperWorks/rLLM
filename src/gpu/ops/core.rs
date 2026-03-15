// ---------------------------------------------------------------------------
// GpuCore — device management, tensor memory, and synchronisation.
//
// This is the foundation trait that all other Gpu* traits extend.  It owns
// the associated `Tensor` type and provides the basic infrastructure every
// backend needs: device queries, tensor allocation/upload/download, and
// the flush/submit sync primitives that control when the CPU blocks on
// GPU work.
//
// Metal impl: gpu/metal/kernels/core.rs
// ---------------------------------------------------------------------------

use super::super::TensorDtype;

pub(crate) trait GpuCore: Send + Sync {
    /// Opaque tensor handle.  Each backend defines its own type wrapping
    /// the platform's buffer handle, tensor shape, and dtype metadata.
    type Tensor;

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

    /// Return the total byte count of a tensor's data.
    fn tensor_byte_count(&self, tensor: &Self::Tensor) -> usize;
}
