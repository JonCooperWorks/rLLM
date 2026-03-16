// ---------------------------------------------------------------------------
// CudaTensor — the CUDA backend's opaque tensor type.
//
// Wraps a cudarc device buffer plus shape and dtype metadata.  Unlike Metal's
// unified memory model, CUDA uses discrete host/device memory — all tensor
// data lives in device (GPU) memory and requires explicit copies for host
// access via `cuMemcpyDtoH` / `cuMemcpyHtoD`.
//
// The buffer is stored as `CudaSlice<u8>` (byte-typed) because tensors may
// hold bf16, f32, or Q4 data — the kernel dispatch layer reinterprets the
// raw bytes based on `dtype`.
// ---------------------------------------------------------------------------

use crate::gpu::TensorDtype;
use cudarc::driver::CudaSlice;

pub(crate) struct CudaTensor {
    pub buf: CudaSlice<u8>,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

impl CudaTensor {
    pub fn byte_count(&self) -> usize {
        match self.dtype {
            TensorDtype::Q4 => {
                assert!(self.shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                crate::gpu::q4_byte_count(self.shape[0], self.shape[1])
            }
            _ => self.shape.iter().product::<usize>() * self.dtype.byte_size(),
        }
    }
}
