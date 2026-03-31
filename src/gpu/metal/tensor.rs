// ---------------------------------------------------------------------------
// MetalTensor — the backend's opaque tensor type.
//
// Wraps a Metal buffer plus shape and dtype metadata.  The buffer is
// allocated with `StorageModeShared` (unified memory), so the CPU can
// read/write it directly via `buffer.contents()` and the GPU accesses
// the same physical memory.
// ---------------------------------------------------------------------------

use crate::gpu::TensorDtype;

pub(crate) struct MetalTensor {
    pub buffer: metal::Buffer,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

impl MetalTensor {
    pub fn byte_count(&self) -> usize {
        match self.dtype {
            TensorDtype::Q4 => {
                assert!(self.shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                crate::gpu::q4_byte_count(self.shape[0], self.shape[1])
            }
            TensorDtype::Q8 => {
                assert!(self.shape.len() == 2, "Q8 tensors must be 2D [m, k]");
                crate::gpu::q8_byte_count(self.shape[0], self.shape[1])
            }
            TensorDtype::TQ3 => {
                assert!(self.shape.len() == 2, "TQ3 tensors must be 2D [m, k]");
                crate::gpu::tq3_byte_count(self.shape[0], self.shape[1])
            }
            _ => self.shape.iter().product::<usize>() * self.dtype.byte_size(),
        }
    }
}
