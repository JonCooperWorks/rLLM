// ---------------------------------------------------------------------------
// Metal impl: GpuCore — tensor lifecycle and command buffer control.
//
// Trait contract: gpu/ops/core.rs
//
// This is the foundation — every other kernel file depends on MetalBackend
// having a Tensor type and the ability to alloc/upload/copy tensors.
//
// Memory model: all buffers are StorageModeShared (CPU+GPU visible), which
// lets us avoid explicit blit copies on Apple Silicon's unified memory.
// copy_to_host() calls flush() first to ensure all queued GPU work completes
// before the CPU reads the buffer contents.
// ---------------------------------------------------------------------------

use metal::MTLResourceOptions;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuCore;
use crate::gpu::TensorDtype;

impl GpuCore for MetalBackend {
    type Tensor = MetalTensor;

    fn device_name(&self) -> &str {
        &self.name
    }

    fn recommended_max_memory(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }

    fn flush(&self) {
        MetalBackend::flush(self);
    }

    fn submit(&self) {
        MetalBackend::submit(self);
    }

    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> MetalTensor {
        let byte_count = match dtype {
            TensorDtype::Q4 => {
                assert!(shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                crate::gpu::q4_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        let buffer = self
            .device
            .new_buffer(byte_count as u64, MTLResourceOptions::StorageModeShared);
        MetalTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> MetalTensor {
        let expected = match dtype {
            TensorDtype::Q4 => {
                assert!(shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                crate::gpu::q4_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        assert_eq!(
            data.len(),
            expected,
            "upload_tensor: data length {} != expected {}",
            data.len(),
            expected
        );
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn copy_to_tensor(&self, tensor: &MetalTensor, src: &[u8]) {
        let byte_count = tensor.byte_count();
        assert!(
            src.len() <= byte_count,
            "copy_to_tensor: src too large ({} > {})",
            src.len(),
            byte_count
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                tensor.buffer.contents() as *mut u8,
                src.len(),
            );
        }
    }

    fn tensor_byte_count(&self, tensor: &MetalTensor) -> usize {
        tensor.byte_count()
    }

    fn copy_to_host(&self, tensor: &MetalTensor, dst: &mut [u8]) {
        self.flush();

        let byte_count = tensor.byte_count();
        assert!(
            dst.len() >= byte_count,
            "copy_to_host: dst too small ({} < {})",
            dst.len(),
            byte_count
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor.buffer.contents() as *const u8,
                dst.as_mut_ptr(),
                byte_count,
            );
        }
    }
}
