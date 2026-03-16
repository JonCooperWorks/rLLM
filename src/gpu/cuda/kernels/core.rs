// ---------------------------------------------------------------------------
// CUDA impl: GpuCore — tensor lifecycle and stream control.
//
// Trait contract: gpu/ops/core.rs
//
// This is the foundation — every other kernel file depends on CudaBackend
// having a Tensor type and the ability to alloc/upload/copy tensors.
//
// Memory model: all buffers live in device (HBM) memory.  Unlike Metal's
// unified memory, CUDA requires explicit host↔device copies.  `flush()`
// synchronises the stream, ensuring all queued GPU work completes before
// the CPU reads device data.
//
// ---------------------------------------------------------------------------

use cudarc::driver::DevicePtr;

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuCore;
use crate::gpu::TensorDtype;

impl GpuCore for CudaBackend {
    type Tensor = CudaTensor;

    fn device_name(&self) -> &str {
        &self.name
    }

    fn recommended_max_memory(&self) -> u64 {
        // Use ~90% of total device memory (reserve some for CUDA runtime).
        (self.device_total_memory() as f64 * 0.9) as u64
    }

    fn flush(&self) {
        self.stream.synchronize().expect("CUDA stream synchronize failed");
    }

    fn submit(&self) {
        // CUDA stream submissions are already committed to the GPU immediately.
        // No explicit commit needed (unlike Metal's command buffer model).
    }

    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> CudaTensor {
        let byte_count = match dtype {
            TensorDtype::Q4 => {
                assert!(shape.len() == 2, "Q4 tensors must be 2D [m, k]");
                crate::gpu::q4_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        let buf = self.stream.alloc_zeros::<u8>(byte_count)
            .expect("CUDA alloc_zeros failed");
        CudaTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> CudaTensor {
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
        let buf = self.stream.clone_htod(data)
            .expect("CUDA clone_htod failed");
        CudaTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn copy_to_tensor(&self, tensor: &CudaTensor, src: &[u8]) {
        let byte_count = tensor.byte_count();
        assert!(
            src.len() <= byte_count,
            "copy_to_tensor: src too large ({} > {})",
            src.len(),
            byte_count
        );
        // Use raw driver API since we have &CudaTensor (immutable) but need
        // to write to the device buffer.  This is safe because:
        //   1. We're the only writer (single stream, serialised dispatch)
        //   2. The device pointer itself is unchanged (only data at that address)
        let (dptr, _sync) = tensor.buf.device_ptr(&self.stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dptr, src, self.stream.cu_stream())
        }.expect("CUDA memcpy_htod failed");
    }

    fn tensor_byte_count(&self, tensor: &CudaTensor) -> usize {
        tensor.byte_count()
    }

    fn copy_to_host(&self, tensor: &CudaTensor, dst: &mut [u8]) {
        // Synchronise to ensure all GPU work is complete before reading.
        self.flush();

        let byte_count = tensor.byte_count();
        assert!(
            dst.len() >= byte_count,
            "copy_to_host: dst too small ({} < {})",
            dst.len(),
            byte_count
        );
        self.stream.memcpy_dtoh(&tensor.buf, &mut dst[..byte_count])
            .expect("CUDA memcpy_dtoh failed");
    }

    // quantize_upload: uses the default CPU quantization from GpuCore.
    // GPU-side quantization was considered but doubles peak VRAM during loading
    // (holds bf16 temp + Q4 output simultaneously), which causes OOM on models
    // that nearly fill device memory.  CPU quantize + upload-Q4 only is safer.
}
