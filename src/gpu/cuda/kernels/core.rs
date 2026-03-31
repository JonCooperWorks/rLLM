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
use crate::gpu::TensorDtype;
use crate::gpu::ops::GpuCore;

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
        self.stream
            .synchronize()
            .expect("CUDA stream synchronize failed");
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
            TensorDtype::Q8 => {
                assert!(shape.len() == 2, "Q8 tensors must be 2D [m, k]");
                crate::gpu::q8_byte_count(shape[0], shape[1])
            }
            TensorDtype::TQ3 => {
                assert!(shape.len() == 2, "TQ3 tensors must be 2D [m, k]");
                crate::gpu::tq3_byte_count(shape[0], shape[1])
            }
            _ => shape.iter().product::<usize>() * dtype.byte_size(),
        };
        let buf = self
            .stream
            .alloc_zeros::<u8>(byte_count)
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
            TensorDtype::Q8 => {
                assert!(shape.len() == 2, "Q8 tensors must be 2D [m, k]");
                crate::gpu::q8_byte_count(shape[0], shape[1])
            }
            TensorDtype::TQ3 => {
                assert!(shape.len() == 2, "TQ3 tensors must be 2D [m, k]");
                crate::gpu::tq3_byte_count(shape[0], shape[1])
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
        let buf = self
            .stream
            .clone_htod(data)
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
        // Bind this device's CUDA context to the calling thread.
        // Required for multi-GPU: the main thread may call this for any device,
        // and the raw memcpy needs the correct context to be active.
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");

        // Use raw driver API since we have &CudaTensor (immutable) but need
        // to write to the device buffer.  This is safe because:
        //   1. We're the only writer (single stream, serialised dispatch)
        //   2. The device pointer itself is unchanged (only data at that address)
        let (dptr, _sync) = tensor.buf.device_ptr(&self.stream);
        unsafe { cudarc::driver::result::memcpy_htod_async(dptr, src, self.stream.cu_stream()) }
            .expect("CUDA memcpy_htod failed");
    }

    fn tensor_byte_count(&self, tensor: &CudaTensor) -> usize {
        tensor.byte_count()
    }

    fn copy_tensor_region(
        &self,
        src: &CudaTensor,
        src_byte_offset: usize,
        dst: &CudaTensor,
        dst_byte_offset: usize,
        byte_count: usize,
    ) {
        // Device-to-device async copy on the backend's CUDA stream.
        //
        // CUDA streams provide ordering guarantees: all prior kernel launches
        // on this stream will complete before this copy starts, and the copy
        // will complete before any subsequent kernels.  This makes it safe to
        // extract rows from a batched tensor right after a GEMM wrote to it.
        //
        // Unlike Metal's unified memory, CUDA device memory is NOT CPU-visible,
        // so we must use the driver's DtoD memcpy (no CPU roundtrip).
        //
        // We use raw device pointers (u64) with arithmetic because cudarc's
        // DevicePtr doesn't expose offset methods — same pattern as allreduce.rs.
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");

        let (src_dptr, _s) =
            cudarc::driver::DevicePtr::<u8>::device_ptr(&src.buf, &self.stream);
        let (dst_dptr, _d) =
            cudarc::driver::DevicePtr::<u8>::device_ptr(&dst.buf, &self.stream);
        let src_addr = src_dptr as u64 + src_byte_offset as u64;
        let dst_addr = dst_dptr as u64 + dst_byte_offset as u64;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst_addr,
                src_addr,
                byte_count,
                self.stream.cu_stream(),
            )
        }
        .expect("CUDA memcpy_dtod_async failed");
    }

    fn copy_to_tensor_from_host(&self, src: &[u8], dst: &CudaTensor) {
        // Bind context for multi-GPU support.
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");

        let byte_count = src.len();
        assert!(
            dst.byte_count() >= byte_count,
            "copy_to_tensor_from_host: dst too small ({} < {})",
            dst.byte_count(),
            byte_count
        );

        let (dst_dptr, _sync) = dst.buf.device_ptr(&self.stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dst_dptr,
                src,
                self.stream.cu_stream(),
            )
        }
        .expect("CUDA memcpy_htod_async failed");
    }

    fn copy_to_host(&self, tensor: &CudaTensor, dst: &mut [u8]) {
        // Bind context for multi-GPU support.
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");
        // Synchronise to ensure all GPU work is complete before reading.
        self.flush();

        let byte_count = tensor.byte_count();
        assert!(
            dst.len() >= byte_count,
            "copy_to_host: dst too small ({} < {})",
            dst.len(),
            byte_count
        );
        self.stream
            .memcpy_dtoh(&tensor.buf, &mut dst[..byte_count])
            .expect("CUDA memcpy_dtoh failed");
    }

    // quantize_upload: uses the default CPU quantization from GpuCore.
    // GPU-side quantization was considered but doubles peak VRAM during loading
    // (holds bf16 temp + Q4 output simultaneously), which causes OOM on models
    // that nearly fill device memory.  CPU quantize + upload-Q4 only is safer.

    fn alloc_pinned_buf(&self, byte_count: usize) -> Option<crate::gpu::PinnedBuf> {
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");

        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result =
            unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, byte_count) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return None;
        }
        Some(crate::gpu::PinnedBuf {
            ptr: ptr as *mut u8,
            len: byte_count,
            _ctx: self.ctx.clone(),
        })
    }

    fn copy_to_tensor_async(&self, tensor: &CudaTensor, src: &[u8]) {
        let byte_count = tensor.byte_count();
        assert!(
            src.len() <= byte_count,
            "copy_to_tensor_async: src too large ({} > {})",
            src.len(),
            byte_count
        );
        // Bind context for multi-GPU safety (same pattern as copy_to_tensor).
        self.ctx
            .bind_to_thread()
            .expect("CUDA bind_to_thread failed");

        // Enqueue HtoD on the dedicated transfer stream (not the compute stream).
        // For true async behavior, `src` must point to pinned (page-locked) memory.
        // Unpinned memory silently falls back to synchronous — still correct, just
        // slower.  Expert streaming allocates pinned buffers via alloc_pinned_buf().
        let (dptr, _sync) = tensor.buf.device_ptr(&self.transfer_stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                dptr,
                src,
                self.transfer_stream.cu_stream(),
            )
        }
        .expect("CUDA async memcpy_htod on transfer stream failed");
    }

    fn sync_transfers(&self) {
        // Record event on transfer stream — marks when all prior transfers complete.
        // Then make compute stream wait on that event.  Both operations are GPU-side
        // only — the CPU returns immediately.  The compute stream will stall only if
        // it reaches this wait point before the DMA engine finishes.
        self.transfer_event
            .record(&self.transfer_stream)
            .expect("CUDA event record failed");
        self.stream
            .wait(&self.transfer_event)
            .expect("CUDA stream wait failed");
    }
}
