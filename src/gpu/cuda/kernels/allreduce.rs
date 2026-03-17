// ---------------------------------------------------------------------------
// CUDA impl: GpuAllReduce — NCCL-backed collective operations for
// multi-GPU tensor parallelism.
//
// When nccl_comm is Some (multi-GPU), uses NCCL AllReduce/AllGather via
// the raw result API.  When nccl_comm is None (single GPU), these are no-ops.
//
// We use the raw NCCL API (cudarc::nccl::result) instead of the safe Comm
// wrapper because:
//   1. Our tensors are CudaSlice<u8> (byte-typed, type-erased) but NCCL
//      needs bf16 element semantics.
//   2. The GpuAllReduce trait takes &Self::Tensor (shared reference) for
//      in-place operations, but the safe API requires &mut for recv buffers.
//
// Trait contract: gpu/ops/allreduce.rs
// ---------------------------------------------------------------------------

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuAllReduce;

impl GpuAllReduce for CudaBackend {
    fn all_reduce_sum(&self, tensor: &CudaTensor, size: u32) {
        let Some(ref nccl) = self.nccl_comm else {
            return; // Single GPU — no-op.
        };

        // Get the raw device pointer from the tensor buffer.
        let (dptr, _guard) = cudarc::driver::DevicePtr::<u8>::device_ptr(
            &tensor.buf, &self.stream,
        );

        // In-place AllReduce: src == dst, count = number of bf16 elements.
        unsafe {
            cudarc::nccl::result::all_reduce(
                dptr as *const std::ffi::c_void,
                dptr as *mut std::ffi::c_void,
                size as usize,
                cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
                cudarc::nccl::sys::ncclRedOp_t::ncclSum,
                nccl.comm,
                self.stream.cu_stream() as cudarc::nccl::sys::cudaStream_t,
            )
            .expect("NCCL AllReduce failed");
        }
    }

    fn all_gather(
        &self,
        tensor: &CudaTensor,
        output: &CudaTensor,
        local_size: u32,
        _full_size: u32,
    ) {
        let Some(ref nccl) = self.nccl_comm else {
            return; // Single GPU — no-op.
        };

        let (src_ptr, _guard1) = cudarc::driver::DevicePtr::<u8>::device_ptr(
            &tensor.buf, &self.stream,
        );
        let (dst_ptr, _guard2) = cudarc::driver::DevicePtr::<u8>::device_ptr(
            &output.buf, &self.stream,
        );

        unsafe {
            cudarc::nccl::result::all_gather(
                src_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                local_size as usize,
                cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
                nccl.comm,
                self.stream.cu_stream() as cudarc::nccl::sys::cudaStream_t,
            )
            .expect("NCCL AllGather failed");
        }
    }
}
