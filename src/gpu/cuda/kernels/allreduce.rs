// ---------------------------------------------------------------------------
// CUDA impl: GpuAllReduce — no-op stubs for single-GPU inference.
//
// Future: NCCL integration for multi-GPU tensor parallelism.
//   ncclAllReduce (ring or tree algorithm over NVLink/PCIe) would replace
//   these no-ops.  The NCCL communicator would be initialised at startup
//   and stored in CudaBackend.
//
// Trait contract: gpu/ops/allreduce.rs
// ---------------------------------------------------------------------------

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::ops::GpuAllReduce;

impl GpuAllReduce for CudaBackend {
    fn all_reduce_sum(&self, _tensor: &CudaTensor, _size: u32) {
        // No-op: single GPU, nothing to reduce.
    }

    fn all_gather(
        &self,
        _tensor: &CudaTensor,
        _output: &CudaTensor,
        _local_size: u32,
        _full_size: u32,
    ) {
        // No-op: single GPU, tensor is already the full result.
    }
}
