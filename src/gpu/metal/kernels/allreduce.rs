// ---------------------------------------------------------------------------
// Metal impl: GpuAllReduce — no-op stubs for single-GPU inference.
//
// LEARNING OVERVIEW
//
// Why no-ops?
//   Apple Silicon Macs have a single GPU.  True tensor parallelism requires
//   multiple GPUs communicating over a bus (NVLink, PCIe).  Metal on a
//   single device has no peer to communicate with.
//
// Future: simulated multi-rank on unified memory.
//   For correctness testing, we could partition a single buffer into
//   logical rank regions and run a reduction kernel across them.  This
//   exercises the kernel logic without real multi-GPU hardware.  See
//   gpu/parallel.rs for the design.
//
// Trait contract: gpu/ops/allreduce.rs
// ---------------------------------------------------------------------------

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::ops::GpuAllReduce;

impl GpuAllReduce for MetalBackend {
    fn all_reduce_sum(&self, _tensor: &MetalTensor, _size: u32) {
        // No-op: single GPU, nothing to reduce.
    }

    fn all_gather(
        &self,
        _tensor: &MetalTensor,
        _output: &MetalTensor,
        _local_size: u32,
        _full_size: u32,
    ) {
        // No-op: single GPU, tensor is already the full result.
    }
}
