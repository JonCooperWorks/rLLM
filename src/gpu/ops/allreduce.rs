// ---------------------------------------------------------------------------
// GpuAllReduce — collective communication for tensor parallelism.
//
// LEARNING OVERVIEW
//
// Why this exists:
//   In tensor parallelism, each GPU computes a PARTIAL result for certain
//   operations (row-split matmuls like o_proj and down_proj).  AllReduce
//   sums these partial results across GPUs so every GPU has the correct
//   full output.
//
// When is it called?
//   Exactly twice per transformer layer:
//     1. After o_proj matmul (attention output projection) — each GPU has
//        a partial hidden-state contribution from its subset of heads.
//     2. After down_proj matmul (FFN down projection) — each GPU has a
//        partial hidden-state contribution from its slice of intermediate dim.
//
// Why a no-op for world_size=1?
//   Single-GPU inference is the common case.  Making AllReduce a no-op
//   means we can insert the calls unconditionally in primitives without
//   adding any overhead to single-GPU users.  The compiler can inline-
//   eliminate empty function bodies entirely.
//
// Future implementations:
//   CUDA: NCCL's ncclAllReduce (ring or tree algorithm over NVLink/PCIe).
//   Metal: Simulated via buffer-region reduction on unified memory (useful
//          for correctness testing of the kernel logic, but Apple Silicon
//          has a single GPU so there's no real peer to communicate with).
//
// Related files:
//   Trait organisation:    gpu/ops/mod.rs
//   Sharding plan:         gpu/parallel.rs
//   Primitives (callers):  model/primitives.rs
//   Metal no-op impl:      gpu/metal/kernels/allreduce.rs
//   CUDA stub impl:        gpu/cuda/kernels/allreduce.rs
//   CPU impl:              gpu/cpu/mod.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuAllReduce: GpuCore {
    /// In-place sum-reduce `tensor` (first `size` elements) across all ranks.
    ///
    /// After this call every rank holds the element-wise sum of all ranks'
    /// partial results.  Used after row-split matmuls (o_proj, down_proj)
    /// where each rank computed a partial output.
    ///
    /// No-op when world_size=1.
    fn all_reduce_sum(&self, tensor: &Self::Tensor, size: u32);

    /// Gather shards from all ranks into a single contiguous output.
    ///
    /// Each rank contributes `local_size` elements; the result is
    /// `full_size = local_size * world_size` elements in `output`.
    /// Used for vocab-parallel logits where each rank computes a slice
    /// of the vocabulary.
    ///
    /// No-op when world_size=1 (tensor and output are the same buffer).
    #[allow(dead_code)] // CUDA multi-GPU path only
    fn all_gather(
        &self,
        tensor: &Self::Tensor,
        output: &Self::Tensor,
        local_size: u32,
        full_size: u32,
    );
}
