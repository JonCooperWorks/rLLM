// ---------------------------------------------------------------------------
// GpuMatmul — matrix multiplication kernels.
//
// The single most performance-critical ops.  LLM inference is memory-
// bandwidth-bound: each token's forward pass does ~145 matmuls, each
// reading the entire weight matrix.  `matmul` is the single-token
// mat-vec; `matmul_batch` is the GEMM used during prefill.
// Both auto-detect Q4 vs BF16 weights and dispatch the right kernel.
//
// Metal shaders: shaders/matmul.metal
// Metal impl:    gpu/metal/kernels/matmul.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuMatmul: GpuCore {
    /// Matrix-vector multiply: out[i] = dot(weight[i, :], input)
    fn matmul(
        &self,
        weight: &Self::Tensor,
        input: &Self::Tensor,
        out: &Self::Tensor,
        m: u32,
        k: u32,
    );

    /// Batched GEMM: out = input @ weight^T for batch_size input rows.
    fn matmul_batch(
        &self,
        weight: &Self::Tensor,
        input: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        m: u32,
        k: u32,
    );
}
