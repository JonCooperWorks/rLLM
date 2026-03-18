// ---------------------------------------------------------------------------
// Metal impl: GpuMatmul — matrix-vector and batched GEMM kernels.
//
// Trait contract: gpu/ops/matmul.rs
// Metal shader:   metal/shaders/matmul.metal
//
// Two code paths based on weight dtype:
//   - BF16: standard SIMD-cooperative matvec (32 threads per output row,
//     simd_sum reduction, 4x unrolled inner loop)
//   - Q4: dequantize-on-the-fly from 4-bit blocks (20 bytes per 32 weights)
//
// Batched variants (gemm) are used during prefill; single-vector (matvec)
// during token-by-token decode.  Both dispatch 256-thread groups with 32
// threads collaborating per output row via SIMD shuffle.
// ---------------------------------------------------------------------------

use metal::MTLSize;

use super::super::backend::MetalBackend;
use super::super::tensor::MetalTensor;
use crate::gpu::TensorDtype;
use crate::gpu::ops::GpuMatmul;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatvecParams {
    m: u32,
    k: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    batch_size: u32,
    m: u32,
    k: u32,
}

impl GpuMatmul for MetalBackend {
    fn matmul(&self, weight: &MetalTensor, input: &MetalTensor, out: &MetalTensor, m: u32, k: u32) {
        let params = MatvecParams { m, k };
        let pipeline = match weight.dtype {
            TensorDtype::Q4 => &self.pipeline_matvec_q4,
            _ => &self.pipeline_matvec,
        };
        self.dispatch_async(
            pipeline,
            &params,
            &[(&weight.buffer, 1), (&input.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(m as u64 * 32, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn matmul_batch(
        &self,
        weight: &MetalTensor,
        input: &MetalTensor,
        out: &MetalTensor,
        batch_size: u32,
        m: u32,
        k: u32,
    ) {
        let params = GemmParams { batch_size, m, k };
        let pipeline = match weight.dtype {
            TensorDtype::Q4 => &self.pipeline_gemm_q4,
            _ => &self.pipeline_gemm_bf16,
        };
        self.dispatch_async(
            pipeline,
            &params,
            &[(&weight.buffer, 1), (&input.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(batch_size as u64 * m as u64 * 32, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
