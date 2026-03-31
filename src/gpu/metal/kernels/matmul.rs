// ---------------------------------------------------------------------------
// Metal impl: GpuMatmul — matrix-vector and batched GEMM kernels.
//
// Trait contract: gpu/ops/matmul.rs
// Metal shader:   metal/shaders/matmul.metal
//
// Both BF16 and Q4 use multi-row SIMD (2 rows per SIMD group, 16 rows per
// threadgroup) with shared memory x caching.  Q4 additionally uses fused
// dequant, packed uint loads, and split accumulators.
//
// Multi-row dispatch: grid = ceil(M/2) × 32 threads.  Each SIMD group
// computes 2 output rows, sharing x loads via threadgroup shared memory.
//
// Batched variants (gemm) are used during prefill; single-vector (matvec)
// during token-by-token decode.
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

#[repr(C)]
#[derive(Clone, Copy)]
struct WhtParams {
    k: u32,
}

/// Number of output rows per SIMD group in bf16 matvec kernel.
/// Must match ROWS_PER_SIMD in matmul.metal.
const ROWS_PER_SIMD_BF16: u64 = 2;

impl GpuMatmul for MetalBackend {
    fn matmul(&self, weight: &MetalTensor, input: &MetalTensor, out: &MetalTensor, m: u32, k: u32) {
        // TQ3: two-step dispatch — WHT-rotate x first (O(K)), then matvec (O(M×K)).
        // Scratch buffer is cached and reused across calls.
        if weight.dtype == TensorDtype::TQ3 {
            let k_padded = ((k + 255) / 256) * 256;
            let scratch = self.tq3_scratch(k_padded as u64);
            let wht_buf = scratch.as_ref().unwrap();

            let wht_params = WhtParams { k };
            self.dispatch_async(
                &self.pipeline_wht_rotate_x,
                &wht_params,
                &[(wht_buf, 1), (&input.buffer, 2)],
                MTLSize::new(k_padded as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            let params = MatvecParams { m, k };
            let num_simd_groups = (m as u64 + 1) / 2;
            self.dispatch_async(
                &self.pipeline_matvec_tq3,
                &params,
                &[(&weight.buffer, 1), (wht_buf, 2), (&out.buffer, 3)],
                MTLSize::new(num_simd_groups * 32, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            return;
        }

        let params = MatvecParams { m, k };
        let (pipeline, rows_per_simd) = match weight.dtype {
            // Q4: multi-row SIMD (2 rows per SIMD group) — fused dequant
            // reduces register pressure enough to match bf16.
            TensorDtype::Q4 => (&self.pipeline_matvec_q4, 2u64),
            TensorDtype::Q8 => (&self.pipeline_matvec_q8, 2u64),
            // FP8: not supported on Metal — no FP8 hardware.  Should never be
            // reached because the loader selects Q8 blocks on Metal.
            TensorDtype::FP8 => panic!("FP8 tensors not supported on Metal — use Q8 block format"),
            // BF16: multi-row SIMD (2 rows per SIMD group) — x loaded once,
            // used for both rows, giving free ILP via independent accumulators.
            _ => (&self.pipeline_matvec, ROWS_PER_SIMD_BF16),
        };
        let num_simd_groups = (m as u64 + rows_per_simd - 1) / rows_per_simd;
        self.dispatch_async(
            pipeline,
            &params,
            &[(&weight.buffer, 1), (&input.buffer, 2), (&out.buffer, 3)],
            MTLSize::new(num_simd_groups * 32, 1, 1),
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
        // TQ3: two-step dispatch — batch WHT-rotate X first, then GEMM.
        if weight.dtype == TensorDtype::TQ3 {
            let total_elements = batch_size as u64 * k as u64;
            let total_padded = ((total_elements + 255) / 256) * 256;
            let scratch = self.tq3_scratch(total_padded);
            let wht_buf = scratch.as_ref().unwrap();

            let wht_params = GemmParams { batch_size, m, k };
            self.dispatch_async(
                &self.pipeline_wht_rotate_x_batch,
                &wht_params,
                &[(wht_buf, 1), (&input.buffer, 2)],
                MTLSize::new(total_padded, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            let params = GemmParams { batch_size, m, k };
            self.dispatch_async(
                &self.pipeline_gemm_tq3,
                &params,
                &[(&weight.buffer, 1), (wht_buf, 2), (&out.buffer, 3)],
                MTLSize::new(batch_size as u64 * m as u64 * 32, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            return;
        }

        let params = GemmParams { batch_size, m, k };
        let pipeline = match weight.dtype {
            TensorDtype::Q4 => &self.pipeline_gemm_q4,
            TensorDtype::Q8 => &self.pipeline_gemm_q8,
            TensorDtype::FP8 => panic!("FP8 tensors not supported on Metal — use Q8 block format"),
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
