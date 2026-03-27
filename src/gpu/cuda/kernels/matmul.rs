// ---------------------------------------------------------------------------
// CUDA impl: GpuMatmul — matrix-vector and batched GEMM kernels.
//
// Trait contract: gpu/ops/matmul.rs
// CUDA shader:    cuda/shaders/matmul.cu
//
// Two code paths based on weight dtype:
//   - BF16: standard warp-cooperative matvec (32 threads per output row,
//     warp shuffle reduction, 4x unrolled inner loop)
//   - Q4: dequantize-on-the-fly from 4-bit blocks (20 bytes per 32 weights)
//
// Batched variants (gemm) are used during prefill; single-vector (matvec)
// during token-by-token decode.  Both dispatch 256-thread blocks with 32
// threads collaborating per output row via warp shuffle.
// ---------------------------------------------------------------------------

use cudarc::driver::{DeviceRepr, PushKernelArg};

use super::super::backend::CudaBackend;
use super::super::tensor::CudaTensor;
use crate::gpu::TensorDtype;
use crate::gpu::ops::GpuMatmul;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatvecParams {
    m: u32,
    k: u32,
}
unsafe impl DeviceRepr for MatvecParams {}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    batch_size: u32,
    m: u32,
    k: u32,
}
unsafe impl DeviceRepr for GemmParams {}

impl GpuMatmul for CudaBackend {
    fn matmul(&self, weight: &CudaTensor, input: &CudaTensor, out: &CudaTensor, m: u32, k: u32) {
        let params = MatvecParams { m, k };
        let func = match weight.dtype {
            TensorDtype::Q4 => &self.fn_matvec_q4,
            TensorDtype::Q8 => todo!("Q8 CUDA kernels not yet implemented"),
            _ => &self.fn_matvec_bf16,
        };
        // M rows × 32 threads per row = M*32 total threads.
        let cfg = CudaBackend::cfg_1d(m * 32, 256);
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&params)
                .arg(&weight.buf)
                .arg(&input.buf)
                .arg(&out.buf)
                .launch(cfg)
        }
        .expect("matmul launch failed");
    }

    fn matmul_batch(
        &self,
        weight: &CudaTensor,
        input: &CudaTensor,
        out: &CudaTensor,
        batch_size: u32,
        m: u32,
        k: u32,
    ) {
        let params = GemmParams { batch_size, m, k };

        // Use tensor-core WMMA path on sm_80+ when batch is large enough to
        // fill the 128×128 output tiles.  Small batches (< 4 rows) have mostly
        // empty tiles and are faster with the scalar warp-cooperative GEMM.
        const TC_BATCH_THRESHOLD: u32 = 4;
        let use_tc = self.fn_gemm_bf16_tc.is_some() && batch_size >= TC_BATCH_THRESHOLD;

        if use_tc {
            let func = match weight.dtype {
                TensorDtype::Q4 => self.fn_gemm_q4_tc.as_ref().unwrap(),
                TensorDtype::Q8 => todo!("Q8 CUDA kernels not yet implemented"),
                _ => self.fn_gemm_bf16_tc.as_ref().unwrap(),
            };
            // 2D grid: tiles over (M, batch_size), 256 threads (8 warps) per tile.
            // Shared memory is statically allocated in the kernel (32 KB for
            // double-buffered A and B tiles), so no dynamic smem needed here.
            let grid_m = (m + 127) / 128;
            let grid_n = (batch_size + 127) / 128;
            let cfg = CudaBackend::cfg_2d_smem(grid_m, grid_n, 256, 0);
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&params)
                    .arg(&weight.buf)
                    .arg(&input.buf)
                    .arg(&out.buf)
                    .launch(cfg)
            }
            .expect("matmul_batch TC launch failed");
        } else {
            let func = match weight.dtype {
                TensorDtype::Q4 => &self.fn_gemm_q4,
                TensorDtype::Q8 => todo!("Q8 CUDA kernels not yet implemented"),
                _ => &self.fn_gemm_bf16,
            };
            // Scalar path: batch × M rows × 32 threads per row.
            let total = batch_size * m * 32;
            let cfg = CudaBackend::cfg_1d(total, 256);
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&params)
                    .arg(&weight.buf)
                    .arg(&input.buf)
                    .arg(&out.buf)
                    .launch(cfg)
            }
            .expect("matmul_batch launch failed");
        }
    }
}
