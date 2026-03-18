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
        let func = match weight.dtype {
            TensorDtype::Q4 => &self.fn_gemm_q4,
            _ => &self.fn_gemm_bf16,
        };
        // batch × M rows × 32 threads per row.
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
