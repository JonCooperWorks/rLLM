// ===========================================================================
// CPU reference backend for testing.
//
// Implements all GPU traits with pure Rust math on Vec<u8> tensors.
// Compiled during tests and as a runtime fallback when no GPU backend
// (Metal or CUDA) is available.  Zero impact on production GPU builds.
//
// Purposes:
//   1. Ground truth for validating Metal/CUDA kernel correctness
//   2. Enables `cargo test` on any platform without GPU hardware
//   3. Tests the model forward pass without a real GPU
//
// Interior mutability:
//   GPU traits take `&Self::Tensor` (not `&mut`) because GPU buffers have
//   their own synchronization.  CpuTensor uses raw pointer writes to match
//   this interface — safe because tests are single-threaded.
// ===========================================================================

use half::bf16;

use super::TensorDtype;
use super::ops::*;

pub(crate) struct CpuBackend;

#[allow(dead_code)]
pub(crate) struct CpuTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

// ---------------------------------------------------------------------------
// Helpers for reading/writing bf16 and f32 data in CpuTensor buffers.
// ---------------------------------------------------------------------------

/// Read bf16 elements from a tensor's raw bytes as f32 values.
fn read_bf16(tensor: &CpuTensor, count: usize) -> Vec<f32> {
    let slice: &[bf16] = bytemuck::cast_slice(&tensor.data[..count * 2]);
    slice.iter().map(|v| v.to_f32()).collect()
}

/// Read bf16 elements from a byte slice as f32 values.
fn read_bf16_bytes(data: &[u8], count: usize) -> Vec<f32> {
    let slice: &[bf16] = bytemuck::cast_slice(&data[..count * 2]);
    slice.iter().map(|v| v.to_f32()).collect()
}

/// Read f32 elements from a tensor's raw bytes.
fn read_f32(tensor: &CpuTensor, count: usize) -> Vec<f32> {
    let slice: &[f32] = bytemuck::cast_slice(&tensor.data[..count * 4]);
    slice.to_vec()
}

/// Write f32 values as bf16 into a tensor's buffer (unsafe interior mutability).
fn write_bf16(tensor: &CpuTensor, values: &[f32]) {
    let bf16_values: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    let bytes = bytemuck::cast_slice::<bf16, u8>(&bf16_values);
    let dst = tensor.data.as_ptr() as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
}

/// Write f32 values as bf16 at a byte offset into a tensor's buffer.
fn write_bf16_at(tensor: &CpuTensor, offset: usize, values: &[f32]) {
    let bf16_values: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    let bytes = bytemuck::cast_slice::<bf16, u8>(&bf16_values);
    let dst = unsafe { (tensor.data.as_ptr() as *mut u8).add(offset) };
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
}

/// Write f32 values directly into a tensor's buffer.
fn write_f32(tensor: &CpuTensor, values: &[f32]) {
    let bytes = bytemuck::cast_slice::<f32, u8>(values);
    let dst = tensor.data.as_ptr() as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
}

/// Write raw bytes at an offset into a tensor.
fn write_bytes_at(tensor: &CpuTensor, offset: usize, src: &[u8]) {
    let dst = unsafe { (tensor.data.as_ptr() as *mut u8).add(offset) };
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

// ===========================================================================
// GpuCore — tensor lifecycle
// ===========================================================================

impl GpuCore for CpuBackend {
    type Tensor = CpuTensor;

    fn device_name(&self) -> &str {
        "cpu-reference"
    }
    fn recommended_max_memory(&self) -> u64 {
        16 * 1024 * 1024 * 1024
    }
    fn flush(&self) {}
    fn submit(&self) {}

    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> CpuTensor {
        let byte_count = match dtype {
            TensorDtype::Q4 => {
                let total_elements: usize = shape.iter().product();
                (total_elements / 32) * 20
            }
            other => {
                let total_elements: usize = shape.iter().product();
                total_elements * other.byte_size()
            }
        };
        CpuTensor {
            data: vec![0u8; byte_count],
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> CpuTensor {
        CpuTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn copy_to_host(&self, tensor: &CpuTensor, dst: &mut [u8]) {
        dst[..tensor.data.len()].copy_from_slice(&tensor.data);
    }

    fn copy_to_tensor(&self, tensor: &CpuTensor, src: &[u8]) {
        let dst = tensor.data.as_ptr() as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
    }

    fn tensor_byte_count(&self, tensor: &CpuTensor) -> usize {
        tensor.data.len()
    }

    fn copy_tensor_region(
        &self,
        src: &CpuTensor,
        src_byte_offset: usize,
        dst: &CpuTensor,
        dst_byte_offset: usize,
        byte_count: usize,
    ) {
        // CPU backend: direct byte copy between Vec<u8> buffers.
        // Uses raw pointers because GpuCore takes &Self::Tensor (not &mut),
        // matching the GPU model where buffers have their own sync semantics.
        unsafe {
            let src_ptr = src.data.as_ptr().add(src_byte_offset);
            let dst_ptr = (dst.data.as_ptr() as *mut u8).add(dst_byte_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, byte_count);
        }
    }
}

// ===========================================================================
// GpuNorm — RMSNorm
// ===========================================================================

impl GpuNorm for CpuBackend {
    fn rms_norm(&self, input: &CpuTensor, weight: &CpuTensor, eps: f32, out: &CpuTensor) {
        let size = input.data.len() / 2; // bf16
        let x = read_bf16(input, size);
        let w = read_bf16(weight, size);

        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / size as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();

        let result: Vec<f32> = x
            .iter()
            .zip(w.iter())
            .map(|(xi, wi)| xi * scale * wi)
            .collect();
        write_bf16(out, &result);
    }

    fn rms_norm_batch(
        &self,
        input: &CpuTensor,
        weight: &CpuTensor,
        eps: f32,
        out: &CpuTensor,
        batch_size: u32,
    ) {
        let dim = weight.data.len() / 2; // bf16 weight size = hidden dim
        let w = read_bf16(weight, dim);

        for b in 0..batch_size as usize {
            let offset = b * dim * 2;
            let x = read_bf16_bytes(&input.data[offset..], dim);
            let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            let result: Vec<f32> = x
                .iter()
                .zip(w.iter())
                .map(|(xi, wi)| xi * scale * wi)
                .collect();
            write_bf16_at(out, offset, &result);
        }
    }
}

// ===========================================================================
// GpuElementwise — point-wise operations
// ===========================================================================

impl GpuElementwise for CpuBackend {
    fn silu_mul(&self, gate: &CpuTensor, up: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let g = read_bf16(gate, n);
        let u = read_bf16(up, n);
        let result: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gi, ui)| {
                let silu = gi / (1.0 + (-gi).exp()); // silu(x) = x * sigmoid(x)
                silu * ui
            })
            .collect();
        write_bf16(out, &result);
    }

    fn gelu_mul(&self, gate: &CpuTensor, up: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let g = read_bf16(gate, n);
        let u = read_bf16(up, n);
        let result: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gi, ui)| {
                // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                let gelu = 0.5 * gi * (1.0 + (c * (gi + 0.044715 * gi * gi * gi)).tanh());
                gelu * ui
            })
            .collect();
        write_bf16(out, &result);
    }

    fn scalar_mul(&self, input: &CpuTensor, out: &CpuTensor, scalar: f32, size: u32) {
        let n = size as usize;
        let x = read_bf16(input, n);
        let result: Vec<f32> = x.iter().map(|v| v * scalar).collect();
        write_bf16(out, &result);
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let va = read_bf16(a, n);
        let vb = read_bf16(b, n);
        let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(ai, bi)| ai + bi).collect();
        write_bf16(out, &result);
    }

    fn scale_add(&self, dst: &CpuTensor, src: &CpuTensor, scale: f32, size: u32) {
        let n = size as usize;
        let d = read_bf16(dst, n);
        let s = read_bf16(src, n);
        let result: Vec<f32> = d
            .iter()
            .zip(s.iter())
            .map(|(di, si)| di + si * scale)
            .collect();
        write_bf16(dst, &result);
    }

    fn fill_zero(&self, dst: &CpuTensor, size: u32) {
        let byte_count = size as usize * 2; // bf16
        let ptr = dst.data.as_ptr() as *mut u8;
        unsafe { std::ptr::write_bytes(ptr, 0, byte_count) };
    }

    fn bias_add_batch(
        &self,
        input: &CpuTensor,
        bias: &CpuTensor,
        out: &CpuTensor,
        batch_size: u32,
        dim: u32,
    ) {
        let d = dim as usize;
        let b_vec = read_bf16(bias, d);
        for b in 0..batch_size as usize {
            let offset = b * d * 2;
            let x = read_bf16_bytes(&input.data[offset..], d);
            let result: Vec<f32> = x.iter().zip(b_vec.iter()).map(|(xi, bi)| xi + bi).collect();
            write_bf16_at(out, offset, &result);
        }
    }

    fn sigmoid(&self, input: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let x = read_f32(input, n);
        let result: Vec<f32> = x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        write_f32(out, &result);
    }

    fn sigmoid_bf16(&self, input: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let x = read_bf16(input, n);
        let result: Vec<f32> = x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        write_bf16(out, &result);
    }

    fn silu(&self, input: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let x = read_bf16(input, n);
        let result: Vec<f32> = x.iter().map(|v| v / (1.0 + (-v).exp())).collect();
        write_bf16(out, &result);
    }

    fn mul(&self, a: &CpuTensor, b: &CpuTensor, out: &CpuTensor, size: u32) {
        let n = size as usize;
        let va = read_bf16(a, n);
        let vb = read_bf16(b, n);
        let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(ai, bi)| ai * bi).collect();
        write_bf16(out, &result);
    }

    fn silu_mul_clamp(
        &self,
        gate: &CpuTensor,
        up: &CpuTensor,
        out: &CpuTensor,
        size: u32,
        limit: f32,
    ) {
        let n = size as usize;
        let g = read_bf16(gate, n);
        let u = read_bf16(up, n);
        let result: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gi, ui)| {
                let silu = gi / (1.0 + (-gi).exp());
                (silu * ui).clamp(-limit, limit)
            })
            .collect();
        write_bf16(out, &result);
    }

    fn gpt_oss_gated_act(
        &self,
        gate: &CpuTensor,
        up: &CpuTensor,
        out: &CpuTensor,
        size: u32,
        alpha: f32,
        limit: f32,
    ) {
        let n = size as usize;
        let g = read_bf16(gate, n);
        let u = read_bf16(up, n);
        let result: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(gi, ui)| {
                let g_c = gi.min(limit);
                let u_c = ui.clamp(-limit, limit);
                let glu = g_c / (1.0 + (-g_c * alpha).exp());
                (u_c + 1.0) * glu
            })
            .collect();
        write_bf16(out, &result);
    }

    fn top_k_softmax(&self, logits: &CpuTensor, output: &CpuTensor, num_experts: u32, k: u32) {
        let n = num_experts as usize;
        let kk = k as usize;
        let x = read_f32(logits, n);

        // Find top-k indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_indices = &indices[..kk];

        // Softmax over top-k
        let max_val = x[top_indices[0]];
        let exps: Vec<f32> = top_indices
            .iter()
            .map(|&i| (x[i] - max_val).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Output format: [index0_f32, weight0_f32, index1_f32, weight1_f32, ...]
        let mut result = vec![0.0f32; kk * 2];
        for (i, (&idx, &prob)) in top_indices.iter().zip(probs.iter()).enumerate() {
            result[i * 2] = idx as f32;
            result[i * 2 + 1] = prob;
        }
        write_f32(output, &result);
    }
}

// ===========================================================================
// GpuEmbed — embedding table lookup
// ===========================================================================

impl GpuEmbed for CpuBackend {
    fn embed_lookup(&self, table: &CpuTensor, token_id: u32, out: &CpuTensor, hidden_dim: u32) {
        let dim = hidden_dim as usize;
        let offset = token_id as usize * dim * 2; // bf16
        let row = &table.data[offset..offset + dim * 2];
        write_bytes_at(out, 0, row);
    }

    fn embed_lookup_batch(
        &self,
        table: &CpuTensor,
        token_ids: &CpuTensor,
        out: &CpuTensor,
        batch_size: u32,
        hidden_dim: u32,
    ) {
        let dim = hidden_dim as usize;
        let ids: &[u32] = bytemuck::cast_slice(&token_ids.data[..batch_size as usize * 4]);
        for (b, &tid) in ids.iter().enumerate() {
            let src_offset = tid as usize * dim * 2;
            let dst_offset = b * dim * 2;
            let row = &table.data[src_offset..src_offset + dim * 2];
            write_bytes_at(out, dst_offset, row);
        }
    }
}

// ===========================================================================
// GpuMatmul — matrix-vector and batched GEMM
// ===========================================================================

impl GpuMatmul for CpuBackend {
    fn matmul(&self, weight: &CpuTensor, input: &CpuTensor, out: &CpuTensor, m: u32, k: u32) {
        let mm = m as usize;
        let kk = k as usize;

        match weight.dtype {
            TensorDtype::BF16 => {
                let inp = read_bf16(input, kk);
                let mut result = vec![0.0f32; mm];
                for row in 0..mm {
                    let w_offset = row * kk * 2;
                    let w_row = read_bf16_bytes(&weight.data[w_offset..], kk);
                    let mut acc = 0.0f32;
                    for j in 0..kk {
                        acc += w_row[j] * inp[j];
                    }
                    result[row] = acc;
                }
                write_bf16(out, &result);
            }
            TensorDtype::Q4 => {
                let inp = read_bf16(input, kk);
                let blocks_per_row = kk / 32;
                let mut result = vec![0.0f32; mm];

                for row in 0..mm {
                    let row_offset = row * blocks_per_row * 20;
                    let mut acc = 0.0f32;
                    for block in 0..blocks_per_row {
                        let block_offset = row_offset + block * 20;
                        let scale: f32 = bytemuck::cast_slice::<u8, f32>(
                            &weight.data[block_offset..block_offset + 4],
                        )[0];
                        let nibbles = &weight.data[block_offset + 4..block_offset + 20];
                        for i in 0..16 {
                            let byte = nibbles[i];
                            let lo = (byte & 0x0F) as i8 - 8;
                            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                            let j = block * 32 + i * 2;
                            acc += (lo as f32 * scale) * inp[j];
                            acc += (hi as f32 * scale) * inp[j + 1];
                        }
                    }
                    result[row] = acc;
                }
                write_bf16(out, &result);
            }
            _ => unimplemented!("CpuBackend::matmul for {:?}", weight.dtype),
        }
    }

    fn matmul_batch(
        &self,
        weight: &CpuTensor,
        input: &CpuTensor,
        out: &CpuTensor,
        batch_size: u32,
        m: u32,
        k: u32,
    ) {
        let mm = m as usize;
        let kk = k as usize;
        let bs = batch_size as usize;

        // Weight is [m, k], input is [batch_size, k], out is [batch_size, m]
        match weight.dtype {
            TensorDtype::BF16 => {
                for b in 0..bs {
                    let inp_offset = b * kk * 2;
                    let inp = read_bf16_bytes(&input.data[inp_offset..], kk);

                    let mut row_result = vec![0.0f32; mm];
                    for row in 0..mm {
                        let w_offset = row * kk * 2;
                        let w_row = read_bf16_bytes(&weight.data[w_offset..], kk);
                        let mut acc = 0.0f32;
                        for j in 0..kk {
                            acc += w_row[j] * inp[j];
                        }
                        row_result[row] = acc;
                    }
                    write_bf16_at(out, b * mm * 2, &row_result);
                }
            }
            TensorDtype::Q4 => {
                let blocks_per_row = kk / 32;
                for b in 0..bs {
                    let inp_offset = b * kk * 2;
                    let inp = read_bf16_bytes(&input.data[inp_offset..], kk);

                    let mut row_result = vec![0.0f32; mm];
                    for row in 0..mm {
                        let row_offset = row * blocks_per_row * 20;
                        let mut acc = 0.0f32;
                        for block in 0..blocks_per_row {
                            let block_offset = row_offset + block * 20;
                            let scale: f32 = bytemuck::cast_slice::<u8, f32>(
                                &weight.data[block_offset..block_offset + 4],
                            )[0];
                            let nibbles =
                                &weight.data[block_offset + 4..block_offset + 20];
                            for i in 0..16 {
                                let byte = nibbles[i];
                                let lo = (byte & 0x0F) as i8 - 8;
                                let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                                let j = block * 32 + i * 2;
                                acc += (lo as f32 * scale) * inp[j];
                                acc += (hi as f32 * scale) * inp[j + 1];
                            }
                        }
                        row_result[row] = acc;
                    }
                    write_bf16_at(out, b * mm * 2, &row_result);
                }
            }
            _ => unimplemented!("CpuBackend::matmul_batch for {:?}", weight.dtype),
        }
    }
}

// ===========================================================================
// GpuRope — Rotary Positional Embeddings
// ===========================================================================

impl GpuRope for CpuBackend {
    fn rope(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        self.rope_partial(
            q,
            k,
            pos,
            rope_theta,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
        );
    }

    fn rope_batch(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        positions: &CpuTensor,
        rope_theta: f32,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let bs = batch_size as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let half_dim = hd / 2;
        let q_stride = nh * hd;
        let k_stride = nkv * hd;

        let pos_data: &[u32] = bytemuck::cast_slice(&positions.data[..bs * 4]);
        let mut q_data = read_bf16(q, bs * q_stride);
        let mut k_data = read_bf16(k, bs * k_stride);

        for b in 0..bs {
            let pos = pos_data[b] as f32;

            // Apply RoPE to Q heads
            for h in 0..nh {
                let base = b * q_stride + h * hd;
                for pair in 0..half_dim {
                    let freq_exp = 2.0 * pair as f32 / head_dim as f32;
                    let inv_freq = 1.0 / rope_theta.powf(freq_exp);
                    let angle = pos * inv_freq;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let a = q_data[base + pair];
                    let b_val = q_data[base + pair + half_dim];
                    q_data[base + pair] = a * cos_a - b_val * sin_a;
                    q_data[base + pair + half_dim] = a * sin_a + b_val * cos_a;
                }
            }

            // Apply RoPE to K heads
            for h in 0..nkv {
                let base = b * k_stride + h * hd;
                for pair in 0..half_dim {
                    let freq_exp = 2.0 * pair as f32 / head_dim as f32;
                    let inv_freq = 1.0 / rope_theta.powf(freq_exp);
                    let angle = pos * inv_freq;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let a = k_data[base + pair];
                    let b_val = k_data[base + pair + half_dim];
                    k_data[base + pair] = a * cos_a - b_val * sin_a;
                    k_data[base + pair + half_dim] = a * sin_a + b_val * cos_a;
                }
            }
        }

        write_bf16(q, &q_data);
        write_bf16(k, &k_data);
    }

    fn rope_partial(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    ) {
        let hd = head_dim as usize;
        let rd = rotary_dim as usize;
        let half_rd = rd / 2;

        // Apply RoPE to Q heads using HALVED pairing convention:
        // element i pairs with element i + D/2 (matches HuggingFace rotate_half)
        let q_total = num_heads as usize * hd;
        let mut q_data = read_bf16(q, q_total);
        for h in 0..num_heads as usize {
            let base = h * hd;
            for pair in 0..half_rd {
                let freq_exp = 2.0 * pair as f32 / head_dim as f32;
                let inv_freq = 1.0 / rope_theta.powf(freq_exp);
                let angle = pos as f32 * inv_freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let a = q_data[base + pair];
                let b = q_data[base + pair + half_rd];
                q_data[base + pair] = a * cos_a - b * sin_a;
                q_data[base + pair + half_rd] = a * sin_a + b * cos_a;
            }
        }
        write_bf16(q, &q_data);

        // Apply RoPE to K heads using HALVED pairing convention
        let k_total = num_kv_heads as usize * hd;
        let mut k_data = read_bf16(k, k_total);
        for h in 0..num_kv_heads as usize {
            let base = h * hd;
            for pair in 0..half_rd {
                let freq_exp = 2.0 * pair as f32 / head_dim as f32;
                let inv_freq = 1.0 / rope_theta.powf(freq_exp);
                let angle = pos as f32 * inv_freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let a = k_data[base + pair];
                let b = k_data[base + pair + half_rd];
                k_data[base + pair] = a * cos_a - b * sin_a;
                k_data[base + pair + half_rd] = a * sin_a + b * cos_a;
            }
        }
        write_bf16(k, &k_data);
    }

    fn rope_yarn(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_pos: u32,
    ) {
        let hd = head_dim as usize;
        let half_dim = hd / 2;

        // Helper: compute YaRN-scaled inv_freq for a pair index.
        let yarn_inv_freq = |pair: usize| -> f32 {
            let freq_exp = 2.0 * pair as f32 / head_dim as f32;
            let inv_freq_base = 1.0 / rope_theta.powf(freq_exp);
            let wavelength = 2.0 * std::f32::consts::PI / inv_freq_base;
            let low_freq_wavelen = original_max_pos as f32 / beta_slow;
            let high_freq_wavelen = original_max_pos as f32 / beta_fast;

            if wavelength < high_freq_wavelen {
                inv_freq_base
            } else if wavelength > low_freq_wavelen {
                inv_freq_base / factor
            } else {
                let smooth = (low_freq_wavelen / wavelength - 1.0) / (beta_fast / beta_slow - 1.0);
                (1.0 - smooth) * (inv_freq_base / factor) + smooth * inv_freq_base
            }
        };

        // Apply to Q heads.
        let q_total = num_heads as usize * hd;
        let mut q_data = read_bf16(q, q_total);
        for h in 0..num_heads as usize {
            let base = h * hd;
            for pair in 0..half_dim {
                let inv_freq = yarn_inv_freq(pair);
                let angle = pos as f32 * inv_freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let a = q_data[base + pair];
                let b = q_data[base + pair + half_dim];
                q_data[base + pair] = a * cos_a - b * sin_a;
                q_data[base + pair + half_dim] = a * sin_a + b * cos_a;
            }
        }
        write_bf16(q, &q_data);

        // Apply to K heads.
        let k_total = num_kv_heads as usize * hd;
        let mut k_data = read_bf16(k, k_total);
        for h in 0..num_kv_heads as usize {
            let base = h * hd;
            for pair in 0..half_dim {
                let inv_freq = yarn_inv_freq(pair);
                let angle = pos as f32 * inv_freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let a = k_data[base + pair];
                let b = k_data[base + pair + half_dim];
                k_data[base + pair] = a * cos_a - b * sin_a;
                k_data[base + pair + half_dim] = a * sin_a + b * cos_a;
            }
        }
        write_bf16(k, &k_data);
    }

    fn rope_yarn_batch(
        &self,
        _q: &CpuTensor,
        _k: &CpuTensor,
        _positions: &CpuTensor,
        _rope_theta: f32,
        _batch_size: u32,
        _num_heads: u32,
        _num_kv_heads: u32,
        _head_dim: u32,
        _factor: f32,
        _beta_fast: f32,
        _beta_slow: f32,
        _original_max_pos: u32,
    ) {
        unimplemented!("CpuBackend::rope_yarn_batch")
    }
}

// ===========================================================================
// GpuAttention — attention and KV cache operations
// ===========================================================================

impl GpuAttention for CpuBackend {
    fn attention(
        &self,
        q: &CpuTensor,
        k_cache: &CpuTensor,
        v_cache: &CpuTensor,
        out: &CpuTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    ) {
        let sl = seq_len as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let heads_per_group = nh / nkv;

        let q_data = read_bf16(q, nh * hd);
        let kv_dim = nkv * hd;

        let mut out_data = vec![0.0f32; nh * hd];

        for h in 0..nh {
            let kv_h = h / heads_per_group;
            let q_head = &q_data[h * hd..(h + 1) * hd];

            // Compute attention scores: Q @ K^T
            let effective_len = if window_size > 0 && (window_size as usize) < sl {
                window_size as usize
            } else {
                sl
            };
            let start = sl - effective_len;

            let mut scores = vec![f32::NEG_INFINITY; effective_len];
            for s in 0..effective_len {
                let pos = start + s;
                let k_offset = pos * kv_dim + kv_h * hd;
                let k_head = read_bf16_bytes(&k_cache.data[k_offset * 2..], hd);
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_head[d] * k_head[d];
                }
                scores[s] = dot * attn_scale;
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V
            let mut head_out = vec![0.0f32; hd];
            for s in 0..effective_len {
                let pos = start + s;
                let v_offset = pos * kv_dim + kv_h * hd;
                let v_head = read_bf16_bytes(&v_cache.data[v_offset * 2..], hd);
                for d in 0..hd {
                    head_out[d] += scores[s] * v_head[d];
                }
            }
            out_data[h * hd..(h + 1) * hd].copy_from_slice(&head_out);
        }

        write_bf16(out, &out_data);
    }

    fn copy_to_kv_cache(
        &self,
        src: &CpuTensor,
        cache: &CpuTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let kv_dim = num_kv_heads as usize * head_dim as usize;
        let byte_offset = pos as usize * kv_dim * 2; // bf16
        let byte_count = kv_dim * 2;
        write_bytes_at(cache, byte_offset, &src.data[..byte_count]);
    }

    fn copy_to_paged_kv_cache(
        &self,
        src: &CpuTensor,
        pool: &CpuTensor,
        block_table: &CpuTensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let kv_dim = num_kv_heads as usize * head_dim as usize;
        let block_size = 16usize; // BLOCK_SIZE
        let block_idx = pos as usize / block_size;
        let block_offset = pos as usize % block_size;

        let table: &[u32] = bytemuck::cast_slice(&block_table.data);
        let physical_block = table[block_idx] as usize;

        let dst_offset = (physical_block * block_size + block_offset) * kv_dim * 2;
        write_bytes_at(pool, dst_offset, &src.data[..kv_dim * 2]);
    }

    fn paged_attention(
        &self,
        q: &CpuTensor,
        k_pool: &CpuTensor,
        v_pool: &CpuTensor,
        block_table: &CpuTensor,
        out: &CpuTensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&CpuTensor>,
    ) {
        let sl = seq_len as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let heads_per_group = nh / nkv;
        let kv_dim = nkv * hd;
        let block_size = 16usize;

        let q_data = read_bf16(q, nh * hd);
        let table: &[u32] = bytemuck::cast_slice(&block_table.data);

        let mut out_data = vec![0.0f32; nh * hd];

        for h in 0..nh {
            let kv_h = h / heads_per_group;
            let q_head = &q_data[h * hd..(h + 1) * hd];

            let effective_len = if window_size > 0 && (window_size as usize) < sl {
                window_size as usize
            } else {
                sl
            };
            let start = sl - effective_len;

            let mut scores = vec![0.0f32; effective_len];
            for s in 0..effective_len {
                let pos = start + s;
                let blk = pos / block_size;
                let off = pos % block_size;
                let physical = table[blk] as usize;
                let k_offset = (physical * block_size + off) * kv_dim + kv_h * hd;
                let k_head = read_bf16_bytes(&k_pool.data[k_offset * 2..], hd);
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_head[d] * k_head[d];
                }
                scores[s] = dot * attn_scale;
            }

            // Attention sinks: include per-head sink logit as extra softmax entry.
            // The sink has no V vector — it just absorbs probability mass.
            let sink_score = sinks.map(|s| {
                let sink_data = read_bf16(s, nh);
                sink_data[h]
            });

            // Softmax (including sink if present)
            let mut max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if let Some(ss) = sink_score {
                max_score = max_score.max(ss);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if let Some(ss) = sink_score {
                sum += (ss - max_score).exp();
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V (paged) — sink weight is excluded (no V vector).
            let mut head_out = vec![0.0f32; hd];
            for s in 0..effective_len {
                let pos = start + s;
                let blk = pos / block_size;
                let off = pos % block_size;
                let physical = table[blk] as usize;
                let v_offset = (physical * block_size + off) * kv_dim + kv_h * hd;
                let v_head = read_bf16_bytes(&v_pool.data[v_offset * 2..], hd);
                for d in 0..hd {
                    head_out[d] += scores[s] * v_head[d];
                }
            }
            out_data[h * hd..(h + 1) * hd].copy_from_slice(&head_out);
        }

        write_bf16(out, &out_data);
    }

    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &CpuTensor,
        pool: &CpuTensor,
        block_table: &CpuTensor,
        positions: &CpuTensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let bs = batch_size as usize;
        let kv_dim = num_kv_heads as usize * head_dim as usize;
        let block_size = 16usize;
        let token_bytes = kv_dim * 2; // bf16

        let pos_data: &[u32] = bytemuck::cast_slice(&positions.data[..bs * 4]);
        let table: &[u32] = bytemuck::cast_slice(&block_table.data);

        for b in 0..bs {
            let pos = pos_data[b] as usize;
            let block_idx = pos / block_size;
            let block_offset = pos % block_size;
            let physical_block = table[block_idx] as usize;

            let src_offset = b * token_bytes;
            let dst_offset = (physical_block * block_size + block_offset) * kv_dim * 2;
            write_bytes_at(
                pool,
                dst_offset,
                &src.data[src_offset..src_offset + token_bytes],
            );
        }
    }

    fn prefill_attention(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        out: &CpuTensor,
        chunk_size: u32,
        _start_pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&CpuTensor>,
    ) {
        let cs = chunk_size as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let heads_per_group = nh / nkv;
        let q_stride = nh * hd;
        let kv_stride = nkv * hd;

        let q_data = read_bf16(q, cs * q_stride);
        let k_data = read_bf16(k, cs * kv_stride);
        let v_data = read_bf16(v, cs * kv_stride);

        let mut out_data = vec![0.0f32; cs * q_stride];

        for qi in 0..cs {
            for h in 0..nh {
                let kv_h = h / heads_per_group;
                let q_head = &q_data[qi * q_stride + h * hd..qi * q_stride + (h + 1) * hd];

                // Causal mask: attend to positions 0..=qi within chunk.
                // With sliding window: attend_start = max(0, qi + 1 - window_size).
                let attend_len = qi + 1;
                let attend_start = if window_size > 0 && attend_len > window_size as usize {
                    attend_len - window_size as usize
                } else {
                    0
                };

                let scale = if attn_scale > 0.0 {
                    attn_scale
                } else {
                    1.0 / (hd as f32).sqrt()
                };

                // Compute attention scores
                let effective_len = attend_len - attend_start;
                let mut scores = vec![0.0f32; effective_len];
                for s in 0..effective_len {
                    let pos = attend_start + s;
                    let k_head =
                        &k_data[pos * kv_stride + kv_h * hd..pos * kv_stride + (kv_h + 1) * hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q_head[d] * k_head[d];
                    }
                    scores[s] = dot * scale;
                }

                // Attention sinks: include per-head sink logit in softmax.
                let sink_score = sinks.map(|s| {
                    let sink_data = read_bf16(s, nh);
                    sink_data[h]
                });

                // Softmax (including sink if present)
                let mut max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if let Some(ss) = sink_score {
                    max_score = max_score.max(ss);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if let Some(ss) = sink_score {
                    sum += (ss - max_score).exp();
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }

                // Weighted sum of V — sink weight is excluded (no V vector).
                let out_offset = qi * q_stride + h * hd;
                for s in 0..effective_len {
                    let pos = attend_start + s;
                    let v_head =
                        &v_data[pos * kv_stride + kv_h * hd..pos * kv_stride + (kv_h + 1) * hd];
                    for d in 0..hd {
                        out_data[out_offset + d] += scores[s] * v_head[d];
                    }
                }
            }
        }

        write_bf16(out, &out_data);
    }
}

// ===========================================================================
// GpuDeltaNet — stubs (lowest priority, Qwen 3.5 only)
// ===========================================================================

impl GpuDeltaNet for CpuBackend {
    fn conv1d_depthwise_single(
        &self,
        _input: &CpuTensor,
        _history: &CpuTensor,
        _weight: &CpuTensor,
        _out: &CpuTensor,
        _dim: u32,
        _kernel_size: u32,
    ) {
        unimplemented!("CpuBackend::conv1d_depthwise_single")
    }
    fn conv1d_shift_history(
        &self,
        _history: &CpuTensor,
        _input: &CpuTensor,
        _dim: u32,
        _kernel_size: u32,
    ) {
        unimplemented!("CpuBackend::conv1d_shift_history")
    }
    fn l2_normalize_heads(
        &self,
        _data: &CpuTensor,
        _num_heads: u32,
        _head_dim: u32,
        _elem_offset: u32,
    ) {
        unimplemented!("CpuBackend::l2_normalize_heads")
    }
    fn deltanet_decay_gate(
        &self,
        _x: &CpuTensor,
        _dt_bias: &CpuTensor,
        _a_log: &CpuTensor,
        _out: &CpuTensor,
        _size: u32,
    ) {
        unimplemented!("CpuBackend::deltanet_decay_gate")
    }
    fn deltanet_step(
        &self,
        _state: &CpuTensor,
        _q: &CpuTensor,
        _k: &CpuTensor,
        _v: &CpuTensor,
        _alpha: &CpuTensor,
        _beta: &CpuTensor,
        _out: &CpuTensor,
        _num_qk_heads: u32,
        _num_v_heads: u32,
        _head_dim: u32,
        _q_offset: u32,
        _k_offset: u32,
        _v_offset: u32,
    ) {
        unimplemented!("CpuBackend::deltanet_step")
    }
}

// ===========================================================================
// GpuAllReduce — no-op (single process, no communication needed)
// ===========================================================================

impl GpuAllReduce for CpuBackend {
    fn all_reduce_sum(&self, _tensor: &CpuTensor, _size: u32) {
        // No-op: single process, nothing to reduce.
    }

    fn all_gather(
        &self,
        _tensor: &CpuTensor,
        _output: &CpuTensor,
        _local_size: u32,
        _full_size: u32,
    ) {
        // No-op: single process, tensor is already the full result.
    }
}

// ===========================================================================
// GpuMoe — fused MoE kernels (reference CPU implementation).
// ===========================================================================

impl GpuMoe for CpuBackend {
    fn fused_gate_up_swiglu(
        &self,
        w_gate: &CpuTensor,
        w_up: &CpuTensor,
        input: &CpuTensor,
        output: &CpuTensor,
        m: u32,
        k: u32,
    ) {
        let mm = m as usize;
        let kk = k as usize;
        let inp = read_bf16(input, kk);
        let mut result = vec![0.0f32; mm];

        match w_gate.dtype {
            TensorDtype::BF16 => {
                for row in 0..mm {
                    let g_offset = row * kk * 2;
                    let u_offset = row * kk * 2;
                    let g_row = read_bf16_bytes(&w_gate.data[g_offset..], kk);
                    let u_row = read_bf16_bytes(&w_up.data[u_offset..], kk);

                    let mut acc_gate = 0.0f32;
                    let mut acc_up = 0.0f32;
                    for j in 0..kk {
                        acc_gate += g_row[j] * inp[j];
                        acc_up += u_row[j] * inp[j];
                    }

                    let silu = acc_gate / (1.0 + (-acc_gate).exp());
                    result[row] = silu * acc_up;
                }
            }
            TensorDtype::Q4 => {
                let blocks_per_row = kk / 32;
                for row in 0..mm {
                    let row_offset = row * blocks_per_row * 20;
                    let mut acc_gate = 0.0f32;
                    let mut acc_up = 0.0f32;

                    for block in 0..blocks_per_row {
                        let g_off = row_offset + block * 20;
                        let g_scale: f32 =
                            bytemuck::cast_slice::<u8, f32>(&w_gate.data[g_off..g_off + 4])[0];
                        let u_off = row_offset + block * 20;
                        let u_scale: f32 =
                            bytemuck::cast_slice::<u8, f32>(&w_up.data[u_off..u_off + 4])[0];

                        for i in 0..16 {
                            let gb = w_gate.data[g_off + 4 + i];
                            let ub = w_up.data[u_off + 4 + i];
                            let x_idx = block * 32 + i * 2;

                            let glo = (gb & 0xF) as i32 - 8;
                            let ghi = (gb >> 4) as i32 - 8;
                            let ulo = (ub & 0xF) as i32 - 8;
                            let uhi = (ub >> 4) as i32 - 8;

                            acc_gate += glo as f32 * g_scale * inp[x_idx];
                            acc_gate += ghi as f32 * g_scale * inp[x_idx + 1];
                            acc_up += ulo as f32 * u_scale * inp[x_idx];
                            acc_up += uhi as f32 * u_scale * inp[x_idx + 1];
                        }
                    }

                    let silu = acc_gate / (1.0 + (-acc_gate).exp());
                    result[row] = silu * acc_up;
                }
            }
            _ => panic!("fused_gate_up_swiglu: unsupported dtype {:?}", w_gate.dtype),
        }

        write_bf16(output, &result);
    }

    fn moe_combine_residual(
        &self,
        residual: &CpuTensor,
        expert_outputs: &CpuTensor,
        weights: &[f32],
        output: &CpuTensor,
        hidden_size: u32,
        k: u32,
    ) {
        let hs = hidden_size as usize;
        let kk = k as usize;
        let res = read_bf16(residual, hs);
        let experts = read_bf16(expert_outputs, hs * kk);

        let mut result = vec![0.0f32; hs];
        for j in 0..hs {
            let mut sum = res[j];
            for i in 0..kk {
                sum += weights[i] * experts[i * hs + j];
            }
            result[j] = sum;
        }
        write_bf16(output, &result);
    }
}

// ===========================================================================
// Tests — GPU kernel correctness via CpuBackend
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let bf16_values: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        bytemuck::cast_slice(&bf16_values).to_vec()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    fn assert_bf16_close(tensor: &CpuTensor, expected: &[f32], tol: f32) {
        let actual = read_bf16(tensor, expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "element {i}: actual={a}, expected={e}, diff={}",
                (a - e).abs()
            );
        }
    }

    fn assert_f32_close(tensor: &CpuTensor, expected: &[f32], tol: f32) {
        let actual = read_f32(tensor, expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "element {i}: actual={a}, expected={e}, diff={}",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // GpuCore tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_core_alloc_upload_roundtrip() {
        let b = CpuBackend;
        let data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let t = b.upload_tensor(&data, &[4], TensorDtype::BF16);
        let mut dst = vec![0u8; data.len()];
        b.copy_to_host(&t, &mut dst);
        assert_eq!(dst, data);
    }

    #[test]
    fn test_core_alloc_zero_initialized() {
        let b = CpuBackend;
        let t = b.alloc_tensor(&[4], TensorDtype::BF16);
        assert_bf16_close(&t, &[0.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_core_copy_to_tensor() {
        let b = CpuBackend;
        let t = b.alloc_tensor(&[4], TensorDtype::BF16);
        let data = bf16_bytes(&[5.0, 6.0, 7.0, 8.0]);
        b.copy_to_tensor(&t, &data);
        assert_bf16_close(&t, &[5.0, 6.0, 7.0, 8.0], 0.01);
    }

    #[test]
    fn test_core_tensor_byte_count() {
        let b = CpuBackend;
        let t = b.alloc_tensor(&[4], TensorDtype::BF16);
        assert_eq!(b.tensor_byte_count(&t), 8); // 4 * 2

        let t2 = b.alloc_tensor(&[4], TensorDtype::F32);
        assert_eq!(b.tensor_byte_count(&t2), 16); // 4 * 4
    }

    #[test]
    fn test_core_q4_alloc() {
        let b = CpuBackend;
        // 2 rows, 32 elements each = 2 blocks = 40 bytes
        let t = b.alloc_tensor(&[64], TensorDtype::Q4);
        assert_eq!(b.tensor_byte_count(&t), 40);
    }

    // -----------------------------------------------------------------------
    // GpuElementwise tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add() {
        let b = CpuBackend;
        let a = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let bv = b.upload_tensor(
            &bf16_bytes(&[10.0, 20.0, 30.0, 40.0]),
            &[4],
            TensorDtype::BF16,
        );
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.add(&a, &bv, &out, 4);
        assert_bf16_close(&out, &[11.0, 22.0, 33.0, 44.0], 0.1);
    }

    #[test]
    fn test_scalar_mul() {
        let b = CpuBackend;
        let input = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.scalar_mul(&input, &out, 3.0, 4);
        assert_bf16_close(&out, &[3.0, 6.0, 9.0, 12.0], 0.1);
    }

    #[test]
    fn test_fill_zero() {
        let b = CpuBackend;
        let t = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        b.fill_zero(&t, 4);
        assert_bf16_close(&t, &[0.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_silu_mul() {
        let b = CpuBackend;
        let gate = b.upload_tensor(&bf16_bytes(&[0.0, 1.0, 2.0, -1.0]), &[4], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.silu_mul(&gate, &up, &out, 4);
        let result = read_bf16(&out, 4);
        // silu(0) = 0, silu(1) ≈ 0.731, silu(2) ≈ 1.762, silu(-1) ≈ -0.269
        assert!((result[0] - 0.0).abs() < 0.05);
        assert!((result[1] - 0.731).abs() < 0.05);
        assert!((result[2] - 1.762).abs() < 0.05);
        assert!((result[3] - (-0.269)).abs() < 0.05);
    }

    #[test]
    fn test_sigmoid() {
        let b = CpuBackend;
        let input = b.upload_tensor(&f32_bytes(&[0.0, 10.0, -10.0]), &[3], TensorDtype::F32);
        let out = b.alloc_tensor(&[3], TensorDtype::F32);
        b.sigmoid(&input, &out, 3);
        assert_f32_close(&out, &[0.5, 1.0, 0.0], 0.01);
    }

    #[test]
    fn test_mul() {
        let b = CpuBackend;
        let a = b.upload_tensor(&bf16_bytes(&[2.0, 3.0, 4.0]), &[3], TensorDtype::BF16);
        let bv = b.upload_tensor(&bf16_bytes(&[5.0, 6.0, 7.0]), &[3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[3], TensorDtype::BF16);
        b.mul(&a, &bv, &out, 3);
        assert_bf16_close(&out, &[10.0, 18.0, 28.0], 0.1);
    }

    #[test]
    fn test_gelu_mul() {
        let b = CpuBackend;
        let gate = b.upload_tensor(&bf16_bytes(&[0.0, 1.0, -1.0]), &[3], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0]), &[3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[3], TensorDtype::BF16);
        b.gelu_mul(&gate, &up, &out, 3);
        let result = read_bf16(&out, 3);
        // gelu(0) = 0, gelu(1) ≈ 0.841, gelu(-1) ≈ -0.159
        assert!((result[0] - 0.0).abs() < 0.05);
        assert!((result[1] - 0.841).abs() < 0.05);
        assert!((result[2] - (-0.159)).abs() < 0.05);
    }

    #[test]
    fn test_scale_add() {
        let b = CpuBackend;
        let dst = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0]), &[3], TensorDtype::BF16);
        let src = b.upload_tensor(&bf16_bytes(&[10.0, 20.0, 30.0]), &[3], TensorDtype::BF16);
        b.scale_add(&dst, &src, 0.5, 3);
        // dst = [1 + 10*0.5, 2 + 20*0.5, 3 + 30*0.5] = [6, 12, 18]
        assert_bf16_close(&dst, &[6.0, 12.0, 18.0], 0.1);
    }

    #[test]
    fn test_silu() {
        let b = CpuBackend;
        let input = b.upload_tensor(&bf16_bytes(&[0.0, 1.0, -1.0]), &[3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[3], TensorDtype::BF16);
        b.silu(&input, &out, 3);
        let result = read_bf16(&out, 3);
        assert!((result[0] - 0.0).abs() < 0.05);
        assert!((result[1] - 0.731).abs() < 0.05);
        assert!((result[2] - (-0.269)).abs() < 0.05);
    }

    #[test]
    fn test_bias_add_batch() {
        let b = CpuBackend;
        // 2 batches, dim=3
        let input = b.upload_tensor(
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            &[2, 3],
            TensorDtype::BF16,
        );
        let bias = b.upload_tensor(&bf16_bytes(&[0.1, 0.2, 0.3]), &[3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[2, 3], TensorDtype::BF16);
        b.bias_add_batch(&input, &bias, &out, 2, 3);
        assert_bf16_close(&out, &[1.1, 2.2, 3.3, 4.1, 5.2, 6.3], 0.1);
    }

    #[test]
    fn test_top_k_softmax() {
        let b = CpuBackend;
        // 4 experts, top-2
        let logits = b.upload_tensor(&f32_bytes(&[1.0, 3.0, 2.0, 0.5]), &[4], TensorDtype::F32);
        let output = b.alloc_tensor(&[4], TensorDtype::F32); // 2*2 f32s
        b.top_k_softmax(&logits, &output, 4, 2);
        let result = read_f32(&output, 4);
        // Top-2 should be indices 1 (3.0) and 2 (2.0)
        assert_eq!(result[0] as u32, 1); // first expert index
        assert_eq!(result[2] as u32, 2); // second expert index
        // Weights should sum to 1.0
        let weight_sum = result[1] + result[3];
        assert!((weight_sum - 1.0).abs() < 0.01);
        // Expert 1 (logit=3) should have higher weight than expert 2 (logit=2)
        assert!(result[1] > result[3]);
    }

    // -----------------------------------------------------------------------
    // GpuNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_uniform() {
        let b = CpuBackend;
        let input = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let weight = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.rms_norm(&input, &weight, 1e-5, &out);
        // RMS of [1,1,1,1] = sqrt(1) = 1, so output ≈ [1,1,1,1]
        assert_bf16_close(&out, &[1.0, 1.0, 1.0, 1.0], 0.02);
    }

    #[test]
    fn test_rms_norm_varying() {
        let b = CpuBackend;
        let input = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let weight = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.rms_norm(&input, &weight, 1e-5, &out);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v / rms).collect();
        assert_bf16_close(&out, &expected, 0.02);
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let b = CpuBackend;
        let input = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let weight = b.upload_tensor(&bf16_bytes(&[2.0, 2.0, 2.0, 2.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.rms_norm(&input, &weight, 1e-5, &out);
        let rms = (7.5f32).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v / rms * 2.0).collect();
        assert_bf16_close(&out, &expected, 0.02);
    }

    #[test]
    fn test_rms_norm_batch() {
        let b = CpuBackend;
        // 2 rows of dim=4
        let input = b.upload_tensor(
            &bf16_bytes(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            &[2, 4],
            TensorDtype::BF16,
        );
        let weight = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[2, 4], TensorDtype::BF16);
        b.rms_norm_batch(&input, &weight, 1e-5, &out, 2);
        // Row 0: all-ones → output ≈ [1,1,1,1]
        // Row 1: all-twos → RMS=2, output ≈ [1,1,1,1]
        let result = read_bf16(&out, 8);
        for v in &result {
            assert!((v - 1.0).abs() < 0.02, "value {v} not close to 1.0");
        }
    }

    // -----------------------------------------------------------------------
    // GpuEmbed tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_embed_lookup() {
        let b = CpuBackend;
        // Embedding table: 4 tokens, dim=3
        let table_data = bf16_bytes(&[
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
            2.1, 2.2, 2.3, // token 2
            3.1, 3.2, 3.3, // token 3
        ]);
        let table = b.upload_tensor(&table_data, &[4, 3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[3], TensorDtype::BF16);
        b.embed_lookup(&table, 2, &out, 3);
        assert_bf16_close(&out, &[2.1, 2.2, 2.3], 0.05);
    }

    #[test]
    fn test_embed_lookup_batch() {
        let b = CpuBackend;
        let table_data = bf16_bytes(&[
            0.1, 0.2, // token 0
            1.1, 1.2, // token 1
            2.1, 2.2, // token 2
        ]);
        let table = b.upload_tensor(&table_data, &[3, 2], TensorDtype::BF16);
        let token_ids =
            b.upload_tensor(bytemuck::cast_slice(&[2u32, 0u32]), &[2], TensorDtype::F32);
        let out = b.alloc_tensor(&[2, 2], TensorDtype::BF16);
        b.embed_lookup_batch(&table, &token_ids, &out, 2, 2);
        assert_bf16_close(&out, &[2.1, 2.2, 0.1, 0.2], 0.05);
    }

    // -----------------------------------------------------------------------
    // GpuMatmul tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_identity_like() {
        let b = CpuBackend;
        // 2x2 "identity-like" weight (diagonal = 1, rest = 0)
        let weight = b.upload_tensor(
            &bf16_bytes(&[1.0, 0.0, 0.0, 1.0]),
            &[2, 2],
            TensorDtype::BF16,
        );
        let input = b.upload_tensor(&bf16_bytes(&[3.0, 7.0]), &[2], TensorDtype::BF16);
        let out = b.alloc_tensor(&[2], TensorDtype::BF16);
        b.matmul(&weight, &input, &out, 2, 2);
        assert_bf16_close(&out, &[3.0, 7.0], 0.05);
    }

    #[test]
    fn test_matmul_known() {
        let b = CpuBackend;
        // Weight [2, 3]:
        // [1 2 3]
        // [4 5 6]
        // Input [3]: [1, 1, 1]
        // Expected: [6, 15]
        let weight = b.upload_tensor(
            &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            &[2, 3],
            TensorDtype::BF16,
        );
        let input = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0]), &[3], TensorDtype::BF16);
        let out = b.alloc_tensor(&[2], TensorDtype::BF16);
        b.matmul(&weight, &input, &out, 2, 3);
        assert_bf16_close(&out, &[6.0, 15.0], 0.1);
    }

    #[test]
    fn test_matmul_batch() {
        let b = CpuBackend;
        // Weight [2, 3], batch of 2 inputs
        let weight = b.upload_tensor(
            &bf16_bytes(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), // extract dim 0 and dim 1
            &[2, 3],
            TensorDtype::BF16,
        );
        let input = b.upload_tensor(
            &bf16_bytes(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            &[2, 3],
            TensorDtype::BF16,
        );
        let out = b.alloc_tensor(&[2, 2], TensorDtype::BF16);
        b.matmul_batch(&weight, &input, &out, 2, 2, 3);
        assert_bf16_close(&out, &[5.0, 6.0, 8.0, 9.0], 0.1);
    }

    #[test]
    fn test_matmul_q4() {
        let b = CpuBackend;
        // Create a Q4 weight with 1 row of 32 elements (1 block = 20 bytes)
        // Scale = 1.0, all quantized values = 1 (stored as 1+8 = 9)
        let mut block = vec![0u8; 20];
        // Write scale as f32
        let scale_bytes = 1.0f32.to_le_bytes();
        block[0..4].copy_from_slice(&scale_bytes);
        // Pack nibbles: each byte has (lo, hi), value 9 means dequant = 9-8 = 1
        // 0x99 = lo=9, hi=9
        for i in 4..20 {
            block[i] = 0x99;
        }
        let weight = b.upload_tensor(&block, &[32], TensorDtype::Q4);

        // Input: all ones
        let input = b.upload_tensor(&bf16_bytes(&vec![1.0; 32]), &[32], TensorDtype::BF16);
        let out = b.alloc_tensor(&[1], TensorDtype::BF16);
        b.matmul(&weight, &input, &out, 1, 32);

        // Expected: 32 * (1 * 1.0) = 32.0
        let result = read_bf16(&out, 1);
        assert!(
            (result[0] - 32.0).abs() < 1.0,
            "Q4 matmul result: {}",
            result[0]
        );
    }

    // -----------------------------------------------------------------------
    // GpuRope tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_position_zero() {
        let b = CpuBackend;
        // At position 0, all angles = 0, so cos=1, sin=0 → identity
        let q = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let k = b.upload_tensor(&bf16_bytes(&[5.0, 6.0, 7.0, 8.0]), &[4], TensorDtype::BF16);
        b.rope(&q, &k, 0, 10000.0, 1, 1, 4);
        assert_bf16_close(&q, &[1.0, 2.0, 3.0, 4.0], 0.02);
        assert_bf16_close(&k, &[5.0, 6.0, 7.0, 8.0], 0.02);
    }

    #[test]
    fn test_rope_modifies_at_nonzero_position() {
        let b = CpuBackend;
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let q = b.upload_tensor(&bf16_bytes(&original), &[4], TensorDtype::BF16);
        let k = b.upload_tensor(&bf16_bytes(&[1.0, 0.0, 0.0, 0.0]), &[4], TensorDtype::BF16);
        b.rope(&q, &k, 5, 10000.0, 1, 1, 4);
        let q_after = read_bf16(&q, 4);
        // Values should differ from original at non-zero position
        let differs = q_after
            .iter()
            .zip(original.iter())
            .any(|(a, o)| (a - o).abs() > 0.01);
        assert!(differs, "RoPE at pos=5 should modify values");
    }

    #[test]
    fn test_rope_partial() {
        let b = CpuBackend;
        // head_dim=4, rotary_dim=2: only first 2 dims rotated, last 2 unchanged
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let q = b.upload_tensor(&bf16_bytes(&original), &[4], TensorDtype::BF16);
        let k = b.upload_tensor(&bf16_bytes(&[0.0; 4]), &[4], TensorDtype::BF16);
        b.rope_partial(&q, &k, 5, 10000.0, 1, 1, 4, 2);
        let q_after = read_bf16(&q, 4);
        // Dims 2,3 should be unchanged
        assert!((q_after[2] - 3.0).abs() < 0.02);
        assert!((q_after[3] - 4.0).abs() < 0.02);
        // Dims 0,1 should be different (rotated at pos=5)
        let d01_changed = (q_after[0] - 1.0).abs() > 0.01 || (q_after[1] - 2.0).abs() > 0.01;
        assert!(d01_changed, "First 2 dims should be rotated");
    }

    // -----------------------------------------------------------------------
    // GpuAttention tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_copy_to_kv_cache_and_attention() {
        let b = CpuBackend;
        // 1 head, dim=2, sequence of 3 tokens
        let num_heads = 1u32;
        let head_dim = 2u32;
        let kv_dim = num_heads as usize * head_dim as usize;
        let max_seq = 4;

        // Allocate KV caches
        let k_cache = b.alloc_tensor(&[max_seq * kv_dim], TensorDtype::BF16);
        let v_cache = b.alloc_tensor(&[max_seq * kv_dim], TensorDtype::BF16);

        // Write 3 tokens into cache
        for pos in 0..3u32 {
            let k_val = bf16_bytes(&[(pos as f32 + 1.0) * 0.1, (pos as f32 + 1.0) * 0.2]);
            let v_val = bf16_bytes(&[1.0, 0.0]); // uniform V for simplicity
            let k_src = b.upload_tensor(&k_val, &[kv_dim], TensorDtype::BF16);
            let v_src = b.upload_tensor(&v_val, &[kv_dim], TensorDtype::BF16);
            b.copy_to_kv_cache(&k_src, &k_cache, pos, num_heads, head_dim);
            b.copy_to_kv_cache(&v_src, &v_cache, pos, num_heads, head_dim);
        }

        // Query
        let q = b.upload_tensor(&bf16_bytes(&[0.3, 0.6]), &[kv_dim], TensorDtype::BF16);
        let out = b.alloc_tensor(&[kv_dim], TensorDtype::BF16);

        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        b.attention(
            &q, &k_cache, &v_cache, &out, 3, num_heads, num_heads, head_dim, 0, attn_scale,
        );

        // Since all V vectors are [1, 0], the output should be close to [1, 0]
        // (softmax of any scores still sums to 1, so weighted sum of identical V = V)
        assert_bf16_close(&out, &[1.0, 0.0], 0.05);
    }

    #[test]
    fn test_attention_single_kv() {
        let b = CpuBackend;
        // 1 head, dim=2, seq_len=1 → attention is just Q@K * V (with softmax=1)
        let q = b.upload_tensor(&bf16_bytes(&[1.0, 0.0]), &[2], TensorDtype::BF16);
        let k_cache = b.upload_tensor(&bf16_bytes(&[1.0, 0.0]), &[2], TensorDtype::BF16);
        let v_cache = b.upload_tensor(&bf16_bytes(&[3.0, 4.0]), &[2], TensorDtype::BF16);
        let out = b.alloc_tensor(&[2], TensorDtype::BF16);

        let scale = 1.0 / 2.0f32.sqrt();
        b.attention(&q, &k_cache, &v_cache, &out, 1, 1, 1, 2, 0, scale);
        // With seq_len=1, softmax([score]) = [1.0], so output = V = [3, 4]
        assert_bf16_close(&out, &[3.0, 4.0], 0.1);
    }

    // -----------------------------------------------------------------------
    // GPT-OSS specific: silu_mul_clamp + rope_yarn tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_mul_clamp() {
        let b = CpuBackend;
        // Large gate values should be clamped.
        let gate = b.upload_tensor(
            &bf16_bytes(&[0.0, 1.0, 10.0, -10.0]),
            &[4],
            TensorDtype::BF16,
        );
        let up = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0, 1.0]), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.silu_mul_clamp(&gate, &up, &out, 4, 2.0);
        let result = read_bf16(&out, 4);
        // silu(0)*1 = 0, silu(1)*1 ≈ 0.731, silu(10)*1 ≈ 10 (clamped to 2.0),
        // silu(-10)*1 ≈ 0 (stays within [-2, 2])
        assert!(
            (result[0] - 0.0).abs() < 0.05,
            "silu_clamp[0]={}",
            result[0]
        );
        assert!(
            (result[1] - 0.731).abs() < 0.05,
            "silu_clamp[1]={}",
            result[1]
        );
        assert!(
            (result[2] - 2.0).abs() < 0.05,
            "silu_clamp[2]={} should be clamped to 2.0",
            result[2]
        );
        assert!(
            result[3].abs() < 0.05,
            "silu_clamp[3]={} should be near 0",
            result[3]
        );
    }

    #[test]
    fn test_silu_mul_clamp_no_clamp() {
        let b = CpuBackend;
        // With a high limit, results should match silu_mul.
        let gate = b.upload_tensor(&bf16_bytes(&[0.0, 1.0, 2.0]), &[3], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&[1.0, 1.0, 1.0]), &[3], TensorDtype::BF16);
        let out_clamped = b.alloc_tensor(&[3], TensorDtype::BF16);
        let out_plain = b.alloc_tensor(&[3], TensorDtype::BF16);
        b.silu_mul_clamp(&gate, &up, &out_clamped, 3, 100.0);
        b.silu_mul(&gate, &up, &out_plain, 3);
        let clamped = read_bf16(&out_clamped, 3);
        let plain = read_bf16(&out_plain, 3);
        for i in 0..3 {
            assert!(
                (clamped[i] - plain[i]).abs() < 0.05,
                "silu_mul_clamp with high limit should match silu_mul: [{}] {} vs {}",
                i,
                clamped[i],
                plain[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // GPT-OSS gated activation tests
    // -----------------------------------------------------------------------

    /// CPU reference for gpt_oss_gated_act.
    fn gpt_oss_gated_act_cpu(gate: &[f32], up: &[f32], alpha: f32, limit: f32) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(&g, &u)| {
                let g_c = g.min(limit);
                let u_c = u.clamp(-limit, limit);
                let glu = g_c / (1.0 + (-g_c * alpha).exp());
                (u_c + 1.0) * glu
            })
            .collect()
    }

    #[test]
    fn test_gpt_oss_gated_act_basic() {
        let b = CpuBackend;
        let gate_vals = [1.0f32, -2.0, 0.0, 5.0, -8.0, 3.5];
        let up_vals = [0.5f32, 1.0, -1.0, 0.0, 10.0, -3.0];
        let alpha = 1.702;
        let limit = 7.0;

        let gate = b.upload_tensor(&bf16_bytes(&gate_vals), &[6], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&up_vals), &[6], TensorDtype::BF16);
        let out = b.alloc_tensor(&[6], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate, &up, &out, 6, alpha, limit);

        let result = read_bf16(&out, 6);
        let expected = gpt_oss_gated_act_cpu(&gate_vals, &up_vals, alpha, limit);
        for i in 0..6 {
            let tol = expected[i].abs() * 0.01 + 0.01;
            assert!(
                (result[i] - expected[i]).abs() < tol,
                "gpt_oss_gated_act[{}]: got {} expected {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_gpt_oss_gated_act_clamping() {
        let b = CpuBackend;
        // gate=10 should be upper-clamped to 7, gate=-10 should NOT be clamped (no lower clamp).
        // up=10 should be clamped to 7, up=-10 should be clamped to -7.
        let gate_vals = [10.0f32, -10.0, 0.0, 0.0];
        let up_vals = [0.0f32, 0.0, 10.0, -10.0];
        let alpha = 1.702;
        let limit = 7.0;

        let gate = b.upload_tensor(&bf16_bytes(&gate_vals), &[4], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&up_vals), &[4], TensorDtype::BF16);
        let out = b.alloc_tensor(&[4], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate, &up, &out, 4, alpha, limit);
        let result = read_bf16(&out, 4);

        // gate=10 → g_c=7, glu=7*sigmoid(7*1.702) ≈ 7.0 (sigmoid ≈ 1)
        // up=0 → u_c=0, out=(0+1)*glu = glu ≈ 7.0
        assert!(
            (result[0] - 7.0).abs() < 0.1,
            "gate clamped: got {}",
            result[0]
        );

        // gate=-10 → g_c=-10 (no lower clamp), sigmoid(-10*1.702) ≈ 0, glu ≈ 0
        assert!(result[1].abs() < 0.01, "negative gate: got {}", result[1]);

        // gate=0 → glu=0, out=0 regardless of up
        assert!(
            result[2].abs() < 0.01,
            "zero gate, large up: got {}",
            result[2]
        );
        assert!(
            result[3].abs() < 0.01,
            "zero gate, negative up: got {}",
            result[3]
        );
    }

    #[test]
    fn test_gpt_oss_gated_act_edge_cases() {
        let b = CpuBackend;

        // gate=0, up=0 → out=0
        let gate = b.upload_tensor(&bf16_bytes(&[0.0]), &[1], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&[0.0]), &[1], TensorDtype::BF16);
        let out = b.alloc_tensor(&[1], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate, &up, &out, 1, 1.702, 7.0);
        let result = read_bf16(&out, 1);
        assert!(result[0].abs() < 0.001, "zero/zero: got {}", result[0]);

        // up=0 → out = (0+1)*glu = glu (non-zero if gate != 0)
        let gate2 = b.upload_tensor(&bf16_bytes(&[2.0]), &[1], TensorDtype::BF16);
        let up2 = b.upload_tensor(&bf16_bytes(&[0.0]), &[1], TensorDtype::BF16);
        let out2 = b.alloc_tensor(&[1], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate2, &up2, &out2, 1, 1.702, 7.0);
        let result2 = read_bf16(&out2, 1);
        // glu = 2 / (1 + exp(-2*1.702)) ≈ 2 * 0.967 ≈ 1.934
        assert!(
            result2[0] > 1.5,
            "up=0 with gate=2 should give non-zero: got {}",
            result2[0]
        );

        // Very negative gate (-100): sigmoid(-100*1.702) ≈ 0, so glu ≈ 0
        let gate3 = b.upload_tensor(&bf16_bytes(&[-100.0]), &[1], TensorDtype::BF16);
        let up3 = b.upload_tensor(&bf16_bytes(&[5.0]), &[1], TensorDtype::BF16);
        let out3 = b.alloc_tensor(&[1], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate3, &up3, &out3, 1, 1.702, 7.0);
        let result3 = read_bf16(&out3, 1);
        assert!(
            result3[0].abs() < 0.01,
            "very negative gate: got {}",
            result3[0]
        );
    }

    #[test]
    fn test_gpt_oss_gated_act_large_size() {
        let b = CpuBackend;
        let n = 4096;
        let gate_vals: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.01).collect();
        let up_vals: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.005).collect();
        let alpha = 1.702;
        let limit = 7.0;

        let gate = b.upload_tensor(&bf16_bytes(&gate_vals), &[n], TensorDtype::BF16);
        let up = b.upload_tensor(&bf16_bytes(&up_vals), &[n], TensorDtype::BF16);
        let out = b.alloc_tensor(&[n], TensorDtype::BF16);
        b.gpt_oss_gated_act(&gate, &up, &out, n as u32, alpha, limit);

        let result = read_bf16(&out, n);
        let expected = gpt_oss_gated_act_cpu(&gate_vals, &up_vals, alpha, limit);
        for i in 0..n {
            let tol = expected[i].abs() * 0.02 + 0.01;
            assert!(
                (result[i] - expected[i]).abs() < tol,
                "large size [{}]: got {} expected {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_rope_yarn_position_zero() {
        let b = CpuBackend;
        // At position 0, YaRN RoPE should be identity (same as standard RoPE).
        let q = b.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
        let k = b.upload_tensor(&bf16_bytes(&[5.0, 6.0, 7.0, 8.0]), &[4], TensorDtype::BF16);
        b.rope_yarn(&q, &k, 0, 10000.0, 1, 1, 4, 32.0, 32.0, 1.0, 4096);
        assert_bf16_close(&q, &[1.0, 2.0, 3.0, 4.0], 0.02);
        assert_bf16_close(&k, &[5.0, 6.0, 7.0, 8.0], 0.02);
    }

    #[test]
    fn test_rope_yarn_modifies_at_nonzero_position() {
        let b = CpuBackend;
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let q = b.upload_tensor(&bf16_bytes(&original), &[4], TensorDtype::BF16);
        let k = b.upload_tensor(&bf16_bytes(&[1.0, 0.0, 0.0, 0.0]), &[4], TensorDtype::BF16);
        b.rope_yarn(&q, &k, 5, 10000.0, 1, 1, 4, 32.0, 32.0, 1.0, 4096);
        let q_after = read_bf16(&q, 4);
        let differs = q_after
            .iter()
            .zip(original.iter())
            .any(|(a, o)| (a - o).abs() > 0.01);
        assert!(differs, "YaRN RoPE at pos=5 should modify values");
    }

    // -----------------------------------------------------------------------
    // Metal cross-validation tests (macOS only)
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    mod metal_cross {
        use super::*;
        use crate::gpu::metal::MetalBackend;

        fn assert_tensors_close(
            _cpu_backend: &CpuBackend,
            cpu_tensor: &CpuTensor,
            metal_backend: &MetalBackend,
            metal_tensor: &<MetalBackend as GpuCore>::Tensor,
            count: usize,
            tol: f32,
            label: &str,
        ) {
            let cpu_values = read_bf16(cpu_tensor, count);
            let mut metal_bytes = vec![0u8; count * 2];
            metal_backend.copy_to_host(metal_tensor, &mut metal_bytes);
            let metal_values = read_bf16_bytes(&metal_bytes, count);

            for (i, (c, m)) in cpu_values.iter().zip(metal_values.iter()).enumerate() {
                assert!(
                    (c - m).abs() < tol,
                    "{label} element {i}: cpu={c}, metal={m}, diff={}",
                    (c - m).abs()
                );
            }
        }

        #[test]
        fn test_metal_vs_cpu_add() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let a_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let b_data = bf16_bytes(&[10.0, 20.0, 30.0, 40.0]);

            let ca = cpu.upload_tensor(&a_data, &[4], TensorDtype::BF16);
            let cb = cpu.upload_tensor(&b_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.add(&ca, &cb, &cout, 4);

            let ma = metal.upload_tensor(&a_data, &[4], TensorDtype::BF16);
            let mb = metal.upload_tensor(&b_data, &[4], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[4], TensorDtype::BF16);
            metal.add(&ma, &mb, &mout, 4);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 4, 0.1, "add");
        }

        #[test]
        fn test_metal_vs_cpu_rms_norm() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let input_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let weight_data = bf16_bytes(&[1.0, 1.0, 1.0, 1.0]);

            let ci = cpu.upload_tensor(&input_data, &[4], TensorDtype::BF16);
            let cw = cpu.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.rms_norm(&ci, &cw, 1e-5, &cout);

            let mi = metal.upload_tensor(&input_data, &[4], TensorDtype::BF16);
            let mw = metal.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[4], TensorDtype::BF16);
            metal.rms_norm(&mi, &mw, 1e-5, &mout);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 4, 0.05, "rms_norm");
        }

        #[test]
        fn test_metal_vs_cpu_matmul() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            // 2x3 weight, 3-element input
            let w_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let i_data = bf16_bytes(&[1.0, 1.0, 1.0]);

            let cw = cpu.upload_tensor(&w_data, &[2, 3], TensorDtype::BF16);
            let ci = cpu.upload_tensor(&i_data, &[3], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[2], TensorDtype::BF16);
            cpu.matmul(&cw, &ci, &cout, 2, 3);

            let mw = metal.upload_tensor(&w_data, &[2, 3], TensorDtype::BF16);
            let mi = metal.upload_tensor(&i_data, &[3], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[2], TensorDtype::BF16);
            metal.matmul(&mw, &mi, &mout, 2, 3);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 2, 0.5, "matmul");
        }

        #[test]
        fn test_metal_vs_cpu_silu_mul() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let gate_data = bf16_bytes(&[0.0, 1.0, 2.0, -1.0]);
            let up_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

            let cg = cpu.upload_tensor(&gate_data, &[4], TensorDtype::BF16);
            let cu = cpu.upload_tensor(&up_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.silu_mul(&cg, &cu, &cout, 4);

            let mg = metal.upload_tensor(&gate_data, &[4], TensorDtype::BF16);
            let mu = metal.upload_tensor(&up_data, &[4], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[4], TensorDtype::BF16);
            metal.silu_mul(&mg, &mu, &mout, 4);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 4, 0.1, "silu_mul");
        }

        #[test]
        fn test_metal_vs_cpu_scalar_mul() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

            let ci = cpu.upload_tensor(&data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.scalar_mul(&ci, &cout, 2.5, 4);

            let mi = metal.upload_tensor(&data, &[4], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[4], TensorDtype::BF16);
            metal.scalar_mul(&mi, &mout, 2.5, 4);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 4, 0.1, "scalar_mul");
        }

        #[test]
        fn test_metal_vs_cpu_embed_lookup() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let table = bf16_bytes(&[0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]);

            let ct = cpu.upload_tensor(&table, &[3, 3], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[3], TensorDtype::BF16);
            cpu.embed_lookup(&ct, 1, &cout, 3);

            let mt = metal.upload_tensor(&table, &[3, 3], TensorDtype::BF16);
            let mout = metal.alloc_tensor(&[3], TensorDtype::BF16);
            metal.embed_lookup(&mt, 1, &mout, 3);

            assert_tensors_close(&cpu, &cout, &metal, &mout, 3, 0.05, "embed_lookup");
        }

        #[test]
        fn test_metal_vs_cpu_rope() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();
            let q_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let k_data = bf16_bytes(&[5.0, 6.0, 7.0, 8.0]);

            let cq = cpu.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let ck = cpu.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cpu.rope(&cq, &ck, 3, 10000.0, 1, 1, 4);

            let mq = metal.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let mk = metal.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            metal.rope(&mq, &mk, 3, 10000.0, 1, 1, 4);

            assert_tensors_close(&cpu, &cq, &metal, &mq, 4, 0.05, "rope_q");
            assert_tensors_close(&cpu, &ck, &metal, &mk, 4, 0.05, "rope_k");
        }

        // -----------------------------------------------------------------------
        // Attention cross-validation tests
        //
        // Each test exercises Metal vs CPU for a specific attention variant.
        // We test both head_dim=64 (Llama) and head_dim=128 (Qwen/Mistral)
        // to catch hardcoded assumptions in the Metal shaders.
        // -----------------------------------------------------------------------

        /// Deterministic pseudo-random data for tests.  Simple LCG seeded per call.
        fn make_test_data(count: usize, seed: u32) -> Vec<f32> {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    // Map to [-1, 1] range — realistic for normalised hidden states.
                    ((state >> 16) as f32 / 32768.0) - 1.0
                })
                .collect()
        }

        /// Test flat attention (head_dim=64): Metal vs CPU.
        #[test]
        fn test_metal_vs_cpu_attention_hd64() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 64;
            let seq_len: u32 = 20;
            let max_seq: u32 = 32;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));
            let k_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 123));
            let v_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 456));

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let ck = cpu.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.attention(
                &cq,
                &ck,
                &cv,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            let mq = metal.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let mk = metal.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let mv = metal.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let mout =
                metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            metal.attention(
                &mq,
                &mk,
                &mv,
                &mout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                (num_heads * head_dim) as usize,
                0.15,
                "attention_hd64",
            );
        }

        /// Test flat attention (head_dim=128): catches hardcoded head_dim=64 in Metal.
        #[test]
        fn test_metal_vs_cpu_attention_hd128() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let seq_len: u32 = 20;
            let max_seq: u32 = 32;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));
            let k_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 123));
            let v_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 456));

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let ck = cpu.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.attention(
                &cq,
                &ck,
                &cv,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            let mq = metal.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let mk = metal.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let mv = metal.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let mout =
                metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            metal.attention(
                &mq,
                &mk,
                &mv,
                &mout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                (num_heads * head_dim) as usize,
                0.15,
                "attention_hd128",
            );
        }

        /// Test paged attention (head_dim=64): Metal vs CPU with block table.
        #[test]
        fn test_metal_vs_cpu_paged_attention_hd64() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 64;
            let seq_len: u32 = 20;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            // 2 blocks needed for 20 tokens (block 0: pos 0-15, block 1: pos 16-19).
            // Map logical blocks 0,1 to physical blocks 3,1 (non-sequential to test indirection).
            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;
            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));

            // Build paged KV cache by writing each position.
            let k_pool_data = vec![0u8; pool_size * 2];
            let v_pool_data = vec![0u8; pool_size * 2];

            let ck_pool = cpu.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let cv_pool = cpu.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let mk_pool = metal.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let mv_pool = metal.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let mbt =
                metal.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            for pos in 0..seq_len {
                let kv_vec = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let csrc = cpu.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                cpu.copy_to_paged_kv_cache(&csrc, &ck_pool, &cbt, pos, num_kv_heads, head_dim);
                cpu.copy_to_paged_kv_cache(&csrc, &cv_pool, &cbt, pos, num_kv_heads, head_dim);

                let msrc = metal.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                metal.copy_to_paged_kv_cache(&msrc, &mk_pool, &mbt, pos, num_kv_heads, head_dim);
                metal.copy_to_paged_kv_cache(&msrc, &mv_pool, &mbt, pos, num_kv_heads, head_dim);
            }

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.paged_attention(
                &cq,
                &ck_pool,
                &cv_pool,
                &cbt,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let mq = metal.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let mout =
                metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            metal.paged_attention(
                &mq,
                &mk_pool,
                &mv_pool,
                &mbt,
                &mout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                (num_heads * head_dim) as usize,
                0.15,
                "paged_attn_hd64",
            );
        }

        /// Test paged attention (head_dim=128): catches hardcoded head_dim=64.
        #[test]
        fn test_metal_vs_cpu_paged_attention_hd128() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let seq_len: u32 = 20;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;
            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));

            let k_pool_data = vec![0u8; pool_size * 2];
            let v_pool_data = vec![0u8; pool_size * 2];

            let ck_pool = cpu.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let cv_pool = cpu.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let mk_pool = metal.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let mv_pool = metal.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let mbt =
                metal.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            for pos in 0..seq_len {
                let kv_vec = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let csrc = cpu.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                cpu.copy_to_paged_kv_cache(&csrc, &ck_pool, &cbt, pos, num_kv_heads, head_dim);
                cpu.copy_to_paged_kv_cache(&csrc, &cv_pool, &cbt, pos, num_kv_heads, head_dim);

                let msrc = metal.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                metal.copy_to_paged_kv_cache(&msrc, &mk_pool, &mbt, pos, num_kv_heads, head_dim);
                metal.copy_to_paged_kv_cache(&msrc, &mv_pool, &mbt, pos, num_kv_heads, head_dim);
            }

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.paged_attention(
                &cq,
                &ck_pool,
                &cv_pool,
                &cbt,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let mq = metal.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let mout =
                metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            metal.paged_attention(
                &mq,
                &mk_pool,
                &mv_pool,
                &mbt,
                &mout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                (num_heads * head_dim) as usize,
                0.15,
                "paged_attn_hd128",
            );
        }

        /// Test fused paged attention (head_dim=64): KV write + attention in one dispatch.
        #[test]
        fn test_metal_vs_cpu_paged_attention_fused_hd64() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 64;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;

            let ck_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cv_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let mk_pool =
                metal.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let mv_pool =
                metal.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let mbt =
                metal.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            // Write 9 positions via fused path, check the 10th.
            for pos in 0..10u32 {
                let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42 + pos));
                let k_data = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let v_data = bf16_bytes(&make_test_data(kv_dim, 2000 + pos));

                let cq = cpu.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let ck = cpu.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let cv_in = cpu.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let cout =
                    cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                cpu.paged_attention_fused(
                    &cq,
                    &ck,
                    &cv_in,
                    &ck_pool,
                    &cv_pool,
                    &cbt,
                    &cout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                let mq = metal.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let mk = metal.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let mv_in = metal.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let mout =
                    metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                metal.paged_attention_fused(
                    &mq,
                    &mk,
                    &mv_in,
                    &mk_pool,
                    &mv_pool,
                    &mbt,
                    &mout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                if pos == 9 {
                    assert_tensors_close(
                        &cpu,
                        &cout,
                        &metal,
                        &mout,
                        (num_heads * head_dim) as usize,
                        0.15,
                        "paged_attn_fused_hd64",
                    );
                }
            }
        }

        /// Test fused paged attention (head_dim=128): catches hardcoded head_dim=64.
        #[test]
        fn test_metal_vs_cpu_paged_attention_fused_hd128() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;

            let ck_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cv_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let mk_pool =
                metal.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let mv_pool =
                metal.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let mbt =
                metal.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            for pos in 0..10u32 {
                let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42 + pos));
                let k_data = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let v_data = bf16_bytes(&make_test_data(kv_dim, 2000 + pos));

                let cq = cpu.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let ck = cpu.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let cv_in = cpu.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let cout =
                    cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                cpu.paged_attention_fused(
                    &cq,
                    &ck,
                    &cv_in,
                    &ck_pool,
                    &cv_pool,
                    &cbt,
                    &cout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                let mq = metal.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let mk = metal.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let mv_in = metal.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let mout =
                    metal.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                metal.paged_attention_fused(
                    &mq,
                    &mk,
                    &mv_in,
                    &mk_pool,
                    &mv_pool,
                    &mbt,
                    &mout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                if pos == 9 {
                    assert_tensors_close(
                        &cpu,
                        &cout,
                        &metal,
                        &mout,
                        (num_heads * head_dim) as usize,
                        0.15,
                        "paged_attn_fused_hd128",
                    );
                }
            }
        }

        /// Test prefill attention (head_dim=64): causal self-attention on dense Q/K/V.
        #[test]
        fn test_metal_vs_cpu_prefill_attention_hd64() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 64;
            let chunk_size: u32 = 8;
            let start_pos: u32 = 0;
            let q_stride = (num_heads * head_dim) as usize;
            let kv_stride = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data(chunk_size as usize * q_stride, 42));
            let k_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 123));
            let v_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 456));

            let cq =
                cpu.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let ck = cpu.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cout = cpu.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            cpu.prefill_attention(
                &cq,
                &ck,
                &cv,
                &cout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let mq =
                metal.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let mk = metal.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let mv = metal.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let mout = metal.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            metal.prefill_attention(
                &mq,
                &mk,
                &mv,
                &mout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                chunk_size as usize * q_stride,
                0.15,
                "prefill_attn_hd64",
            );
        }

        /// Test prefill attention (head_dim=128): catches hardcoded head_dim=64.
        #[test]
        fn test_metal_vs_cpu_prefill_attention_hd128() {
            let cpu = CpuBackend;
            let metal = MetalBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let chunk_size: u32 = 8;
            let start_pos: u32 = 0;
            let q_stride = (num_heads * head_dim) as usize;
            let kv_stride = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data(chunk_size as usize * q_stride, 42));
            let k_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 123));
            let v_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 456));

            let cq =
                cpu.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let ck = cpu.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cout = cpu.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            cpu.prefill_attention(
                &cq,
                &ck,
                &cv,
                &cout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let mq =
                metal.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let mk = metal.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let mv = metal.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let mout = metal.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            metal.prefill_attention(
                &mq,
                &mk,
                &mv,
                &mout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close(
                &cpu,
                &cout,
                &metal,
                &mout,
                chunk_size as usize * q_stride,
                0.15,
                "prefill_attn_hd128",
            );
        }
    }

    // -----------------------------------------------------------------------
    // CUDA cross-validation tests (Linux only)
    //
    // Same pattern as Metal tests: run the same operation on both CPU and
    // CUDA backends, compare results within tolerance.  This validates that
    // the CUDA kernels produce correct results by cross-referencing against
    // the CPU reference implementation.
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod cuda_cross {
        use super::*;
        use crate::gpu::cuda::CudaBackend;

        fn assert_tensors_close_bf16(
            _cpu_backend: &CpuBackend,
            cpu_tensor: &CpuTensor,
            cuda_backend: &CudaBackend,
            cuda_tensor: &<CudaBackend as GpuCore>::Tensor,
            count: usize,
            tol: f32,
            label: &str,
        ) {
            let cpu_values = read_bf16(cpu_tensor, count);
            let mut cuda_bytes = vec![0u8; count * 2];
            cuda_backend.copy_to_host(cuda_tensor, &mut cuda_bytes);
            let cuda_values = read_bf16_bytes(&cuda_bytes, count);

            for (i, (c, g)) in cpu_values.iter().zip(cuda_values.iter()).enumerate() {
                assert!(
                    (c - g).abs() < tol,
                    "{label} element {i}: cpu={c}, cuda={g}, diff={}",
                    (c - g).abs()
                );
            }
        }

        fn assert_tensors_close_f32(
            cuda_backend: &CudaBackend,
            cuda_tensor: &<CudaBackend as GpuCore>::Tensor,
            expected: &[f32],
            tol: f32,
            label: &str,
        ) {
            let mut cuda_bytes = vec![0u8; expected.len() * 4];
            cuda_backend.copy_to_host(cuda_tensor, &mut cuda_bytes);
            let cuda_values: &[f32] = bytemuck::cast_slice(&cuda_bytes);

            for (i, (g, e)) in cuda_values.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() < tol,
                    "{label} element {i}: cuda={g}, expected={e}, diff={}",
                    (g - e).abs()
                );
            }
        }

        /// Deterministic pseudo-random data for tests.
        fn make_test_data(count: usize, seed: u32) -> Vec<f32> {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    ((state >> 16) as f32 / 32768.0) - 1.0
                })
                .collect()
        }

        // -------------------------------------------------------------------
        // GpuCore tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_core_upload_roundtrip() {
            let cuda = CudaBackend::new().unwrap();
            let data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let t = cuda.upload_tensor(&data, &[4], TensorDtype::BF16);
            let mut dst = vec![0u8; data.len()];
            cuda.copy_to_host(&t, &mut dst);
            assert_eq!(dst, data);
        }

        #[test]
        fn test_cuda_core_alloc_zero() {
            let cuda = CudaBackend::new().unwrap();
            let t = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            let mut dst = vec![0u8; 8];
            cuda.copy_to_host(&t, &mut dst);
            let vals = read_bf16_bytes(&dst, 4);
            for v in &vals {
                assert!((v - 0.0).abs() < 0.001, "expected zero, got {v}");
            }
        }

        #[test]
        fn test_cuda_core_copy_to_tensor() {
            let cuda = CudaBackend::new().unwrap();
            let t = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            let data = bf16_bytes(&[5.0, 6.0, 7.0, 8.0]);
            cuda.copy_to_tensor(&t, &data);
            let mut dst = vec![0u8; 8];
            cuda.copy_to_host(&t, &mut dst);
            let vals = read_bf16_bytes(&dst, 4);
            for (i, (a, e)) in vals.iter().zip([5.0, 6.0, 7.0, 8.0].iter()).enumerate() {
                assert!((a - e).abs() < 0.1, "element {i}: {a} != {e}");
            }
        }

        // -------------------------------------------------------------------
        // GpuElementwise tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_add() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let a_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let b_data = bf16_bytes(&[10.0, 20.0, 30.0, 40.0]);

            let ca = cpu.upload_tensor(&a_data, &[4], TensorDtype::BF16);
            let cb = cpu.upload_tensor(&b_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.add(&ca, &cb, &cout, 4);

            let ga = cuda.upload_tensor(&a_data, &[4], TensorDtype::BF16);
            let gb = cuda.upload_tensor(&b_data, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            cuda.add(&ga, &gb, &gout, 4);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 4, 0.1, "add");
        }

        #[test]
        fn test_cuda_vs_cpu_scalar_mul() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

            let ci = cpu.upload_tensor(&data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.scalar_mul(&ci, &cout, 2.5, 4);

            let gi = cuda.upload_tensor(&data, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            cuda.scalar_mul(&gi, &gout, 2.5, 4);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 4, 0.1, "scalar_mul");
        }

        #[test]
        fn test_cuda_vs_cpu_silu_mul() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let gate_data = bf16_bytes(&[0.0, 1.0, 2.0, -1.0]);
            let up_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);

            let cg = cpu.upload_tensor(&gate_data, &[4], TensorDtype::BF16);
            let cu = cpu.upload_tensor(&up_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.silu_mul(&cg, &cu, &cout, 4);

            let gg = cuda.upload_tensor(&gate_data, &[4], TensorDtype::BF16);
            let gu = cuda.upload_tensor(&up_data, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            cuda.silu_mul(&gg, &gu, &gout, 4);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 4, 0.1, "silu_mul");
        }

        #[test]
        fn test_cuda_vs_cpu_gelu_mul() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let gate_data = bf16_bytes(&[0.0, 1.0, -1.0]);
            let up_data = bf16_bytes(&[1.0, 1.0, 1.0]);

            let cg = cpu.upload_tensor(&gate_data, &[3], TensorDtype::BF16);
            let cu = cpu.upload_tensor(&up_data, &[3], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[3], TensorDtype::BF16);
            cpu.gelu_mul(&cg, &cu, &cout, 3);

            let gg = cuda.upload_tensor(&gate_data, &[3], TensorDtype::BF16);
            let gu = cuda.upload_tensor(&up_data, &[3], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[3], TensorDtype::BF16);
            cuda.gelu_mul(&gg, &gu, &gout, 3);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 3, 0.1, "gelu_mul");
        }

        #[test]
        fn test_cuda_vs_cpu_scale_add() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let dst_data = bf16_bytes(&[1.0, 2.0, 3.0]);
            let src_data = bf16_bytes(&[10.0, 20.0, 30.0]);

            let cdst = cpu.upload_tensor(&dst_data, &[3], TensorDtype::BF16);
            let csrc = cpu.upload_tensor(&src_data, &[3], TensorDtype::BF16);
            cpu.scale_add(&cdst, &csrc, 0.5, 3);

            let gdst = cuda.upload_tensor(&dst_data, &[3], TensorDtype::BF16);
            let gsrc = cuda.upload_tensor(&src_data, &[3], TensorDtype::BF16);
            cuda.scale_add(&gdst, &gsrc, 0.5, 3);

            assert_tensors_close_bf16(&cpu, &cdst, &cuda, &gdst, 3, 0.1, "scale_add");
        }

        #[test]
        fn test_cuda_vs_cpu_fill_zero() {
            let cuda = CudaBackend::new().unwrap();
            let t = cuda.upload_tensor(&bf16_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], TensorDtype::BF16);
            cuda.fill_zero(&t, 4);
            let mut dst = vec![0u8; 8];
            cuda.copy_to_host(&t, &mut dst);
            let vals = read_bf16_bytes(&dst, 4);
            for v in &vals {
                assert!((v - 0.0).abs() < 0.001, "expected zero, got {v}");
            }
        }

        #[test]
        fn test_cuda_vs_cpu_bias_add() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let input_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let bias_data = bf16_bytes(&[0.1, 0.2, 0.3]);

            let ci = cpu.upload_tensor(&input_data, &[2, 3], TensorDtype::BF16);
            let cb = cpu.upload_tensor(&bias_data, &[3], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[2, 3], TensorDtype::BF16);
            cpu.bias_add_batch(&ci, &cb, &cout, 2, 3);

            let gi = cuda.upload_tensor(&input_data, &[2, 3], TensorDtype::BF16);
            let gb = cuda.upload_tensor(&bias_data, &[3], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[2, 3], TensorDtype::BF16);
            cuda.bias_add_batch(&gi, &gb, &gout, 2, 3);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 6, 0.1, "bias_add");
        }

        #[test]
        fn test_cuda_vs_cpu_top_k_softmax() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            // CPU takes f32 logits, CUDA takes bf16 logits.
            let logits_f32 = f32_bytes(&[1.0, 3.0, 2.0, 0.5]);
            let logits_bf16 = bf16_bytes(&[1.0, 3.0, 2.0, 0.5]);

            let cl = cpu.upload_tensor(&logits_f32, &[4], TensorDtype::F32);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::F32);
            cpu.top_k_softmax(&cl, &cout, 4, 2);
            let cpu_result = read_f32(&cout, 4);

            let gl = cuda.upload_tensor(&logits_bf16, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[4], TensorDtype::F32);
            cuda.top_k_softmax(&gl, &gout, 4, 2);

            // Check indices and weights match
            assert_tensors_close_f32(&cuda, &gout, &cpu_result, 0.01, "top_k_softmax");
        }

        // -------------------------------------------------------------------
        // GpuNorm tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_rms_norm() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let input_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let weight_data = bf16_bytes(&[1.0, 1.0, 1.0, 1.0]);

            let ci = cpu.upload_tensor(&input_data, &[4], TensorDtype::BF16);
            let cw = cpu.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[4], TensorDtype::BF16);
            cpu.rms_norm(&ci, &cw, 1e-5, &cout);

            let gi = cuda.upload_tensor(&input_data, &[4], TensorDtype::BF16);
            let gw = cuda.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[4], TensorDtype::BF16);
            cuda.rms_norm(&gi, &gw, 1e-5, &gout);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 4, 0.05, "rms_norm");
        }

        #[test]
        fn test_cuda_vs_cpu_rms_norm_batch() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let input_data = bf16_bytes(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
            let weight_data = bf16_bytes(&[1.0, 1.0, 1.0, 1.0]);

            let ci = cpu.upload_tensor(&input_data, &[2, 4], TensorDtype::BF16);
            let cw = cpu.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[2, 4], TensorDtype::BF16);
            cpu.rms_norm_batch(&ci, &cw, 1e-5, &cout, 2);

            let gi = cuda.upload_tensor(&input_data, &[2, 4], TensorDtype::BF16);
            let gw = cuda.upload_tensor(&weight_data, &[4], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[2, 4], TensorDtype::BF16);
            cuda.rms_norm_batch(&gi, &gw, 1e-5, &gout, 2);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 8, 0.05, "rms_norm_batch");
        }

        // -------------------------------------------------------------------
        // GpuEmbed tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_embed_lookup() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let table = bf16_bytes(&[0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]);

            let ct = cpu.upload_tensor(&table, &[3, 3], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[3], TensorDtype::BF16);
            cpu.embed_lookup(&ct, 1, &cout, 3);

            let gt = cuda.upload_tensor(&table, &[3, 3], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[3], TensorDtype::BF16);
            cuda.embed_lookup(&gt, 1, &gout, 3);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 3, 0.05, "embed_lookup");
        }

        #[test]
        fn test_cuda_vs_cpu_embed_lookup_batch() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let table = bf16_bytes(&[0.1, 0.2, 1.1, 1.2, 2.1, 2.2]);
            let token_ids: Vec<u8> = bytemuck::cast_slice(&[2u32, 0u32]).to_vec();

            let ct = cpu.upload_tensor(&table, &[3, 2], TensorDtype::BF16);
            let cids = cpu.upload_tensor(&token_ids, &[2], TensorDtype::F32);
            let cout = cpu.alloc_tensor(&[2, 2], TensorDtype::BF16);
            cpu.embed_lookup_batch(&ct, &cids, &cout, 2, 2);

            let gt = cuda.upload_tensor(&table, &[3, 2], TensorDtype::BF16);
            let gids = cuda.upload_tensor(&token_ids, &[2], TensorDtype::F32);
            let gout = cuda.alloc_tensor(&[2, 2], TensorDtype::BF16);
            cuda.embed_lookup_batch(&gt, &gids, &gout, 2, 2);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 4, 0.05, "embed_lookup_batch");
        }

        // -------------------------------------------------------------------
        // GpuMatmul tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_matmul() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            // Weight [2, 128], input [128] — must be multiple of 128 for warp-cooperative kernel
            let w_data = bf16_bytes(&make_test_data(2 * 128, 42));
            let i_data = bf16_bytes(&make_test_data(128, 123));

            let cw = cpu.upload_tensor(&w_data, &[2, 128], TensorDtype::BF16);
            let ci = cpu.upload_tensor(&i_data, &[128], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[2], TensorDtype::BF16);
            cpu.matmul(&cw, &ci, &cout, 2, 128);

            let gw = cuda.upload_tensor(&w_data, &[2, 128], TensorDtype::BF16);
            let gi = cuda.upload_tensor(&i_data, &[128], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[2], TensorDtype::BF16);
            cuda.matmul(&gw, &gi, &gout, 2, 128);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 2, 0.5, "matmul");
        }

        #[test]
        fn test_cuda_vs_cpu_matmul_batch() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            let w_data = bf16_bytes(&make_test_data(2 * 128, 42));
            let i_data = bf16_bytes(&make_test_data(3 * 128, 123));

            let cw = cpu.upload_tensor(&w_data, &[2, 128], TensorDtype::BF16);
            let ci = cpu.upload_tensor(&i_data, &[3, 128], TensorDtype::BF16);
            let cout = cpu.alloc_tensor(&[3, 2], TensorDtype::BF16);
            cpu.matmul_batch(&cw, &ci, &cout, 3, 2, 128);

            let gw = cuda.upload_tensor(&w_data, &[2, 128], TensorDtype::BF16);
            let gi = cuda.upload_tensor(&i_data, &[3, 128], TensorDtype::BF16);
            let gout = cuda.alloc_tensor(&[3, 2], TensorDtype::BF16);
            cuda.matmul_batch(&gw, &gi, &gout, 3, 2, 128);

            assert_tensors_close_bf16(&cpu, &cout, &cuda, &gout, 6, 0.5, "matmul_batch");
        }

        // -------------------------------------------------------------------
        // GpuRope tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_rope() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let q_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let k_data = bf16_bytes(&[5.0, 6.0, 7.0, 8.0]);

            let cq = cpu.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let ck = cpu.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cpu.rope(&cq, &ck, 3, 10000.0, 1, 1, 4);

            let gq = cuda.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let gk = cuda.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cuda.rope(&gq, &gk, 3, 10000.0, 1, 1, 4);

            assert_tensors_close_bf16(&cpu, &cq, &cuda, &gq, 4, 0.05, "rope_q");
            assert_tensors_close_bf16(&cpu, &ck, &cuda, &gk, 4, 0.05, "rope_k");
        }

        #[test]
        fn test_cuda_vs_cpu_rope_position_zero() {
            let cuda = CudaBackend::new().unwrap();
            let q_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let k_data = bf16_bytes(&[5.0, 6.0, 7.0, 8.0]);

            let gq = cuda.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let gk = cuda.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cuda.rope(&gq, &gk, 0, 10000.0, 1, 1, 4);

            // At position 0, angles = 0 → identity transform
            let mut q_out = vec![0u8; 8];
            let mut k_out = vec![0u8; 8];
            cuda.copy_to_host(&gq, &mut q_out);
            cuda.copy_to_host(&gk, &mut k_out);
            let q_vals = read_bf16_bytes(&q_out, 4);
            let k_vals = read_bf16_bytes(&k_out, 4);
            for (i, (a, e)) in q_vals.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
                assert!((a - e).abs() < 0.02, "rope q[{i}]: {a} != {e}");
            }
            for (i, (a, e)) in k_vals.iter().zip([5.0, 6.0, 7.0, 8.0].iter()).enumerate() {
                assert!((a - e).abs() < 0.02, "rope k[{i}]: {a} != {e}");
            }
        }

        #[test]
        fn test_cuda_vs_cpu_rope_partial() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();
            let q_data = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
            let k_data = bf16_bytes(&[0.0; 4]);

            let cq = cpu.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let ck = cpu.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cpu.rope_partial(&cq, &ck, 5, 10000.0, 1, 1, 4, 2);

            let gq = cuda.upload_tensor(&q_data, &[4], TensorDtype::BF16);
            let gk = cuda.upload_tensor(&k_data, &[4], TensorDtype::BF16);
            cuda.rope_partial(&gq, &gk, 5, 10000.0, 1, 1, 4, 2);

            assert_tensors_close_bf16(&cpu, &cq, &cuda, &gq, 4, 0.05, "rope_partial_q");
        }

        // -------------------------------------------------------------------
        // GpuAttention tests
        // -------------------------------------------------------------------

        #[test]
        fn test_cuda_vs_cpu_attention_hd128() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let seq_len: u32 = 20;
            let max_seq: u32 = 32;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));
            let k_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 123));
            let v_cache_data = bf16_bytes(&make_test_data(max_seq as usize * kv_dim, 456));

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let ck = cpu.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.attention(
                &cq,
                &ck,
                &cv,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            let gq = cuda.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let gk = cuda.upload_tensor(
                &k_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let gv = cuda.upload_tensor(
                &v_cache_data,
                &[max_seq as usize, kv_dim],
                TensorDtype::BF16,
            );
            let gout =
                cuda.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cuda.attention(
                &gq,
                &gk,
                &gv,
                &gout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
            );

            assert_tensors_close_bf16(
                &cpu,
                &cout,
                &cuda,
                &gout,
                (num_heads * head_dim) as usize,
                0.15,
                "attention_hd128",
            );
        }

        #[test]
        fn test_cuda_vs_cpu_paged_attention_hd128() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let seq_len: u32 = 20;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;
            let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42));

            let k_pool_data = vec![0u8; pool_size * 2];
            let v_pool_data = vec![0u8; pool_size * 2];

            let ck_pool = cpu.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let cv_pool = cpu.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let gk_pool = cuda.upload_tensor(&k_pool_data, &[pool_size], TensorDtype::BF16);
            let gv_pool = cuda.upload_tensor(&v_pool_data, &[pool_size], TensorDtype::BF16);
            let gbt =
                cuda.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            for pos in 0..seq_len {
                let kv_vec = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let csrc = cpu.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                cpu.copy_to_paged_kv_cache(&csrc, &ck_pool, &cbt, pos, num_kv_heads, head_dim);
                cpu.copy_to_paged_kv_cache(&csrc, &cv_pool, &cbt, pos, num_kv_heads, head_dim);

                let gsrc = cuda.upload_tensor(&kv_vec, &[kv_dim], TensorDtype::BF16);
                cuda.copy_to_paged_kv_cache(&gsrc, &gk_pool, &gbt, pos, num_kv_heads, head_dim);
                cuda.copy_to_paged_kv_cache(&gsrc, &gv_pool, &gbt, pos, num_kv_heads, head_dim);
            }

            let cq = cpu.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let cout =
                cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cpu.paged_attention(
                &cq,
                &ck_pool,
                &cv_pool,
                &cbt,
                &cout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let gq = cuda.upload_tensor(
                &q_data,
                &[num_heads as usize, head_dim as usize],
                TensorDtype::BF16,
            );
            let gout =
                cuda.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
            cuda.paged_attention(
                &gq,
                &gk_pool,
                &gv_pool,
                &gbt,
                &gout,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close_bf16(
                &cpu,
                &cout,
                &cuda,
                &gout,
                (num_heads * head_dim) as usize,
                0.15,
                "paged_attn_hd128",
            );
        }

        #[test]
        fn test_cuda_vs_cpu_paged_attention_fused_hd128() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let block_size: u32 = 16;
            let kv_dim = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let num_physical_blocks: u32 = 4;
            let block_table: Vec<u32> = vec![3, 1];
            let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_table).to_vec();

            let pool_size = (num_physical_blocks * block_size) as usize * kv_dim;

            let ck_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cv_pool =
                cpu.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let cbt = cpu.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);
            let gk_pool =
                cuda.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let gv_pool =
                cuda.upload_tensor(&vec![0u8; pool_size * 2], &[pool_size], TensorDtype::BF16);
            let gbt =
                cuda.upload_tensor(&block_table_bytes, &[block_table.len()], TensorDtype::F32);

            for pos in 0..10u32 {
                let q_data = bf16_bytes(&make_test_data((num_heads * head_dim) as usize, 42 + pos));
                let k_data = bf16_bytes(&make_test_data(kv_dim, 1000 + pos));
                let v_data = bf16_bytes(&make_test_data(kv_dim, 2000 + pos));

                let cq = cpu.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let ck = cpu.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let cv_in = cpu.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let cout =
                    cpu.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                cpu.paged_attention_fused(
                    &cq,
                    &ck,
                    &cv_in,
                    &ck_pool,
                    &cv_pool,
                    &cbt,
                    &cout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                let gq = cuda.upload_tensor(
                    &q_data,
                    &[num_heads as usize, head_dim as usize],
                    TensorDtype::BF16,
                );
                let gk = cuda.upload_tensor(&k_data, &[kv_dim], TensorDtype::BF16);
                let gv_in = cuda.upload_tensor(&v_data, &[kv_dim], TensorDtype::BF16);
                let gout =
                    cuda.alloc_tensor(&[num_heads as usize, head_dim as usize], TensorDtype::BF16);
                cuda.paged_attention_fused(
                    &gq,
                    &gk,
                    &gv_in,
                    &gk_pool,
                    &gv_pool,
                    &gbt,
                    &gout,
                    pos,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    0,
                    attn_scale,
                    None,
                );

                if pos == 9 {
                    assert_tensors_close_bf16(
                        &cpu,
                        &cout,
                        &cuda,
                        &gout,
                        (num_heads * head_dim) as usize,
                        0.15,
                        "paged_attn_fused_hd128",
                    );
                }
            }
        }

        #[test]
        fn test_cuda_vs_cpu_prefill_attention_hd128() {
            let cpu = CpuBackend;
            let cuda = CudaBackend::new().unwrap();

            let num_heads: u32 = 4;
            let num_kv_heads: u32 = 2;
            let head_dim: u32 = 128;
            let chunk_size: u32 = 8;
            let start_pos: u32 = 0;
            let q_stride = (num_heads * head_dim) as usize;
            let kv_stride = (num_kv_heads * head_dim) as usize;
            let attn_scale = 1.0 / (head_dim as f32).sqrt();

            let q_data = bf16_bytes(&make_test_data(chunk_size as usize * q_stride, 42));
            let k_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 123));
            let v_data = bf16_bytes(&make_test_data(chunk_size as usize * kv_stride, 456));

            let cq =
                cpu.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let ck = cpu.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cv = cpu.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let cout = cpu.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            cpu.prefill_attention(
                &cq,
                &ck,
                &cv,
                &cout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            let gq =
                cuda.upload_tensor(&q_data, &[chunk_size as usize, q_stride], TensorDtype::BF16);
            let gk = cuda.upload_tensor(
                &k_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let gv = cuda.upload_tensor(
                &v_data,
                &[chunk_size as usize, kv_stride],
                TensorDtype::BF16,
            );
            let gout = cuda.alloc_tensor(&[chunk_size as usize, q_stride], TensorDtype::BF16);
            cuda.prefill_attention(
                &gq,
                &gk,
                &gv,
                &gout,
                chunk_size,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                0,
                attn_scale,
                None,
            );

            assert_tensors_close_bf16(
                &cpu,
                &cout,
                &cuda,
                &gout,
                chunk_size as usize * q_stride,
                0.15,
                "prefill_attn_hd128",
            );
        }
    }
}
