// ---------------------------------------------------------------------------
// WeightQuantiser — format-agnostic weight quantisation trait.
//
// LEARNING OVERVIEW
//
// Weight quantisation compresses bf16 weight matrices into smaller block-wise
// formats (Q4, Q8, etc.) to reduce memory bandwidth and GPU memory usage.
// Each format defines:
//   - A block size (e.g. 32 weights per block)
//   - A block layout (scale + compressed values)
//   - A dequantisation rule (how GPU kernels reconstruct weights)
//
// This trait abstracts over the format so the loader, quantize command, and
// memory estimator work with any format without knowing its internals.
//
// CURRENT FORMATS
//
//   Q4 — 4-bit symmetric, 18 bytes per block of 32 weights.
//         Block: [bf16 scale | 16 packed nibble bytes]
//         scale = max(|w|) / 7, q = clamp(round(w/scale), -8, 7) + 8
//         GPU kernels: matvec_q4, gemm_q4, fused_gate_up_swiglu_q4
//
//   Q8 — 8-bit symmetric, 34 bytes per block of 32 weights.
//         Block: [bf16 scale | 32 signed int8 values]
//         scale = max(|w|) / 127, q = clamp(round(w/scale), -128, 127)
//         GPU kernels: matvec_q8, gemm_q8, fused_gate_up_swiglu_q8
//
// ADDING A NEW FORMAT
//
//   1. Add a variant to QuantFormat (below)
//   2. Add a variant to TensorDtype in gpu/mod.rs
//   3. Implement WeightQuantiser for the new format
//   4. Add GPU kernels (Metal + CUDA shaders) for dequant-matmul
//   5. Wire the format into pipeline selection (matmul.rs, moe.rs dispatch)
//   6. Add metadata key to the quantize command (commands/quantize.rs)
//
// Related files:
//   gpu/mod.rs              — TensorDtype enum, Q4 quantise function
//   gpu/ops/core.rs         — GpuCore::quantize_upload (calls into this trait)
//   model/loader/         — weight loading, pre-quantized detection
//   commands/quantize.rs    — offline quantization CLI
//   metal/shaders/matmul.metal — GPU dequant-matmul kernels
//   metal/kernels/matmul.rs    — pipeline dispatch by dtype
// ---------------------------------------------------------------------------

use super::super::TensorDtype;

// ---------------------------------------------------------------------------
// QuantFormat — identifies a weight quantisation format.
//
// Used by the CLI (`--quant q4`), the loader (to detect pre-quantized files),
// and the quantise command (to select the output format).
// ---------------------------------------------------------------------------

/// Weight quantisation format selector.
///
/// Each variant corresponds to a `WeightQuantiser` implementation and a
/// `TensorDtype` variant.  The mapping is:
///   Q4 → TensorDtype::Q4 (18 bytes per block of 32 weights)
///   Q8 → TensorDtype::Q8 (34 bytes per block of 32 weights)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantFormat {
    /// Block-wise 4-bit symmetric quantisation.
    /// 32 weights per block, 18 bytes per block (2-byte bf16 scale + 16 packed nibbles).
    Q4,
    /// Block-wise 8-bit symmetric quantisation.
    /// 32 weights per block, 34 bytes per block (2-byte bf16 scale + 32 signed int8 values).
    Q8,
}

impl QuantFormat {
    /// Parse a format name from CLI arguments or config.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "q4" | "Q4" => Some(Self::Q4),
            "q8" | "Q8" => Some(Self::Q8),
            _ => None,
        }
    }

    /// Short name for display and metadata keys.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Q4 => "q4",
            Self::Q8 => "q8",
        }
    }

    /// The `TensorDtype` variant that GPU kernels use to dispatch.
    pub fn dtype(&self) -> TensorDtype {
        match self {
            Self::Q4 => TensorDtype::Q4,
            Self::Q8 => TensorDtype::Q8,
        }
    }

    /// Metadata marker stored in safetensors files (e.g. `"rllm-q4"`).
    pub fn metadata_tag(&self) -> &'static str {
        match self {
            Self::Q4 => "rllm-q4",
            Self::Q8 => "rllm-q8",
        }
    }

    /// Prefix for per-tensor metadata keys (e.g. `"rllm_q4:"`).
    pub fn metadata_prefix(&self) -> &'static str {
        match self {
            Self::Q4 => "rllm_q4:",
            Self::Q8 => "rllm_q8:",
        }
    }

    /// Detect format from safetensors metadata "quantization" value.
    pub fn from_metadata_tag(tag: &str) -> Option<Self> {
        match tag {
            "rllm-q4" => Some(Self::Q4),
            "rllm-q8" => Some(Self::Q8),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightQuantiser — the core trait for quantisation formats.
//
// Implementations are stateless — all parameters are in the method signatures.
// This trait covers the CPU-side operations: quantise, byte counting, and
// format metadata.  The GPU-side dequantisation lives in the Metal/CUDA
// shaders and is selected by `TensorDtype` dispatch in the kernel impls.
// ---------------------------------------------------------------------------

/// Trait for block-wise weight quantisation formats.
///
/// Each format (Q4, Q8, ...) implements this trait.  The loader and quantize
/// command call these methods through `QuantFormat::quantiser()` without
/// knowing which format is in use.
pub(crate) trait WeightQuantiser: Send + Sync {
    /// Which format this quantiser produces.
    fn format(&self) -> QuantFormat;

    /// Number of weights per quantisation block.
    fn block_size(&self) -> usize;

    /// Bytes per block in the quantised representation.
    fn bytes_per_block(&self) -> usize;

    /// Total byte count for a quantised [m, k] weight matrix.
    ///
    /// Panics on overflow to prevent silent undersized GPU buffer allocations.
    fn byte_count(&self, m: usize, k: usize) -> usize {
        let bs = self.block_size();
        assert!(
            k % bs == 0,
            "K ({k}) must be divisible by block_size ({})",
            bs
        );
        let blocks_per_row = k / bs;
        let bpb = self.bytes_per_block();
        m.checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(bpb))
            .unwrap_or_else(|| {
                panic!(
                    "{} byte_count overflow: [{m}, {k}] too large",
                    self.format().name()
                )
            })
    }

    /// Row stride in bytes (bytes per row in the quantised layout).
    fn row_bytes(&self, k: usize) -> usize {
        (k / self.block_size()) * self.bytes_per_block()
    }

    /// Quantise a bf16 weight matrix [m, k] to this format.
    ///
    /// Input: raw bf16 bytes (little-endian, m*k*2 bytes).
    /// Output: quantised bytes ready for GPU upload.
    fn quantise(&self, bf16_data: &[u8], m: usize, k: usize) -> Vec<u8>;

    /// The `TensorDtype` for GPU kernel dispatch.
    fn dtype(&self) -> TensorDtype {
        self.format().dtype()
    }
}

// ---------------------------------------------------------------------------
// Q4 quantiser — delegates to the existing quantize_bf16_to_q4().
// ---------------------------------------------------------------------------

pub(crate) struct Q4Quantiser;

impl WeightQuantiser for Q4Quantiser {
    fn format(&self) -> QuantFormat {
        QuantFormat::Q4
    }

    fn block_size(&self) -> usize {
        32
    }

    fn bytes_per_block(&self) -> usize {
        18
    }

    fn quantise(&self, bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
        super::super::quantize_bf16_to_q4(bf16_data, m, k)
    }
}

// ---------------------------------------------------------------------------
// Q8 quantiser — delegates to quantize_bf16_to_q8().
// ---------------------------------------------------------------------------

pub(crate) struct Q8Quantiser;

impl WeightQuantiser for Q8Quantiser {
    fn format(&self) -> QuantFormat {
        QuantFormat::Q8
    }

    fn block_size(&self) -> usize {
        32
    }

    fn bytes_per_block(&self) -> usize {
        34
    }

    fn quantise(&self, bf16_data: &[u8], m: usize, k: usize) -> Vec<u8> {
        super::super::quantize_bf16_to_q8(bf16_data, m, k)
    }
}

// ---------------------------------------------------------------------------
// Factory — get the quantiser for a given format.
// ---------------------------------------------------------------------------

/// Returns a static reference to the quantiser for the given format.
pub(crate) fn quantiser(format: QuantFormat) -> &'static dyn WeightQuantiser {
    match format {
        QuantFormat::Q4 => &Q4Quantiser,
        QuantFormat::Q8 => &Q8Quantiser,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_byte_count_matches_legacy() {
        let q = quantiser(QuantFormat::Q4);
        // Compare against the standalone q4_byte_count function.
        assert_eq!(q.byte_count(1, 32), crate::gpu::q4_byte_count(1, 32));
        assert_eq!(q.byte_count(1, 64), crate::gpu::q4_byte_count(1, 64));
        assert_eq!(q.byte_count(4, 64), crate::gpu::q4_byte_count(4, 64));
        assert_eq!(q.byte_count(2048, 2048), crate::gpu::q4_byte_count(2048, 2048));
    }

    #[test]
    fn test_q4_row_bytes() {
        let q = quantiser(QuantFormat::Q4);
        // 32 weights / 32 per block = 1 block * 18 bytes = 18
        assert_eq!(q.row_bytes(32), 18);
        // 2048 weights / 32 per block = 64 blocks * 18 bytes = 1152
        assert_eq!(q.row_bytes(2048), 64 * 18);
    }

    #[test]
    fn test_format_round_trip() {
        let fmt = QuantFormat::from_name("q4").unwrap();
        assert_eq!(fmt.name(), "q4");
        assert_eq!(fmt.dtype(), TensorDtype::Q4);
        assert_eq!(fmt.metadata_tag(), "rllm-q4");

        let fmt2 = QuantFormat::from_metadata_tag("rllm-q4").unwrap();
        assert_eq!(fmt, fmt2);
    }

    #[test]
    fn test_format_from_name_case_insensitive() {
        assert!(QuantFormat::from_name("Q4").is_some());
        assert!(QuantFormat::from_name("q4").is_some());
        assert!(QuantFormat::from_name("Q8").is_some());
        assert!(QuantFormat::from_name("q8").is_some());
        assert!(QuantFormat::from_name("q16").is_none());
    }

    #[test]
    fn test_q8_byte_count_matches_legacy() {
        let q = quantiser(QuantFormat::Q8);
        assert_eq!(q.byte_count(1, 32), crate::gpu::q8_byte_count(1, 32));
        assert_eq!(q.byte_count(1, 64), crate::gpu::q8_byte_count(1, 64));
        assert_eq!(q.byte_count(4, 64), crate::gpu::q8_byte_count(4, 64));
        assert_eq!(q.byte_count(2048, 2048), crate::gpu::q8_byte_count(2048, 2048));
    }

    #[test]
    fn test_q8_row_bytes() {
        let q = quantiser(QuantFormat::Q8);
        // 32 weights / 32 per block = 1 block * 34 bytes = 34
        assert_eq!(q.row_bytes(32), 34);
        // 2048 weights / 32 per block = 64 blocks * 34 bytes = 2176
        assert_eq!(q.row_bytes(2048), 64 * 34);
    }

    #[test]
    fn test_q8_format_round_trip() {
        let fmt = QuantFormat::from_name("q8").unwrap();
        assert_eq!(fmt.name(), "q8");
        assert_eq!(fmt.dtype(), TensorDtype::Q8);
        assert_eq!(fmt.metadata_tag(), "rllm-q8");

        let fmt2 = QuantFormat::from_metadata_tag("rllm-q8").unwrap();
        assert_eq!(fmt, fmt2);
    }

    #[test]
    fn test_q8_quantise_round_trip_accuracy() {
        use half::bf16;

        let m = 2;
        let k = 64;
        let values: Vec<bf16> = (0..m * k)
            .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
            .collect();
        let bf16_bytes: &[u8] = bytemuck::cast_slice(&values);

        let q = quantiser(QuantFormat::Q8);
        let q8_data = q.quantise(bf16_bytes, m, k);
        assert_eq!(q8_data.len(), q.byte_count(m, k));

        // Dequantize on CPU and check max error is within one quantisation step.
        let blocks_per_row = k / 32;
        let mut max_err: f32 = 0.0;
        for row in 0..m {
            for block in 0..blocks_per_row {
                let offset = (row * blocks_per_row + block) * 34;
                let scale_bits = u16::from_le_bytes([q8_data[offset], q8_data[offset + 1]]);
                let scale = bf16::from_bits(scale_bits).to_f32();
                for i in 0..32 {
                    let q_val = q8_data[offset + 2 + i] as i8;
                    let dequant = q_val as f32 * scale;
                    let orig = values[row * k + block * 32 + i].to_f32();
                    let err = (orig - dequant).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        // Max error should be at most one quantisation step (scale).
        // The largest scale in our test data is about 0.8/127 ≈ 0.0063.
        assert!(max_err < 0.02, "max Q8 round-trip error {max_err} too large");
    }

    #[test]
    #[should_panic(expected = "byte_count overflow")]
    fn test_byte_count_overflow_panics() {
        let q = quantiser(QuantFormat::Q4);
        let _ = q.byte_count(usize::MAX / 2, 64);
    }

    #[test]
    #[should_panic(expected = "must be divisible by block_size")]
    fn test_byte_count_misaligned_k_panics() {
        let q = quantiser(QuantFormat::Q4);
        let _ = q.byte_count(1, 17); // not divisible by 32
    }
}
