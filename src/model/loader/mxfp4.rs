// ===========================================================================
// MXFP4 dequantization — converts microscaling FP4 packed format to bf16.
//
// Related files:
//   loader/mod.rs — load_mxfp4_experts() calls dequantize_mxfp4()
// ===========================================================================

#[allow(unused_imports)]
use half::bf16;

// ---------------------------------------------------------------------------
// MXFP4 dequantization — converts microscaling FP4 packed format to bf16.
//
// MXFP4 (Microscaling FP4) stores weights using 4-bit FP (E2M1 format):
//   - 1 sign bit, 2 exponent bits, 1 mantissa bit → 16 distinct values
//   - Two values packed per byte (low nibble = even index, high nibble = odd)
//   - Block scaling: every `block_size` elements share one scale factor
//
// Layout on disk (for a weight of shape [rows, cols]):
//   blocks: [rows, cols/2] bytes (packed pairs)
//   scales: [rows, cols/block_size] bf16 scale factors
//   bias:   [rows] bf16 per-row bias (optional, added after dequant)
//
// FP4 E2M1 encoding:
//   nibble  value       nibble  value
//   0b0000   0.0        0b1000  -0.0
//   0b0001   0.5        0b1001  -0.5
//   0b0010   1.0        0b1010  -1.0
//   0b0011   1.5        0b1011  -1.5
//   0b0100   2.0        0b1100  -2.0
//   0b0101   3.0        0b1101  -3.0
//   0b0110   4.0        0b1110  -4.0
//   0b0111   6.0        0b1111  -6.0
// ---------------------------------------------------------------------------

/// Lookup table: FP4 E2M1 nibble → f32 value.
pub(crate) const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive (sign=0)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative (sign=1)
];

/// Decode an E8M0 scale byte to f32.
///
/// E8M0 is an 8-bit exponent-only format used by MXFP4 for per-block scales:
///   value = 2^(byte - 127)
/// Special cases: 0 → 0.0, 255 → NaN.
#[inline]
pub(crate) fn e8m0_to_f32(byte: u8) -> f32 {
    match byte {
        0 => 0.0,
        255 => f32::NAN,
        e => f32::from_bits((e as u32) << 23), // 2^(e-127) via IEEE 754
    }
}

/// Dequantize MXFP4 packed blocks to bf16 using per-block E8M0 scaling.
///
/// Arguments:
///   - `blocks`: packed fp4 data, 2 values per byte, shape [rows, cols/2]
///   - `scales`: E8M0 block scales (1 byte each), shape [rows, num_scale_blocks]
///   - `rows`, `cols`: logical weight shape
///   - `block_size`: number of elements per scale block (typically 32)
///
/// Returns: bf16 bytes for [rows, cols] weight tensor.
pub(crate) fn dequantize_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block_size: usize,
) -> Vec<u8> {
    let num_scale_blocks = (cols + block_size - 1) / block_size;
    let mut out = vec![half::bf16::ZERO; rows * cols];

    for r in 0..rows {
        let row_block_offset = r * (cols / 2);
        let row_scale_offset = r * num_scale_blocks;
        for c in 0..cols {
            let byte_idx = row_block_offset + c / 2;
            let nibble = if c % 2 == 0 {
                blocks[byte_idx] & 0x0F // low nibble
            } else {
                (blocks[byte_idx] >> 4) & 0x0F // high nibble
            };
            let fp4_val = FP4_E2M1_LUT[nibble as usize];
            let scale_idx = row_scale_offset + c / block_size;
            let scale = e8m0_to_f32(scales[scale_idx]);
            out[r * cols + c] = half::bf16::from_f32(fp4_val * scale);
        }
    }

    bytemuck::cast_slice(&out).to_vec()
}
