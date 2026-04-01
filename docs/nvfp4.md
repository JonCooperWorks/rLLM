# NVFP4 Weight Quantization (NVIDIA Blackwell)

NVFP4 E2M1 is a 4-bit floating-point format for weight storage on NVIDIA
Blackwell GPUs (SM 100+).  It replaces Q4 block format on these platforms,
using the same block layout (18 bytes per 32 weights) but with E2M1 float
values instead of symmetric integers.

**Key files:**
- `src/gpu/mod.rs` — `TensorDtype::NVFP4`, `quantize_bf16_to_nvfp4()`, `nvfp4_byte_count()`
- `src/gpu/ops/quant.rs` — `QuantFormat::NVFP4`, `NVFP4Quantiser`
- `src/gpu/cuda/shaders/matmul.cu` — `matvec_nvfp4`, `gemm_nvfp4` CUDA kernels
- `src/gpu/cuda/shaders/matmul_tc.cu` — `gemm_nvfp4_tc` tensor-core GEMM
- `src/gpu/cuda/shaders/moe.cu` — `fused_gate_up_swiglu_nvfp4` MoE kernel
- `src/commands/quantize.rs` — platform-aware `--format q4` dispatch
- `src/model/loader/` — pre-quantized NVFP4 detection (`"rllm-nvfp4"` metadata tag)

---

## NVFP4 E2M1 Format

| Field | Bits | Description |
|-------|------|-------------|
| Sign | 1 | 0 = positive, 1 = negative |
| Exponent | 2 | Biased by 1 (range 0–3) |
| Mantissa | 1 | Implicit leading 1 for normals |

**Representable values (16 entries):**

| Nibble | Hex | Value | Nibble | Hex | Value |
|--------|-----|-------|--------|-----|-------|
| 0000 | 0 | +0.0 | 1000 | 8 | -0.0 |
| 0001 | 1 | +0.5 | 1001 | 9 | -0.5 |
| 0010 | 2 | +1.0 | 1010 | A | -1.0 |
| 0011 | 3 | +1.5 | 1011 | B | -1.5 |
| 0100 | 4 | +2.0 | 1100 | C | -2.0 |
| 0101 | 5 | +3.0 | 1101 | D | -3.0 |
| 0110 | 6 | +4.0 | 1110 | E | -4.0 |
| 0111 | 7 | +6.0 | 1111 | F | -6.0 |

**Properties:**
- Range: ±6.0
- 16 distinct values (8 positive + 8 negative)
- Per-block bf16 scale factor compensates for limited range
- Same encoding as MXFP4 E2M1 (see `model/loader/mxfp4.rs`)

---

## Block Layout

Same as Q4 — 18 bytes per block of 32 weights:

```
[0..2]:   bf16 scale (little-endian)
[2..18]:  16 packed nibble bytes (2 E2M1 values per byte)
          byte[i] = nibble[2i] | (nibble[2i+1] << 4)
```

**Byte count:** `m × (k / 32) × 18` — identical to Q4.

**Quantization:**
1. `scale = max(|w_i|) / 6.0` per block of 32 weights
2. Each weight: `w / scale` → nearest E2M1 nibble via 16-entry LUT search
3. Pack two nibbles per byte (low nibble first)

**Dequantization (GPU):**
```
weight = FP4_E2M1_LUT[nibble] × scale
```

The 16-entry LUT is stored in CUDA `__constant__` memory for single-cycle
access.  No arithmetic dequant needed — just a table lookup + multiply.

---

## Platform Detection

`--format q4` transparently uses NVFP4 on Blackwell:

```
rllm quantize --format q4 --model /path/to/model --output /path/to/output
```

On SM 100+ (Blackwell), this produces NVFP4 files with `"rllm-nvfp4"` metadata.
On older GPUs and Metal, standard Q4 symmetric integer format is used.

`--format nvfp4` can also be specified explicitly.

---

## Comparison with Q4

| Property | Q4 (symmetric int) | NVFP4 (E2M1) |
|----------|-------------------|---------------|
| Bits per weight | 4 | 4 |
| Block size | 32 | 32 |
| Bytes per block | 18 | 18 |
| Scale type | bf16 | bf16 |
| Dequant | `(nibble - 8) × scale` | `LUT[nibble] × scale` |
| Value distribution | Uniform [-8, 7] | Non-uniform, denser near 0 |
| Platform | All | CUDA SM 100+ |

NVFP4's non-uniform value spacing (denser near zero) better matches the
typical weight distribution in neural networks, where small values dominate.

---

## CUDA Kernels

| Kernel | Dispatch | Purpose |
|--------|----------|---------|
| `matvec_nvfp4` | M×32 threads | Single-token decode |
| `gemm_nvfp4` | batch×M×32 threads | Batched prefill (scalar) |
| `gemm_nvfp4_tc` | 128×128 tiles, 256 threads | Tensor-core prefill (SM 80+) |
| `fused_gate_up_swiglu_nvfp4` | M×32 threads | MoE expert FFN |

All kernels use the same block iteration as Q4 — only the dequant line differs.

---

## References

1. NVIDIA. "FP4 Formats for Deep Learning." Technical Brief, 2024.
2. Rouhani et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537, 2023.
