# FP8 Weight Quantization (NVIDIA)

FP8 E4M3 is an IEEE 8-bit floating-point format used for weight storage on
NVIDIA Ada and Hopper GPUs (SM 89+).  It replaces Q8 block format on these
platforms for better memory efficiency and native hardware support.

**Key files:**
- `src/gpu/mod.rs` — `TensorDtype::FP8`, `quantize_bf16_to_fp8()`, `fp8_byte_count()`
- `src/gpu/ops/quant.rs` — `QuantFormat::FP8`, `FP8Quantiser`
- `src/gpu/cuda/shaders/matmul.cu` — `matvec_fp8`, `gemm_fp8` CUDA kernels
- `src/gpu/cuda/shaders/matmul_tc.cu` — `gemm_fp8_tc` tensor-core GEMM
- `src/gpu/cuda/shaders/moe.cu` — `fused_gate_up_swiglu_fp8` MoE kernel
- `src/commands/quantize.rs` — platform-aware `--format q8` dispatch
- `src/model/loader/` — pre-quantized FP8 detection (`"rllm-fp8"` metadata tag)

---

## FP8 E4M3 Format

| Field | Bits | Description |
|-------|------|-------------|
| Sign | 1 | 0 = positive, 1 = negative |
| Exponent | 4 | Biased by 7 (range 0–15) |
| Mantissa | 3 | Implicit leading 1 for normals |

**Properties:**
- Range: ±448 (max normal: `(1 + 6/8) × 2^8 = 448`)
- Precision: ~0.125 at magnitude 1.0 (3-bit mantissa)
- Min subnormal: `2^-9 ≈ 0.00195`
- No infinity representation (E=1111 with M<111 are valid normals)
- NaN: `0x7F` (E=1111, M=111)

---

## Why FP8 on NVIDIA, Not Metal

| Property | FP8 E4M3 | Q8 Blocks |
|----------|---------|-----------|
| Bytes per weight | 1 | 34/32 ≈ 1.06 |
| Block structure | None | 32-weight blocks with bf16 scale |
| NVIDIA hardware support | Native on SM 89+ | Software dequant |
| Apple Silicon support | None | Full software support |
| Dequant complexity | Format conversion only | Scale multiply per weight |

FP8 eliminates the per-block scale overhead and uses simpler linear addressing.
On SM 89+, the hardware can process FP8 values natively in tensor cores.
On Metal, there is no FP8 hardware — Q8 blocks with software dequant perform
identically, so FP8 would add complexity with zero benefit.

---

## Platform Dispatch

The user always types `q8`.  The system selects the physical format:

```
--quant q8  or  --format q8
  │
  ├── NVIDIA SM 89+ (Ada/Hopper)  →  FP8 E4M3  (metadata: "rllm-fp8")
  ├── NVIDIA SM < 89 (Ampere etc) →  Q8 blocks  (metadata: "rllm-q8")
  └── Apple Silicon (Metal)       →  Q8 blocks  (metadata: "rllm-q8")
```

Detection happens at quantization time (`rllm quantize`) by querying the GPU's
compute capability.  Pre-quantized models carry the format in safetensors
metadata, so loading is automatic — the loader detects `"rllm-fp8"` and
uploads raw FP8 bytes with `TensorDtype::FP8`.

---

## Byte Layout

FP8 has no block structure.  A weight matrix `[m, k]` is stored as `m × k`
bytes, where each byte is an independent FP8 E4M3 value:

```
Weight[0,0]  Weight[0,1]  ...  Weight[0,k-1]  Weight[1,0]  ...
  1 byte       1 byte            1 byte         1 byte
```

Byte count: `m * k` (available as `gpu::fp8_byte_count(m, k)`).

---

## Conversion Algorithm (bf16 → FP8 E4M3)

```
1. Read bf16 value (2 bytes LE)
2. Convert to f32
3. Handle special cases:
   - Zero → FP8 zero (preserving sign)
   - NaN/Inf → 0x7F (FP8 NaN)
   - |value| > 448 → clamp to max (0x7E or 0xFE)
   - |value| < min_subnormal → flush to zero
4. Normal conversion:
   - Extract sign, exponent, mantissa from f32
   - Rebias exponent: f32 bias (127) → E4M3 bias (7)
   - Round mantissa from 23 bits to 3 bits (round-to-nearest-even)
   - Handle mantissa overflow (increment exponent)
5. Pack: [sign:1][exp:4][man:3] → 1 byte
```

Implemented in `src/gpu/mod.rs::f32_to_fp8_e4m3()` as pure bit manipulation
(no external crate dependencies).

---

## CUDA Kernels

### Dequantization

Each kernel contains an inline `fp8_e4m3_to_float()` function that converts
FP8 bytes to f32 via bit manipulation:

```cuda
__device__ float fp8_e4m3_to_float(unsigned char bits) {
    // Normal: rebias exponent from 7 to 127, shift mantissa
    unsigned int f32_bits = (sign << 31) | ((exp - 7 + 127) << 23) | (man << 20);
    return __uint_as_float(f32_bits);
}
```

### Kernel Variants

| Kernel | Use Case | Pattern |
|--------|----------|---------|
| `matvec_fp8` | Decode (single token) | 32 threads/row, warp-cooperative |
| `gemm_fp8` | Prefill (batched, scalar) | batch × M × 32 threads |
| `gemm_fp8_tc` | Prefill (tensor-core WMMA) | Dequant during tile load |
| `fused_gate_up_swiglu_fp8` | MoE FFN | Fused gate+up+SiLU |

All follow the same warp-cooperative pattern as Q8 but with simpler addressing
(no block headers, no scale multiplication).

---

## Safetensors Metadata

Pre-quantized FP8 files use these metadata keys:

```
__metadata__.quantization = "rllm-fp8"
__metadata__.rllm_fp8:<tensor_name> = "m,k"
```

The loader detects `"rllm-fp8"` and populates `TensorStore::fp8_map` with
the original logical shapes for byte count validation during upload.

---

See also: [Quantization](quantization.md) · [GPU Backend](gpu-backend.md) ·
[Expert Streaming](expert-streaming.md)
