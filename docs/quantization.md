# Quantization

rLLM uses 4-bit symmetric quantization (Q4) to compress model weights ~3.5x,
cutting memory bandwidth and directly increasing decode throughput on
bandwidth-bound workloads.

**Key files:**
- `src/gpu/mod.rs` — Q4 block format, `quantize_bf16_to_q4()`, `q4_byte_count()`
- `src/commands/quantize.rs` — CLI `rllm quantize` command
- `src/model/loader/` — pre-quantized model detection and loading
- `src/gpu/metal/shaders/matmul.metal` — Metal Q4 dequantization kernels
- `src/gpu/cuda/shaders/matmul.cu` — CUDA Q4 dequantization kernels
- `src/gpu/cpu/mod.rs` — CPU reference Q4 implementation

---

## Block Format

Weights are quantized in blocks of 32.  Each block is 18 bytes:

| Offset | Size | Content |
|--------|------|---------|
| 0 | 2 bytes | bf16 scale factor (little-endian) |
| 2 | 16 bytes | 32 packed 4-bit nibbles (2 per byte) |

Nibble packing: `byte[i] = (q[2i] & 0xF) | (q[2i+1] << 4)`.

Compared to bf16 (64 bytes for 32 weights), Q4 is 3.56x smaller.

### Why bf16 Scale

The scale could be stored as f32 (4 bytes) for slightly more precision, but
bf16 (2 bytes) saves 10% I/O per block.  This matters for
[expert streaming](expert-streaming.md) where NVMe bandwidth is the bottleneck —
3.5x less I/O per expert load with Q4 vs bf16 weights.

---

## Quantization Scheme

Symmetric quantization maps the weight range to 4-bit signed integers:

```
scale = max(|w_i|) / 7.0
q_i   = clamp(round(w_i / scale), -8, 7)
u_i   = q_i + 8                             // offset to unsigned [0, 15]
```

Dequantization (in GPU kernels):

```
weight = (nibble - 8) * scale
```

The scale is computed per block (32 weights), so each block has independent
dynamic range.  This is coarser than per-channel quantization but simpler and
fast to dequantize in a single FMA per weight.

---

## Pre-Quantizing Models

The `rllm quantize` command converts bf16 safetensors to Q4 on disk:

```bash
rllm quantize --model models/llama-3.1-8b --output models/llama-3.1-8b-q4
```

### What gets quantized

Only 2D weight matrices with K divisible by 32 and matching projection
keywords: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, `in_proj`, `out_proj`, `lm_head`.

3D fused expert tensors like `[num_experts, rows, K]` are flattened to
`[num_experts * rows, K]` and quantized per-row — identical results to
quantizing each expert separately.

**Not quantized:** norms, embeddings, conv1d, router gates, biases.  These are
small and sensitive to precision loss.

### Output format

The quantized safetensors files store Q4 tensors as raw `U8` bytes with
metadata that records the original logical shape:

```
__metadata__.quantization = "rllm-q4"
__metadata__.rllm_q4:<tensor_name> = "m,k"
```

Multi-shard models produce a standard `model.safetensors.index.json`.  Default
shard limit is 5 GB.

---

## Loading Pre-Quantized Models

The loader auto-detects pre-quantized models by checking the safetensors
metadata for `quantization = "rllm-q4"`.  No flags needed — just point at
the quantized directory:

```bash
rllm run --model models/llama-3.1-8b-q4 --prompt "Hello"
```

For each Q4 tensor, the loader:
1. Reads the original `(m, k)` shape from metadata
2. Validates byte count: `q4_byte_count(m, k) == data.len()`
3. Uploads raw Q4 bytes to the GPU with shape `[m, k]` and dtype `Q4`

Non-quantized tensors (norms, embeddings) are loaded normally as bf16.

### Runtime quantization

Models can also be quantized on-the-fly during loading (without pre-quantizing
to disk).  The `GpuCore::quantize_upload()` method converts bf16 weights to Q4
in CPU memory and uploads the result.  This is slower than loading
pre-quantized files but avoids the separate quantize step.

---

## GPU Kernel Dequantization

Q4 weights are dequantized on-the-fly during matrix multiplication — no
separate dequantization pass.  Each platform has optimised kernels.

### Metal (`matvec_q4`)

The Metal kernel uses SIMD-cooperative loading with shared memory:

- 256 threads (8 SIMD groups) cooperatively load the input vector into
  threadgroup memory, tiled in 4096-element chunks
- Each thread handles blocks for its assigned output row
- Dequantization uses FMA: `acc = fma(nibble, scale*x, -8*scale*x)` —
  precomputing `scale*x` eliminates a multiply per weight
- Input vector loaded once per 8 rows instead of once per row (8x bandwidth
  reduction on the input side)

### CUDA (`matvec_q4`)

The CUDA kernel selects between two access patterns at runtime:

- **Per-lane** (K ≤ 2048): each warp lane owns separate blocks, strided by 32.
  Better L1 locality for small matrices (MoE experts, attention projections)
- **Cooperative** (K > 2048): all 32 lanes cooperate on the same block, using
  `__shfl_sync` to broadcast the scale.  Better memory coalescing for large K

Both patterns read bf16 scale from the first 2 bytes, nibbles from bytes 2–17.

### Batched GEMM (`gemm_q4`)

Prefill uses batched GEMM (matrix × matrix) instead of mat-vec.  The Q4 GEMM
kernels apply the same per-block dequantization but across multiple input
vectors simultaneously.

---

## Performance Impact

Decode is memory-bandwidth-bound — the GPU reads every weight matrix once per
token.  Q4 cuts weight memory ~3.5x, which translates almost directly to
throughput gains:

| Model | bf16 tok/s | Q4 tok/s | Speedup |
|-------|-----------|---------|---------|
| Qwen3.5 122B-A10B (M4 Max) | 1.0 | 8.0 | 8.0x |
| Qwen3.5 397B-A17B (M4 Max) | 0.3 | 4.4 | 14.7x |
| Qwen3.5 122B-A10B (RTX 4090) | 1.2 | 4.0 | 3.3x |

The speedup exceeds 3.5x on expert-streamed models because Q4 also reduces
NVMe I/O volume per expert load.

### What quantization does NOT improve

- **Prefill throughput** — prefill is compute-bound (GEMM), not bandwidth-bound.
  Q4 prefill is only marginally faster than bf16
- **Quality** — some accuracy loss is inherent.  Sensitive layers (attention
  projections) lose more than FFN weights.  Always eval after quantizing

---

## Expert Streaming with Q4

Pre-quantized Q4 is critical for [expert streaming](expert-streaming.md)
performance.  When experts are Q4 on disk, `pread()` loads 3.5x fewer bytes
per cache miss — no CPU-side quantization needed, the raw Q4 bytes go straight
to the GPU buffer.

| Model | Expert I/O (bf16) | Expert I/O (Q4) |
|-------|------------------|-----------------|
| Qwen3.5-35B (gate+up per expert) | ~20 MB | ~6 MB |
| Qwen3.5-122B (gate+up per expert) | ~60 MB | ~17 MB |

For NVMe-bound workloads, this is the difference between usable and unusable
generation speed.

---

## Byte Count Calculation

Given a weight matrix `[m, k]`:

```
blocks_per_row = k / 32
bytes = m * blocks_per_row * 18
```

Available as `gpu::q4_byte_count(m, k)` in code.

---

See also: [GPU Backend](gpu-backend.md) · [Expert Streaming](expert-streaming.md) ·
[Production Considerations](production-considerations.md)
