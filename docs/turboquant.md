# TurboQuant — KV Cache Vector Quantization

TurboQuant is an online, data-oblivious vector quantization algorithm for
compressing the KV cache during inference.  At 4 bits per channel it matches
full-precision quality while reducing KV cache memory by ~4x.

**Paper:** ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"](https://arxiv.org/abs/2504.19874)
Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind),
Vahab Mirrokni (Google Research).  arXiv:2504.19874v1, April 2025.

**Blog:** [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Usage

TurboQuant 4-bit is **on by default**.  No flags needed:

```bash
# Default: TurboQuant 4-bit (~4x KV cache compression, quality-neutral)
rllm run --model models/llama-3.2-1b --prompt "Hello"
rllm serve --model models/llama-3.2-1b

# Override: more aggressive compression
rllm run --model models/llama-3.2-1b --kv-quant turbo3  # ~5x compression
rllm run --model models/llama-3.2-1b --kv-quant turbo2  # ~7.5x compression

# Disable: BF16 KV cache (debugging/benchmarking only)
rllm run --model models/llama-3.2-1b --kv-quant none
```

## Algorithm Overview

TurboQuant exploits a property of high-dimensional geometry: when a unit vector
is multiplied by a random orthogonal matrix, each coordinate becomes approximately
Gaussian and nearly independent.  This allows optimal scalar quantization per
coordinate using precomputed codebook centroids.

### Step 1: Random Rotation

A random orthogonal matrix Pi (d x d) is generated once at model load time via
QR decomposition of a random Gaussian matrix.  The same matrix is used across
all layers and all tokens.  A fixed seed (42) ensures reproducibility and prefix
cache compatibility.

### Step 2: Scalar Quantization

Each coordinate of the rotated vector is quantized to the nearest centroid from
a Max-Lloyd codebook optimised for the Gaussian distribution:

| Bits | Centroids | MSE Distortion | Compression |
|------|-----------|----------------|-------------|
| 2    | 4         | ~0.117         | ~7.5x       |
| 3    | 8         | ~0.03          | ~5.1x       |
| 4    | 16        | ~0.009         | ~3.9x       |

### Step 3: Efficient Attention

The rotation is folded into the attention computation:

1. **K dot products:** Pre-rotate Q once (q_rot = Pi * Q), then
   `<q_rot, dequant(Pi*K)> ≈ <Q, K>`.  No inverse rotation per position.
2. **V accumulation:** Each thread accumulates all head_dim V dimensions
   in rotated space, then Pi^T is applied once per query head at the end.

Per-position cost is just a codebook table lookup (16 entries for 4-bit).
Memory bandwidth drops ~4x — the decode bottleneck on Apple Silicon.

## Storage Format

Per KV head per position:

```
[2 bytes bf16 norm] [ceil(head_dim * bits / 8) bytes packed codes]
```

For 4-bit with head_dim=128: 2 + 64 = 66 bytes per head (vs 256 bytes BF16).

The pool layout uses the same paged block structure as BF16, addressed via
block table indirection.  The only difference is bytes-per-position is smaller.

## Theoretical Guarantees

TurboQuant's distortion is within a factor of sqrt(3)*pi/2 ~ 2.7 of the
information-theoretic lower bound (Shannon's distortion-rate function).

For inner product estimation (the operation attention performs on K), the
paper proves unbiasedness and bounded variance.

## Production Considerations

### Economics

KV cache memory is the primary constraint on concurrent request throughput.
Each active sequence holds its KV cache for the duration of generation —
for a 100-token prompt generating 500 tokens, that's 600 positions × num_layers
× 2 (K+V) × num_kv_heads × head_dim × 2 bytes.

**Concrete example** (Qwen 3.5 9B, 32 layers, 4 KV heads, head_dim=256):
- BF16: 600 × 32 × 2 × 4 × 256 × 2 = 100 MB per sequence
- TurboQuant 4-bit: 600 × 32 × 2 × 4 × 130 = 20 MB per sequence (~5x less)

On a 64GB M4 Max after loading 9B weights (~18 GB), ~40 GB remains for KV cache.
That's **400 concurrent sequences** with TurboQuant vs **~400** with BF16 but
at much shorter context.  For long-context workloads (8K+ tokens), the savings
compound — TurboQuant enables sequences that wouldn't fit at all in BF16.

### Prefill vs Decode

During **prefill** (processing the prompt), K/V are quantized into the paged pool
but attention uses the full BF16 Q/K/V directly.  There is no quality loss during
prefill — quantization only affects the stored cache.

During **decode** (generating tokens), attention reads from the quantized cache
using inline dequantization.  The 4-bit mode has been verified to produce
identical output to BF16 across Llama, Qwen 2.5, and Qwen 3.5 model families.

### Hybrid Architectures

Models with non-standard attention layers (e.g., Qwen 3.5's DeltaNet linear
attention) are handled correctly: TurboQuant is only applied to standard GQA
attention layers.  DeltaNet layers maintain their own recurrent state and never
interact with the quantized KV pool.

### Code Packing

Sub-byte quantization codes (2-4 bits) share device memory bytes across GPU
threads.  Naive per-thread read-modify-write causes data races on Apple Silicon
SIMD groups — the GPU deterministically picks one lane's write, silently zeroing
the other threads' codes.  The `pack_codes_shared()` helper avoids this by
collecting all codes in threadgroup shared memory and having each thread write
one complete byte.

## Architecture

```
model/turboquant.rs           — KvQuantMode, TurboQuantConfig, rotation matrix, TurboContext
gpu/ops/turboquant.rs         — GpuTurboQuant trait (GPU kernel interface)
gpu/metal/shaders/turboquant.metal — Metal shader kernels
gpu/metal/kernels/turboquant.rs    — Metal dispatch code (#[repr(C)] params)
model/primitives.rs           — paged_kv_and_attention_maybe_quantized() (decode)
                                paged_kv_and_prefill_attention_maybe_quantized() (prefill)
model/kv_cache.rs             — KvPool with quantized buffer allocation
engine/loader.rs              — CLI flag threading, TurboContext creation
```

### Data Flow

```
Forward pass (prefill):
  QKV projection + RoPE → k_buf, v_buf, q_buf (bf16, batched)
    ↓
  turbo_quantize_to_paged_batch(k_buf) → k_pool (for future decode)
  turbo_quantize_to_paged_batch(v_buf) → v_pool (for future decode)
  prefill_attention(q_buf, k_buf, v_buf) → attn_out (BF16, full precision)

Forward pass (decode):
  QKV projection + RoPE → k_buf, v_buf, q_buf (bf16)
    ↓
  turbo_quantize_to_paged(k_buf) → k_pool (packed codes + norm)
  turbo_quantize_to_paged(v_buf) → v_pool (packed codes + norm)
  turbo_rotate_q(q_buf) → q_rot (f32)
    ↓
  turbo_paged_attention(q_rot, k_pool, v_pool)
    → inline dequant: code → centroid lookup → scale by norm
    → online softmax: Q_rot · K_approx / sqrt(d)
    → V accumulation in rotated space (all dims per thread)
    → cross-thread SIMD reduction of V partials
    → Pi^T × v_rot (once per head, fused with reduction)
    → attn_out (bf16)
```

### Metal Kernels

| Kernel | Threadgroups | Threads/Group | Purpose |
|--------|-------------|---------------|---------|
| turbo_quantize_paged | num_kv_heads | head_dim | Rotate + quantize + pack + write |
| turbo_quantize_paged_batch | batch × num_kv_heads | head_dim | Batched prefill variant |
| turbo_rotate_q | num_heads | head_dim | Pre-rotate query |
| turbo_paged_attention | num_heads | 256 | Attention with inline dequant |

## Configuration

The `--kv-quant` flag accepts:

| Value | Bits | Centroids | Compression | Quality |
|-------|------|-----------|-------------|---------|
| `turbo4` (default) | 4 | 16 | ~3.9x | Quality-neutral |
| `turbo3` | 3 | 8 | ~5.1x | Near-lossless |
| `turbo2` | 2 | 4 | ~7.5x | Marginal degradation |
| `none` | 16 (bf16) | N/A | 1x | Full precision |

## References

1. Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization
   with Near-optimal Distortion Rate." arXiv:2504.19874, 2025.
2. Lloyd. "Least squares quantization in PCM." IEEE Trans. IT, 1982.
3. Max. "Quantizing for minimum distortion." IRE Trans. IT, 1960.
4. Shannon. "A mathematical theory of communication." Bell System Technical
   Journal, 1948.
