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
   `<q_rot, dequant(Pi*K)> approx <Q, K>`.  No inverse rotation per position.
2. **V accumulation:** Accumulate dequantized centroids in rotated space,
   apply Pi^T once per query head at the end.

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

## Architecture

```
model/turboquant.rs           — KvQuantMode, TurboQuantConfig, rotation matrix, TurboContext
gpu/ops/turboquant.rs         — GpuTurboQuant trait (GPU kernel interface)
gpu/metal/shaders/turboquant.metal — Metal shader kernels
gpu/metal/kernels/turboquant.rs    — Metal dispatch code (#[repr(C)] params)
model/primitives.rs           — paged_kv_and_attention_maybe_quantized()
model/kv_cache.rs             — KvPool with quantized buffer allocation
engine/loader.rs              — CLI flag threading, TurboContext creation
```

### Data Flow

```
Forward pass (decode):
  QKV projection + RoPE → k_buf, v_buf, q_buf (bf16)
    ↓
  turbo_quantize_to_paged(k_buf) → k_pool (packed codes + norm)
  turbo_quantize_to_paged(v_buf) → v_pool (packed codes + norm)
  turbo_rotate_q(q_buf) → q_rot (f32)
    ↓
  turbo_paged_attention(q_rot, k_pool, v_pool)
    → inline dequant: code → centroid lookup → scale by norm
    → online softmax: Q_rot . K_approx / sqrt(d)
    → V accumulation in rotated space
    → Pi^T * v_acc (once per head)
    → attn_out (bf16)
```

### Metal Kernels

| Kernel | Threadgroups | Threads/Group | Purpose |
|--------|-------------|---------------|---------|
| turbo_quantize_paged | num_kv_heads | head_dim | Rotate + quantize + pack + write |
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
