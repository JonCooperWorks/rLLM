# TurboQuant — KV Cache Vector Quantization

TurboQuant is an online, data-oblivious vector quantization algorithm for
compressing the KV cache during inference.  At 4 bits per channel it matches
full-precision quality while reducing KV cache memory by ~4x.

**Paper:** ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"](https://arxiv.org/abs/2504.19874)
— Zandieh, Daliri, Hadian, Mirrokni.  arXiv:2504.19874, April 2025.

## Usage

TurboQuant 4-bit is **on by default**.  No flags needed:

```bash
rllm run --model models/llama-3.2-1b --prompt "Hello"
rllm serve --model models/llama-3.2-1b

# More aggressive compression
rllm run --model models/llama-3.2-1b --kv-quant turbo3  # ~5x
rllm run --model models/llama-3.2-1b --kv-quant turbo2  # ~7.5x

# Explicit asymmetric: BF16 K, TurboQuant V (auto-selected for bias models)
rllm run --model models/qwen2-3b --kv-quant none:turbo4

# Disable entirely
rllm run --model models/llama-3.2-1b --kv-quant none
```

### `--kv-quant` values

| Value | Bits | Compression | Quality |
|-------|------|-------------|---------|
| `turbo4` (default) | 4 | ~3.9x | Quality-neutral |
| `turbo3` | 3 | ~5.1x | Near-lossless |
| `turbo2` | 2 | ~7.5x | Marginal degradation |
| `none:turbo4` | K=16, V=4 | ~1.7x (V only) | Quality-neutral |
| `none` | 16 (bf16) | 1x | Full precision |

The `K:V` syntax sets independent modes for K and V caches.

## Algorithm

TurboQuant exploits high-dimensional geometry: when a unit vector is multiplied
by a random orthogonal matrix, each coordinate becomes approximately Gaussian
and nearly independent.  This allows optimal scalar quantization per coordinate
using precomputed Max-Lloyd codebook centroids.

1. **Rotate:** `y = Pi × (x / ||x||)` — random orthogonal Pi, generated once
   at load time from seed 42 (reproducible, prefix-cache-compatible).
2. **Quantize:** Each y_j → nearest centroid from a Max-Lloyd codebook for N(0,1).
3. **Pack:** b-bit codes + bf16 norm → `2 + ceil(d×b/8)` bytes per head.

### Efficient Attention

The rotation folds into attention computation:

- **K scoring:** Pre-rotate Q once (`q_rot = Pi × Q`), then
  `<q_rot, dequant(Pi×K)> ≈ <Q, K>`.  No inverse rotation per position.
- **V accumulation:** Accumulate in rotated space, apply Pi^T once per query
  head at the end.

Per-position cost: a codebook table lookup (16 entries for 4-bit).
Memory bandwidth drops ~4x — the decode bottleneck on Apple Silicon.

## Asymmetric Mode (K=BF16, V=Turbo)

Models with QKV bias (Qwen2, GPT-OSS) and sparse-attention hybrids (Qwen 3.5,
Nemotron-H) can't tolerate K quantization — the bias reduces angular diversity
of normalized K vectors, causing correlated quantization errors that softmax
amplifies.  V is tolerant because errors average out in weighted sums.

**Asymmetric mode** keeps K at BF16 and turbo-quantizes V only.  Auto-selected
for affected architectures on both Metal and CUDA.

```
Qwen 3.5 27B Q8 (M4 Max 64GB):
  BF16 KV:        6,753 MB  →  108K max tokens
  Asymmetric V4:   5,136 MB  →  131K max tokens   (24% smaller, 21% more context)
```

### V-Only Kernel

`turbo_paged_attention_v_only` splices:
- **K scoring** from `paged_attention` — standard bf16 dot product, no rotation
- **V accumulation** from `turbo_paged_attention` — centroid dequant in rotated
  space, Pi^T inverse rotation at the end

Same edge-case handling: window attention, attention sinks, GQA.

## Data Flow

```
Prefill:
  QKV + RoPE → k_buf, v_buf, q_buf (bf16, batched)
  ├─ quantize K/V → paged pool (for future decode)     [turbo or bf16 copy]
  └─ prefill_attention(q, k, v) → attn_out             [always full BF16]

Decode (symmetric — both K and V quantized):
  QKV + RoPE → k, v, q (bf16)
  ├─ turbo_quantize_to_paged(k) → k_pool
  ├─ turbo_quantize_to_paged(v) → v_pool
  ├─ turbo_rotate_q(q) → q_rot (f32)
  └─ turbo_paged_attention(q_rot, k_pool, v_pool) → attn_out

Decode (asymmetric — K=BF16, V=turbo):
  QKV + RoPE → k, v, q (bf16)
  ├─ copy_to_paged_kv_cache(k) → k_pool                [bf16 write]
  ├─ turbo_quantize_to_paged(v) → v_pool                [turbo quantize]
  └─ turbo_paged_attention_v_only(q, k_pool, v_pool) → attn_out
```

## Storage Format

Per KV head per position:

```
[2 bytes bf16 norm] [ceil(head_dim × bits / 8) bytes packed codes]
```

4-bit, head_dim=128: 2 + 64 = 66 bytes per head (vs 256 bytes BF16 = 3.9x).

Same paged block structure as BF16, addressed via block table indirection.

## Production Notes

**Prefill vs decode:** Prefill uses full BF16 attention — quantization only
affects the stored cache.  Zero quality loss during prompt processing.

**Code packing:** Sub-byte codes (2-4 bits) share device memory bytes.  Naive
per-thread read-modify-write causes data races on Apple Silicon — the GPU picks
one lane's write, zeroing others.  `pack_codes_shared()` collects codes in
threadgroup shared memory and has each thread write one complete byte.

**Theoretical bound:** Distortion is within √3·π/2 ≈ 2.7× of the
information-theoretic lower bound (Shannon's distortion-rate function).

## Architecture

```
model/turboquant.rs                    — KvQuantMode, KvQuantPair, TurboQuantConfig, TurboContext
gpu/ops/turboquant.rs                  — GpuTurboQuant trait
gpu/metal/shaders/turboquant.metal     — Metal kernels (quantize, rotate_q, paged_attention, v_only)
gpu/metal/kernels/turboquant.rs        — Metal dispatch (#[repr(C)] param structs)
gpu/cuda/shaders/turboquant.cu         — CUDA kernels (same five kernels as Metal)
gpu/cuda/kernels/turboquant.rs         — CUDA dispatch
model/primitives.rs                    — paged_kv_and_attention_maybe_quantized() (3-path dispatch)
                                         paged_kv_and_prefill_attention_maybe_quantized()
model/kv_cache.rs                      — KvPool with asymmetric K/V buffer allocation
engine/loader.rs                       — Auto-select symmetric/asymmetric, TurboContext creation
```

### GPU Kernels (Metal & CUDA)

| Kernel | Blocks/Threadgroups | Threads | Purpose |
|--------|-------------|---------|---------|
| `turbo_quantize_paged` | num_kv_heads | head_dim | Rotate + quantize + pack |
| `turbo_quantize_paged_batch` | batch × num_kv_heads | head_dim | Batched prefill variant |
| `turbo_rotate_q` | num_heads | head_dim | Pre-rotate query |
| `turbo_paged_attention` | num_heads | 256 | Symmetric: both K/V quantized |
| `turbo_paged_attention_v_only` | num_heads | 256 | Asymmetric: BF16 K + turbo V |

## References

1. Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization
   with Near-optimal Distortion Rate." arXiv:2504.19874, 2025.
2. Lloyd. "Least squares quantization in PCM." IEEE Trans. IT, 1982.
3. Max. "Quantizing for minimum distortion." IRE Trans. IT, 1960.
