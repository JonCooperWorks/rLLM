# Flash Attention v2 — Tiled Prefill

Tiled prefill is a Metal kernel optimisation that **halves K/V memory traffic**
during the prefill phase by grouping adjacent query positions into a single
threadgroup.  Instead of each query loading K/V vectors independently from
device memory, two queries share every K/V load.

**Paper:** ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691)
Tri Dao.  arXiv:2307.08691, July 2023.

## Why Prefill Attention is Memory-Bound

During prefill, the model processes all prompt tokens at once.  Each query
position `Q[i]` attends to all positions `0..i` (causal) or `0..N` (bidirectional).
The naive approach launches one threadgroup per (query, head) pair, and each
threadgroup independently reads the K/V vectors for its attend range from
device memory:

```
Naive prefill (1024 tokens, causal):
  Q[0]   reads K/V positions 0..0         → 1 K/V load
  Q[1]   reads K/V positions 0..1         → 2 K/V loads
  Q[2]   reads K/V positions 0..2         → 3 K/V loads
  ...
  Q[1023] reads K/V positions 0..1023     → 1024 K/V loads
  ─────────────────────────────────────────
  Total: 1024 × 1025 / 2 = 524,800 K/V loads (each = head_dim × 2 bytes)
```

Adjacent queries attend to nearly identical K/V ranges — `Q[i]` and `Q[i+1]`
differ by exactly one position.  This means ~99.9% of K/V reads are duplicated
between consecutive queries.

## How Tiling Fixes This

The tiled kernel groups `TILE_Q = 2` adjacent queries per threadgroup.  Each
K/V position is loaded **once** and used for both queries:

```
Tiled prefill (TILE_Q = 2):
  Threadgroup 0: Q[0], Q[1]  → load K/V once for positions 0..1
  Threadgroup 1: Q[2], Q[3]  → load K/V once for positions 0..3
  ...
  Threadgroup 511: Q[1022], Q[1023] → load K/V once for positions 0..1023
  ─────────────────────────────────────────
  Total K/V loads: ~262,400 (halved)
```

The two queries share the same K/V load in the inner loop:

```metal
for (uint pos = loop_start + tid; pos < max_attend; pos += tg_size) {
    // Load K[pos] ONCE from device memory
    device const bfloat4* k4 = ...;
    float4 dot_0 = float4(0), dot_1 = float4(0);
    for (uint i = 0; i < hd4; i++) {
        float4 ki = float4(k4[i]);    // K loaded once
        dot_0 += q0_4[i] * ki;        // Reused for Q[0]·K
        dot_1 += q1_4[i] * ki;        // Reused for Q[1]·K
    }

    // ... online softmax update for both Q's ...

    // Load V[pos] ONCE, accumulate for both Q's
    device const bfloat4* v4 = ...;
    for (uint i = 0; i < hd4; i++) {
        float4 vi = float4(v4[i]);    // V loaded once
        v_acc_0[i] += w0 * vi;        // Reused for Q[0]
        v_acc_1[i] += w1 * vi;        // Reused for Q[1]
    }
}
```

## Register Budget

Each thread maintains **two** sets of V accumulators (one per query position),
doubling the register pressure compared to the non-tiled kernel:

| Component | Non-tiled | Tiled (TILE_Q=2) |
|---|---|---|
| V accumulators | 32 float4 = 512 B | 64 float4 = 1024 B |
| Softmax state | 2 floats | 4 floats |
| Temporaries | ~12 floats | ~16 floats |
| **Per-thread total** | ~530 B | ~1060 B |
| **256 threads total** | ~133 KB | ~265 KB |

Apple Silicon has generous register files (~96 KB per SIMD group × 8 groups
= 768 KB theoretical).  Some spilling to L1-backed stack is expected, but the
2× bandwidth savings more than compensates.

For `head_dim = 256` (Gemma 3 27B), the doubled registers would cause excessive
spilling, so the non-tiled kernel is used instead.

## Shared Memory Layout

```
q_shared[TILE_Q × MAX_HD]          2 × 128 × 4 = 1,024 bytes (both Q vectors)
shared_reduce[NUM_SIMD_GROUPS × MAX_HD]  8 × 128 × 4 = 4,096 bytes (reused)
──────────────────────────────────────────────────────
Total: 5,120 bytes (well within 32 KB threadgroup limit)
```

## Dispatch Model

```
Non-tiled:  chunk_size × num_heads threadgroups  (one per query × head)
Tiled:      ceil(chunk_size / 2) × num_heads     (one per query-pair × head)
```

When `chunk_size` is odd, the last threadgroup processes only one query position,
degenerating to non-tiled behaviour for that block.

## Causal Mask Handling

For `Q[qi_base]` and `Q[qi_base + 1]`:
- `Q[qi_base]` attends to positions `0..qi_base` (attend_len = qi_base + 1)
- `Q[qi_base + 1]` attends to positions `0..qi_base + 1` (one extra position)

The loop runs up to the larger attend length.  For the one extra position that
only `Q[1]` sees, `Q[0]`'s weight is set to zero — at most one wasted position
per tile, negligible overhead.

Sliding window is handled per-query: adjacent Q positions may have window starts
that differ by at most 1.  Both are checked independently in the loop.

## When It Activates

The Metal dispatch code (`metal/kernels/attention.rs`) selects the tiled kernel
when:

```rust
if chunk_size >= 2 && head_dim <= 128 {
    // Flash Attention v2: tiled kernel, ceil(chunk_size / TILE_Q) threadgroups
    pipeline_prefill_attention_tiled
} else if head_dim > 128 {
    pipeline_prefill_attention_hd256   // Gemma 27B: non-tiled, hd=256
} else {
    pipeline_prefill_attention         // Fallback: chunk_size == 1
}
```

## Files

```
gpu/metal/shaders/attention.metal       — prefill_attention_tiled kernel
gpu/metal/kernels/attention.rs          — dispatch logic (selects tiled vs non-tiled)
gpu/metal/backend.rs                    — pipeline_prefill_attention_tiled compilation
gpu/ops/attention.rs                    — trait (unchanged — tiling is transparent)
```

## Future Work

- **TILE_Q = 4** for `head_dim ≤ 64` (Llama 1B/3B): register pressure is
  manageable at small head_dim, and 4× K/V bandwidth savings would help.
- **K/V tiling into shared memory**: load blocks of K/V into threadgroup memory
  for faster dot-product reads.  Currently K/V are read directly from device
  memory; shared memory would trade bandwidth for latency.
- **Flash-Decoding**: split long KV sequences across multiple threadgroups for
  single-query decode.  Helps when seq_len >> 256 threads, adding parallelism
  beyond what a single threadgroup can exploit.
