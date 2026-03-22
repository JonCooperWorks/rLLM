# Expert Streaming (SSD-backed MoE)

How rLLM runs Mixture-of-Experts models that don't fit in GPU memory by streaming
expert weights from NVMe SSD on demand.

---

## Problem

MoE models store hundreds of expert FFN sub-networks per layer.  A model like
Qwen3.5-35b-a3b has 256 experts across 40 layers — 60 GB of expert weights in bf16.
On a 64 GB Mac, there's no room left for KV cache and scratch buffers.  Qwen3.5-397B
has 512 experts across 60 layers — 720 GB of expert weights that don't fit on any
consumer machine.

But MoE routing only activates K experts per token (K=8 for Qwen3.5-35b).  At any
given moment, 248 of 256 experts per layer are idle.  This is the key insight:
**we only need K experts in GPU memory at a time, not all N**.

## Solution: On-Demand Expert Streaming

Instead of loading all expert weights to GPU at startup, rLLM records their file
locations and loads only the router-selected experts from SSD during inference.

```
Traditional:  Load 256 × 40 experts to GPU → 60 GB VRAM
Streaming:    Load 8 buffer slots to GPU   → 15 MB VRAM
              + pread K=8 experts from SSD per layer per token
```

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Model Loading                         │
│                                                          │
│  Non-expert weights (attention, norms, embeddings,       │
│  router gates, shared experts) → uploaded to GPU         │
│                                                          │
│  Expert weights → NOT uploaded, file offsets recorded    │
│  in ExpertIndex (shard file + byte offset per expert)    │
│                                                          │
│  K buffer slots pre-allocated on GPU (reused every       │
│  layer, every token)                                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                Per-Layer MoE Dispatch                     │
│                                                          │
│  1. Router matmul → per-expert scores         [GPU]      │
│  2. Top-K softmax → select K experts          [GPU]      │
│  3. Copy routing indices to CPU               [sync]     │
│  4. pread() K experts from SSD into slots     [I/O]      │
│  5. Fused gate+up+SwiGLU per expert           [GPU]      │
│  6. Down projection per expert                [GPU]      │
│  7. Weighted sum → residual add               [GPU]      │
└──────────────────────────────────────────────────────────┘
```

### How File Offsets Work

Safetensors files have a simple layout:

```
[8-byte LE header_len][JSON header][tensor data region]
```

The JSON header contains `data_offsets` for each tensor — byte ranges relative to
the start of the tensor data region.  For fused expert tensors like Qwen3.5's
`gate_up_proj [num_experts, 2*moe_inter, hidden]`, each expert is a contiguous
slice along dimension 0:

```
Expert j gate data: file_offset + j × (2 × moe_inter × hidden × 2)
Expert j up data:   file_offset + j × (2 × moe_inter × hidden × 2) + (moe_inter × hidden × 2)
Expert j down data: down_file_offset + j × (hidden × moe_inter × 2)
```

This means **no preprocessing** is needed — we compute byte offsets from the
safetensors header and `pread()` directly from the original weight files.

### I/O Strategy

Expert loading uses the Unix `pread()` syscall (via Rust's `FileExt::read_exact_at`):

- **Thread-safe**: no shared file position (unlike `seek + read`)
- **Single syscall**: kernel handles the offset seek internally
- **OS page cache**: the kernel caches recently-read pages automatically;
  flash-moe reports ~71% hit rate on repeated queries, meaning most expert
  reads come from RAM after the first pass

For Q4 streaming, experts are quantized on the CPU after reading from disk,
then copied to the pre-allocated Q4 GPU buffer.  For bf16 streaming, the raw
bytes are copied directly to the GPU buffer — no CPU processing needed, which
is why bf16 streaming is ~6x faster than Q4 streaming.

---

## Key Types

### ExpertIndex (`model/expert_stream.rs`)

Records where every expert lives on disk, built during model loading:

```rust
ExpertIndex {
    layers: Vec<Vec<ExpertLocation>>,  // [layer][expert] → file location
    shard_files: Vec<File>,            // open file handles for pread
    hidden: usize,                     // hidden dimension
    moe_inter: usize,                  // expert intermediate dimension
    quantize: bool,                    // Q4 quantize on load?
}
```

### ExpertStreamer (`model/expert_stream.rs`)

Manages K GPU buffer slots and loads experts on demand:

```rust
ExpertStreamer {
    index: ExpertIndex,                // file locations
    slots: Vec<ExpertSlot>,            // K pre-allocated GPU buffers
    read_bufs: UnsafeCell<Vec<Vec<u8>>>,  // CPU scratch for pread
}
```

The `UnsafeCell` on `read_bufs` allows `load_experts(&self, ...)` to take a shared
reference (matching the GPU tensor interior mutability pattern), since inference is
single-threaded within a model.

### ExpertSlot

One expert's worth of GPU-resident weight buffers, reused each layer:

```rust
ExpertSlot {
    gate_proj: Tensor,  // [moe_inter, hidden]
    up_proj: Tensor,    // [moe_inter, hidden]
    down_proj: Tensor,  // [hidden, moe_inter]
}
```

---

## Integration Points

### Loader (`model/loader.rs`)

When `--stream-experts` is passed:

1. `build_expert_index_from_safetensors()` reads safetensors headers and computes
   per-expert byte offsets without loading expert data
2. `load_weights_inner()` is called with `skip_experts=true`, which loads router
   gates, shared experts, and all non-MoE weights normally, but skips the expert
   upload entirely
3. The `ExpertIndex` is returned alongside the weights

Supports two expert storage formats:
- **Fused** (Qwen3.5): `gate_up_proj [N, 2*inter, hidden]` + `down_proj [N, hidden, inter]`
- **Per-expert** (Qwen3-MoE, Mixtral): separate `gate_proj`, `up_proj`, `down_proj` per expert

### Primitives (`model/primitives.rs`)

`moe_expert_dispatch_streamed()` mirrors `moe_expert_dispatch()` but calls
`streamer.load_experts()` instead of indexing into `Vec<ExpertWeights>`.
Uses the fused gate+up+SwiGLU kernel from the `GpuMoe` trait.

### Model Registry (`model/registry/qwen3_5.rs`, `mixtral.rs`, `qwen3_moe.rs`)

Each MoE model's `moe_ffn_block` checks `m.expert_streamer.is_some()` and dispatches
to the streamed or resident path accordingly.

---

## Memory Budget

### Qwen3.5-35b-a3b (256 experts, K=8)

| Component | bf16 | Q4 |
|-----------|------|-----|
| Non-expert weights | ~7 GB | ~3 GB |
| K=8 expert buffer slots | 48 MB | 15 MB |
| KV cache | 80 MB - 2.5 GB | 80 MB - 2.5 GB |
| **Total GPU** | **~7-10 GB** | **~3-6 GB** |
| Without streaming | 67 GB (doesn't fit) | ~22 GB |

### Qwen3.5-397B (512 experts, K=10)

| Component | bf16 |
|-----------|------|
| Non-expert weights | ~20 GB |
| K=10 expert buffer slots | 240 MB |
| KV cache | ~2 GB |
| **Total GPU** | **~22 GB** |
| Without streaming | 720+ GB (impossible) |

---

## Performance

### Benchmarks (Apple M4 Max, 64 GB, NVMe ~7 GB/s)

| Model | Mode | Load Time | Decode |
|-------|------|-----------|--------|
| Qwen3.5-35b-a3b | bf16 streaming | 0.7s | 3.0 tok/s |
| Qwen3.5-35b-a3b | Q4 streaming | 7.1s | 0.5 tok/s |
| Qwen3.5-35b-a3b | resident (no streaming) | N/A | N/A (doesn't fit) |

The bf16 path is 6x faster than Q4 because it skips CPU-side quantization per expert.
Each token reads K × 3 tensors from SSD per layer — for Qwen3.5-35b (K=8, 40 layers),
that's ~1.9 GB of I/O per token from NVMe.

### Bottleneck Analysis

Per-layer time breakdown (Qwen3.5-35b-a3b, bf16):
- Attention (DeltaNet/GQA): ~1 ms
- Routing: ~0.1 ms
- **Expert I/O**: ~6 ms (8 × 6 MB from SSD)
- Expert compute: ~0.5 ms

Expert I/O dominates.  Implemented optimisations:
1. **Parallel pread** — K threads reading experts concurrently (std::thread::scope)
2. **Fused gate+up read** — single pread for contiguous gate+up (Qwen3.5 fused format)

Future optimisations:
1. **Q4 pre-quantized streaming** — 3.2x less I/O volume per expert
2. **Pipeline overlap** — start next layer's attention while experts still loading
3. **Expert caching** — keep recently-used experts in CPU RAM instead of re-reading

---

## Usage

```bash
# Stream experts from SSD (bf16 — fastest)
rllm run --model models/qwen3.5-35b-a3b \
  --prompt "Hello" --stream-experts

# Stream experts from a pre-quantized Q4 model
rllm run --model models/qwen3.5-35b-a3b-q4 \
  --prompt "Hello" --stream-experts
```

The `--stream-experts` flag works with any MoE model (Qwen3.5, Qwen3-MoE, Mixtral).
Dense models ignore the flag.

---

## Related Files

| File | Purpose |
|------|---------|
| `model/expert_stream.rs` | ExpertIndex, ExpertStreamer, pread loading, index building |
| `model/loader.rs` | Expert index construction from safetensors headers |
| `model/primitives.rs` | `moe_expert_dispatch_streamed()`, `moe_ffn_block_streamed()` |
| `model/mod.rs` | `Model.expert_streamer` field |
| `gpu/ops/moe.rs` | `GpuMoe` trait (fused kernels used by both streaming and resident paths) |
| `gpu/metal/shaders/moe.metal` | Fused gate+up+SwiGLU and combine+residual Metal shaders |
| `commands/run.rs` | `--stream-experts` CLI flag |
| `engine/loader.rs` | Threads streaming flag through model construction |

## Design Inspiration

The SSD streaming architecture is inspired by
[flash-moe](https://github.com/danveloper/flash-moe), which runs Qwen3.5-397B
at 4.4 tok/s on a 48GB MacBook using parallel `pread()`, double-buffered expert
slots, and a pipelined GPU command buffer architecture.  rLLM uses parallel
pread with fused gate+up reads (0.28 tok/s on 397B bf16, M4 Max 64 GB).
We tested mmap-based streaming but found it slower for large shard files
(751 GB across 94 files) — page fault overhead (16 KB granularity) exceeds
the cost of targeted 8-16 MB pread calls.
