# Multi-GPU MoE: Expert Parallelism

rLLM uses a **Hybrid** parallelism strategy for Mixture-of-Experts models on
multi-GPU: tensor parallelism (TP) for attention layers, expert parallelism (EP)
for MoE FFN layers.  This gives the best of both worlds — TP splits the large
attention matrices across GPUs, while EP assigns whole experts to individual GPUs
without splitting their (relatively small) weight matrices.

**Key files:**
- `src/gpu/parallel.rs` — `ParallelStrategy`, `ShardingPlan`, expert assignment
- `src/gpu/multi_gpu.rs` — `MultiGpuInference`, per-rank setup
- `src/model/primitives.rs` — `moe_expert_dispatch` with EP skip/remap/allreduce
- `src/model/forward.rs` — `MoeBuffers` (local_expert_start/count)
- `src/engine/loader.rs` — `alloc_moe_buffers`, EP-aware buffer sizing
- `src/model/loader/mod.rs` — weight loading: skip non-local experts
- `src/model/loader/upload.rs` — Q4/Q8/FP8 tensor sharding

---

## Why Hybrid, Not Pure TP?

With pure tensor parallelism, each expert's gate/up/down weight matrices get
split across GPUs.  This requires an AllReduce after every expert's down
projection — expensive for MoE models where multiple experts fire per token.

```
Pure TP (bad for MoE):
  For each of k selected experts:
    gate_proj: Column split → no allreduce
    up_proj:   Column split → no allreduce
    down_proj: Row split    → AllReduce!     ← k allreduces per layer
```

Expert parallelism avoids this entirely.  Each GPU owns a subset of whole
experts.  The router runs identically on all GPUs (same input, same weights →
same routing decisions).  Each GPU computes only its local experts, then a
single AllReduce combines the partial outputs:

```
Hybrid EP (one allreduce per MoE layer):
  Router: replicated on all GPUs (same routing decisions)
  For each of k selected experts:
    if expert is local → compute full FFN
    if expert is remote → skip (other GPU handles it)
  AllReduce(moe_output)                      ← 1 allreduce per layer
```

---

## Architecture

```
                        GPU 0                              GPU 1
                   ┌──────────────┐                  ┌──────────────┐
   Token           │  Attention   │                  │  Attention   │
   embedding       │  (TP: heads  │                  │  (TP: heads  │
   (replicated)    │   0..N/2)    │                  │   N/2..N)    │
                   └──────┬───────┘                  └──────┬───────┘
                          │                                 │
                          │         AllReduce (o_proj)       │
                          ├─────────────────────────────────┤
                          │                                 │
                   ┌──────┴───────┐                  ┌──────┴───────┐
                   │   Router     │                  │   Router     │
                   │ (replicated) │                  │ (replicated) │
                   │ same top-k   │                  │ same top-k   │
                   │ on both GPUs │                  │ on both GPUs │
                   └──────┬───────┘                  └──────┴───────┘
                          │                                 │
                   ┌──────┴───────┐                  ┌──────┴───────┐
                   │ Experts 0-3  │                  │ Experts 4-7  │
                   │ (EP: local   │                  │ (EP: local   │
                   │  only, skip  │                  │  only, skip  │
                   │  4-7)        │                  │  0-3)        │
                   └──────┬───────┘                  └──────┬───────┘
                          │                                 │
                          │     AllReduce (moe_output)       │
                          ├─────────────────────────────────┤
                          │                                 │
                   ┌──────┴───────┐                  ┌──────┴───────┐
                   │ + Residual   │                  │ + Residual   │
                   └──────────────┘                  └──────────────┘
```

Both GPUs see the same hidden state after the attention AllReduce.  The router
produces identical top-k selections on both GPUs.  Each GPU only computes its
local experts and zeros out the rest.  The MoE output AllReduce sums each GPU's
partial contribution, producing the correct weighted sum.

---

## Sharding Plan

`ShardingPlan::derive()` in `parallel.rs` assigns weights to ranks:

| Weight | Dense TP | Hybrid (Attention) | Hybrid (MoE) |
|--------|----------|-------------------|--------------|
| q_proj | Column | Column | Column |
| k_proj | Column | Column | Column |
| v_proj | Column | Column | Column |
| o_proj | Row + AllReduce | Row + AllReduce | Row + AllReduce |
| router_gate | — | — | Replicated |
| expert.gate_proj | — | — | ExpertParallel |
| expert.up_proj | — | — | ExpertParallel |
| expert.down_proj | — | — | ExpertParallel |

Expert assignment is contiguous: rank 0 gets experts `[0, N/ws)`, rank 1 gets
`[N/ws, 2N/ws)`, etc.  `N % world_size == 0` is validated at plan derivation
time.

---

## Weight Loading

The loader detects EP from the sharding plan by scanning for `ExpertParallel`
entries.  Non-local experts are skipped entirely — never read from disk, never
uploaded to GPU.

Three expert storage formats are handled:

1. **Per-expert** (Mixtral, Qwen3-MoE): separate safetensors per expert.
   The loader iterates only local expert indices.

2. **Fused** (Qwen3.5): one tensor for all experts, sliced by byte offset.
   The loader extracts only local expert slices from the fused tensor.

3. **MXFP4** (GPT-OSS): packed fp4 blocks with E8M0 scales.
   The loader dequantizes only local experts to bf16.

### Quantized weight sharding (Q4/Q8/FP8)

Attention weights (q/k/v/o_proj) are pre-quantized and must be sliced in the
quantized domain.  The slicer treats Q4 blocks (18 bytes per 32 weights) and
Q8 blocks (34 bytes per 32 weights) as atomic units:

```
Q4 tensor [m, k]:
  On disk: m rows × (k/32) blocks × 18 bytes/block
  Column split: take m/ws contiguous rows
  Row split:    take (k/32)/ws blocks per row
```

FP8 has no block structure (1 byte per weight) and slices element-wise.

Expert weights with `ExpertParallel` split pass through unchanged — the whole
expert is loaded as-is onto its assigned GPU.

---

## MoE Dispatch

`moe_expert_dispatch()` in `primitives.rs` handles both single-GPU and EP:

```rust
// Parameters: local_expert_start, local_expert_count
// Single-GPU / TP: pass (0, num_experts) — all experts are local
// EP:              pass (start, count)   — only local experts computed

for (expert_idx, weight) in router_top_k {
    if expert_idx not in [local_start .. local_start + local_count] {
        continue;  // skip — other GPU handles this expert
    }
    let local_idx = expert_idx - local_start;
    compute_expert_ffn(experts[local_idx], weight);
    moe_output += weight * expert_output;
}

if EP {
    all_reduce_sum(moe_output);  // combine partial outputs
}
```

The EP state is stored in `MoeBuffers::local_expert_start` and
`MoeBuffers::local_expert_count`, set at model load time based on the
parallelism strategy.

---

## Buffer Sizing

EP changes buffer requirements compared to pure TP:

| Buffer | Pure TP | Hybrid EP |
|--------|---------|-----------|
| `moe_gate_buf` | `moe_inter / ws` | `moe_inter` (full — experts aren't split) |
| `moe_up_buf` | `moe_inter / ws` | `moe_inter` (full) |
| `moe_output` | `hidden_size` | `hidden_size` (same) |
| `gate_buf` (Model) | `inter / ws` | `max(inter / ws, hidden_size)` |

The `gate_buf` in `Model` is dual-purpose: dense FFN gate output (`inter / ws`)
and MoE down_proj scratch (`hidden_size`).  With EP, `inter / ws` can be smaller
than `hidden_size`, so we take the max.

---

## Strategy Selection

`MultiGpuInference::new()` automatically selects the strategy:

```rust
let strategy = if config.is_moe() {
    ParallelStrategy::Hybrid   // TP attention + EP MoE
} else {
    ParallelStrategy::TensorParallel  // standard TP
};
```

No user configuration needed — MoE models get EP automatically.

---

## Constraints

- **`num_experts % world_size == 0`** — validated at plan derivation time
- **Router is replicated** — no communication needed for routing decisions
- **Expert streaming + EP is not supported** — all local experts must fit in VRAM
- **Shared experts (Qwen3.5) are replicated** — active on every GPU, not split
