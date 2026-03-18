# Evaluation: Multi-Dispatch Pipeline

## Overview

rLLM's inference pipeline chains four distinct dispatch layers to route an HTTP
request through tokenization, scheduling, model-specific forward passes, and GPU
kernel execution.  This evaluation examines each layer's design, identifies
strengths, and flags areas for improvement.

---

## 1. Dispatch Layers

### Layer 1: API → Worker Thread (channel dispatch)

**Files:** `src/api/mod.rs:324-435`

HTTP handlers tokenize on async Tokio threads, then send `WorkerRequest` structs
over a `std::sync::mpsc::SyncSender` to a dedicated worker thread.  The worker
drains all pending requests via `try_recv()` before each engine step, achieving
natural request batching.

| Aspect | Assessment |
|--------|-----------|
| Async/sync bridge | Clean — avoids `block_in_place` or `spawn_blocking` |
| Backpressure | `SyncSender` with bounded channel provides implicit backpressure |
| Batching | Good — `try_recv()` loop drains all arrivals before GPU work |
| Error propagation | Adequate — engine errors broadcast to all active requests |

**Potential improvements:**
- The worker thread currently processes prefill sequentially per sequence (line
  148-188 of engine/mod.rs).  True batch-prefill across multiple sequences would
  amortize kernel launch overhead.
- `SyncSender` capacity isn't visible in the code read — should be tuned to
  avoid stalling HTTP handlers under burst load.

### Layer 2: Engine → Model (architecture dispatch)

**Files:** `src/engine/mod.rs:126-251`, `src/model/mod.rs:595-663`

The engine's `step()` method implements a three-phase loop:
1. **Admit** — scheduler moves waiting requests to active set (KV block check)
2. **Prefill** — full-prompt GEMM forward pass per new sequence
3. **Decode** — single-token mat-vec forward pass per active sequence

Model dispatch uses a `match` on `ModelArch` enum (9 variants) to route to the
correct registry module.

| Aspect | Assessment |
|--------|-----------|
| Phase separation | Excellent — prefill (GEMM) vs decode (mat-vec) cleanly split |
| Dispatch mechanism | Simple match on enum — zero overhead, exhaustive |
| Extensibility | Adding a model = 1 new registry file + 2 match arms |
| Borrow structure | Well-handled — ID collection avoids borrow-checker conflicts |

**Potential improvements:**
- Decode is currently one-sequence-at-a-time (lines 199-227).  Batching decode
  across multiple sequences into a single padded mat-mul call would improve GPU
  utilization at high concurrency.
- The prefill loop (lines 148-188) processes sequences serially.  While each
  prefill is already batched (GEMM), multiple concurrent prefills could be merged.

### Layer 3: Model → Primitives → GPU Traits (trait dispatch)

**Files:** `src/model/primitives.rs`, `src/gpu/ops/*.rs`, `src/gpu/mod.rs:61-85`

Each model family composes shared primitives (`embed_token`, `qkv_projection`,
`apply_rope`, `paged_kv_and_attention`, `fused_ffn`, etc.).  Primitives call
methods on the `B: GpuBackend` bound, which is a blanket supertrait of 9
composable sub-traits.

| Aspect | Assessment |
|--------|-----------|
| Composability | Excellent — sub-traits are independent, minimal bounds per primitive |
| Zero-cost abstraction | Monomorphization eliminates vtable overhead |
| Separation of concerns | Each trait maps 1:1 to a kernel family |
| Code reuse | High — most models compose the same primitives with different config |

**Strengths:**
- `ArchFeatures` pattern lets Qwen, Phi, Mistral delegate to Llama's forward
  pass with config flags (QKV bias, tied embeddings, etc.) — minimal duplication.
- Primitive functions declare minimal trait bounds (e.g., `rms_norm` only needs
  `GpuNorm`), enabling fine-grained testing and future backend porting.

### Layer 4: GPU Backend (kernel dispatch)

**Files:** `src/gpu/metal/backend.rs:447-536`, `src/gpu/metal/kernels/*.rs`

The Metal backend accumulates all GPU work into a single `CommandBuffer` via
`dispatch_async()`.  Each call creates a compute encoder, binds params/buffers,
dispatches threads, and ends the encoder — but does NOT commit.  Only `flush()`
commits and waits.

| Aspect | Assessment |
|--------|-----------|
| Batching | Excellent — single command buffer per engine step |
| Overhead reduction | Dramatic: 2500 → 1 command buffer for MoE models |
| Pipeline specialization | Smart — HEAD_DIM=128 vs 256 avoids register waste |
| dtype dispatch | Q4 vs BF16 pipeline selection at runtime |
| `submit()` vs `flush()` | Good — allows CPU/GPU overlap when needed |

**Potential improvements:**
- `dispatch_async` allocates a new `param_buf` per kernel dispatch (line 504-508).
  A ring buffer or pool of param buffers would reduce allocation pressure.
- The Mutex on `current_cmd` is uncontested (single worker thread) but still has
  overhead.  An `UnsafeCell` with a single-thread assertion could eliminate it.

---

## 2. KV Cache & Scheduling (cross-cutting)

**Files:** `src/model/kv_cache.rs`, `src/engine/scheduler.rs:99-129`

The paged KV cache is the throughput gatekeeper.  The scheduler checks free block
count before admitting sequences, and `ensure_slots()` / `ensure_slot()`
allocate blocks on demand during prefill/decode.

| Aspect | Assessment |
|--------|-----------|
| Admission control | Conservative — checks total blocks needed upfront |
| Block reuse | Immediate — freed on sequence completion |
| GPU sync | Lazy — block table synced only when dirty |
| Max concurrency | Configurable (default 32 sequences) |

**Potential improvement:**
- No preemption: if all KV blocks are consumed by long sequences, new requests
  starve.  A preemption policy (evict lowest-priority sequence, spill to host)
  would improve tail latency under memory pressure.

---

## 3. Overall Assessment

### Strengths

1. **Clean layering** — each dispatch layer has a single responsibility and clear
   interface boundaries (channels → enum match → trait methods → command buffer).
2. **Zero-cost generics** — the trait hierarchy monomorphizes away, so the
   dispatch chain compiles to direct function calls with no vtable indirection.
3. **Command buffer batching** — the single-command-buffer pattern is the single
   biggest performance win, eliminating thousands of Metal API round-trips per
   token.
4. **Extensibility** — adding a new model or kernel family is mechanical and
   well-documented (CLAUDE.md checklists).
5. **Borrow-safety** — the ID-collection pattern in the engine avoids unsafe code
   while satisfying Rust's aliasing rules.

### Areas for Improvement

| Priority | Issue | Impact |
|----------|-------|--------|
| High | Single-sequence decode loop | GPU underutilized at high concurrency |
| Medium | Per-dispatch param buffer allocation | Allocation pressure on MoE models |
| Medium | No sequence preemption | Head-of-line blocking under memory pressure |
| Low | Serial prefill across sequences | Minor — prefill is already GEMM-batched |
| Low | Mutex on single-threaded command buffer | Negligible overhead in practice |

### Verdict

The multi-dispatch pipeline is well-architected.  The four-layer design achieves
both clean separation of concerns and high performance through Rust's zero-cost
abstractions and Metal command buffer batching.  The main scaling bottleneck is
the per-sequence decode loop — batching decode across concurrent sequences would
be the highest-impact improvement for serving workloads.
