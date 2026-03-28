# rvLLM-Inspired CUDA Optimizations

## Attribution

These optimizations are inspired by **rvLLM** — a from-scratch Rust reimplementation
of vLLM by **Andy Norris (m0at)** that achieves 16x throughput over Python vLLM on
A100 (3,191 tok/s) and 3,946 tok/s on B200.

**Paper**: "rvLLM: A High-Performance LLM Inference Engine in Rust" — Andy Norris (m0at),
March 28, 2026.  https://github.com/m0at/rvllm

The speedup comes from three reinforcing mechanisms: GPU-resident greedy decoding,
CUDA graph replay, and zero Python overhead.  We adapted the first two kernel-level
optimizations for rLLM's backend-agnostic architecture.

---

## Optimization 1: GPU-Resident Argmax Kernel

### The Problem

During greedy decode (temperature=0), the standard pipeline copies the **full logits
tensor** from GPU to host, then does argmax on CPU:

```
Model → logits [batch, vocab_size] bf16
  ↓ DtoH copy (PCIe)
Host buffer
  ↓ CPU argmax scan
Token ID (u32)
```

For Qwen 2.5 (vocab 151,936) at batch size 128:
- **Before**: 128 × 151,936 × 2 bytes = **37 MB per decode step** over PCIe
- **After**: 128 × 4 bytes = **512 bytes per decode step**
- **Reduction**: ~72,000x less DtoH transfer

At PCIe Gen4 (~25 GB/s), 37 MB takes ~1.5ms.  Over 32 decode steps, that's
~48ms of pure PCIe stall time eliminated.

### Implementation

**GPU Kernel** (`argmax_gpu` in elementwise.cu / elementwise.metal):
- One block per row (batch element), block size = min(vocab_size, 1024)
- Each thread strides across its row finding a local max
- Warp-level reduction via shuffle intrinsics
- Cross-warp reduction via shared memory
- Output: `u32[batch_size]` — one token ID per sequence

**Greedy Gate** (engine/mod.rs):
- Before sampling, checks if temperature == 0.0 for single-sequence decode
- For batched decode, checks if ALL sequences have temperature == 0.0
- When the gate triggers, calls `sample_greedy_gpu()` / `sample_batch_greedy_gpu()`
  instead of the full sampling pipeline (no DtoH, no softmax, no top-p)

**Files**:
```
src/gpu/ops/elementwise.rs           — argmax_gpu trait method
src/gpu/cuda/shaders/elementwise.cu  — CUDA argmax kernel
src/gpu/cuda/kernels/elementwise.rs  — CUDA dispatch
src/gpu/cuda/backend.rs              — fn_argmax_gpu kernel handle
src/gpu/metal/shaders/elementwise.metal — Metal argmax kernel
src/gpu/metal/kernels/elementwise.rs    — Metal dispatch
src/gpu/metal/backend.rs                — pipeline_argmax_gpu
src/gpu/cpu/mod.rs                      — CPU reference impl
src/model/sampler.rs                    — sample_greedy_gpu, sample_batch_greedy_gpu
src/engine/dispatch.rs                  — Dispatch trait: sample_greedy_gpu, sample_batch_greedy_gpu
src/engine/mod.rs                       — greedy gate in sample_and_finish + batched decode
```

---

## Optimization 2: Fused Residual + RMSNorm Kernel

### The Problem

Each transformer layer does two operations back-to-back:

```
1. hidden += attention_output    (residual connection — reads hidden, reads proj, writes hidden)
2. norm = rms_norm(hidden)       (normalisation — reads hidden AGAIN, writes norm)
```

The `hidden` tensor is read twice: once for the add, once for the norm.  Fusing
eliminates one full read of `hidden` per call site.

### Impact

- 1 fused call site per layer (post-attention residual + FFN norm)
- For a 32-layer model: 32 fewer hidden-tensor reads per token
- Hidden tensor size: 4096 × 2 bytes = 8 KB (Llama 3.2 1B) to 8192 × 2 = 16 KB (Llama 3.1 70B)
- Also halves kernel launch overhead at each site (1 dispatch instead of 2)

### Implementation

**GPU Kernel** (`fused_residual_rms_norm` in rms_norm.cu / rms_norm.metal):
- Single pass: read hidden + residual → add in-place → accumulate sum-of-squares
- Same reduction pattern as standard RMSNorm (warp shuffle + shared memory)
- Second pass: read updated hidden → scale by weight * rsqrt(mean_sq + eps) → write output
- `_batch` variant: one block per row of [batch_size, hidden_size]

**Model Integration**:
- `o_proj_fused_residual_norm_qdim()` in primitives.rs: replaces the O-projection
  residual add + FFN's initial RMSNorm with a single fused kernel
- `ffn_block_pre_normed()` in primitives.rs: FFN block that skips the initial RMSNorm
  (already done by the fused kernel)
- Currently enabled for the Llama/Qwen single-token decode path

**Files**:
```
src/gpu/ops/norm.rs                  — fused_residual_rms_norm trait methods
src/gpu/cuda/shaders/rms_norm.cu     — CUDA fused kernel
src/gpu/cuda/kernels/norm.rs         — CUDA dispatch
src/gpu/cuda/backend.rs              — kernel handles
src/gpu/metal/shaders/rms_norm.metal — Metal fused kernel
src/gpu/metal/kernels/norm.rs        — Metal dispatch
src/gpu/metal/backend.rs             — pipeline handles
src/gpu/cpu/mod.rs                   — CPU reference impl
src/model/primitives.rs              — o_proj_fused_residual_norm_qdim, ffn_block_pre_normed
src/model/registry/llama.rs          — updated single-token decode path
```

---

## Future Work: CUDA Graph Replay

**Not yet implemented.**  The third major optimization from rvLLM is CUDA graph
capture/replay, which records the entire decode forward pass and replays it as a
single `cuGraphLaunch` call (~5μs total vs ~3.2ms of kernel launch overhead for
a 32-layer model).

This requires:
- Stable input tensor addresses between capture and replay
- Separate graph captures for different padded batch sizes (1, 2, 4, 8, 16, 32)
- KV cache pointer updates via `cuGraphExecKernelNodeSetParams`

Deferred until optimizations 1 and 2 are validated in production.

---

## Verification

All optimizations maintain numerical equivalence with the original code paths:
- GPU argmax produces identical token IDs to CPU argmax
- Fused residual+RMSNorm produces identical outputs to separate add+rms_norm
  within bf16 precision
- Existing `cargo test` suite validates correctness
