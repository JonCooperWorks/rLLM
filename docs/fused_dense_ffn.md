# Fused Dense FFN — Gate + Up + SwiGLU in One Kernel

The fused dense FFN optimisation replaces three separate kernel dispatches
(gate projection, up projection, SiLU activation + multiply) with a single
fused kernel that reads the input vector once instead of twice.

## Background: How SwiGLU FFN Works

Most modern LLMs (Llama, Qwen, Mistral, Phi) use SwiGLU as their FFN
activation.  The computation for a single token at each transformer layer is:

```
ffn_out = down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
```

Where:
- `x` is the hidden state after attention (size: hidden_size, e.g. 4096)
- `gate_proj` is [inter_size × hidden_size] (e.g. [14336 × 4096])
- `up_proj` is [inter_size × hidden_size] (same shape as gate)
- `down_proj` is [hidden_size × inter_size]
- `silu(z) = z / (1 + exp(-z))` (Sigmoid Linear Unit)

## The Problem: Redundant Input Reads

The naive implementation uses three separate kernel dispatches:

```
Dispatch 1: gate_buf = gate_proj @ x      ← reads x from device memory
Dispatch 2: up_buf   = up_proj   @ x      ← reads x from device memory AGAIN
Dispatch 3: gate_buf = silu(gate_buf) * up_buf
```

The input vector `x` (hidden_size × 2 bytes = 8 KB for Llama 8B) is read
from device memory twice — once for each mat-vec.  Additionally, each dispatch
has Metal encoding overhead (~2–3 µs).

Over a 32-layer model, this adds up to:
- 64 redundant `x` reads (256 KB total)
- 64 extra dispatch overheads (~150 µs total)

## The Fix: One Fused Kernel

The fused kernel (`fused_gate_up_swiglu`) computes both dot products in a
single pass:

```metal
// Each SIMD group (32 threads) handles one output row
float gate_acc = 0.0f, up_acc = 0.0f;
for (uint j = lane; j < k; j += 32) {
    float xi = float(input[j]);           // x loaded ONCE
    gate_acc += xi * float(w_gate[row * k + j]);  // gate dot product
    up_acc   += xi * float(w_up[row * k + j]);    // up dot product (reuses xi)
}
// SIMD reduction for both accumulators
gate_acc = simd_sum(gate_acc);
up_acc   = simd_sum(up_acc);
// Apply SiLU and multiply in-place
if (lane == 0) {
    float silu = gate_acc / (1.0f + exp(-gate_acc));
    output[row] = bfloat(silu * up_acc);
}
```

This kernel:
1. **Reads `x` once** — each thread loads `x[j]` and uses it for both gate
   and up dot products.  Halves input bandwidth.
2. **Eliminates 2 dispatches** — one kernel instead of three (gate matmul +
   up matmul + silu_mul).  Saves ~2–3 µs × 2 = 4–6 µs per layer.
3. **Fuses the activation** — SiLU + multiply happens in registers, avoiding
   a round-trip through device memory for the intermediate gate/up results.

Both bf16 and Q4 weight formats are supported.  The Q4 variant additionally
caches the input vector in threadgroup shared memory, which is especially
beneficial since Q4 dequantisation makes each weight access more expensive.

## Where It's Used

The fused kernel was originally written for MoE (Mixture of Experts) models,
where each expert runs a small gate+up+SwiGLU.  We now also use it for
**dense model decode** (single-token mat-vec):

```rust
// primitives.rs — ffn_block() for single-token decode
backend.fused_gate_up_swiglu(
    &layer.gate_proj,
    &layer.up_proj,
    norm_buf,       // input: hidden state after RMSNorm
    gate_buf,       // output: silu(gate @ x) * (up @ x)
    inter_size,     // m: output dimension
    hidden_size,    // k: input dimension
);
// Then: down_proj matmul on the fused result
backend.matmul(&layer.down_proj, gate_buf, norm_buf, hidden_size, inter_size);
```

The **batched prefill** path (`ffn_block_batch`) still uses separate GEMM
dispatches, since the fused kernel is a mat-vec (single-vector) kernel.  A
fused GEMM variant is a potential future optimisation.

## Impact

For a model generating at ~60 tok/s (~16.7 ms per token):
- Dispatch savings: ~150 µs per token → ~1% decode throughput improvement
- Bandwidth savings: ~256 KB per token → negligible at 400 GB/s
- Primary benefit: reduced kernel launch overhead, cleaner dispatch pipeline

The improvement is modest but comes at zero code complexity cost — we're
reusing an existing kernel for a new code path.

## Models Affected

All models using SwiGLU FFN in their single-token decode path:
- Llama 3.x (1B, 3B, 8B, 70B)
- Qwen 2.5 (3B, 7B, 14B, 32B, 72B)
- Qwen 3.5 dense variants (9B, 27B)
- Mistral 7B
- Phi-4

Models using GeGLU (Gemma 3) are **not** affected — a separate fused GELU
kernel would be needed.

## Files

```
model/primitives.rs                — ffn_block() uses fused_gate_up_swiglu
gpu/ops/moe.rs                     — GpuMoe::fused_gate_up_swiglu trait
gpu/metal/kernels/moe.rs           — Metal dispatch (bf16 + Q4 pipelines)
gpu/metal/shaders/moe.metal        — fused_gate_up_swiglu_bf16, _q4 kernels
model/registry/llama.rs            — GpuMoe bound on forward_single_impl
model/forward.rs                   — ModelForward trait (GpuBackend bound includes GpuMoe)
```
