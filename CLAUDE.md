# rLLM — Project Conventions

Rust LLM inference engine with a Metal GPU backend.  Educational codebase — every file should
be annotated so a reader can understand *why* it exists, not just *what* it does.

## Architecture

### GPU Backend Traits (`src/gpu/ops/`)

The GPU interface is a set of composable sub-traits, not a monolithic god-trait:

```
GpuCore          — tensor lifecycle, device info, command buffer control
GpuNorm          — RMS normalization
GpuMatmul        — matrix-vector + batched GEMM
GpuRope          — rotary positional embeddings
GpuAttention     — attention + KV cache
GpuElementwise   — activations, arithmetic, MoE routing
GpuEmbed         — embedding lookup
GpuMoe           — fused MoE kernels (gate+up+SwiGLU, combine+residual)
GpuDeltaNet      — Qwen 3.5 linear attention kernels
GpuAllReduce     — collective communication for tensor parallelism
```

All extend `GpuCore` (which owns the associated `Tensor` type).  `GpuBackend` is a blanket
supertrait: `impl<T> GpuBackend for T where T: GpuCore + GpuNorm + ...`.  Model code uses
`B: GpuBackend` as a single bound; calling code that only needs device info imports `GpuCore`.

### Metal Backend (`src/gpu/metal/`)

```
metal/
├── mod.rs         — re-exports MetalBackend, MetalTensor
├── backend.rs     — struct, pipeline compilation, async dispatch infrastructure
├── tensor.rs      — MetalTensor (buffer + shape + dtype)
├── kernels/       — one file per ops/ sub-trait (impl GpuFoo for MetalBackend)
└── shaders/       — .metal shader sources (embedded via include_str!)
```

### Models (`src/model/registry/`)

Nine model families, each implementing a forward pass generic over `B: GpuBackend`:

```
llama.rs       — Llama 3.x
qwen.rs        — Qwen 2.5
qwen3_moe.rs   — Qwen 3 Mixture-of-Experts
qwen3_5.rs     — Qwen 3.5 hybrid (DeltaNet + GQA)
phi.rs         — Phi
gemma.rs       — Gemma 3
mistral.rs     — Mistral
mixtral.rs     — Mixtral (MoE)
gpt_oss.rs     — GPT Open Source
```

Config parsing: `src/model/config.rs` (`ModelArch` enum).
Weight loading: `src/model/loader.rs` (safetensors, single + multi-shard, pre-quantized Q4 auto-detection).
Expert streaming: `src/model/expert_stream.rs` (SSD-backed on-demand expert loading for large MoE models).

### Inference Engine (`src/engine/`)

```
engine/
├── mod.rs         — inference engine with continuous batching
└── scheduler.rs   — multi-sequence scheduler + paged KV cache
```

Prefill and decode phases are separated.  `src/model/kv_cache.rs` implements a paged KV cache
with block tables and a shared `KvPool` across concurrent sequences.

### API Server (`src/api/`)

OpenAI- and Anthropic-compatible HTTP API:

```
api/
├── mod.rs         — server impl, continuous batching via tokio + worker threads
├── openai.rs      — /v1/chat/completions, /v1/completions, /v1/models
├── anthropic.rs   — /v1/messages
└── tls.rs         — TLS support (manual certs, Let's Encrypt)
```

Tool/function calling lives in `src/model/tools.rs` with architecture-specific prompt formatting
and output parsing (Llama 3.1, Qwen, Mistral, Anthropic formats).

### CLI Commands (`src/commands/`)

```
rllm run    — single-prompt inference
rllm batch  — batched inference from file
rllm serve  — HTTP API server
```

### Adding a New Kernel Family

1. Create the trait in `gpu/ops/new_family.rs`
2. Re-export from `gpu/ops/mod.rs`
3. Create the shader in `metal/shaders/new_family.metal`
4. Add pipeline fields + compilation in `metal/backend.rs`
5. Create `metal/kernels/new_family.rs` implementing the trait
6. Add `mod new_family;` to `metal/kernels/mod.rs`
7. Add the trait to the `GpuBackend` supertrait bound in `gpu/mod.rs`
8. Add a stub impl in `gpu/cuda/mod.rs`

### Adding a New Model

1. **`src/model/config.rs`**: Add `ModelArch` variant + `from_model_type()` case + feature
   methods (e.g. `has_qkv_bias()`) if the new arch has novel traits.
2. **`src/model/loader.rs`**: Set the right flags in `LoaderHints::new()`.  If the model
   uses standard weight formats, zero additional loader code is needed.  Novel formats
   (like MXFP4) get their own helper function (see `load_mxfp4_experts()`).
3. **`src/model/registry/new_model.rs`**: Forward pass implementation.  Standard dense
   transformers can delegate to `registry/llama.rs` via `ArchFeatures`.
4. **`src/model/mod.rs`**: Add dispatch arms in `forward_single_paged()` and
   `forward_prefill_paged()`.
5. **`src/model/chat.rs`**: Add chat template match arm (or reuse ChatML).
6. **`src/model/tools.rs`**: Add tool-call format match arm if applicable.

## Code Style

- **Annotate every file** with a header comment block explaining what lives there, why,
  and cross-references to related files (trait ↔ impl, Rust ↔ shader).
- **`#[repr(C)]` param structs** must match the corresponding Metal shader struct layout
  byte-for-byte.  Name them after the kernel (e.g., `RmsNormParams`, `MatvecParams`).
- **OS-conditional compilation** (`#[cfg(target_os = "macos")]`) instead of feature flags.
- **`pub(crate)` visibility** — nothing is `pub` outside the crate.
- Prefer trait indirection over tight coupling.
- Keep files focused — split when a file serves multiple unrelated concerns.

## Platform

- macOS: Metal backend (`src/gpu/metal/`)
- Linux: CUDA backend (`src/gpu/cuda/`) — NVRTC runtime compilation, async streams
- Tests: CPU backend (`src/gpu/cpu/`) — reference impl in pure Rust
- `gpu::Backend` type alias in `src/gpu/mod.rs` resolves to the platform-specific backend

## Q4 Quantization Format

Block size 32 weights, 18 bytes per block: 2-byte bf16 scale + 16 bytes packed nibbles.
Symmetric: `scale = max(abs) / 7`, `q = clamp(round(v / scale), -8, 7)`, stored as `q + 8`.
bf16 scale (vs f32) saves 10% I/O per block — critical for NVMe-bound expert streaming.
