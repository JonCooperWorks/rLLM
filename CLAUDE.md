# rLLM — Project Conventions

Rust LLM inference engine with a Metal GPU backend.  Educational codebase — every file should
be annotated so a reader can understand *why* it exists, not just *what* it does.

## Architecture

### GPU Backend Traits (`src/gpu/ops/`)

The GPU interface is a set of composable sub-traits, not a monolithic god-trait:

```
GpuCore          — tensor lifecycle, device info, command buffer control
GpuNorm          — RMS normalization + LayerNorm (vision encoder)
GpuMatmul        — matrix-vector + batched GEMM
GpuRope          — rotary positional embeddings
GpuAttention     — attention + KV cache (causal + bidirectional)
GpuElementwise   — activations (SiLU, GELU, GeGLU), arithmetic, MoE routing
GpuEmbed         — embedding lookup
GpuMoe           — fused MoE kernels (gate+up+SwiGLU, combine+residual)
GpuDeltaNet      — Qwen 3.5 linear attention kernels
GpuAllReduce     — collective communication for tensor parallelism
GpuVision        — vision encoder kernels (spatial merge, token scatter)
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

Ten model families, each implementing the `ModelForward` trait (defined in
`src/model/forward.rs`).  The engine holds `Box<dyn ModelForward<B>>` — one
match at construction time in `engine/loader.rs`, no match dispatch after that.

Standard dense transformers (Llama, Phi, Qwen, Mistral) share `LlamaForward`
parameterised by `ArchFeatures`.  Complex architectures have their own structs:

```
llama.rs       — LlamaForward (Llama 3.x, also used by Phi/Qwen/Mistral)
gemma.rs       — GemmaForward (sandwich norms, GeGLU, embed scaling)
mixtral.rs     — MixtralForward (MoE FFN, holds MoeBuffers)
qwen3_moe.rs   — Qwen3MoeForward (QK-norm + MoE, holds MoeBuffers)
qwen3_5.rs     — Qwen35Forward (DeltaNet + GQA + MoE, holds DeltaNetBuffers + MoeBuffers)
gpt_oss.rs     — GptOssForward (YaRN + MoE + biases, holds MoeBuffers)
nemotron_h.rs  — NemotronForward (Mamba-2 + attention + MoE, holds Mamba2Buffers + MoeBuffers)
qwen.rs        — documentation only (uses LlamaForward with has_qkv_bias: true)
phi.rs         — documentation only (uses LlamaForward)
mistral.rs     — documentation only (uses LlamaForward)
```

Config parsing: `src/model/config.rs` (`ModelArch` enum + `VisionConfig` for VLMs).
Weight loading: `src/model/loader/` (safetensors, single + multi-shard, pre-quantized Q4 auto-detection,
  split into store, upload, mxfp4, vision, expert_index submodules).
Expert streaming: `src/model/expert_stream.rs` (SSD-backed on-demand expert loading for large MoE models).
Vision encoder: `src/model/vision.rs` (SigLIP ViT forward pass, image preprocessing with tiling, patch embedding).

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

### TurboQuant KV Cache Quantization (`src/model/turboquant.rs`)

Online vector quantization for the KV cache, based on Zandieh et al. (arXiv:2504.19874).
**On by default at 4-bit** (~4x KV compression, quality-neutral).  Override with `--kv-quant`.

Algorithm: random orthogonal rotation → Max-Lloyd scalar quantization per coordinate.
Efficiency: Q is pre-rotated once, K dequant is a centroid lookup per position, V inverse
rotation happens once per query head (not per position).

Both prefill and decode paths quantize K/V into the paged pool.  During prefill, attention
uses full BF16 Q/K/V directly (no quality loss).  During decode, attention reads from the
quantized cache with inline dequantization.

```
model/turboquant.rs                — KvQuantMode, TurboQuantConfig, rotation matrix, TurboContext
gpu/ops/turboquant.rs              — GpuTurboQuant trait
gpu/metal/shaders/turboquant.metal — Metal kernels (quantize, rotate_q, paged_attention)
gpu/metal/kernels/turboquant.rs    — Metal dispatch
model/primitives.rs                — paged_kv_and_attention_maybe_quantized() (decode)
                                     paged_kv_and_prefill_attention_maybe_quantized() (prefill)
docs/turboquant.md                 — Full documentation
```

### Vision Encoder (`src/model/vision.rs`)

Vision-language models (VLMs) like Qwen 3.5 and Gemma 3 include a SigLIP-based Vision
Transformer (ViT) that converts images into token embeddings the LLM can process.

**Architecture:**
```
Image [3, H, W]
  ↓ CPU: decode, resize, CLIP-normalise, patchify (rayon-parallel)
Patches [N, 768]             (768 = 3 channels × 16² pixels per patch)
  ↓ GPU: patch_embed matmul + positional embedding add
Patch embeddings [N, 1152]
  ↓ GPU: 27× ViT blocks (LayerNorm → fused QKV matmul → bidirectional attention → GELU FFN)
Vision features [N, 1152]
  ↓ GPU: fused spatial merge + LayerNorm (single kernel) + MLP projector
Vision tokens [M, 5120]      (M = N/4 after merge, matches LLM hidden_size)
  ↓ GPU: scatter into text embedding buffer at <|image_pad|> position
```

**Key design decisions:**
- Fused QKV matmul — single [3\*hd, hd] weight, one matmul per ViT block instead of three
- Fused spatial merge + LayerNorm — one kernel dispatch instead of two, avoids intermediate buffer round-trip
- LayerNorm (not RMSNorm) — matches the frozen SigLIP encoder's training
- Bidirectional attention (`causal=false`) — images have no left-to-right ordering
- Plain GELU (not GeGLU/SwiGLU) — standard ViT activation
- Patch embedding via matmul — equivalent to Conv2D with stride==kernel_size
- Spatial merge (Qwen 3.5) — 2×2 adjacent patches concatenated to reduce token count 4×
- All projections have bias — SigLIP design choice (unlike LLM layers)
- Rayon-parallel preprocessing — patch rows processed in parallel on CPU

**Data flow through the system:**
```
API request (base64 image)
  → chat.rs: deserialise content array, extract ImageData
  → api/mod.rs: preprocess_images() on handler thread (CPU, rayon-parallel)
  → WorkerRequest.images: Vec<ProcessedImage>
  → engine: SequenceRequest.images → Sequence.images
  → ModelForward::prefill_preamble: embed_lookup → vision_encode → scatter
  → ModelForward::forward_prefill: transformer layers
```

**Files:**
```
model/vision.rs         — VisionWeights, VisionBuffers, preprocess_image(), vision_encode()
model/config.rs         — VisionConfig (parsed from vision_config in config.json)
model/loader/vision.rs  — load_vision_weights() (handles f32→bf16, fused QKV concat, temporal avg)
model/chat.rs           — ImageData, vision placeholder tokens in chat templates
model/forward.rs        — prefill_preamble() default impl (vision scatter integration)
gpu/ops/vision.rs       — GpuVision trait (spatial_merge, spatial_merge_norm, scatter_vision_tokens)
gpu/ops/attention.rs    — prefill_attention_fused_qkv (interleaved QKV attention)
gpu/ops/norm.rs         — layer_norm_batch (LayerNorm for ViT)
gpu/metal/shaders/vision.metal    — spatial merge, fused merge+norm, scatter kernels
gpu/metal/shaders/attention.metal — fused QKV bidirectional attention kernel
```

### CLI Commands (`src/commands/`)

```
rllm run    — single-prompt inference (--image <path> for vision models)
rllm batch  — batched inference from file
rllm serve  — HTTP API server (accepts images via OpenAI/Anthropic multimodal format)
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
2. **`src/model/loader/mod.rs`**: Set the right flags in `LoaderHints::new()`.  If the model
   uses standard weight formats, zero additional loader code is needed.  Novel formats
   (like MXFP4) get their own helper function (see `loader/mxfp4.rs`).
3. **`src/model/registry/new_model.rs`**: Implement the `ModelForward` trait
   (`forward_decode`, `forward_prefill`, optionally `forward_decode_batch`).
   Standard dense transformers can use `LlamaForward` from `registry/llama.rs`.
4. **`src/engine/loader.rs`**: Add match arm in `create_forward()` to construct
   the new `ModelForward` implementor (with any arch-specific buffers).
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
