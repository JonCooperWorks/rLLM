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
GpuDeltaNet      — Qwen 3.5 linear attention kernels
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

Models live in `src/model/`.  Each model implements a forward pass generic over `B: GpuBackend`.
Config parsing goes in `src/config.rs`.  Weight loading goes in `src/loader.rs`.

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
- Linux: CUDA backend (`src/gpu/cuda/`) — currently stubbed with `unreachable!()`
- `gpu::Backend` type alias in `src/gpu/mod.rs` resolves to the platform-specific backend

## Q4 Quantization Format

Block size 32 weights, 20 bytes per block: 4-byte f32 scale + 16 bytes packed nibbles.
Symmetric: `scale = max(abs) / 7`, `q = clamp(round(v / scale), -8, 7)`, stored as `q + 8`.
