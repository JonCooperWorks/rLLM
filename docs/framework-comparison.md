# LLM Inference Framework Comparison

A comparison of rLLM against vLLM, Ollama, and llama.cpp across
ease of understanding, features, and code quality. rLLM builds on ideas
pioneered by these projects.

---

## Ease of Understanding

| Dimension | rLLM | vLLM | Ollama | llama.cpp |
|---|---|---|---|---|
| **Language** | Rust | Python + C++/CUDA | Go + C/C++ | C/C++ |
| **Codebase size** | ~40K lines, 88 files | ~587K+ lines, 2,600+ files | Medium (Go wrapper over llama.cpp) | Large, 1,200+ contributors |
| **In-code docs** | Every file annotated with purpose, rationale, cross-references | Extensive official docs site, blog posts, RFCs | Minimal in-code, good user-facing docs | Community wiki, GitHub discussions |
| **Architecture** | Composable GPU sub-traits, clear module separation | Multi-layer (scheduler, engine, executor, distributed) | Simple (thin CLI, HTTP server, scheduler, runner) | Multi-backend with ongoing modularization |
| **Learning curve** | Moderate (Rust + small annotated codebase) | High (large codebase + distributed systems) | Low (Go is approachable, CLI-first UX) | High (C++ complexity, large codebase) |

---

## Features

| Feature | rLLM | vLLM | Ollama | llama.cpp |
|---|---|---|---|---|
| **Model architectures** | 9 families | ~100+ | Inherits llama.cpp | Dozens via GGUF |
| **Quantization** | Q4 (bf16 scale, block-32) | FP8, AWQ, GPTQ, INT4/8 | Q2–Q8, K-quants | Q2–Q8, K-quants, 1.58-bit |
| **API compatibility** | OpenAI + Anthropic | OpenAI | OpenAI | OpenAI |
| **Continuous batching** | Yes (paged KV cache) | Yes (PagedAttention pioneer) | Basic scheduling | Yes (server mode) |
| **Hardware** | Metal, CUDA, CPU (test) | NVIDIA, AMD, Intel, TPU, ARM | CUDA, ROCm, Vulkan, Metal, CPU | 12+ backends |
| **Vision/multimodal** | Yes (SigLIP ViT) | Yes (image, audio, video) | Yes (image-to-text) | Yes (LLaVA) |
| **Tool calling** | Yes (4 formats) | Yes | Yes | Yes |
| **Distributed** | Tensor parallelism | TP, PP, DP, EP; multi-node | Ollama Cloud | RPC multi-node |
| **Expert streaming** | Yes (SSD-backed MoE) | No | No | No |
| **Speculative decoding** | No | Yes | No | Yes |
| **LoRA** | No | Yes (multi-LoRA batching) | Yes | Yes |

**rLLM's unique feature**: Expert streaming loads MoE experts on-demand from NVMe with
GPU-side LRU caching, enabling models larger than GPU memory without full offloading.

---

## Code Quality

### rLLM
- Rust with compile-time memory safety and `pub(crate)` visibility discipline
- Trait-based GPU abstraction (9 composable sub-traits)
- `#[repr(C)]` param structs byte-matched to shader layouts
- CPU backend serves as reference implementation for GPU kernel testing
- Single crate, ~40K lines — feasible to read end-to-end
- Every file has mandatory header annotation

### vLLM
- Python (high-level logic) + C++/CUDA (kernels) — rapid iteration with strong kernel implementations
- Pioneered PagedAttention, a foundational contribution to LLM serving
- Large codebase reflecting broad model and hardware support
- Active V1 engine refactoring to simplify architecture
- Extensive CI/CD with Buildkite, pytest, and RFC-driven development process

### Ollama
- Clean Go architecture with process isolation
- Content-addressed model storage with SHA256 deduplication
- Excellent developer experience — `ollama run` is the simplest way to get started with local LLMs
- Dual engine approach (llama.cpp + native Go) provides migration flexibility

### llama.cpp
- Zero external dependencies; compiles to single binary — remarkable portability
- 1,200+ contributors, ~4,000 releases — massive community and velocity
- Defined the GGUF format, now the de facto standard for distributing quantized models
- 12+ hardware backends providing the broadest device support of any framework
- Active modularization effort to improve maintainability

---

## Summary

| Dimension | rLLM | vLLM | Ollama | llama.cpp |
|---|---|---|---|---|
| **Understandability** | High | Moderate | High | Moderate |
| **Features** | Focused | Broadest | Broad (inherited) | Broad |
| **Code quality** | High | High | Good | Good |

**rLLM** trades breadth for depth: fewer model architectures and hardware backends, but
a codebase small enough to understand fully and unique capabilities like expert streaming.
It is an educational project that aims to make LLM inference internals accessible.

**vLLM** leads on features and pioneered PagedAttention, which fundamentally changed how
LLM serving manages memory. The go-to choice for high-throughput production GPU serving.

**Ollama** made local LLM inference accessible to everyone. Its CLI-first design and model
library set the standard for developer experience.

**llama.cpp** proved that efficient LLM inference doesn't require a GPU or a heavy framework.
GGUF and its quantization ecosystem are foundational contributions that the entire community
builds on.
