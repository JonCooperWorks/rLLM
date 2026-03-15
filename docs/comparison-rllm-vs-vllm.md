# rLLM vs vLLM: A Detailed Comparison

## Overview

| Dimension | rLLM | vLLM |
|---|---|---|
| **Language** | Rust + Metal shaders | Python + C++/CUDA kernels |
| **Codebase size** | ~15,700 lines (Rust + Metal) | ~500,000+ lines (Python + C++ + CUDA) |
| **GPU backends** | Metal (macOS), CUDA (stubbed) | CUDA (NVIDIA), ROCm (AMD), TPU, CPU |
| **Primary target** | Educational / single-user inference on Apple Silicon | Production serving at scale |
| **Maturity** | Solo project, v0.1.0 | Industry standard, hundreds of contributors |
| **License** | (not specified) | Apache 2.0 |

---

## Architecture Comparison

### GPU Abstraction

**rLLM** uses a composable trait system with eight sub-traits (`GpuCore`, `GpuNorm`, `GpuMatmul`, `GpuRope`, `GpuAttention`, `GpuElementwise`, `GpuEmbed`, `GpuDeltaNet`) unified under a blanket `GpuBackend` supertrait. Model code is fully generic over `B: GpuBackend` — zero platform-specific imports in the model layer. This is elegant Rust design: fine-grained trait bounds, `#[cfg(target_os)]` conditional compilation, and a single `gpu::Backend` type alias resolving to the platform backend.

**vLLM** uses a `ModelRunner` / execution engine pattern with a platform abstraction layer, but the architecture is more deeply coupled to CUDA. Custom CUDA kernels (PagedAttention, FlashAttention integration) are wrapped via pybind11/torch extensions. The abstraction is pragmatic rather than principled — it works across CUDA/ROCm/TPU but carries significant platform-specific branching.

**Verdict**: rLLM's trait-based design is *architecturally cleaner*. vLLM's is *more battle-tested*.

### KV Cache & Memory Management

**rLLM** implements a paged KV cache with block size 16, free-list allocation (LIFO stack), per-sequence block tables synced to GPU, and support for up to 128K context (8192 blocks × 16 positions). The implementation is ~240 lines of clear, well-documented Rust.

**vLLM** pioneered PagedAttention — the same concept but taken further. Variable block sizes, cross-sequence block sharing (for parallel sampling / beam search), prefix caching (shared prompt blocks across requests), copy-on-write semantics, and GPU memory profiling to auto-size the pool. Memory waste drops from 60-80% (naive approach) to under 4%.

**Verdict**: Same core idea. vLLM's implementation is far more sophisticated (sharing, CoW, prefix caching). rLLM's is a clean, correct educational implementation.

### Scheduling & Batching

**rLLM** has a continuous batching scheduler (~170 lines) using FCFS (first-come, first-served). Sequences join/leave mid-batch. Prefilling sequences get their entire prompt processed via GEMM in one shot. Decode sequences run one token per step. KV block availability gates admission.

**vLLM** has a multi-level scheduler with preemption (can evict running sequences to admit higher-priority ones), chunked prefill (interleaves prefill chunks with decode tokens in the same batch for better GPU utilization), priority-based scheduling, and fairness policies. The V1 engine flattens all sequences into a single concatenated "super sequence" for maximal GPU occupancy.

**Verdict**: rLLM's scheduler is correct and functional. vLLM's scheduler is production-grade with features like preemption and chunked prefill that are critical at scale.

### Model Support

**rLLM** supports 8 architectures: Llama, Mistral, Mixtral (MoE), Phi, Qwen2, Qwen3-MoE, Qwen 3.5 (hybrid DeltaNet + GQA), and Gemma3. Notable: Qwen 3.5 hybrid support with custom DeltaNet linear attention kernels — this is cutting-edge and unusual for a small project.

**vLLM** supports 100+ model architectures with automatic weight loading from HuggingFace, including vision-language models, encoder-decoder models, embedding models, and multi-modal models.

**Verdict**: vLLM wins on breadth. rLLM's Qwen 3.5 DeltaNet support is impressively niche.

### Quantization

**rLLM** uses a custom Q4 format: block size 32, symmetric quantization, 20 bytes per block (4-byte f32 scale + 16 bytes packed nibbles). Simple, efficient, hand-written Metal kernels.

**vLLM** supports GPTQ, AWQ, SqueezeLLM, FP8, INT8, GGUF, Marlin kernels, and more. Deep integration with quantization-aware training frameworks.

**Verdict**: vLLM vastly more complete. rLLM's Q4 is clean but limited to one format.

### API & Serving

**rLLM** exposes OpenAI-compatible and Anthropic-compatible HTTP APIs via axum, with TLS support (rustls-acme for automatic HTTPS certificates). Streaming responses via async-stream.

**vLLM** exposes an OpenAI-compatible API server (FastAPI), supports guided/structured generation (JSON mode, grammar-constrained decoding), tool calling, vision inputs, and integrates with inference orchestrators (Triton, TGI, etc.).

**Verdict**: rLLM's dual API compatibility (OpenAI + Anthropic) is a nice touch. vLLM's feature set is vastly larger.

---

## Quantitative Comparison

| Metric | rLLM | vLLM |
|---|---|---|
| **Throughput (single user)** | Competitive on Apple Silicon (Metal-native) | 14-24x over HF Transformers (CUDA) |
| **GPU utilization** | Good for single sequence; batching helps | >90% at scale with continuous batching |
| **Max concurrent requests** | Limited by Apple GPU memory (~10s) | Thousands (with sufficient GPU memory) |
| **Startup time** | Fast (native binary, no Python) | Slower (Python import, model compilation) |
| **Memory overhead** | Minimal (no Python runtime, no PyTorch) | ~2-4 GB Python/PyTorch overhead |
| **Speculative decoding** | No | Yes |
| **Multi-GPU** | No | Yes (tensor + pipeline parallelism) |
| **Prefix caching** | No | Yes |
| **Structured output** | No | Yes (grammar-constrained) |

---

## What rLLM Does Better

1. **Zero-dependency GPU path**: No PyTorch, no Python runtime. The binary directly talks to Metal via `metal-rs`. Cold start is nearly instant.

2. **Educational clarity**: Every file has a learning-oriented header explaining *why* it exists. The paged KV cache implementation is one of the clearest explanations of PagedAttention available in code.

3. **Trait-based backend abstraction**: The composable sub-trait design (`GpuNorm`, `GpuMatmul`, etc.) is architecturally superior to vLLM's more ad-hoc platform branching. Adding a new kernel family is a well-defined 8-step process.

4. **Apple Silicon native**: Metal shaders handwritten for the target hardware. No CUDA-to-Metal translation layer, no MPS compatibility shims.

5. **Memory footprint**: ~15K lines of Rust vs ~500K+ lines. The entire project fits in your head.

6. **Compile-time safety**: Rust's type system enforces correct tensor lifetime management, prevents data races in the async serving path, and catches shape mismatches at compile time (via the trait bounds).

---

## What vLLM Does Better

1. **Production readiness**: Battle-tested at scale by thousands of deployments. Robust error handling, monitoring, metrics, logging.

2. **Advanced scheduling**: Preemption, chunked prefill, fairness policies, priority queues — essential for multi-tenant serving.

3. **Memory optimization**: Prefix caching, cross-request block sharing, copy-on-write, automatic memory profiling. These multiply effective GPU memory by 2-4x at scale.

4. **Model coverage**: 100+ architectures vs 8. Automatic weight loading from HuggingFace.

5. **Quantization ecosystem**: GPTQ, AWQ, FP8, Marlin, etc. vs a single Q4 format.

6. **Multi-GPU / distributed**: Tensor parallelism, pipeline parallelism, multi-node inference. rLLM is single-GPU only.

7. **Speculative decoding**: 2-3x latency reduction for interactive workloads.

8. **Structured generation**: JSON mode, regex-constrained decoding, tool calling support.

9. **Ecosystem**: Integration with HuggingFace, LangChain, OpenRouter, Triton Inference Server, etc.

---

## Rating

### rLLM: 7.5 / 10

| Category | Score | Notes |
|---|---|---|
| **Architecture & Design** | 9/10 | Exemplary trait-based GPU abstraction; clean separation of concerns |
| **Code Quality** | 9/10 | Excellent annotations, consistent style, well-tested scheduler/KV cache |
| **Correctness** | 8/10 | Paged KV, continuous batching, GEMM prefill — all correctly implemented |
| **Performance** | 6/10 | Good single-user Metal perf; no speculative decoding, no multi-GPU |
| **Feature Completeness** | 5/10 | 8 models, 1 quantization format, no distributed, no structured output |
| **Production Readiness** | 4/10 | Single-user focused; limited error recovery, no preemption |
| **Documentation** | 9/10 | In-code learning annotations are best-in-class |
| **Portability** | 5/10 | Metal only (CUDA stubbed); no CPU fallback for inference |

**Summary**: An *outstanding educational project* and a *competent single-user inference engine*. The Rust trait architecture is genuinely better-designed than vLLM's equivalent abstraction layer. The DeltaNet (Qwen 3.5) support shows real ambition. For what it sets out to do — teach someone how LLM inference works while being usable on a MacBook — it succeeds admirably.

### vLLM: 9.0 / 10

| Category | Score | Notes |
|---|---|---|
| **Architecture & Design** | 7/10 | Pragmatic but complex; V0→V1 migration shows growing pains |
| **Code Quality** | 7/10 | Variable — core kernels are excellent, Python glue is sprawling |
| **Correctness** | 9/10 | Battle-tested across thousands of production deployments |
| **Performance** | 9/10 | Near-optimal GPU utilization; PagedAttention, chunked prefill, speculative decoding |
| **Feature Completeness** | 10/10 | 100+ models, every quantization format, multi-GPU, structured output |
| **Production Readiness** | 10/10 | Industry standard for LLM serving |
| **Documentation** | 7/10 | Good docs but the codebase itself is harder to learn from |
| **Portability** | 8/10 | CUDA, ROCm, TPU, CPU — but CUDA is first-class, others lag |

**Summary**: The *industry standard* for LLM inference serving. Unmatched feature breadth and production battle-testing. The Python/C++ architecture is pragmatic but creates a high barrier to deep understanding compared to rLLM's transparent Rust.

---

## The Interesting Takeaway

rLLM demonstrates that the *core ideas* behind production LLM serving (PagedAttention, continuous batching, paged KV cache, GEMM prefill) can be expressed in ~15K lines of Rust with zero external ML framework dependencies. The remaining ~485K lines in vLLM are: (a) model coverage breadth, (b) production hardening, (c) advanced optimizations (speculative decoding, prefix caching, distributed), and (d) Python ecosystem integration.

For someone trying to *understand* how LLM inference works at the GPU kernel level, rLLM is the better codebase to read. For someone trying to *deploy* an LLM in production, vLLM is the clear choice.

---

*Sources:*
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [vLLM Deep Dive — Architecture, Features, and Production Best Practices](https://martinuke0.github.io/posts/2025-12-19-vllm-deep-dive-architecture-features-and-production-best-practices/)
- [Under the Hood of vLLM: Memory, Scheduling & Batching Strategies](https://www.javacodegeeks.com/2025/10/under-the-hood-of-vllm-memory-scheduling-batching-strategies.html)
- [vLLM Quickstart: High-Performance LLM Serving in 2026](https://www.glukhov.org/post/2026/01/vllm-quickstart/)
- [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/stable/design/paged_attention/)
