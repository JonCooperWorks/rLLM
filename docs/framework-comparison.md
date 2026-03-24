# LLM Inference Framework Comparison

A comparison of rLLM against vLLM, Ollama, and llama.cpp across security,
ease of understanding, features, and code quality.

---

## Security

### rLLM — Strong

- **Memory safety**: Rust eliminates buffer overflows, use-after-free, and data races
  at compile time. Only ~18 controlled `unsafe` blocks, all justified (mmap, GPU buffer
  writes, interior mutability with documented invariants).
- **TLS**: Built-in via rustls (no OpenSSL). Two modes: manual PEM certificates and
  automatic Let's Encrypt (ACME TLS-ALPN-01). External interfaces require TLS by default;
  bypassing requires explicit `--dangerous_no_tls`.
- **Authentication**: Pluggable system with OIDC JWT validation (JWKS caching + rotation)
  and static API key auth (argon2id, constant-time comparison). Auth middleware covers all
  routes except `/health`. AuthUser identity flows through the request lifecycle for audit.
- **Threat model**: Documented STRIDE analysis in `docs/threat-model.md`.
- **No known CVEs**.

### vLLM — Weak

- **Memory safety**: Python + C++/CUDA. Python pickle deserialization and dynamic code
  loading create a large attack surface.
- **TLS**: Built-in SSL/TLS via Python's ssl module.
- **Authentication**: `--api-key` flag, but only protects `/v1` endpoints. Other endpoints
  (`/invocations`, `/pooling`, `/classify`) bypass auth. API key comparison vulnerable to
  timing attacks.
- **Critical CVEs**:
  - CVE-2026-22778 (CVSS 9.8) — unauthenticated RCE via crafted video URL
  - CVE-2025-62164 (CVSS 8.8) — RCE via unsafe `torch.load()` deserialization
  - CVE-2025-66448 — RCE via model config auto-mapping
  - CVE-2025-30165 — RCE via ZeroMQ pickle deserialization
- **Verdict**: Deploy behind a reverse proxy. Do not expose directly to the internet.

### Ollama — Weak

- **TLS**: None built-in. Requires a reverse proxy (nginx, Caddy) for HTTPS.
- **Authentication**: None by default. API is open on port 11434.
- **CVEs**: CVE-2025-63389 (missing auth enables unauthorized model management),
  out-of-bounds write in model parsing (fixed by rewriting C++ to Go), path traversal
  (CVE-2024-39722), file existence disclosure.
- **Verdict**: Designed for local/personal use. Must be hardened with firewall rules and
  reverse proxy for any network exposure.

### llama.cpp — Moderate

- **TLS**: Optional via OpenSSL (compile-time flag).
- **Authentication**: API key support (single key, comma-separated list, or key file).
  Sensitive endpoints disabled by default.
- **CVEs**: CVE-2025-49847 (CVSS 8.8) — buffer overflow via malicious GGUF vocabulary.
  Additional RPC server vulnerabilities. C/C++ codebase inherently susceptible to memory
  corruption.
- **Verdict**: Security-conscious endpoint design, but C/C++ memory safety risks persist.

---

## Ease of Understanding

| Dimension | rLLM | vLLM | Ollama | llama.cpp |
|---|---|---|---|---|
| **Language** | Rust | Python + C++/CUDA | Go + C/C++ | C/C++ |
| **Codebase size** | ~40K lines, 88 files | ~587K+ lines, 2,600+ files | Medium (Go wrapper over llama.cpp) | Very large, files routinely 10K+ lines |
| **In-code docs** | Every file annotated with purpose, rationale, cross-references | Variable | Minimal in-code, good user-facing docs | Variable, community-driven |
| **Architecture** | Composable GPU sub-traits, clear module separation, CLAUDE.md maps entire codebase | Complex multi-layer (scheduler, engine, executor, distributed), V0/V1 transition adds confusion | Simple (thin CLI, HTTP server, scheduler, runner) | Monolithic files, ongoing modularization effort |
| **Learning curve** | Moderate (Rust + small annotated codebase) | High (huge codebase + distributed systems) | Low (Go is approachable, CLI-first UX) | High (C++ complexity, massive codebase) |

**Key insight**: nano-vllm reproduced vLLM's core inference in ~1,200 lines with comparable
performance, suggesting significant complexity overhead in the full vLLM codebase.

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

### rLLM — High
- Trait-based GPU abstraction (9 composable sub-traits vs monolithic interface)
- `pub(crate)` visibility enforced — nothing leaks outside the crate
- `#[repr(C)]` param structs byte-matched to shader layouts
- CPU backend serves as reference implementation for GPU kernel testing
- Single crate, ~40K lines — feasible to read end-to-end
- Every file has mandatory header annotation

### vLLM — Mixed
- Strong CUDA kernel implementations
- Python enables rapid iteration but sacrifices type safety
- 587K+ lines with acknowledged bloat; Docker images exceed 10GB
- CI flakiness is a known problem (developers suspect CI before code)
- Active V0→V1 refactoring creating parallel code paths

### Ollama — Good
- Clean Go architecture with process isolation
- Rewriting C++ model handling to Go fixed a critical OOB vulnerability — a
  practical demonstration of memory-safe language benefits
- CGO boundary between Go and C/C++ creates synchronization risks
- Content-addressed model storage with SHA256 deduplication

### llama.cpp — Mixed
- Zero external dependencies; compiles to single binary
- 1,200+ contributors, ~4,000 releases — high velocity
- Monolithic files (routinely 10K+ lines) acknowledged as maintainability problem
- Organic growth from single-file proof of concept
- 12+ hardware backends create combinatorial testing burden

---

## Summary

| Dimension | rLLM | vLLM | Ollama | llama.cpp |
|---|---|---|---|---|
| **Security** | Strong | Weak | Weak | Moderate |
| **Understandability** | High | Low | High | Low |
| **Features** | Focused | Broadest | Broad (inherited) | Broad |
| **Code quality** | High | Mixed | Good | Mixed |

**rLLM** trades breadth for depth: fewer model architectures and hardware backends, but
stronger security guarantees, a codebase small enough to understand fully, and unique
capabilities like expert streaming. It is the only framework with no known CVEs, built-in
OIDC authentication, and a formal threat model.

**vLLM** leads on features (100+ models, speculative decoding, multi-LoRA) but carries
significant security debt and codebase complexity.

**Ollama** excels at user experience and simplicity but lacks built-in security, making it
unsuitable for network-exposed deployments without additional infrastructure.

**llama.cpp** offers the broadest hardware support and quantization options but inherits
C/C++ memory safety risks that have produced real vulnerabilities.
