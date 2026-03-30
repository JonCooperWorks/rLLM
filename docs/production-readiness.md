# rLLM Production Readiness Assessment

Comparison against vLLM, SGLang, llama.cpp, and LM Studio.

## What rLLM Does Well

| Area | Status | Notes |
|------|--------|-------|
| Continuous batching | Strong | Prefill-first scheduling, chunked prefill (2048 default) |
| Paged KV cache | Strong | Block allocation, copy-on-write, prefix caching, preemption |
| KV cache quantization | Strong | TurboQuant 4-bit on by default (~4x compression) — ahead of most competitors |
| API compatibility | Strong | OpenAI + Anthropic endpoints, streaming, tools, thinking |
| Auth | Strong | OIDC (JWT/JWKS), static API keys (argon2id), hot reload |
| TLS | Strong | Manual certs + Let's Encrypt ACME |
| Observability | Good | Prometheus metrics (TTFT, tok/s, queue depth, etc.), structured logging |
| Graceful lifecycle | Good | Health endpoint (503 during drain), SIGTERM handling, 30s shutdown |
| Quantization | Good | Q4, Q8, FP8 (Ada/Hopper), MXFP4, BF16 |
| Model breadth | Good | 10 architectures including MoE (Mixtral, Qwen3-MoE), hybrid (Nemotron), VLMs |
| Expert streaming | Unique | NVMe-backed on-demand expert loading for large MoE — none of the competitors have this |
| Vision | Good | SigLIP ViT, fused kernels, multimodal API support |
| Backpressure | Adequate | Bounded queue → 503 when full, per-request timeouts |

## Critical Gaps vs. Competitors

### 1. Decoding Features (High Impact)

| Feature | vLLM | SGLang | llama.cpp | LM Studio | rLLM |
|---------|------|--------|-----------|-----------|------|
| Speculative decoding | Yes | Yes | Yes | — | **No** |
| Beam search | Yes | — | Yes | — | **No** |
| Guided generation (grammar/JSON schema) | Yes (outlines) | Yes (grammar) | Yes (GBNF) | — | **No** |
| Logprobs / top_logprobs | Yes | Yes | Yes | Yes | **No** |
| Logit bias | Yes | Yes | Yes | — | **No** |
| Top-k sampling | Yes | Yes | Yes | Yes | **No** |
| Min-p sampling | — | — | Yes | Yes | **No** |
| Frequency/presence penalty | Yes | Yes | Yes | Yes | **No** |
| Repetition penalty | Yes | Yes | Yes | Yes | **No** |
| Best-of-n (n>1) | Yes | — | — | — | **No** |
| LoRA/adapter hot-swap | Yes | Yes | — | Yes | **No** |

**Speculative decoding** is likely the single highest-impact missing feature — it can 2-3x decode throughput for suitable draft models. **Structured output** (grammar-constrained generation) is increasingly table-stakes for agent/tool-use workloads. **Logprobs** are required by many evaluation frameworks and classification pipelines.

Note: the engine header comments reference top-k, min-p, and repetition penalty but the actual `sampler.rs` implementation only has temperature + top-p + greedy argmax.

### 2. Distributed Serving (High Impact at Scale)

| Feature | vLLM | SGLang | llama.cpp | rLLM |
|---------|------|--------|-----------|------|
| Tensor parallelism | Yes | Yes | — | **CUDA only** (Metal = single GPU) |
| Pipeline parallelism | Yes | Yes | — | **No** |
| Disaggregated prefill/decode | Experimental | Yes | — | **No** |
| Multi-instance coordination | Ray | — | — | **No** |
| CPU/disk swap for evicted KV | Yes | — | — | **No** |
| Distributed KV cache | — | Experimental | — | **No** |

rLLM is single-node only. For models that fit on one machine this is fine, but there's no path to horizontal scaling.

### 3. Operational & Deployment (Medium Impact)

| Feature | vLLM | SGLang | llama.cpp | LM Studio | rLLM |
|---------|------|--------|-----------|-----------|------|
| Docker images | Yes | Yes | Yes | N/A | **No** |
| CI/CD pipeline | Yes | Yes | Yes | Proprietary | **No** |
| Config file (YAML/TOML) | Yes | Yes | Yes | GUI | **No** (CLI args only) |
| Kubernetes/Helm | Yes | Yes | Community | N/A | **No** |
| Auto model download | Yes (HF) | Yes (HF) | Yes (HF/GGUF) | Yes | **No** |
| Pre-built packages | pip | pip | brew/releases | Installer | **No** |
| GPU health monitoring | Yes | Yes | — | Yes | **No** |
| GGUF format support | — | — | Yes | Yes | **No** |
| GPTQ/AWQ support | Yes | Yes | — | — | **No** |

### 4. API Feature Gaps (Medium Impact)

| Feature | Status |
|---------|--------|
| `n > 1` (multiple completions) | Not supported |
| `logprobs` / `top_logprobs` | Not supported |
| `logit_bias` | Not supported |
| `frequency_penalty` / `presence_penalty` | Not supported |
| `response_format: json_schema` (structured output) | Not supported (only prompt-based `json_object`) |
| `/v1/embeddings` | Not supported |
| `/v1/tokenize` / `/v1/detokenize` | Not supported |
| Rate limiting (per-user) | Not supported |
| Request idempotency keys | Not supported |

### 5. Testing & Quality (Medium Impact)

| Area | Status |
|------|--------|
| CI/CD | None — no automated test runs |
| Rust integration tests | None (only unit tests in source files) |
| Load/stress testing | No dedicated suite |
| Fuzz testing | None |
| Performance regression tracking | None |
| Code coverage | Not measured |
| Custom error types | Uses `anyhow` everywhere — no structured error codes |

### 6. Missing Production Infrastructure

| Area | Details |
|------|---------|
| Rate limiting | No per-user quotas, no token bucket, no burst control |
| Distributed tracing | No OpenTelemetry, no span propagation |
| Request deduplication | No idempotency keys |
| Response compression | No gzip/brotli |
| Config file | CLI args only, no YAML/TOML, limited env var support |
| Dynamic config reload | Only auth provider supports hot reload |
| A/B testing / canary | No traffic splitting, no shadow mode |
| Circuit breaker | Worker errors abort all in-flight sequences |

## Priority Recommendations

### Tier 1 — Blocking for Production Use

1. **CI/CD pipeline** — Without automated tests, regressions are invisible. GitHub Actions running `cargo test` + `cargo clippy` + Python integration tests is the minimum.
2. **Dockerfile** — Required for any cloud/k8s deployment. Multi-stage build (compile → slim runtime image).
3. **Sampling parity** — Implement top-k, min-p, repetition penalty, frequency/presence penalty. These are referenced in comments but not in the actual sampler.
4. **Logprobs support** — Many downstream tools (evals, classifiers, routers) depend on logprobs. This is a compatibility blocker.
5. **Structured output (grammar/JSON schema)** — Increasingly required for agent and tool-use workloads.

### Tier 2 — Important for Competitive Parity

6. **Speculative decoding** — 2-3x decode speedup for interactive workloads. vLLM, SGLang, and llama.cpp all have this.
7. **Logit bias** — OpenAI API parity. Trivial to implement (logit manipulation before sampling).
8. **Config file support** — CLI-only configuration doesn't scale for deployment automation.
9. **Rate limiting** — Without this, a single client can monopolize the server.
10. **Auto model download** — Manual model preparation is a friction point vs. `vllm serve model-name`.

### Tier 3 — Nice to Have / Differentiators

11. **LoRA hot-swap** — Important for multi-tenant serving.
12. **Pipeline parallelism** — For models too large for TP alone.
13. **OpenTelemetry integration** — Production observability beyond Prometheus scrape.
14. **Kubernetes Helm chart** — Standard deployment pattern.
15. **GGUF support** — Massive ecosystem of pre-quantized models on HuggingFace.

## Honest Assessment

rLLM is a **well-architected single-node inference engine** with some genuinely impressive features (TurboQuant KV cache, expert streaming, dual API compatibility, OIDC auth). The code quality and documentation are notably above average for the space.

However, it's closer to a **"production-capable single-machine server"** than a **"production-ready inference platform"**. The main gaps are:

- **No CI/CD or containers** — can't deploy with confidence in a team setting
- **Sampling gaps** — top-k, min-p, repetition penalty referenced but not implemented in sampler
- **Missing API features** (logprobs, structured output) — breaks compatibility with many downstream tools
- **No speculative decoding** — leaves significant throughput on the table
- **No horizontal scaling** — single node ceiling
- **No automated model acquisition** — manual setup friction

Compared to vLLM (the current industry standard for production serving), rLLM is missing roughly 60% of the feature surface. But much of what it *does* have is implemented cleanly, and the architecture is sound for incremental improvement. The Rust + Metal focus gives it a genuine niche (fast macOS inference) that vLLM and SGLang don't serve well.

### Where rLLM Wins

- **macOS/Metal native** — vLLM and SGLang are CUDA-only. llama.cpp supports Metal but via GGML, not native Metal shaders.
- **TurboQuant KV cache** — 4-bit online KV quantization on by default. Most competitors don't ship this.
- **Expert streaming** — NVMe-backed on-demand expert loading is unique in the space.
- **Dual API compatibility** — OpenAI + Anthropic format support out of the box.
- **Code quality** — Educational annotations, clean trait abstractions, well-documented design decisions.
- **Single binary** — No Python dependency chain, no pip install issues, no CUDA toolkit version conflicts.
