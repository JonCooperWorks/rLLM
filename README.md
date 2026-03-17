# rLLM

A learning project for understanding GPU programming and LLM inference from the ground up. Built from scratch in Rust with no frameworks or GGML — just raw GPU compute. The codebase is heavily annotated as an educational exercise, explaining *why* things work the way they do, not just *what* they do.

## Backends

| Platform | Backend | Status |
|---|---|---|
| macOS / Apple Silicon | **Metal** | Full support — SIMD-cooperative matmul, async command buffer dispatch |
| Linux / NVIDIA | **CUDA** | Shader ports of Metal kernels |

Platform selection uses OS-conditional compilation (`#[cfg(target_os)]`) — no feature flags. Model code is generic over a `GpuBackend` trait with an associated `Tensor` type, so the same forward pass runs on either backend.

## Supported Models

Llama 3, Qwen 2.5, Mistral, Mixtral 8x7B, Qwen3 MoE, Qwen3.5, Phi-4, Gemma 3, DeepSeek-R1-Distill, GPT-OSS-20B — all from the same codebase with bf16 and Q4 quantization.

## Benchmarks

<details>
<summary><b>Apple M4 Max</b> — 16-core CPU, 40-core GPU, 64 GB unified, 546 GB/s</summary>

Measured via `rllm run --chat`, single run.

| Model | Params | bf16 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 119 tok/s | 183 tok/s | 104 ms | 79 ms |
| Llama 3.2 3B Instruct | 3.2B | 56 tok/s | 100 tok/s | 348 ms | 264 ms |
| Qwen 2.5 3B Instruct | 3.1B | 53 tok/s | 91 tok/s | 127 ms | 84 ms |
| Qwen 2.5 7B Instruct | 7.6B | 28 tok/s | 63 tok/s | 356 ms | 166 ms |
| Mistral 7B Instruct | 7.2B | 30 tok/s | 75 tok/s | 535 ms | 311 ms |
| Llama 3.1 8B Instruct | 8.0B | 27 tok/s | 59 tok/s | 474 ms | 324 ms |
| Gemma 3 4B Instruct | 4.3B | 40 tok/s | 62 tok/s | 361 ms | 324 ms |
| Phi-4 | 14.7B | 6 tok/s | 37 tok/s | 5,300 ms | 638 ms |
| Gemma 3 27B Instruct | 27.4B | 2 tok/s | 7 tok/s | 50,000 ms | 4,300 ms |
| Qwen3 Coder 30B-A3B Instruct | 30.5B (3.3B active) | 2 tok/s | 11 tok/s | 40,000 ms | 2,900 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | — | 5 tok/s | — | 4,700 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 5 tok/s | 16 tok/s | 44,600 ms | 2,000 ms |
| GPT-OSS 20B | 20.0B (3.6B active) | 6 tok/s | 34 tok/s | 4,800 ms | 425 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | — | 12 tok/s | — | 5,400 ms |

Q4 quantization (`--quantize`) gives ~1.3-3.5x faster decode by reducing memory bandwidth. Mixtral requires Q4 (bf16 would need ~87 GB). Q4 is strongly recommended for models over ~8B params. Large models (Gemma 3 27B, Phi-4, Qwen3/3.5 MoE) run in bf16 but are slow because the weights consume most of the 64 GB unified memory. Dynamic KV cache sizing automatically adjusts block count based on available GPU memory.

</details>

<details>
<summary><b>NVIDIA H100 NVL 94 GB HBM3</b> — 3.35 TB/s bandwidth</summary>

Benchmarked on [Vast.ai](https://cloud.vast.ai/?ref_id=394548). Measured via `rllm run`, single run, 128 max tokens.

| Model | Params | bf16 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 326 tok/s | 253 tok/s | 14 ms | 9 ms |
| Llama 3.2 3B Instruct | 3.2B | 149 tok/s | 116 tok/s | 27 ms | 22 ms |
| Qwen 2.5 3B Instruct | 3.1B | 125 tok/s | 99 tok/s | 23 ms | 19 ms |
| Gemma 3 4B Instruct | 4.3B | 93 tok/s | 76 tok/s | 20 ms | 25 ms |
| Qwen 2.5 7B Instruct | 7.6B | 97 tok/s | 74 tok/s | 33 ms | 39 ms |
| Mistral 7B Instruct | 7.2B | 100 tok/s | 74 tok/s | 53 ms | 49 ms |
| Llama 3.1 8B Instruct | 8.0B | 94 tok/s | 71 tok/s | 58 ms | 49 ms |
| Qwen3.5 9B | ~9B | 77 tok/s | 60 tok/s | 236 ms | 250 ms |
| Phi-4 | 14.7B | 58 tok/s | 44 tok/s | 78 ms | 76 ms |
| Gemma 3 27B Instruct | 27.4B | 31 tok/s | 24 tok/s | 143 ms | 160 ms |
| Qwen3.5 27B | ~27B | 28 tok/s | 22 tok/s | 625 ms | 1,800 ms |
| Qwen3 Coder 30B-A3B Instruct | 30.5B (3.3B active) | 49 tok/s | 54 tok/s | 158 ms | 157 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | 28 tok/s | 21 tok/s | 166 ms | 165 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 46 tok/s | 38 tok/s | 221 ms | 244 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | 61 tok/s | 45 tok/s | 254 ms | 710 ms |
| Llama 3.1 70B Instruct | 70.6B | — | 11 tok/s | — | 427 ms |
| Qwen 2.5 72B Instruct | 72.7B | — | 11 tok/s | — | 379 ms |
| Qwen3.5 122B-A10B | ~122B (~10B active) | — | 22 tok/s | — | 864 ms |

Q4 is slower than bf16 for decode on H100 — unlike Apple Silicon where Q4 is always faster. The H100's 3.35 TB/s HBM3 bandwidth is so high that bf16 matvec already finishes quickly, and Q4 dequantisation adds ~5 ALU ops per weight (mask, shift, int subtract, int-to-float, scale multiply) that eat into the 3.2x bandwidth savings. For models up to ~14B, the weight working set fits largely in the 50 MB L2 cache, shrinking the bandwidth advantage further. Q4 still wins on memory capacity (Llama 70B and Qwen 72B don't fit in 94 GB as bf16) and on TTFT where it halves prefill data movement. On Apple Silicon (546 GB/s), the system is 6x more memory-bound so Q4's bandwidth reduction dominates and Q4 is faster across the board. MoE models (Qwen3 Coder, Mixtral) are a middle ground — the small expert matrices are cache-friendly in both formats, but Q4 lets larger MoE models fit in VRAM.

</details>

## Features

- **Multi-architecture** — Llama 3, Qwen 2.5, Mistral, Mixtral 8x7B, Qwen3 MoE, Qwen3.5, Phi-4, Gemma 3, and GPT-OSS-20B from the same codebase
- **Metal + CUDA backends** — SIMD-cooperative matmul, async command buffer dispatch
- **Batched prefill** — GEMM-based prompt processing (3-10x faster than token-by-token)
- **Paged KV cache** — on-demand block allocation, shared across sequences
- **Continuous batching** — concurrent multi-sequence inference via engine/scheduler
- **Q4 quantization** — 4-bit block quantization on load (~3.2x memory reduction)
- **bf16 inference** — native half-precision compute
- **Mixture of Experts** — top-k expert routing with per-token dispatch (Mixtral, Qwen3 MoE, GPT-OSS)
- **MXFP4 dequantization** — microscaling FP4 (E2M1 + E8M0 scales) weight loading for GPT-OSS
- **API server** — OpenAI and Anthropic compatible HTTP endpoints with SSE streaming
- **Chat templates** — Llama 3, ChatML (Qwen/GPT-OSS), Mistral/Mixtral, Phi, and Gemma instruct formats
- **Temperature + top-p sampling** — configurable via `--temperature` and `--top-p`

## Usage

### CLI

```bash
# Text completion
cargo run --release -- run --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128

# With Q4 quantization
cargo run --release -- run --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128 --quantize

# Chat mode (auto-detects template per architecture)
cargo run --release -- run --model models/llama-3.2-3b-instruct --prompt "Write a fibonacci function" --chat --temperature 0

# Continuous batching (multiple prompts from a file)
cargo run --release -- batch --model models/llama-3.2-1b --batch-file test_prompts.txt --max-tokens 64
```

### API Server

Start an OpenAI/Anthropic-compatible server:

```bash
cargo run --release -- serve --model models/llama-3.2-1b-instruct --port 8080 --dangerous-no-tls
```

Works as a drop-in backend for any tool that speaks the OpenAI or Anthropic API — just point it at `http://localhost:8080`.

<details>
<summary>Server options</summary>

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to model directory |
| `--port` | `8080` | Port to listen on |
| `--host` | `127.0.0.1` | Host to bind to (use `0.0.0.0` for all interfaces) |
| `--quantize` | off | Quantize weights to Q4 on load |
| `--cert` / `--private-key` | — | TLS certificate and key (PEM) |
| `--letsencrypt` | off | Automatic TLS via Let's Encrypt (requires `--domain`) |
| `--domain` | — | Domain name for Let's Encrypt |
| `--letsencrypt-email` | — | Contact email for Let's Encrypt |
| `--cert-cache-dir` | `.rllm-certs` | Cache directory for Let's Encrypt certs |
| `--dangerous-no-tls` | off | Allow serving over plain HTTP |

</details>

<details>
<summary>Endpoints</summary>

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/models` | GET | List available models |
| `/v1/messages` | POST | Anthropic messages |
| `/health` | GET | Health check |

All POST endpoints support `"stream": true` for SSE streaming.

</details>

<details>
<summary>curl examples</summary>

```bash
# OpenAI chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "temperature": 0
  }'

# OpenAI chat completion (streaming)
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain hash maps"}],
    "max_tokens": 256,
    "stream": true
  }'

# OpenAI text completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Anthropic messages
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "system": "You are a helpful math tutor."
  }'
```

</details>

<details>
<summary>TLS</summary>

The server requires TLS by default. Use `--dangerous-no-tls` for local development.

**Manual certificates:**

```bash
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --cert /path/to/cert.pem --private-key /path/to/key.pem --port 443
```

**Let's Encrypt (automatic):**

```bash
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --letsencrypt --domain example.com --port 443
```

Uses TLS-ALPN-01 challenge validation — no separate port 80 needed. Certificates are automatically provisioned and renewed, cached to `.rllm-certs/` by default.

</details>

## Architecture

```
src/
├── main.rs              — CLI entry point
├── commands/            — run (single prompt), batch (multi-prompt), serve (API server)
├── model/
│   ├── mod.rs           — Transformer forward pass (single-token + batched prefill)
│   ├── config.rs        — HuggingFace config.json parsing, ModelArch detection
│   ├── loader.rs        — Safetensors loading, Q4 on-load quantization
│   ├── tokenizer.rs     — BPE tokenizer with per-model special tokens
│   ├── chat.rs          — Chat template formatter
│   ├── kv_cache.rs      — Paged KV cache (block pool + per-sequence state)
│   └── sampler.rs       — Temperature + top-p sampling
├── engine/              — Continuous batching loop + FCFS scheduler
├── api/                 — axum HTTP server, OpenAI + Anthropic handlers, TLS
└── gpu/
    ├── mod.rs           — GpuBackend trait (platform-generic interface)
    ├── metal/           — Metal backend + .metal compute shaders
    └── cuda/            — CUDA backend + .cu shader ports
```

### Optimisation Stack

| Layer | Technique | Speedup |
|---|---|---|
| Matmul | SIMD-cooperative (32 threads/row, hardware `simd_sum`) | ~2x |
| Dispatch | Async command buffers (eliminates GPU syncs per token) | ~4x |
| Weights | Q4 quantization (3.2x less memory bandwidth) | ~1.5x |
| Prefill | Batched GEMM (load weights once, compute B times) | 3-10x |
| KV cache | Paged allocation (on-demand blocks, no waste) | — |
| Attention | Fused single-pass softmax+V, head_dim-specialised pipelines | 1.3-2.8x |
| Batching | Continuous batching (N sequences share the GPU) | ~Nx |

## Scripts

The scripts make it easy to get running quickly on a rented GPU — install Rust, pull models, and benchmark in a few commands. [Vast.ai](https://cloud.vast.ai/?ref_id=394548) is a good option for cheap H100 access.

| Script | Description |
|---|---|
| `scripts/install-rust.sh` | Installs Rust via rustup |
| `scripts/download-models.sh` | Downloads all model weights from HuggingFace (handles auth, filters by tier) |
| `scripts/benchmark.sh` | Runs each downloaded model in bf16 and Q4, outputs a Markdown results table |

```bash
# Quick setup on a fresh machine
scripts/install-rust.sh
scripts/download-models.sh --small
cargo run --release -- run --model models/llama-3.2-1b-instruct --prompt "Hello" --chat
```

### Download models

```bash
# Small tier — 1B–8B models (~100 GB)
scripts/download-models.sh --small

# Medium tier (default) — all models up to 35B (~500 GB)
scripts/download-models.sh

# Big tier — adds 70B+ models (~1 TB+)
scripts/download-models.sh --big
```

Gated models (Llama, Gemma, Mistral) require a HuggingFace token — set `HF_TOKEN` or run `hf auth login` first. Qwen models are open access.

### Benchmark

```bash
# Benchmark small models only
scripts/benchmark.sh --small

# Q4 only (for models that don't fit in VRAM as bf16)
scripts/benchmark.sh --q4-only

# Multiple runs for more stable numbers
scripts/benchmark.sh --runs 3
```

Auto-detects the GPU (Apple Silicon or NVIDIA), runs each model in bf16 and Q4, and prints a Markdown table with tok/s and TTFT.

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
