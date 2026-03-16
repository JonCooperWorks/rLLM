# rLLM

A learning project for understanding GPU programming and LLM inference from the ground up. Built from scratch in Rust with no frameworks or GGML — just raw GPU compute. The codebase is heavily annotated as an educational exercise, explaining *why* things work the way they do, not just *what* they do.

## Backends

| Platform | Backend | Status |
|---|---|---|
| macOS / Apple Silicon | **Metal** | Full support — SIMD-cooperative matmul, async command buffer dispatch |
| Linux / NVIDIA | **CUDA** | Shader ports of Metal kernels |

Platform selection uses OS-conditional compilation (`#[cfg(target_os)]`) — no feature flags. Model code is generic over a `GpuBackend` trait with an associated `Tensor` type, so the same forward pass runs on either backend.

## Supported Models

Llama 3, Qwen 2.5, Mistral, Mixtral 8x7B, Qwen3 MoE, Qwen3.5, Phi-4, Gemma 3, DeepSeek-R1-Distill — all from the same codebase with bf16 and Q4 quantization.

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
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | — | 12 tok/s | — | 5,400 ms |

Q4 quantization (`--quantize`) gives ~1.3-3.5x faster decode by reducing memory bandwidth. Mixtral requires Q4 (bf16 would need ~87 GB). Q4 is strongly recommended for models over ~8B params. Large models (Gemma 3 27B, Phi-4, Qwen3/3.5 MoE) run in bf16 but are slow because the weights consume most of the 64 GB unified memory. Dynamic KV cache sizing automatically adjusts block count based on available GPU memory.

</details>

<details>
<summary><b>NVIDIA H100 80GB HBM3</b> — 3.35 TB/s bandwidth</summary>

Measured via `rllm run`, single run, 128 max tokens.

| Model | Params | bf16 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 315 tok/s | 258 tok/s | 15 ms | 19 ms |
| Llama 3.2 3B Instruct | 3.2B | 150 tok/s | 110 tok/s | 29 ms | 44 ms |
| Qwen 2.5 3B Instruct | 3.1B | 127 tok/s | 101 tok/s | 25 ms | 35 ms |
| Gemma 3 4B Instruct | 4.3B | 94 tok/s | 73 tok/s | 20 ms | 46 ms |
| Qwen 2.5 7B Instruct | 7.6B | 97 tok/s | 55 tok/s | 43 ms | 89 ms |
| Mistral 7B Instruct | 7.2B | 102 tok/s | 57 tok/s | 55 ms | 111 ms |
| Llama 3.1 8B Instruct | 8.0B | 96 tok/s | 54 tok/s | 52 ms | 111 ms |
| Qwen3.5 9B | ~9B | 76 tok/s | 47 tok/s | 876 ms | 863 ms |
| Phi-4 | 14.7B | 59 tok/s | 30 tok/s | 78 ms | 171 ms |
| Gemma 3 27B Instruct | 27.4B | 32 tok/s | 15 tok/s | 116 ms | 327 ms |
| Qwen3.5 27B | ~27B | 29 tok/s | 16 tok/s | 2,100 ms | 2,300 ms |
| Qwen3 Coder 30B-A3B Instruct | 30.5B (3.3B active) | 49 tok/s | 56 tok/s | 339 ms | 274 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | 29 tok/s | 14 tok/s | 139 ms | 355 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 46 tok/s | 50 tok/s | 531 ms | 513 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | — | 32 tok/s | — | 939 ms |
| Llama 3.1 70B Instruct | 70.6B | — | 7 tok/s | — | 884 ms |
| Qwen 2.5 72B Instruct | 72.7B | — | 6.5 tok/s | — | 768 ms |

Mixtral, Llama 70B, and Qwen 72B exceed 80 GB in bf16. MoE models (Qwen3 Coder, Qwen3.5 35B-A3B) can be faster in Q4 than bf16 because expert weights dominate bandwidth. Qwen3.5 models have high TTFT due to DeltaNet linear attention initialization overhead.

</details>

## Features

- **Multi-architecture** — Llama 3, Qwen 2.5, Mistral, Mixtral 8x7B, Qwen3 MoE, Qwen3.5, Phi-4, and Gemma 3 from the same codebase
- **Metal + CUDA backends** — SIMD-cooperative matmul, async command buffer dispatch
- **Batched prefill** — GEMM-based prompt processing (3-10x faster than token-by-token)
- **Paged KV cache** — on-demand block allocation, shared across sequences
- **Continuous batching** — concurrent multi-sequence inference via engine/scheduler
- **Q4 quantization** — 4-bit block quantization on load (~3.2x memory reduction)
- **bf16 inference** — native half-precision compute
- **Mixture of Experts** — top-k expert routing with per-token dispatch (Mixtral, Qwen3 MoE)
- **API server** — OpenAI and Anthropic compatible HTTP endpoints with SSE streaming
- **Chat templates** — Llama 3, ChatML (Qwen), Mistral/Mixtral, Phi, and Gemma instruct formats
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

## Model Setup

Download models from [Hugging Face](https://huggingface.co) in safetensors format:

```bash
# Install the hf CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Download models
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b-instruct
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen-2.5-7b-instruct
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct
hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
hf download microsoft/phi-4 --local-dir models/phi-4
```

<details>
<summary>All supported model downloads</summary>

```bash
# Llama 3 — base models
hf download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
hf download meta-llama/Llama-3.2-3B --local-dir models/llama-3.2-3b
hf download meta-llama/Llama-3.1-8B --local-dir models/llama-3.1-8b

# Llama 3 — instruct models
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b-instruct
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3.2-3b-instruct
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b-instruct

# Mistral / Mixtral
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct
hf download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir models/mixtral-8x7b-instruct

# Qwen
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen-2.5-3b-instruct
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen-2.5-7b-instruct
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir models/qwen3-coder-30b-a3b-instruct

# Gemma 3
hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
hf download google/gemma-3-27b-it --local-dir models/gemma-3-27b-it

# Phi-4
hf download microsoft/phi-4 --local-dir models/phi-4

# DeepSeek R1 distilled
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir models/DeepSeek-R1-Distill-Qwen-32B
```

</details>

> **Note:** Llama, Gemma, and Mistral models are gated — accept the license on Hugging Face and run `hf auth login` before downloading. Qwen models are open access.

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
