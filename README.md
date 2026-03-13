# rLLM

Minimal LLM inference engine written from scratch in Rust. Metal GPU backend, bf16 and Q4 quantization, multi-architecture support (Llama 3 + Qwen 2.5). Paged KV cache, batched prefill (GEMM), continuous batching. No frameworks, no GGML — just raw GPU compute.

## Performance

**Apple M4 Max** — 16-core CPU, 40-core GPU, 64 GB unified memory, 546 GB/s bandwidth. bf16 precision. Measured via the `/v1/chat/completions` streaming endpoint, averaged over 3 runs.

| Model | Params | Decode | TTFT |
|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 90 tok/s | 45 ms |
| Llama 3.2 3B Instruct | 3.2B | 32 tok/s | 160 ms |
| Qwen 2.5 3B Instruct | 3.1B | 28 tok/s | 139 ms |
| Qwen 2.5 7B Instruct | 7.6B | 21 tok/s | 380 ms |
| Llama 3.1 8B Instruct | 8.0B | 19 tok/s | 453 ms |

Q4 quantization (`--quantize`) reduces memory ~3.2x and speeds up generation ~1.5x.

## Features

- **Multi-architecture** — Llama 3 and Qwen 2.5 from the same codebase
- **Metal GPU backend** — SIMD-cooperative matmul, async command buffer dispatch
- **Batched prefill** — GEMM-based prompt processing (3-10x faster than token-by-token)
- **Paged KV cache** — on-demand block allocation, shared across sequences
- **Continuous batching** — concurrent multi-sequence inference via engine/scheduler
- **Q4 quantization** — 4-bit block quantization on load (~3.2x memory reduction)
- **bf16 inference** — native half-precision on Apple Silicon
- **Safetensors loading** — single and multi-shard weight files
- **BPE tokenizer** — HuggingFace-compatible tokenizer.json
- **Chat templates** — Llama 3 and ChatML (Qwen) instruct formats with `--chat`
- **Temperature + top-p sampling** — configurable via `--temperature` and `--top-p`
- **Streaming output** — tokens printed as generated
- **API server** — OpenAI and Anthropic compatible HTTP endpoints with SSE streaming

## Usage

### CLI inference

```
cargo run --release -- run --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128
```

With Q4 quantization:

```
cargo run --release -- run --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128 --quantize
```

Chat mode (instruct models — auto-detects Llama 3 or ChatML template):

```
cargo run --release -- run --model models/llama-3.2-3b-instruct --prompt "Write a Python program that adds 4 numbers" --chat --temperature 0
cargo run --release -- run --model models/qwen-2.5-7b-instruct --prompt "Explain hash maps" --chat --temperature 0
```

Continuous batching (multiple prompts from a file):

```
cargo run --release -- batch --model models/llama-3.2-1b --batch-file test_prompts.txt --max-tokens 64
```

### API server

Start an OpenAI/Anthropic-compatible API server:

```
cargo run --release -- serve --model models/llama-3.2-1b-instruct --port 8080
```

The server exposes these endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/models` | GET | List available models |
| `/v1/messages` | POST | Anthropic messages |
| `/health` | GET | Health check |

All POST endpoints support `"stream": true` for Server-Sent Events (SSE) streaming.

**OpenAI chat completion:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

**OpenAI chat completion (streaming):**

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain hash maps"}],
    "max_tokens": 256,
    "stream": true
  }'
```

**OpenAI text completion:**

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

**Anthropic messages:**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "system": "You are a helpful math tutor."
  }'
```

**Anthropic messages (streaming):**

```bash
curl -N http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": true
  }'
```

The server works as a drop-in backend for any tool that speaks the OpenAI or Anthropic API — just change the base URL to `http://localhost:8080`.

### TLS

The server supports HTTPS via two modes:

**Manual certificates:**

```
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --cert /path/to/cert.pem --private-key /path/to/key.pem --port 443
```

**Let's Encrypt (automatic):**

```
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --letsencrypt --domain example.com --port 443
```

Let's Encrypt uses TLS-ALPN-01 challenge validation — no separate port 80 needed. Certificates are automatically provisioned and renewed, cached to `.rllm-certs/` by default (configurable via `--cert-cache-dir`). Optionally pass `--letsencrypt-email` for expiry notifications.

## Architecture

```
src/
├── main.rs              — CLI entry point (parse args, dispatch)
├── commands/
│   ├── mod.rs           — Command enum (Run | Batch | Serve)
│   ├── run.rs           — Single-prompt inference
│   ├── batch.rs         — Batched inference from file
│   └── serve.rs         — API server CLI args + launch
├── model/
│   ├── mod.rs           — Transformer forward pass (single-token + batched prefill)
│   ├── config.rs        — HuggingFace config.json parsing, ModelArch detection
│   ├── loader.rs        — Safetensors loading, Q4 on-load quantization
│   ├── tokenizer.rs     — BPE tokenizer with per-model special tokens
│   ├── chat.rs          — Chat template formatter (Llama 3 + ChatML)
│   ├── kv_cache.rs      — Paged KV cache (block pool + per-sequence state)
│   └── sampler.rs       — Temperature + top-p sampling
├── engine/
│   ├── mod.rs           — Continuous batching loop (scheduler + model)
│   └── scheduler.rs     — Sequence management, FCFS admission
├── api/
│   ├── mod.rs           — HTTP server setup, inference worker thread
│   ├── openai.rs        — OpenAI-compatible handlers (chat/completions/models)
│   ├── anthropic.rs     — Anthropic-compatible handler (messages)
│   └── tls.rs           — TLS support (manual certs + Let's Encrypt)
└── gpu/
    ├── mod.rs           — GpuBackend trait (platform-generic interface)
    ├── metal/           — Metal backend + compute shaders
    │   ├── mod.rs         — dispatch, pipeline management
    │   ├── matmul.metal   — SIMD-cooperative matvec + GEMM (bf16 + Q4)
    │   ├── attention.metal — flat, paged, and causal prefill attention
    │   ├── rms_norm.metal  — single + batched RMSNorm
    │   ├── rope.metal      — single + batched RoPE
    │   ├── embed.metal     — single + batched embedding lookup
    │   └── elementwise.metal — add, SwiGLU
    └── cuda/            — CUDA backend
```

Model code is generic over a `GpuBackend` trait with an associated `Tensor` type. Platform selection uses OS-conditional compilation — Metal on macOS, CUDA on Linux. Llama and Qwen share the same forward pass; the only difference is Qwen adds bias to Q/K/V attention projections (handled by a simple `if let Some(bias)` check).

### Inference pipeline

```
Single-sequence mode (rllm run):
  load model → create paged KV pool + prefill buffers
  → encode prompt → batched GEMM prefill → decode loop (stream tokens)

Batch mode (rllm batch):
  load model → create KV pool + scheduler + engine
  → submit all prompts → engine.step() loop until all sequences complete

Server mode (rllm serve):
  spawn worker thread (owns backend + model + KV pool)
  → start axum HTTP server on tokio runtime
  → per request: handler → channel → worker (prefill + decode) → stream tokens back
```

### Optimisation stack

| Layer | Technique | Speedup | Why |
|---|---|---|---|
| Matmul | SIMD-cooperative (32 threads/row) | ~2x | Coalesced reads, hardware `simd_sum` |
| Dispatch | Async command buffers | ~4x | Eliminates 144 GPU syncs per token |
| Weights | Q4 quantization | ~1.5x | 3.2x less memory bandwidth |
| Prefill | Batched GEMM | 3-10x | Weight reuse: load once, compute B times |
| KV cache | Paged allocation | — | On-demand blocks, no wasted memory |
| Batching | Continuous batching | ~Nx | N sequences share the GPU |

## Model Setup

Models are downloaded from [Hugging Face](https://huggingface.co) in safetensors format. Install the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```bash
# macOS / Linux
curl -LsSf https://hf.co/cli/install.sh | bash

# or via pip
pip install huggingface_hub
```

Then download models into a `models/` directory:

```bash
# Llama 3 — base models (text completion)
hf download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
hf download meta-llama/Llama-3.2-3B --local-dir models/llama-3.2-3b
hf download meta-llama/Llama-3.1-8B --local-dir models/llama-3.1-8b

# Llama 3 — instruct models (chat / instruction following)
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b-instruct
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3.2-3b-instruct
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b-instruct

# Qwen 2.5 — instruct models
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen-2.5-3b-instruct
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen-2.5-7b-instruct
```

> **Note:** Llama models are gated — you'll need to [accept the license](https://huggingface.co/meta-llama/Llama-3.2-1B) on Hugging Face and authenticate with `hf auth login` before downloading. Qwen models are open access.

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
