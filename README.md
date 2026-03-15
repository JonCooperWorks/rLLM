# rLLM

Minimal LLM inference engine written from scratch in Rust. Metal GPU backend, bf16 and Q4 quantization, multi-architecture support (Llama 3, Qwen 2.5, Qwen3 MoE, Qwen3.5, Phi-4, Gemma 3, DeepSeek-R1-Distill). Paged KV cache, batched prefill (GEMM), continuous batching. No frameworks, no GGML — just raw GPU compute.

## Performance

**Apple M4 Max** — 16-core CPU, 40-core GPU, 64 GB unified memory, 546 GB/s bandwidth. Measured via CLI (`rllm run --chat`), single run.

| Model | Params | bf16 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 99 tok/s | 139 tok/s | 100 ms | 79 ms |
| Llama 3.2 3B Instruct | 3.2B | 37 tok/s | 51 tok/s | 322 ms | 253 ms |
| Qwen 2.5 3B Instruct | 3.1B | 31 tok/s | 45 tok/s | 242 ms | 98 ms |
| Qwen 2.5 7B Instruct | 7.6B | 23 tok/s | 39 tok/s | 662 ms | 240 ms |
| Llama 3.1 8B Instruct | 8.0B | 21 tok/s | 36 tok/s | 782 ms | 393 ms |
| Gemma 3 4B Instruct | 4.3B | 25 tok/s | 32 tok/s | 400 ms | 330 ms |
| Gemma 3 27B Instruct | 27.4B | 2 tok/s | 7 tok/s | 50,000 ms | 4,300 ms |
| Qwen3 Coder 30B-A3B Instruct | 30.5B (3.3B active) | 2 tok/s | 11 tok/s | 40,000 ms | 2,900 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 5 tok/s | 16 tok/s | 44,600 ms | 2,000 ms |
| Phi-4 | 14.7B | 2 tok/s | 15 tok/s | 5,300 ms | 813 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | — | 5 tok/s | — | 4,700 ms |

Q4 quantization (`--quantize`) gives ~1.3-3.5x faster decode by reducing memory bandwidth. The Qwen3 and Qwen3.5 MoE models use Mixture of Experts with sparse activation (only ~3B params active per token). Qwen3.5 also uses DeltaNet linear attention layers. Large models (Gemma 3 27B, Phi-4, Qwen3/3.5 MoE) run in bf16 but are slow because the weights consume most of the 64 GB unified memory, leaving limited headroom for KV cache. Q4 is strongly recommended for models over ~8B params. Dynamic KV cache sizing automatically adjusts block count based on available GPU memory.

## Features

- **Multi-architecture** — Llama 3, Qwen 2.5, Qwen3 MoE, Qwen3.5, Phi-4, and Gemma 3 from the same codebase
- **Metal GPU backend** — SIMD-cooperative matmul, async command buffer dispatch
- **Batched prefill** — GEMM-based prompt processing (3-10x faster than token-by-token)
- **Paged KV cache** — on-demand block allocation, shared across sequences
- **Continuous batching** — concurrent multi-sequence inference via engine/scheduler
- **Q4 quantization** — 4-bit block quantization on load (~3.2x memory reduction)
- **bf16 inference** — native half-precision on Apple Silicon
- **Safetensors loading** — single and multi-shard weight files
- **BPE tokenizer** — HuggingFace-compatible tokenizer.json
- **Mixture of Experts** — top-k expert routing with per-token dispatch (Qwen3 MoE)
- **Chat templates** — Llama 3, ChatML (Qwen), Phi, and Gemma instruct formats with `--chat`
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
cargo run --release -- run --model models/qwen3-coder-30b-a3b-instruct --prompt "Write a fibonacci function" --chat --quantize --temperature 0
```

Continuous batching (multiple prompts from a file):

```
cargo run --release -- batch --model models/llama-3.2-1b --batch-file test_prompts.txt --max-tokens 64
```

### API server

Start an OpenAI/Anthropic-compatible API server (TLS required by default — see [TLS](#tls) section):

```
cargo run --release -- serve --model models/llama-3.2-1b-instruct --port 8080 --dangerous-no-tls
```

**Server options:**

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

The server requires TLS by default. To run without TLS (e.g. local development), pass `--dangerous-no-tls`:

```
cargo run --release -- serve --model models/llama-3.2-1b-instruct --port 8080 --dangerous-no-tls
```

HTTPS is supported via two modes:

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
│   ├── chat.rs          — Chat template formatter (Llama 3 + ChatML + Phi)
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

Model code is generic over a `GpuBackend` trait with an associated `Tensor` type. Platform selection uses OS-conditional compilation — Metal on macOS, CUDA on Linux. Llama, Qwen, and Phi share the same forward pass primitives; differences are in weight loading (Phi uses fused qkv/gate_up tensors split on load) and QKV bias (Qwen 2.5 only). Gemma 3 has a dedicated forward pass with sandwich norms (4 RMSNorm per layer), GeGLU activation, sliding window attention, QK-norm, and dual RoPE bases.

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

# Qwen3 MoE — 30.5B total, 3.3B active per token (~61 GB download)
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir models/qwen3-coder-30b-a3b-instruct

# Gemma 3 — instruct models (sandwich norms, GeGLU, sliding window attention)
hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
hf download google/gemma-3-27b-it --local-dir models/gemma-3-27b-it

# Phi-4 — 14.7B dense (~28 GB download)
hf download microsoft/phi-4 --local-dir models/phi-4

# DeepSeek R1 distilled — 32.8B dense (~61 GB download)
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir models/DeepSeek-R1-Distill-Qwen-32B
```

> **Note:** Llama and Gemma models are gated — you'll need to accept the license on Hugging Face and authenticate with `hf auth login` before downloading. Qwen models are open access (Apache-2.0).

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
