# rLLM

Minimal LLM inference engine written from scratch in Rust. Metal GPU backend, bf16 and Q4 quantization, multi-architecture support (Llama 3 + Qwen 2.5). Paged KV cache, batched prefill (GEMM), continuous batching. No frameworks, no GGML — just raw GPU compute.

## Performance

**Apple M4 Max** — 16-core CPU, 40-core GPU, 64 GB unified memory, 546 GB/s bandwidth.

### Generation (tok/s)

Generation is memory-bound (mat-vec): each token loads the full weight matrix for a single dot product per row. Q4 reduces data loaded ~3.2x, giving ~1.5x speedup.

| Model | Params | bf16 | Q4 |
|---|---|---|---|
| Llama 3.2 1B | 1.2B | 86 | 114 |
| Llama 3.2 3B | 3.2B | 33 | 46 |
| Qwen 2.5 3B Instruct | 3.1B | 26 | 33 |
| Qwen 2.5 7B Instruct | 7.6B | 20 | 32 |
| Llama 3.1 8B | 8.0B | 20 | 33 |
| Llama 3.1 8B Instruct | 8.0B | 18 | 29 |

### Prefill (tok/s)

Batched prefill uses GEMM (mat-mat) to process the entire prompt in one forward pass. The weight matrix is loaded once and reused across all prompt tokens — shifting from bandwidth-bound to compute-bound.

| Model | bf16 | Q4 | Prompt length |
|---|---|---|---|
| Llama 3.2 1B | 366 | 664 | 76 tokens |
| Llama 3.2 3B | 99 | 266 | 76 tokens |
| Qwen 2.5 3B Instruct | 105 | 268 | 96 tokens |
| Qwen 2.5 7B Instruct | 39 | 116 | 96 tokens |
| Llama 3.1 8B | 32 | 114 | 76 tokens |
| Llama 3.1 8B Instruct | 37 | 114 | 96 tokens |

Prefill throughput increases with prompt length (higher arithmetic intensity — more FLOPs per byte of weight data loaded). Q4 prefill is 3-4x faster than bf16 for larger models due to the combined effect of batching + quantization.

### Continuous batching

| Sequences | Tokens | Total throughput |
|---|---|---|
| 3 (Llama 3.2 1B, bf16) | 96 | 107 tok/s |

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
cargo run --release -- run --model models/llama-3.2-1b --batch-file test_prompts.txt --max-tokens 64
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

## Architecture

```
src/
├── main.rs        — CLI entry point (run / serve subcommands)
├── config.rs      — HuggingFace config.json parsing, ModelArch detection
├── model.rs       — Transformer forward pass (single-token + batched prefill)
├── engine.rs      — Continuous batching loop (scheduler + model)
├── scheduler.rs   — Sequence management, FCFS admission
├── kv_cache.rs    — Paged KV cache (block pool + per-sequence state)
├── loader.rs      — Safetensors loading, Q4 on-load quantization
├── tokenizer.rs   — BPE tokenizer with per-model special tokens
├── chat.rs        — Chat template formatter (Llama 3 + ChatML)
├── sampler.rs     — Temperature + top-p sampling
├── api/
│   ├── mod.rs     — HTTP server setup, inference worker thread
│   ├── openai.rs  — OpenAI-compatible handlers (chat/completions/models)
│   └── anthropic.rs — Anthropic-compatible handler (messages)
└── gpu/
    ├── mod.rs     — GpuBackend trait (platform-generic interface)
    ├── metal/     — Metal backend + compute shaders
    │   ├── mod.rs         — dispatch, pipeline management
    │   ├── matmul.metal   — SIMD-cooperative matvec + GEMM (bf16 + Q4)
    │   ├── attention.metal — flat, paged, and causal prefill attention
    │   ├── rms_norm.metal  — single + batched RMSNorm
    │   ├── rope.metal      — single + batched RoPE
    │   ├── embed.metal     — single + batched embedding lookup
    │   └── elementwise.metal — add, SwiGLU
    └── cuda/      — CUDA backend (stub)
```

Model code is generic over a `GpuBackend` trait with an associated `Tensor` type. Platform selection uses OS-conditional compilation — Metal on macOS, CUDA on Linux. Llama and Qwen share the same forward pass; the only difference is Qwen adds bias to Q/K/V attention projections (handled by a simple `if let Some(bias)` check).

### Inference pipeline

```
Single-sequence mode (rllm run):
  load config → create backend → load tokenizer → load weights → create model
  → create paged KV pool + prefill buffers
  → encode prompt → batched GEMM prefill → decode loop (stream tokens)

Batch mode (rllm run --batch-file):
  load config → create backend → load tokenizer → load weights → create model
  → create KV pool + scheduler + engine
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

Models are downloaded from [Hugging Face](https://huggingface.co) in safetensors format. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```
pip install huggingface-cli
```

Then download models into a `models/` directory:

```
# Llama 3 — base models (text completion)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
huggingface-cli download meta-llama/Llama-3.2-3B --local-dir models/llama-3.2-3b
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir models/llama-3.1-8b

# Llama 3 — instruct models (chat / instruction following)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b-instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3.2-3b-instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b-instruct

# Qwen 2.5 — instruct models
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen-2.5-3b-instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen-2.5-7b-instruct
```

> **Note:** Llama models are gated — you'll need to [accept the license](https://huggingface.co/meta-llama/Llama-3.2-1B) on Hugging Face and authenticate with `huggingface-cli login` before downloading. Qwen models are open access.

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
