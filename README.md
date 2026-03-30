# rLLM

**This is a learning project.** I'm an appsec engineer who wanted to understand how LLM inference actually works — the GPU kernels, the memory management, the scheduling — so I built one from scratch in Rust with Claude as a teaching partner. It is not rated for production use. Every file is annotated with architectural rationale explaining *why* things work the way they do. It builds on ideas from [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://github.com/ollama/ollama), [flash-moe](https://github.com/danveloper/flash-moe), and [rvLLM](https://github.com/m0at/rvllm) — reimplemented from scratch so I can understand how they work. See [docs/framework-comparison.md](docs/framework-comparison.md) for a detailed comparison.

Runs models up to 397B parameters on a MacBook via NVMe expert streaming. 13 architectures, two GPU backends (Metal + CUDA), continuous batching, paged KV cache, and an OpenAI/Anthropic-compatible API server.

### Highlights

- **Expert streaming** — stream hundreds of GB of expert weights from NVMe on demand with GPU-side LRU caching, on both Metal and CUDA ([docs](docs/expert-streaming.md))
- **13 architectures** — Llama 3.x, Qwen 2.5/3/3.5, Mistral, Mixtral, Gemma 3, Phi-4, DeepSeek-R1-Distill, GPT-OSS, Nemotron-H
- **Vision-language models** — SigLIP ViT encoder for Qwen 3.5 VL and Gemma 3 VLMs ([docs](docs/vision.md))
- **Three quantization formats** — Q4 block, Q8 block, and FP8 E4M3 (auto-selected on NVIDIA Ada/Hopper) ([docs](docs/quantization.md), [FP8 docs](docs/fp8.md))
- **Multi-GPU tensor parallelism** — NCCL-based weight sharding + expert parallelism for MoE models ([docs](docs/multi-gpu-moe.md))
- **Hand-written Metal and CUDA shaders** — SIMD-cooperative matmul, WMMA tensor-core GEMM, fused attention, DeltaNet linear attention

## Backends

| Platform | Backend | Status |
|---|---|---|
| macOS / Apple Silicon | **Metal** | Full support — SIMD-cooperative matmul, async dispatch |
| Linux / NVIDIA | **CUDA** | Shader ports of Metal kernels |

Platform selection uses OS-conditional compilation (`#[cfg(target_os)]`). Model code is generic over a `GpuBackend` trait with an associated `Tensor` type.

## Supported Models

| Model | Type | Multi-GPU |
|---|---|---|
| Llama 3.x | Dense | Yes |
| Mistral 7B | Dense | Yes |
| Gemma 3 | Dense + VLM | Yes |
| Qwen 2.5 | Dense | No |
| Qwen 3.5 | Hybrid (DeltaNet + GQA) | No |
| Qwen 3.5 VL | Vision-Language | No |
| Qwen 3.5 MoE | MoE (hybrid DeltaNet) | No |
| Qwen3 MoE | MoE | No |
| Phi-4 | Dense | No |
| DeepSeek-R1-Distill | Dense | No |
| Mixtral 8x7B | MoE | No |
| GPT-OSS | MoE | No |
| Nemotron-H | Hybrid (Mamba-2 + MoE + GQA) | No |

All models support bf16 and Q4. Multi-GPU via `--tp N` requires CUDA + NCCL.

## Benchmarks

<details>
<summary><b>Apple M4 Max</b> — 40-core GPU, 64 GB unified, 546 GB/s (March 30, 2026)</summary>

| Model | Params | bf16 | Q8 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 114 tok/s | 143 tok/s | 169 tok/s | 69 ms | 46 ms |
| Llama 3.2 3B Instruct | 3.2B | 44 tok/s | 58 tok/s | 71 tok/s | 194 ms | 97 ms |
| Qwen 2.5 3B Instruct | 3.1B | 55 tok/s | 77 tok/s | 101 tok/s | 186 ms | 122 ms |
| Gemma 3 4B Instruct | 4.3B | 27 tok/s | 32 tok/s | 37 tok/s | 244 ms | 131 ms |
| Qwen 2.5 7B Instruct | 7.6B | 27 tok/s | 48 tok/s | 74 tok/s | 479 ms | 167 ms |
| Mistral 7B Instruct | 7.2B | 24 tok/s | 37 tok/s | 52 tok/s | 351 ms | 136 ms |
| Llama 3.1 8B Instruct | 8.0B | 23 tok/s | 35 tok/s | 50 tok/s | 468 ms | 167 ms |
| Qwen3.5 9B † | ~9B | 20 tok/s | 31 tok/s | 43 tok/s | 6,484 ms | 3,022 ms |
| Phi-4 | 14.7B | 10 tok/s | 22 tok/s | 32 tok/s | 4,971 ms | 285 ms |
| GPT-OSS 20B | 20.0B (3.6B active) | 18 tok/s | 20 tok/s | 19 tok/s | 6,404 ms | 4,903 ms |
| Nemotron-H 30B ⚡ | 31.6B (3.6B active) | 7.7 tok/s | 33 tok/s | 39 tok/s | 834 ms | 180 ms |
| Gemma 3 27B Instruct | 27.4B | 1.9 tok/s | 8.3 tok/s | 15 tok/s | 50,900 ms | 590 ms |
| Mixtral 8x7B Instruct ⚡ | 46.7B (12.9B active) | 0.4 tok/s | — | 23 tok/s | 20,853 ms | 696 ms |
| Qwen3.5 27B ⚡† | ~27B | 2.0 tok/s | 9.8 tok/s | 16 tok/s | 47,200 ms | 8,194 ms |
| Qwen3.5 122B-A10B ⚡ | 122B (10B active) | — | — | — | 111,078 ms | 21,634 ms |
| Qwen3.5 397B-A27B ⚡ | 397B (17B active) | — | — | — | — | 46,338 ms |

⚡ = SSD expert streaming (`--stream-experts`). † = thinking model (TTFT includes reasoning time). The 397B model (751 GB on disk, 213 GB Q4) runs on 64 GB using ~20 GB GPU memory. Q4 is strongly recommended for models over ~8B. Benchmarked via `tests/bench.py` (HTTP API, 2 runs averaged); bf16 numbers for ⚡ models from `scripts/benchmark.sh` (CLI).

</details>

<details>
<summary><b>NVIDIA RTX 5090 32 GB</b> — 1.79 TB/s GDDR7X (March 29, 2026)</summary>

Benchmarked on [Vast.ai](https://cloud.vast.ai/?ref_id=394548). Q8 uses FP8 E4M3 (native on SM 12.0).

| Model | Params | bf16 | Q8 (FP8) | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|---|
| Llama 3.2 3B Instruct | 3.2B | 108 tok/s | 97 tok/s | 129 tok/s | 100 ms | 102 ms |
| Qwen 2.5 3B Instruct | 3.1B | 122 tok/s | 100 tok/s | 137 tok/s | 109 ms | 127 ms |
| Gemma 3 4B Instruct | 4.3B | 60 tok/s | 58 tok/s | 68 tok/s | 119 ms | 121 ms |
| Qwen 2.5 7B Instruct | 7.6B | 77 tok/s | 77 tok/s | 110 tok/s | 152 ms | 149 ms |
| Mistral 7B Instruct | 7.2B | 67 tok/s | 67 tok/s | 89 tok/s | 158 ms | 162 ms |
| Llama 3.1 8B Instruct | 8.0B | 65 tok/s | 65 tok/s | 86 tok/s | 164 ms | 165 ms |
| Qwen3.5 9B † | ~9B | 55 tok/s | 62 tok/s | 76 tok/s | 2,323 ms | 1,679 ms |
| Phi-4 | 14.7B | 39 tok/s | 40 tok/s | 57 tok/s | 254 ms | 253 ms |
| Nemotron-H 30B | 31.6B (3.6B active) | — | — | 104 tok/s | — | 148 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | — | — | 31 tok/s | — | 597 ms |
| Gemma 3 27B Instruct | 27.4B | — | — | 30 tok/s | — | 465 ms |
| Qwen3.5 27B † | ~27B | — | 22 tok/s | 29 tok/s | — | 4,353 ms |
| Qwen3.5 35B-A3B † | 35.1B (3.3B active) | 3.5 ⚡ | — | 58 tok/s | 38,442 ms | 2,219 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | 0.4 ⚡ | — | 59 tok/s | 21,756 ms | 251 ms |

⚡ = SSD expert streaming (`--stream-experts`). † = thinking model (TTFT includes reasoning time). Q8 column uses FP8 E4M3 format (auto-selected on NVIDIA SM 89+). Q4 MoE models fit entirely in VRAM: Qwen3.5 35B-A3B Q4 at 22 GB, Mixtral Q4 at 25 GB, Nemotron-H Q4 at 19 GB. Benchmarked via `tests/bench.py` (HTTP API, 2 runs averaged).

</details>

<details>
<summary><b>NVIDIA RTX 4080 SUPER 32 GB</b> — 736 GB/s GDDR6X (March 28, 2026)</summary>

Benchmarked on [Vast.ai](https://cloud.vast.ai/?ref_id=394548).

| Model | Params | bf16 | Q8 | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | 205 tok/s | 238 tok/s | 273 tok/s | 54 ms | 70 ms |
| Llama 3.2 3B Instruct | 3.2B | 81 tok/s | 95 tok/s | 120 tok/s | 111 ms | 145 ms |
| Qwen 2.5 3B Instruct | 3.1B | 88 tok/s | 106 tok/s | — | 130 ms | — |
| Gemma 3 4B Instruct | 4.3B | 53 tok/s | 64 tok/s | 70 tok/s | 119 ms | 154 ms |
| Qwen 2.5 7B Instruct | 7.6B | 45 tok/s | 54 tok/s | 81 tok/s | 187 ms | 225 ms |
| Mistral 7B Instruct | 7.2B | 40 tok/s | 48 tok/s | 69 tok/s | 186 ms | 234 ms |
| Llama 3.1 8B Instruct | 8.0B | 38 tok/s | 46 tok/s | 67 tok/s | 188 ms | 233 ms |
| Qwen3.5 9B | ~9B | 36 tok/s | 46 tok/s | 62 tok/s | 664 ms | 482 ms |
| Phi-4 | 14.7B | 22 tok/s | 26 tok/s | 39 tok/s | 313 ms | 375 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | — | — | 19 tok/s | — | 869 ms |
| Gemma 3 27B Instruct | 27.4B | — | — | 19 tok/s | — | 760 ms |
| Qwen3.5 27B | ~27B | — | — | 21 tok/s | — | 1,538 ms |
| Nemotron-H 30B | 31.6B (3.6B active) | — | — | 113 tok/s | — | 326 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 3.6 ⚡ | 6.2 ⚡ | 66 tok/s | 4,014 ms | 446 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | 0.3 ⚡ | 0.6 ⚡ | 41 tok/s | 33,358 ms | 468 ms |

⚡ = SSD expert streaming (`--stream-experts`) — bf16 and Q8 variants exceed 32 GB VRAM. Q4 MoE models fit entirely in VRAM without streaming: Qwen3.5 35B-A3B Q4 at 21 GB and Mixtral Q4 at 25 GB both load fully, delivering 18–110× speedup over their streamed bf16 counterparts. Nemotron-H 30B Q4 also fits without streaming.

</details>

<details>
<summary><b>NVIDIA H100 NVL 94 GB</b> — 3.35 TB/s HBM3 (March 22, 2026 — older build)</summary>

Benchmarked on [Vast.ai](https://cloud.vast.ai/?ref_id=394548). Numbers from an earlier build — expect improvement on current code.

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
| Qwen3 Coder 30B-A3B | 30.5B (3.3B active) | 49 tok/s | 54 tok/s | 158 ms | 157 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | 28 tok/s | 21 tok/s | 166 ms | 165 ms |
| Qwen3.5 35B-A3B | 35.1B (3.3B active) | 46 tok/s | 38 tok/s | 221 ms | 244 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | 61 tok/s | 45 tok/s | 254 ms | 710 ms |
| Llama 3.1 70B Instruct | 70.6B | — | 11 tok/s | — | 427 ms |
| Qwen 2.5 72B Instruct | 72.7B | — | 11 tok/s | — | 379 ms |
| Qwen3.5 122B-A10B | ~122B (~10B active) | — | 22 tok/s | — | 864 ms |

Q4 is slower than bf16 on H100 — the 3.35 TB/s HBM3 bandwidth means bf16 matvec already finishes quickly, and Q4 dequantisation overhead eats into the bandwidth savings. Q4 still wins on memory capacity (70B+ models don't fit as bf16) and on TTFT.

</details>

<details>
<summary><b>2× NVIDIA RTX 4090 (TP=2)</b> — 2× 1.01 TB/s GDDR6X, 48 GB total (March 29, 2026)</summary>

Benchmarked on [RunPod](https://runpod.io?ref=249k2lel). Tensor parallelism across 2 GPUs. MoE models use hybrid strategy (TP for attention, expert parallelism for MoE FFN). Q8 uses FP8 E4M3 (auto-selected on SM 89+).

| Model | Params | bf16 | Q8 (FP8) | Q4 | TTFT (bf16) | TTFT (Q4) |
|---|---|---|---|---|---|---|
| Llama 3.2 3B Instruct | 3.2B | 122.5 tok/s | 121.3 tok/s | 138.9 tok/s | 172 ms | 204 ms |
| Qwen 2.5 3B Instruct | 3.1B | 113.1 tok/s | — | 124.3 tok/s | 171 ms | 215 ms |
| Gemma 3 4B Instruct | 4.3B | 88.1 tok/s | 87.4 tok/s | 98.0 tok/s | 184 ms | 210 ms |
| Qwen 2.5 7B Instruct | 7.6B | 79.8 tok/s | — | 108.9 tok/s | 208 ms | 264 ms |
| Mistral 7B Instruct | 7.2B | 81.0 tok/s | 78.0 tok/s | 111.0 tok/s | 205 ms | 268 ms |
| Llama 3.1 8B Instruct | 8.0B | 74.1 tok/s | 73.1 tok/s | 102.6 tok/s | 221 ms | 276 ms |
| Qwen3.5 9B † | ~9B | 47.1 tok/s | — | 62.3 tok/s | 2,720 ms | 2,053 ms |
| Phi-4 | 14.7B | — | — | 68.4 tok/s | — | 344 ms |
| GPT-OSS 20B | 20.0B (3.6B active) | 73.6 tok/s | 74.9 tok/s | 79.0 tok/s | 504 ms | 519 ms |
| Qwen3.5 27B † | ~27B | — | — | 24.7 tok/s | — | 5,175 ms |
| DeepSeek-R1-Distill-Qwen-32B | 32.8B | — | — | 35.4 tok/s | — | 638 ms |
| Gemma 3 27B Instruct | 27.4B | — | 30.2 tok/s | 33.6 tok/s | — | 573 ms |
| Qwen3.5 35B-A3B † | 35.1B (3.3B active) | — | — | 57.4 tok/s | — | 2,230 ms |
| Mixtral 8x7B Instruct | 46.7B (12.9B active) | — | — | 45.4 tok/s | — | 1,001 ms |
| Qwen 2.5 72B Instruct | 72.7B | — | — | 18.5 tok/s | — | 1,010 ms |
| Llama 3.1 70B Instruct | 70.6B | — | — | 19.6 tok/s | — | 926 ms |

† = thinking model (TTFT includes reasoning time). MoE models (GPT-OSS, Mixtral, Qwen3.5 35B-A3B) use expert parallelism (experts split across GPUs). Llama 70B and Qwen 72B fit as Q4 (~40 GB). Benchmarked via `tests/bench.py` (HTTP API).

</details>


## Usage

### CLI

```bash
# Text completion
cargo run --release -- run --model models/llama-3.2-1b --prompt "The meaning of life is"

# Chat mode (auto-detects template per architecture)
cargo run --release -- run --model models/llama-3.2-3b-instruct --prompt "Write a fibonacci function" --chat --temperature 0

# Pre-quantize to Q4
cargo run --release -- quantize --model models/llama-3.2-1b

# SSD expert streaming (MoE models larger than GPU memory)
cargo run --release -- run --model models/qwen3.5-35b-a3b --prompt "Hello" --chat --stream-experts

# Vision model
cargo run --release -- run --model models/qwen3.5-vl-3b-instruct --prompt "What's in this image?" --chat --image photo.jpg

# Multi-GPU tensor parallelism
cargo run --release -- run --model models/llama-3.1-70b-instruct --prompt "Hello" --chat --tp 2

# Batched inference from file
cargo run --release -- batch --model models/llama-3.2-1b --batch-file prompts.txt
```

### API Server

```bash
cargo run --release -- serve --model models/llama-3.2-1b-instruct --port 8080 --dangerous-no-tls

# Multi-GPU
cargo run --release -- serve --model models/llama-3.1-70b-instruct --tp 2 --port 8080 --dangerous-no-tls
```

Drop-in backend for any tool that speaks the OpenAI or Anthropic API — point it at `http://localhost:8080`.

<details>
<summary>Endpoints & curl examples</summary>

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/models` | GET | List available models |
| `/v1/messages` | POST | Anthropic messages |
| `/health` | GET | Health check |

All POST endpoints support `"stream": true` for SSE streaming.

```bash
# OpenAI chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 64, "temperature": 0}'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain hash maps"}], "max_tokens": 256, "stream": true}'

# Anthropic messages
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 64, "system": "You are a helpful math tutor."}'

# Vision (OpenAI multimodal)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 < photo.jpg)"'"}},
    {"type": "text", "text": "Describe this image."}
  ]}], "max_tokens": 256}'
```

</details>

<details>
<summary>TLS options</summary>

The server requires TLS by default. Use `--dangerous-no-tls` for local development.

```bash
# Manual certs
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --cert cert.pem --private-key key.pem --port 443

# Let's Encrypt (automatic)
cargo run --release -- serve --model models/llama-3.2-1b-instruct \
  --letsencrypt --domain example.com --port 443
```

</details>

## Optimisation Stack

| Layer | Technique | Docs |
|---|---|---|
| Matmul | SIMD-cooperative (32 threads/row, `simd_sum`), WMMA tensor-core GEMM | — |
| Weights | Q4/Q8 block quantization, FP8 E4M3 on NVIDIA SM 89+ | [quantization](docs/quantization.md), [fp8](docs/fp8.md) |
| KV cache | Paged allocation (vLLM-style), TurboQuant 4-bit compression (~4×) | [kv-cache](docs/kv-cache.md), [turboquant](docs/turboquant.md) |
| Prefill | Tiled prefill (Flash Attention v2 style), batched GEMM | [tiled_prefill](docs/tiled_prefill.md) |
| Prefix caching | Shared KV blocks across sequences with common prefixes | [prompt-caching](docs/prompt-caching.md) |
| Attention | Fused single-pass softmax+V, head_dim-specialised kernels | — |
| FFN | Fused gate+up+SwiGLU (single kernel, halves input bandwidth) | [fused_dense_ffn](docs/fused_dense_ffn.md) |
| Decode | GPU-resident argmax, fused residual+RMSNorm | [rvllm_optimizations](docs/rvllm_optimizations.md) |
| Batching | Continuous batching (N sequences share the GPU) | [inference-engine](docs/inference-engine.md) |
| Expert streaming | NVMe pread on demand with GPU-side LRU cache | [expert-streaming](docs/expert-streaming.md) |
| Multi-GPU | NCCL tensor parallelism + expert parallelism for MoE | [multi-gpu-moe](docs/multi-gpu-moe.md) |

## Landscape

| System | Expert offload | Backends | Status |
|---|---|---|---|
| **rLLM** | NVMe streaming | Metal + CUDA | Educational |
| [flash-moe](https://github.com/danveloper/flash-moe) | NVMe streaming | Metal only | Educational |
| vLLM | CPU memory offload | CUDA only | Production |
| llama.cpp | CPU-side expert compute | CPU (no GPU streaming) | Production |
| SGLang / TensorRT-LLM | No expert offloading | CUDA only | — |

## Scripts

| Script | Description |
|---|---|
| `scripts/install-rust.sh` | Install Rust via rustup |
| `scripts/install-hf.sh` | Install the HuggingFace CLI |
| `scripts/download-models.sh` | Download model weights from HuggingFace |
| `scripts/quantize-models.sh` | Pre-quantize downloaded models to Q4 |

```bash
# Quick setup on a fresh machine
scripts/install-rust.sh
scripts/download-models.sh --small   # 1B–8B models (~100 GB)
cargo run --release -- run --model models/llama-3.2-1b-instruct --prompt "Hello" --chat
```

Gated models (Llama, Gemma, Mistral) require a HuggingFace token — set `HF_TOKEN` or run `hf auth login` first.

### Test & Benchmark

Tests and benchmarks use [uv](https://github.com/astral-sh/uv). Dependencies are in `tests/requirements.txt`.

```bash
# Unit tests (CPU backend, no GPU required)
cargo test                     # All tests
cargo test primitives::tests   # MoE buffer sizing
cargo test gpu::cpu::tests     # GPU kernel correctness vs CPU reference

# GPU integration tests — smoke-tests all model families, validates coherent English output
tests/run.sh                              # build + download small tier + test
tests/run.sh --skip-download -k llama     # filter by family
uv run pytest tests/ -v                   # run directly (models must be present)

# Benchmarks — throughput + TTFT + quality validation
uv run python tests/bench.py --models-dir models
uv run python tests/bench.py --filter qwen3.5 --runs 3
tests/run.sh --bench                      # all-in-one (build + bench)
```
