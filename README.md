# rLLM

Minimal LLM inference engine written from scratch in Rust. Metal GPU backend, bf16 and Q4 quantization, Llama architecture. No frameworks, no GGML — just raw GPU compute.

## Performance

Apple M4 Max (64 GB):

| Model | Params | Weights | Mode | Gen tok/s |
|---|---|---|---|---|
| Llama 3.2 1B | 1.2B | 2.6 GB | bf16 | 128 |
| Llama 3.2 1B | 1.2B | 0.8 GB | Q4 | 183 |
| Llama 3.2 3B | 3.2B | 6.4 GB | bf16 | 42 |
| Llama 3.2 3B | 3.2B | 2.0 GB | Q4 | 47 |
| Llama 3.1 8B | 8.0B | 16 GB | bf16 | 24 |
| Llama 3.1 8B | 8.0B | 5 GB | Q4 | 33–51 |

## Features

- **Metal GPU backend** — SIMD-cooperative matmul, async command buffer dispatch
- **Q4 quantization** — 4-bit block quantization on load (~3.2x memory reduction)
- **bf16 inference** — native half-precision on Apple Silicon
- **Safetensors loading** — single and multi-shard weight files
- **BPE tokenizer** — HuggingFace-compatible tokenizer.json
- **Chat template** — Llama 3 instruct format with `--chat` flag
- **Temperature + top-p sampling** — configurable via `--temperature` and `--top-p`
- **Streaming output** — tokens printed as generated

## Usage

```
cargo run --release -- --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128
```

With Q4 quantization:

```
cargo run --release -- --model models/llama-3.2-1b --prompt "The meaning of life is" --max-tokens 128 --quantize
```

Chat mode (instruct models):

```
cargo run --release -- --model models/llama-3.2-3b-instruct --prompt "Write a Python program that adds 4 numbers" --chat --temperature 0
```

## Architecture

```
src/
├── main.rs        — CLI + inference pipeline
├── config.rs      — HuggingFace config.json parsing
├── model.rs       — Llama forward pass, KV cache
├── loader.rs      — Safetensors loading, Q4 on-load quantization
├── tokenizer.rs   — BPE tokenizer
├── chat.rs        — Llama 3 chat template formatter
├── sampler.rs     — Temperature + top-p sampling
└── gpu/
    ├── mod.rs     — GpuBackend trait
    ├── metal/     — Metal backend + compute shaders
    └── cuda/      — CUDA backend (stub)
```

Model code is generic over a `GpuBackend` trait. Platform selection uses OS-conditional compilation — Metal on macOS, CUDA on Linux.

## Model Setup

Models are downloaded from [Hugging Face](https://huggingface.co) in safetensors format. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```
pip install huggingface-cli
```

Then download models into a `models/` directory:

```
# Base models (text completion)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
huggingface-cli download meta-llama/Llama-3.2-3B --local-dir models/llama-3.2-3b
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir models/llama-3.1-8b

# Instruct models (chat / instruction following)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b-instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3.2-3b-instruct
```

> Note: Llama models are gated — you'll need to [accept the license](https://huggingface.co/meta-llama/Llama-3.2-1B) on Hugging Face and authenticate with `huggingface-cli login` before downloading.

Each model directory should contain `config.json`, `tokenizer.json`, and one or more `.safetensors` weight files.
