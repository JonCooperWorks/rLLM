# rLLM Documentation

Guide to the rLLM documentation suite. Start with the architecture overview for the
big picture, then drill into subsystem docs as needed.

## Architecture

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture-overview.md) | End-to-end flow from user request to token generation — how CLI, HTTP API, inference engine, and GPU backend connect |
| [GPU Backend](gpu-backend.md) | Hardware abstraction layer with composable operation traits and platform-specific implementations (Metal, CUDA, CPU) |
| [Inference Engine](inference-engine.md) | Core scheduling and continuous batching across prefill/decode phases for all entry points (run, batch, serve) |
| [Model Layer](model-layer.md) | Config parsing, weight loading, architecture dispatch, and forward pass implementations for thirteen model families |
| [API Server](api-server.md) | HTTP server with OpenAI and Anthropic endpoint compatibility using a worker-thread architecture |

## Optimisations

| Document | Description |
|----------|-------------|
| [KV Cache](kv-cache.md) | Paged key-value cache inspired by vLLM's PagedAttention — fixed-size block allocation instead of contiguous per-sequence buffers |
| [TurboQuant](turboquant.md) | KV cache vector quantization (arXiv:2504.19874) — random rotation + Max-Lloyd scalar quantization, ~4x compression at 4-bit |
| [Quantization](quantization.md) | Q4/Q8 symmetric quantization — block format, pre-quantization workflow, GPU kernel dequantization |
| [FP8](fp8.md) | FP8 E4M3 format on NVIDIA Ada/Hopper — IEEE 8-bit float, auto-selected when `--quant q8` on SM 89+ |
| [Tiled Prefill](tiled_prefill.md) | Flash Attention v2-style tiled prefill — halves K/V memory traffic by grouping adjacent query positions |
| [Fused Dense FFN](fused_dense_ffn.md) | Single-kernel gate+up+SwiGLU — halves input bandwidth for dense FFN layers |
| [rvLLM Optimizations](rvllm_optimizations.md) | GPU-resident argmax and fused residual+RMSNorm kernels |
| [Prompt Caching](prompt-caching.md) | Reuse prefill KV cache across requests that share a common prefix — system prompt caching, hash-based matching |
| [Expert Streaming](expert-streaming.md) | SSD-backed MoE inference — stream expert weights from NVMe on demand for models that don't fit in GPU memory |
| [Multi-GPU MoE](multi-gpu-moe.md) | Hybrid tensor parallelism + expert parallelism for MoE models across multiple GPUs |

## Model Features

| Document | Description |
|----------|-------------|
| [Vision](vision.md) | SigLIP ViT encoder for vision-language models — image preprocessing, patch embedding, spatial merge, and LLM token scatter |
| [Tool Calling](tool-calling.md) | Tool/function calling with per-architecture prompt formatting, output parsing, and both OpenAI and Anthropic API surfaces |

## Operations

| Document | Description |
|----------|-------------|
| [Integration Testing](integration-testing.md) | GPU integration test suite — model smoke tests, quality validation, benchmark harness |
| [Authentication](authentication.md) | Pluggable auth hook system — trait design, OIDC JWT provider, per-user logging, custom provider guide |
| [Threat Model](threat-model.md) | STRIDE analysis — weight theft, customer data, auth-without-TLS risks, what rLLM protects vs what the deployment must provide |
| [Production Considerations](production-considerations.md) | How LLM inference works at scale — gateway architecture, batching, prompt caching economics, tiers, security controls |
| [Framework Comparison](framework-comparison.md) | How rLLM compares to vLLM, Ollama, and llama.cpp — rLLM is an educational codebase, not a production system |
