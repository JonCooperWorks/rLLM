# rLLM Documentation

Guide to the rLLM documentation suite.  Start with the architecture overview for the
big picture, then drill into subsystem docs as needed.

## Documents

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture-overview.md) | End-to-end flow from user request to token generation — how CLI, HTTP API, inference engine, and GPU backend connect |
| [GPU Backend](gpu-backend.md) | Hardware abstraction layer with composable operation traits and platform-specific implementations (Metal, CUDA, CPU) |
| [Inference Engine](inference-engine.md) | Core scheduling and continuous batching across prefill/decode phases for all entry points (run, batch, serve) |
| [KV Cache](kv-cache.md) | Paged key-value cache inspired by vLLM's PagedAttention — fixed-size block allocation instead of contiguous per-sequence buffers |
| [Model Layer](model-layer.md) | Config parsing, weight loading, architecture dispatch, and forward pass implementations for nine model families |
| [API Server](api-server.md) | HTTP server with OpenAI and Anthropic endpoint compatibility using a worker-thread architecture |
| [Tool Calling](tool-calling.md) | Tool/function calling with per-architecture prompt formatting, output parsing, and both OpenAI and Anthropic API surfaces |
| [Quantization](quantization.md) | Q4 symmetric quantization — block format, pre-quantization workflow, GPU kernel dequantization, and performance impact |
| [Expert Streaming](expert-streaming.md) | SSD-backed MoE inference — stream expert weights from NVMe on demand for models that don't fit in GPU memory |
| [Vision](vision.md) | SigLIP ViT encoder for vision-language models — image preprocessing, patch embedding, spatial merge, and LLM token scatter |
| [Prompt Caching](prompt-caching.md) | Reuse prefill KV cache across requests that share a common prefix — system prompt caching, hash-based matching |
| [Authentication](authentication.md) | Pluggable auth hook system — trait design, OIDC JWT provider, per-user logging, custom provider guide |
| [Threat Model](threat-model.md) | STRIDE analysis — weight theft, customer data, auth-without-TLS risks, what rLLM protects vs what the deployment must provide |
| [Production Considerations](production-considerations.md) | How LLM inference works at scale — gateway architecture, batching, prompt caching economics, tiers, security controls |
| [Framework Comparison](framework-comparison.md) | How rLLM compares to vLLM, Ollama, and llama.cpp — rLLM is an educational codebase for learning inference with Claude, not a production system |
