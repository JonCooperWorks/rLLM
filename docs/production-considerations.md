# Production Considerations

Learning notes on how LLM inference might work at scale.  This is my intuition
from building [rLLM](https://github.com/JonCooperWorks/rLLM) and
[Dyson](https://github.com/JonCooperWorks/dyson) — not insider knowledge of
how any particular provider runs their infrastructure.  Take it as one
developer's mental model, not a reference architecture.

---

## The Gateway

rLLM is a single-model inference server.  In production, you'd run many
instances of it (or something like it), each serving one model on one or more
GPUs.  A gateway sits in front of all of them and handles everything that
isn't inference:

```
                    ┌─────────────┐
    client ────────►│   Gateway   │
                    │             │
                    │  • auth     │
                    │  • billing  │
                    │  • routing  │
                    │  • rate lim │
                    └──┬──┬──┬───┘
                       │  │  │
            ┌──────────┘  │  └──────────┐
            ▼             ▼             ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ rLLM       │ │ rLLM       │ │ rLLM       │
     │ Llama 70B  │ │ Qwen 32B   │ │ Gemma 27B  │
     │ 4×H100     │ │ 1×A100     │ │ 1×L40      │
     └────────────┘ └────────────┘ └────────────┘
```

**Authentication and access control.**  The gateway validates API keys,
checks which models a user's plan grants access to, and rejects requests
before they ever reach a GPU.  An inference server should not know or care
about user identity.

**Billing and metering.**  Token counts (prompt + completion) come back from
the inference server.  The gateway records usage against the user's account.
Streaming responses can be metered as tokens arrive — the gateway sees every
SSE chunk.

**Model routing.**  The gateway maps the `model` field in the request to a
pool of backends running that model.  Load balancing across the pool can be
simple round-robin, or smarter — route to the instance with the shortest
queue, or the one that already has a similar prompt prefix cached.

**Image fetching.**  For multimodal requests, the gateway can fetch images
from URLs and convert them to base64 before forwarding to the inference
server.  This keeps the inference server free of HTTP client dependencies and
eliminates SSRF as an attack surface on the GPU machines.

---

## GPU Throughput and Batching

A single LLM forward pass is memory-bandwidth-bound: the GPU reads every
weight matrix once per token.  If you're decoding one token at a time, most
of the GPU's compute capacity sits idle.

**Continuous batching** (what rLLM's engine does) packs multiple sequences
into a single forward pass.  Instead of N separate mat-vec operations, you
run one GEMM of size [N, hidden_dim] × [hidden_dim, vocab_size].  The weight
read is the same, but you produce N tokens of useful work instead of 1.

This is why inference providers batch aggressively.  A 70B model on 4×H100s
might decode 1 token at ~40ms latency for a single user — but at that same
latency, continuous batching can serve 32–128 concurrent sequences.  The GPU
does the same memory reads regardless; you're just doing more useful math per
byte read.

Prefill (processing the prompt) is compute-bound and parallelizes naturally —
large GEMMs already saturate the GPU.  Decode (generating tokens one at a
time) is where batching matters most.

---

## Hardware Tiers

Not every request needs the fastest GPU.  Providers likely segment hardware
into tiers:

**Fast tier (latest GPUs).**  H100s, B200s — used for flagship models and
premium plans.  Higher memory bandwidth means lower latency per token.
Premium pricing funds the hardware cost.

**Standard tier (previous generation).**  A100s, L40s — still fast, but lower
bandwidth.  Run the same models at slightly higher latency, or run smaller
models at full speed.  Good for mid-tier plans.

**Budget tier (older hardware).**  V100s, A10s, or even consumer GPUs — run
small models (7B–14B) or heavily quantized versions of larger ones.  The
models are less capable, but the hardware is dramatically cheaper.  Free
tiers and low-cost plans likely land here.

The gateway routes requests to the appropriate tier based on the user's plan
and the requested model.  A "fast" plan gets routed to H100 pools; a "free"
plan gets routed to older hardware running a smaller model.

---

## Quantization as a Product Lever

Quantization compresses model weights — for example, rLLM's Q4 format packs
32 weights into 18 bytes (vs 64 bytes at bf16).  This has direct product
implications:

**Smaller models are likely quantized large models.**  When a provider offers
a "small" or "fast" variant of a model, it may literally be the same model
with aggressive quantization.  A 70B model quantized to 4-bit fits in the
same memory as a 20B model at bf16 — and often performs comparably.

**Quantization trades quality for cost.**  4-bit quantization reduces memory
bandwidth by ~4× vs bf16, which directly maps to ~4× higher throughput (since
decode is bandwidth-bound).  The quality loss is small for most tasks — good
enough for a cheaper pricing tier.

**Mixed precision as a middle ground.**  Some layers (attention projections)
are more sensitive to quantization than others (FFN weights).  Keeping
sensitive layers at higher precision while quantizing the rest can preserve
quality at most of the cost savings.

This is likely how providers offer "the same model" at different price points
without actually training different models.  The underlying weights are the
same; the precision varies.

---

## What This Means for rLLM

rLLM covers the inference server box in the diagram above.  It handles:
- Model loading and weight management (including Q4 quantization)
- Continuous batching across concurrent sequences
- GPU kernel dispatch (Metal/CUDA)
- Streaming token generation
- OpenAI/Anthropic-compatible API surface

Everything above the inference server — auth, billing, model routing, plan
tiers, image fetching — belongs in a separate gateway service.  rLLM's job
is to turn tokens into tokens as fast as possible.  The gateway's job is to
decide which tokens go where and who pays for them.
