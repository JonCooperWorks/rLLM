# Production Considerations

Notes on how LLM inference works at scale, based on building
[rLLM](https://github.com/JonCooperWorks/rLLM) and
[Dyson](https://github.com/JonCooperWorks/dyson).  One developer's mental
model, not a reference architecture.

---

## The Gateway

rLLM is a single-model inference server.  In production you'd run many
instances, each serving one model on one or more GPUs.  A gateway sits in
front and handles everything that isn't inference:

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

**Auth.**  Validate API keys, check model access, reject bad requests before
they reach a GPU.  Inference servers should not know about user identity.

**Billing.**  Token counts come back from the inference server.  The gateway
records usage against the user's account.  Streaming responses are metered as
tokens arrive — the gateway sees every SSE chunk.

**Routing.**  Map the `model` field to a pool of backends.  Load balance via
round-robin, shortest queue, or prefix-cache affinity.

**Image fetching.**  For multimodal requests, fetch images from URLs and
convert to base64 before forwarding.  Keeps the inference server free of HTTP
client dependencies and eliminates SSRF on GPU machines.

---

## GPU Throughput and Batching

A single LLM forward pass is memory-bandwidth-bound: the GPU reads every
weight matrix once per token.  Decoding one sequence at a time leaves most
compute idle.

**Continuous batching** (what rLLM's engine does) packs multiple sequences
into one forward pass.  Instead of N mat-vec ops, you run one
[N, hidden] × [hidden, vocab] GEMM.  Same weight read, N tokens of output
instead of 1.

A 70B model on 4×H100s might decode at ~40ms for a single user.  With
continuous batching, that same latency serves 32–128 concurrent sequences —
same memory reads, more useful math per byte.

Prefill is compute-bound and parallelizes naturally.  Decode is where
batching matters.

---

## Hardware Tiers

Not every request needs the fastest GPU.

| Tier | Hardware | Use case |
|------|----------|----------|
| Fast | H100, B200 | Flagship models, premium plans |
| Standard | A100, L40 | Same models at slightly higher latency, or smaller models at full speed |
| Budget | V100, A10, consumer | Small/quantized models for free/low-cost plans |

The gateway routes based on user plan and requested model.

---

## Quantization as a Product Lever

Quantization compresses weights — rLLM's Q4 packs 32 weights into 18 bytes
(vs 64 at bf16).  Product implications:

**"Small" models are often quantized large models.**  A 70B at 4-bit fits in
the same memory as a 20B at bf16 and often performs comparably.

**Quantization trades quality for throughput.**  4-bit cuts bandwidth ~4× vs
bf16, which maps directly to ~4× higher throughput (decode is
bandwidth-bound).  Quality loss is small for most tasks.

**Mixed precision helps.**  Attention projections are more sensitive to
quantization than FFN weights.  Keep sensitive layers at higher precision,
quantize the rest.

**Production runs at the lowest precision that passes eval.**  No reason to
serve bf16 if Q4 produces the same scores.  Every bit shaved off is less
bandwidth, more sequences per GPU, lower cost per token.  The model behind an
API is almost certainly not running at training precision.

Same weights, different precision — that's likely how providers offer "the
same model" at different price points.

---

## QA and Eval Infrastructure

The question every team answers continuously: "which models, at which
precisions, on which hardware, are good enough to ship?"

**What providers likely have.**  An automated framework that:
1. Spins up inference servers for a model + quantization + hardware combo
2. Runs evals — accuracy benchmarks, latency percentiles, throughput under load
3. Produces actionable reports: "Llama 70B Q4 on 2×A100: passes quality bars,
   35ms/tok p50, $X/M tokens — ship it"

This is essentially CI/CD for models.  New model drops, the framework tests
it across a matrix of precisions and hardware, and reports results.

**What I have.**  Scripts I run manually on rented GPUs: a prompt suite
(reasoning, code, chat, tool calling), expected-output checks (exact match
for structured outputs, LLM-as-judge for open-ended), latency/throughput
measurement, and cross-run comparison.

A proper eval framework — provisioning, model download, eval execution,
teardown — would be useful but is a separate project.  rLLM is the inference
engine; eval orchestration is infrastructure tooling that points at one.

---

## What This Means for rLLM

rLLM is the inference server box in the diagram.  It handles model loading,
Q4 quantization, continuous batching, GPU kernel dispatch, streaming
generation, and an OpenAI/Anthropic-compatible API.

Everything above — auth, billing, routing, plan tiers, image fetching —
belongs in a gateway.  rLLM turns tokens into tokens as fast as possible.
The gateway decides which tokens go where and who pays.
