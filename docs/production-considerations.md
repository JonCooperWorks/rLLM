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

## Disk Streaming as a Tier Lever

Not every model has to fit in GPU memory.  rLLM already does this for MoE
experts — Qwen3.5-35b has 256 experts totaling ~60GB, but only 8 are active
per token.  Rather than keeping them all on-GPU, rLLM streams the active
experts from NVMe on demand (`src/model/expert_stream.rs`), reducing expert
memory from 60GB to ~15MB of buffer slots.

The same idea generalizes to entire models.  A dense 70B at Q4 is ~35GB of
weights.  A machine with 24GB of VRAM can still run it by streaming layers
from disk — load a few transformer blocks, run them, evict, load the next
batch.  Latency per token goes up (you're now NVMe-bound instead of
VRAM-bandwidth-bound), but the model runs.

**This maps directly to product tiers.**  A "Pro" user gets the model fully
resident on a GPU with enough VRAM — fast decode, no disk I/O in the hot
path.  A "Free" user gets the same model on cheaper hardware where it doesn't
fully fit, with layers streamed from disk.  Same weights, same quality, but
10–50× higher latency per token.  The free user waits longer; the provider
pays less per request.

The spectrum looks like:

| Configuration | Latency | Cost | Tier |
|---------------|---------|------|------|
| Full model in VRAM, latest GPU | Lowest | Highest | Pro / Premium |
| Full model in VRAM, older GPU | Low | Medium | Standard |
| Quantized model in VRAM, older GPU | Low | Lower | Standard / Free |
| Model streamed from disk, minimal VRAM | High | Lowest | Free |

Each step down trades latency for cost.  The gateway picks the configuration
based on the user's plan — same model, same API, different hardware behind it.

---

## Economics

The unit economics of inference come down to: how many tokens can you squeeze
out of a GPU-hour, and what can you charge per token?

**Cost per token drops with utilization.**  A single H100 costs ~$2–3/hr
rented.  If it's decoding one sequence at a time, you're paying that full
rate for ~25 tokens/sec.  With continuous batching at 64 concurrent sequences,
you get ~1600 tokens/sec from the same GPU-hour.  Cost per token falls 64×.
Batching isn't just a performance optimization — it's the entire business
model.

**Faster tiers are about priority, not capability.**  The simplest way to
offer a "faster" tier: give paying users priority in the batch queue.  Free
users wait until there's spare capacity; Pro users get scheduled immediately.
The GPU does the same work either way — the difference is queue position.
More concretely:

- *Pro*: Dedicated GPU pool, low batch sizes (fewer sequences competing for
  bandwidth), aggressive latency SLOs.
- *Standard*: Shared pool, higher batch sizes, best-effort latency.
- *Free*: Overflow pool on cheaper/older hardware, highest batch sizes,
  disk-streamed models when VRAM is scarce, deprioritized in the queue.

**Quantization is pure margin.**  Serving Q4 instead of bf16 means ~4× more
tokens per GPU-hour at nearly the same quality.  If you charge the same price
per token, that's 4× the margin.  If you pass some savings to the user, you
undercut competitors while staying profitable.  This is why every provider
quantizes aggressively.

**Model size vs hardware cost.**  Smaller models are dramatically cheaper to
serve — a 7B model fits on a single consumer GPU, while a 70B needs 4×H100s.
The cost difference is 10–50×.  When a provider offers a small model for
cheap, they're not being generous — the hardware cost is genuinely that low.
The large-model premium pays for the expensive GPU fleet.

**Disk streaming extends the hardware.**  Streaming from NVMe lets you run
larger models on cheaper GPUs at the cost of latency.  A machine that would
otherwise be limited to 14B models can serve a 70B model slowly.  This is
useful for free tiers where latency tolerance is high and the alternative is
not offering the model at all.

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
