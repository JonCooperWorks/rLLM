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

```mermaid
graph TD
    Client([Client]) --> Gateway

    subgraph Gateway[Gateway]
        Auth[Auth]
        Billing[Billing]
        Routing[Routing]
        RateLimit[Rate Limiting]
    end

    Gateway --> A["rLLM<br/>Llama 70B<br/>4×H100"]
    Gateway --> B["rLLM<br/>Qwen 32B<br/>1×A100"]
    Gateway --> C["rLLM<br/>Gemma 27B<br/>1×L40"]
```

**Auth.**  Validate API keys, check model access, reject bad requests before
they reach a GPU.  Inference servers should not know about user identity.

**Billing.**  Token counts come back from the inference server; the gateway
records usage.  Streaming responses are metered as tokens arrive via SSE.

**Routing.**  Map the `model` field to a backend pool.  Balance via
round-robin, shortest queue, or prefix-cache affinity.

**Image fetching.**  For multimodal requests, fetch images from URLs and
convert to base64 before forwarding.  Eliminates SSRF on GPU machines.

---

## Batching

A single forward pass is memory-bandwidth-bound: the GPU reads every weight
matrix once per token.  Decoding one sequence at a time wastes compute.

**Continuous batching** packs N sequences into one forward pass — one
[N, hidden] × [hidden, vocab] GEMM instead of N mat-vec ops.  Same weight
read, N tokens of output.  A 70B on 4×H100s decodes at ~40ms for one user;
with batching, the same latency serves 32–128 concurrent sequences.

```mermaid
xychart-beta
    title "Throughput vs Batch Size (70B model, 4×H100)"
    x-axis "Batch Size" [1, 4, 8, 16, 32, 64, 128]
    y-axis "Tokens/sec" 0 --> 3500
    bar [25, 100, 200, 400, 800, 1600, 3200]
```

Prefill is compute-bound and parallelizes naturally.  Decode is where
batching matters — and where the economics live.

---

## Quantization

Quantization compresses weights — rLLM's Q4 packs 32 weights into 18 bytes
(vs 64 at bf16), cutting memory bandwidth ~4× and directly increasing decode
throughput ~4× (since decode is bandwidth-bound).

```mermaid
xychart-beta
    title "Memory per Block (32 weights)"
    x-axis "Precision" [bf16, Q8, Q4]
    y-axis "Bytes" 0 --> 70
    bar [64, 34, 18]
```

**"Small" models are often quantized large ones.**  A 70B at Q4 fits in the
same memory as a 20B at bf16 and often performs comparably.

**Mixed precision.**  Attention projections are more sensitive than FFN
weights.  Keep sensitive layers higher, quantize the rest.

**Production runs at the lowest precision that passes eval.**  Every bit
shaved is less bandwidth, more sequences per GPU, lower cost per token.  The
model behind an API is almost certainly not at training precision.

---

## Disk Streaming

Not every model has to fit in GPU memory.  rLLM already streams MoE experts
from NVMe on demand (`src/model/expert_stream.rs`) — Qwen3.5-35b has 256
experts (~60GB), but only 8 are active per token, so expert memory drops from
60GB to ~15MB of buffer slots.

```mermaid
graph LR
    subgraph NVMe["NVMe (256 experts, ~60GB)"]
        E1[Expert 1]
        E2[Expert 2]
        E3["..."]
        EN[Expert 256]
    end

    Router["Router<br/>selects K=8"] --> |pread| Buf["GPU Buffer Slots<br/>~15MB"]
    E1 -.-> Router
    EN -.-> Router
    Buf --> FWD[Forward Pass]
    FWD --> |reuse| Buf
```

The same idea generalizes to dense models.  A 70B at Q4 is ~35GB.  A machine
with 24GB of VRAM can run it by streaming layers from disk — load a few
transformer blocks, run them, evict, load the next batch.  Latency goes up
(NVMe-bound instead of VRAM-bandwidth-bound), but the model runs.

---

## Prompt Caching

Most API traffic shares a common prefix.  System prompts, tool definitions,
and few-shot examples appear identically at the start of every request for a
given integration.  Prompt caching exploits this: compute the KV once, share
the physical blocks across all subsequent requests.

### The mechanism

rLLM's paged KV cache already uses block-table indirection — each sequence has
a block table mapping logical blocks to physical blocks in a shared pool.
Prompt caching extends this with a `PrefixCache` that maps token-sequence
hashes to physical block indices:

```
Request 1 (first with this system prompt):
  prefill all 1024 tokens → writes blocks [0..63]
  register blocks [0..63] in PrefixCache, key = hash(tokens[0..1024])

Request 2 (same system prompt + different user message):
  lookup hash(tokens[0..1024]) → hit, blocks [0..63]
  copy [0..63] into new seq's block table, set seq_len = 1024
  prefill only the 50 suffix tokens → writes blocks [64..67]
```

The attention kernel sees a contiguous 1074-position sequence.  It doesn't
know that blocks 0–63 were computed by a different request.  The block table
abstraction makes sharing transparent.

```mermaid
graph LR
    subgraph Pool["Physical KV Block Pool"]
        B0["Block 0<br/>(shared)"]
        B1["Block 1<br/>(shared)"]
        BN["..."]
        B63["Block 63<br/>(shared)"]
        B64["Block 64<br/>(Seq 2 own)"]
        B65["Block 65<br/>(Seq 3 own)"]
    end

    subgraph Seq2["Sequence 2"]
        BT2["Block Table:<br/>0,1,...,63,64"]
    end

    subgraph Seq3["Sequence 3"]
        BT3["Block Table:<br/>0,1,...,63,65"]
    end

    BT2 --> B0
    BT2 --> B64
    BT3 --> B0
    BT3 --> B65
```

Reference counting ensures shared blocks survive until all users are done.
Eviction is LRU among entries with zero active references.

### The economics

Prompt caching saves **prefill compute** — the most expensive per-request
cost.  Prefill is compute-bound (GEMM through every transformer layer for
every prompt token).  For a 70B model on 4×H100, prefilling 1000 tokens takes
~200ms.  Cache that prefix and subsequent requests skip it entirely.

The numbers:

| Metric | Without cache | With cache (90% hit) |
|--------|--------------|---------------------|
| Prefill per request | 1000 tokens | 100 tokens (suffix only) |
| TTFT (time to first token) | ~200 ms | ~20 ms |
| Prefill compute/request | ~200 TFLOP | ~20 TFLOP |
| Max prefills/sec (compute-limited) | ~5 | ~50 |

**Prefill throughput scales inversely with prefix length.**  A 4000-token
system prompt (common for tool-calling agents) takes ~800ms to prefill.  Cache
it and TTFT drops to the user-message prefill time — typically 10–50ms.

**Memory trade-off.**  Cached blocks consume VRAM that could hold more
concurrent sequences.  A 1000-token prefix at kv_dim=1024, bf16, 32 layers
costs `63 blocks × 16 × 1024 × 2 × 2 × 32 = ~128 MB`.  On a 80GB H100, that's
0.16% of VRAM for a cache entry that might serve thousands of requests.

**The pricing angle.**  Anthropic charges cached input tokens at a 90%
discount (10% of base price).  This works because the provider's marginal cost
for a cached token is near zero — the KV already exists in GPU memory.  The
discount incentivises users to structure prompts for cacheability (stable
system prompt first, variable content last), which improves GPU utilisation
for the provider.

```mermaid
xychart-beta
    title "Effective Cost per Request (1000-token prefix)"
    x-axis "Cache Hit Rate" [0%, 25%, 50%, 75%, 90%, 100%]
    y-axis "Relative Prefill Cost" 0 --> 100
    bar [100, 77, 55, 32, 14, 5]
```

At 90% hit rate, prefill cost per request drops to ~14% of the uncached
baseline.  The remaining cost is the suffix prefill (always required) plus
the decode phase (unaffected by caching).

**What caching does NOT improve.**  Decode throughput (tok/s) is unchanged —
each generated token still requires a full forward pass through all layers,
reading the entire KV cache.  Caching is purely a TTFT and prefill-compute
optimisation.

---

## Economics and Tier Differentiation

Unit economics: how many tokens per GPU-hour, and what do you charge per
token?

**Batching is the business model.**  An H100 at ~$2–3/hr decoding one
sequence produces ~25 tok/s.  Batching 64 sequences produces ~1600 tok/s from
the same hour — 64× lower cost per token.

**Quantization is pure margin.**  Q4 vs bf16 is ~4× more tokens per GPU-hour
at nearly the same quality.  Charge the same price: 4× margin.  Pass savings
to users: undercut competitors.

**Prompt caching is throughput multiplication.**  The GPU cycles saved on
prefill are available for more decode batches.  A workload that's 60% prefill-bound
(long system prompts, short responses) can nearly double effective throughput
with a hot cache — same hardware, same price, 2× the requests served.

**Tiers are a mix of hardware, quantization, residency, caching, and priority.**

```mermaid
quadrantChart
    title Tier Strategy — Latency vs Cost
    x-axis "Lower Cost" --> "Higher Cost"
    y-axis "Higher Latency" --> "Lower Latency"
    quadrant-1 "Pro"
    quadrant-2 "Standard"
    quadrant-3 "Free"
    quadrant-4 "Overprovisioned"
    "VRAM resident, H100/B200": [0.85, 0.9]
    "VRAM resident, A100/L40": [0.55, 0.75]
    "Quantized, older GPU": [0.35, 0.7]
    "Disk-streamed, minimal VRAM": [0.15, 0.15]
```

Same model, same API, different hardware behind it.  Pro users get dedicated
GPU pools with low batch sizes and aggressive latency SLOs.  Free users get
overflow pools on older hardware, high batch sizes, disk-streamed models, and
lower queue priority.  The GPU does the same work — the difference is queue
position and how much of the model lives in VRAM.

```mermaid
graph TD
    Req([API Request]) --> GW{Gateway<br/>checks user tier}

    GW -->|Pro| Pro["Dedicated H100 pool<br/>Low batch size<br/>Full VRAM residency<br/>Hot prefix cache"]
    GW -->|Standard| Std["Shared A100/L40 pool<br/>Medium batch size<br/>Quantized weights"]
    GW -->|Free| Free["Overflow pool<br/>High batch size<br/>Disk-streamed layers<br/>Queue deprioritized"]

    Pro --> Resp([Response])
    Std --> Resp
    Free --> Resp
```

**Smaller models are genuinely cheap.**  A 7B fits on a consumer GPU; a 70B
needs 4×H100s.  10–50× cost difference.  The large-model premium pays for the
fleet.

**Prefix-cache affinity routing.**  The gateway can route requests to the
server instance that already has the system prompt cached, avoiding cold
prefills.  Hash the system prompt, use it as a routing key.  This turns prompt
caching from a per-instance optimisation into a fleet-wide one.

---

## QA and Eval

The question: "which models, at which precisions, on which hardware, pass the
quality bar?"

```mermaid
graph LR
    Drop["New model drop"] --> Matrix["Test matrix<br/>precision × hardware"]
    Matrix --> Eval["Run evals<br/>accuracy, latency,<br/>throughput"]
    Eval --> Report["Ship / Hold<br/>decision"]
    Report -->|passes| Deploy[Deploy to tier]
    Report -->|fails| Tune["Try lower precision<br/>or different hardware"]
    Tune --> Matrix
```

**What providers likely have.**  CI/CD for models — spin up servers for each
model + quantization + hardware combo, run evals (accuracy, latency
percentiles, throughput under load), produce reports: "Llama 70B Q4 on
2×A100: passes quality bars, 35ms/tok p50, $X/M tokens — ship it."

**What I have.**  Scripts I run manually on rented GPUs: prompt suite
(reasoning, code, chat, tool calling), expected-output checks, latency/
throughput measurement, cross-run comparison.

A proper eval framework is a separate project — rLLM is the inference engine,
not the orchestration layer.

---

## What This Means for rLLM

rLLM is the inference server box in the diagram: model loading, Q4
quantization, continuous batching, GPU dispatch, streaming generation, and an
OpenAI/Anthropic-compatible API.

Everything above — auth, billing, routing, tiers — belongs in a gateway.
rLLM turns tokens into tokens.  The gateway decides which tokens go where and
who pays.
