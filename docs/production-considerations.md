# Production Considerations

Notes on how LLM inference works at scale, based on building
[rLLM](https://github.com/JonCooperWorks/rLLM) and
[Dyson](https://github.com/JonCooperWorks/dyson).  One developer's mental
model, not a reference architecture.

---

## The Gateway

rLLM is a single-model inference server.  At scale you'd run many instances
behind a gateway that handles routing, billing, and rate limiting.  For local
use, rLLM serves directly on localhost.

```mermaid
graph TD
    Client([Client]) --> A["rLLM<br/>Local Model<br/>GPU"]
```

At scale, a gateway would sit in front of multiple instances and handle
routing (mapping the `model` field to a backend), billing (metering token
counts), and rate limiting.  rLLM itself doesn't implement any of these —
it just runs inference.

---

## Batching

> See [Inference Engine](inference-engine.md) for rLLM's continuous batching
> implementation and step loop.

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

> See [Quantization](quantization.md) for the full Q4 format spec, kernel
> internals, and pre-quantization workflow.

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

> See [Expert Streaming](expert-streaming.md) for the full SSD-backed MoE
> implementation, LRU cache, and platform-specific transfer paths.

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

> See [Prompt Caching](prompt-caching.md) for the implementation: hash-based
> lookup, reference counting, block-aligned sharing, and eviction.

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
    x-axis "Cache Hit Rate" ["0%", "25%", "50%", "75%", "90%", "100%"]
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

## Prefix-Cache Routing

The prefix cache is **local to each engine instance** — an in-process hash map
with no cross-server coordination.  A request only hits a cached prefix if it
lands on the server that computed it.  This makes load-balancer routing the
single biggest lever for cache hit rate.

### Routing strategies

```mermaid
graph TD
    Req([Request]) --> LB{Load Balancer}

    LB -->|Round-robin| RR["Even load<br/>Low hit rate<br/>Each server builds<br/>its own partial cache"]
    LB -->|Hash system prompt| HP["High hit rate<br/>Uneven if prompts<br/>are unbalanced"]
    LB -->|First-party pool| FP["Dedicated servers<br/>One prompt, always hot"]

    style FP fill:#d4edda
    style HP fill:#fff3cd
    style RR fill:#f8d7da
```

**Round-robin** — Even load, but every server independently caches the same
prefixes.  N servers means N× the VRAM spent on duplicate cache entries and N×
the cold prefills.  Fine for low-volume or heterogeneous traffic where no prefix
dominates.

**Hash on system prompt** — Hash the system prompt content (or a stable prefix
of it) and use it as the routing key.  All requests with the same system prompt
land on the same server(s).  High cache hit rate, and naturally groups traffic
by integration rather than by user.  The risk is hot spots if one prompt
dominates — mitigate with consistent hashing across a small pool per prompt.

**Dedicated first-party pools** — The highest-leverage pattern.  First-party
clients (mobile app, web app, internal tools) all share a single system prompt
that you control.  Route them to a dedicated server pool where that prefix is
always hot.  The prefix never competes with other entries for cache slots and
never gets evicted.

### The first-party / API split

In practice, traffic splits into two categories with very different caching
profiles:

```
First-party clients          API customers
─────────────────           ──────────────
You control the prompt       They control the prompt
One system prompt            One per customer (mostly stable)
100% cache hit rate          High hit rate with affinity routing
Dedicated server pool        Shared pool, hash-routed
```

**First-party clients** are the easy case.  You wrote the system prompt, it
never changes mid-session, and every request from every user starts with the
exact same tokens.  A dedicated pool of 2–4 servers with sticky routing gives
near-100% cache hit rate.  The system prompt's KV blocks are computed once at
first request and stay resident indefinitely.

**API customers** converge on the same pattern organically.  Each customer
typically has one or a few system prompts (one per app they've built).  Route
by API key or by hash of the system prompt and each customer's prefix stays
warm on their assigned servers.  The pricing incentive reinforces this:
charging cached input tokens at a discount (e.g., Anthropic's 90% discount)
nudges API users toward stable, cacheable system prompts — which improves
GPU utilisation for the provider.  A virtuous cycle.

### Why this works economically

System prompts are simultaneously:
- **The longest part** of the input (hundreds to thousands of tokens of
  instructions, tool schemas, few-shot examples)
- **The most repetitive** (identical across every request from the same client)
- **The most expensive to compute** (prefill is compute-bound, cost scales
  with prompt length)

Caching the thing that is longest, most repetitive, and most expensive to
compute gives the biggest return.  A 4000-token system prompt at ~800ms
prefill, cached at 90% hit rate, saves ~720ms of GPU compute per request.
At 100 req/s that's 72 seconds of GPU time saved per second of wall time —
the equivalent of adding 72 GPUs worth of prefill capacity for free.

### Capacity planning

The default cache holds 64 prefixes per instance.  In the first-party pool
pattern, you only need one slot.  In the API pattern, 64 slots covers your
top 64 API customers by traffic — which likely accounts for 95%+ of requests
(API traffic follows a power law).

The real constraint is VRAM.  Each cached prefix holds KV blocks that can't
be used for active sequences.  A 1000-token prefix costs ~128MB at typical
dimensions.  64 cached prefixes = ~8GB — significant on a 24GB card, negligible
on an 80GB H100.  Size the cache to the hardware.

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
    Req([API Request]) --> GW{Gateway<br/>checks tier}

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

**Prefix-cache affinity routing.**  The gateway routes requests to the server
that already has the system prompt cached.  See [Prefix-Cache
Routing](#prefix-cache-routing) above for strategies (dedicated first-party
pools, hash-based API routing, and why the economics create a virtuous cycle).

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

## Security

> See [Threat Model](threat-model.md) for the full STRIDE analysis.

rLLM binds to `127.0.0.1` by default — only the local machine can reach it.
It makes no outbound network connections.  Weights are loaded from local
safetensors files via `mmap`/`pread` and are never transmitted over the
network.

For local use, this is sufficient.  rLLM has no auth, no rate limiting, and
no audit logging — it's a compute engine, not a multi-tenant service.

---

## Server-Side Tools

> See [Tool Calling](tool-calling.md) for rLLM's per-architecture prompt
> formatting, output parsing, and API surface.

Tool and function calling — web search, code execution, retrieval,
calculators — looks like it runs on the inference server, but it doesn't.
The inference server's job is to produce a tool-call response (a structured
JSON output saying "call function X with args Y").  Actually executing the
tool is someone else's problem.

```mermaid
graph TD
    Client([Client]) -->|1. prompt| GW[Gateway]
    GW -->|2. hand off| W["Tool Worker<br/>(stateful)"]
    W -->|3. prompt| Inf["Inference Server<br/>(GPU)"]
    Inf -->|"4. tool_call(search, {q: ...})"| W
    W -->|5. dispatch| Tools["Tool Cluster<br/>(Kubernetes)"]
    Tools -->|6. result| W
    W -->|"7. tool_result + continue"| Inf
    Inf -->|8. final completion| W
    W -->|9. response| GW
    GW -->|10. response| Client
```

The **gateway hands off tool-call requests to a worker** so it doesn't get
bogged down.  When a request includes tools, the gateway spawns (or assigns)
a **tool worker** — a lightweight, stateful process that owns the
multi-round-trip loop.  The worker talks to the inference server, dispatches
tool calls to an isolated tool cluster, feeds results back, and repeats
until the model produces a final completion.  The gateway stays free for
routing and billing.  The inference server never talks to the tool
cluster directly.

### Why isolate tool execution

**Different compute profile.**  Inference servers are GPU-bound and
expensive.  Tool execution is CPU/memory-bound — web search, database
lookups, code sandboxes.  Running tools on GPU machines wastes $2–3/hr
hardware on work that a $0.10/hr container can do.

**Different security profile.**  Tools may need network access (web search
hits the internet), filesystem access (code execution), or database
credentials (retrieval).  Inference servers have none of these — they sit on
a locked-down private network with no outbound internet.  Mixing the two
profiles weakens the inference server's security posture.

**Independent scaling.**  Tool traffic is bursty and unpredictable (a
retrieval call might take 50ms or 5s depending on the backend).  The tool
cluster scales on CPU and memory via Kubernetes autoscaling.  The inference
cluster scales on GPU availability and batch utilisation.  Coupling them
means one bottlenecks the other.

**Blast radius.**  A tool that crashes, hangs, or times out affects one step
of one request.  The gateway retries or returns an error to the model.  If
the same tool ran on the inference server, a hang could block a GPU, stall a
batch, and degrade latency for every concurrent user.

### Worker-based orchestration

The tool worker is the agent-loop orchestrator — not the gateway.  For a
single user request, the worker may make multiple round trips between
inference and tools:

```
Gateway → worker → inference → tool_call → tool cluster → tool_result →
inference → tool_call → tool cluster → tool_result →
inference → final completion → worker → gateway → client
```

Each round trip adds latency (~100ms worker overhead + tool execution time),
but the alternative — giving inference servers network access and tool
credentials — breaks the security model.

**Policy enforcement is split** (at scale).  The **gateway** enforces
per-customer policies before handing off: **tool allow-lists** and **rate
limits**.  The **tool worker** enforces
per-request limits during the loop: **max tool calls** (at most N round
trips), **per-tool timeouts** (tool must respond within T seconds), and
**loop deadlines** (total wall-clock cap across all round trips).  The
inference server just produces tool-call JSON and doesn't know or care
whether the tool actually runs.

---

## What This Means for rLLM

rLLM is the inference server box in the diagram: model loading, Q4
quantization, continuous batching, GPU dispatch, streaming generation, and an
OpenAI/Anthropic-compatible API.

Everything above — billing, routing, tiers — would belong in a gateway.
rLLM turns tokens into tokens.

---

## Related Documents

- [Architecture Overview](architecture-overview.md) — end-to-end flow from request to token
- [Inference Engine](inference-engine.md) — step loop, scheduler, continuous batching
- [KV Cache](kv-cache.md) — paged allocation, block tables, generational indices
- [Prompt Caching](prompt-caching.md) — prefix sharing, ref counting, eviction
- [Quantization](quantization.md) — Q4 format, pre-quantization, kernel dequantization
- [Expert Streaming](expert-streaming.md) — SSD-backed MoE, LRU cache, pread I/O
- [API Server](api-server.md) — HTTP endpoints, worker thread, streaming
- [Tool Calling](tool-calling.md) — per-architecture formats, parsing, API surface
- [Threat Model](threat-model.md) — STRIDE analysis, weight storage, residual risks
