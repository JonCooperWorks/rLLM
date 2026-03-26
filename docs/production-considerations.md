# Production Considerations

How LLM inference works at scale.  Based on building
[rLLM](https://github.com/JonCooperWorks/rLLM) and
[Dyson](https://github.com/JonCooperWorks/dyson).  One developer's mental
model, not a reference architecture.

---

## The Gateway

rLLM is a single-model inference server.  In production you run many
instances — one model per process, one or more GPUs per instance.  A gateway
sits in front and owns everything that isn't inference:

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

- **Auth** — validate API keys, check model access, reject bad requests before
  they touch a GPU
- **Billing** — record token counts as they stream back via SSE
- **Routing** — map the `model` field to a backend pool; balance via
  round-robin, shortest queue, or prefix-cache affinity
- **Image fetching** — for multimodal requests, fetch images from URLs and
  convert to base64 before forwarding; eliminates SSRF risk on GPU machines

---

## Batching

> See [Inference Engine](inference-engine.md) for rLLM's continuous batching
> implementation and step loop.

A single forward pass is memory-bandwidth-bound: the GPU reads every weight
once per token.  Decoding one sequence at a time wastes compute.

**Continuous batching** packs N sequences into one forward pass — one
[N, hidden] × [hidden, vocab] GEMM instead of N mat-vec ops.  Same weight
read, N tokens of output.

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

> See [Quantization](quantization.md) for the Q4 format spec, kernel
> internals, and pre-quantization workflow.

Quantization compresses weights.  rLLM's Q4 packs 32 weights into 18 bytes
(vs 64 at bf16) — ~4× less bandwidth, ~4× faster decode (decode is
bandwidth-bound).

```mermaid
xychart-beta
    title "Memory per Block (32 weights)"
    x-axis "Precision" [bf16, Q8, Q4]
    y-axis "Bytes" 0 --> 70
    bar [64, 34, 18]
```

A 70B at Q4 fits in the same memory as a 20B at bf16 and often performs
comparably.  Production runs at the lowest precision that passes eval — every
bit saved is more sequences per GPU.

---

## Disk Streaming

> See [Expert Streaming](expert-streaming.md) for the SSD-backed MoE
> implementation, LRU cache, and platform-specific transfer paths.

Not every model fits in VRAM.  rLLM streams MoE experts from NVMe on demand —
Qwen3.5-35b has 256 experts (~60GB), but only 8 are active per token, so
expert memory drops to ~15MB of buffer slots.

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

The same idea generalizes to dense models.  A 70B at Q4 is ~35GB — a 24GB
card can run it by streaming layers from disk.  Higher latency (NVMe-bound),
but the model runs.

---

## Prompt Caching

> See [Prompt Caching](prompt-caching.md) for the implementation: hash-based
> lookup, reference counting, block-aligned sharing, and eviction.

Most API traffic shares a common prefix.  System prompts, tool definitions,
and few-shot examples repeat identically across requests.  Prompt caching
computes the KV once and shares the physical blocks.

### Mechanism

rLLM's paged KV cache uses block-table indirection.  The `PrefixCache` maps
token-sequence hashes to physical block indices:

```
Request 1 (cold):
  prefill 1024 tokens → writes blocks [0..63]
  register in PrefixCache: hash(tokens[0..1024]) → [0..63]

Request 2 (same system prompt, different user message):
  lookup hash → hit → copy block refs into new block table
  prefill only the 50 suffix tokens → writes blocks [64..67]
```

The attention kernel sees a contiguous sequence.  Block-table indirection
makes sharing transparent.

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

Reference counting keeps shared blocks alive.  Eviction is LRU among entries
with zero active references.

### Cache routing

The prefix cache is **local to each engine instance**.  A request only hits
cache if it lands on the server that computed it.  Routing strategy is the
biggest lever for hit rate.

```mermaid
graph TD
    Req([Request]) --> LB{Load Balancer}

    LB -->|Round-robin| RR["Even load<br/>Low hit rate"]
    LB -->|Hash system prompt| HP["High hit rate<br/>Risk: hot spots"]
    LB -->|First-party pool| FP["Dedicated servers<br/>Always hot"]

    style FP fill:#d4edda
    style HP fill:#fff3cd
    style RR fill:#f8d7da
```

- **Round-robin** — even load, N× duplicate cache entries, N× cold prefills
- **Hash on system prompt** — high hit rate; consistent hashing mitigates hot spots
- **Dedicated first-party pools** — your clients share one prompt, always hot

First-party traffic is easy: one system prompt, 100% hit rate.  API customers
converge organically — each has a few stable prompts.  Route by API key or
prompt hash.

---

## Economics

LLM inference economics come down to one question: how many tokens can you
extract per GPU-hour, and what do you charge per token?  Four levers —
batching, weight quantization, KV cache quantization, and prompt caching —
determine cost per token.  Hardware selection and tier differentiation
determine what you charge.  Everything connects: cheaper cost per token lets
you offer lower prices or higher margins, and the pricing structure itself
shapes user behaviour in ways that improve GPU utilisation.

### Cost levers

```mermaid
graph LR
    subgraph Levers["Cost per Token"]
        B["Batching<br/>64× more tok/s<br/>same GPU-hour"]
        Q["Weight Quantization<br/>4× less bandwidth<br/>4× faster decode"]
        KV["KV Cache Quantization<br/>~4× less KV memory<br/>more sequences or longer context"]
        C["Prompt Caching<br/>skip prefill compute<br/>for repeated prefixes"]
    end

    B --> Low["Lower $/token"]
    Q --> Low
    KV --> Low
    C --> Low
```

**Batching is the business model.**  An H100 at ~$2–3/hr decoding one
sequence produces ~25 tok/s.  Batch 64 sequences from the same GPU-hour:
~1600 tok/s.  64× lower cost per token.  Without batching, LLM inference
APIs don't work economically.

**Weight quantization is pure margin.**  Q4 vs bf16 gives ~4× more tokens
per GPU-hour at near-identical quality.  Charge the same price: 4× margin.
Pass savings to users: undercut competitors.  Either way, it's a direct
multiplier on unit economics.

**[KV cache quantization](turboquant.md) is concurrency multiplication.**
KV cache is the memory that scales with concurrent users × context length.
TurboQuant 4-bit compresses it ~4× with no quality loss — same GPU can hold
~4× more active sequences, or serve ~4× longer contexts, or any mix of
both.  On a 64 GB M4 Max serving Qwen 3.5 9B, that's ~2,000 concurrent
sequences vs ~400 at BF16.  For long-context workloads the effect compounds:
a 32K-token sequence drops from ~5.3 GB to ~1.1 GB.

**Prompt caching is throughput multiplication.**  Prefill (processing the
prompt) is compute-bound and often the dominant per-request cost.  A
4000-token system prompt takes ~800ms to prefill on a 70B model.  Cache that
prefix and subsequent requests skip it entirely — TTFT drops from ~800ms to
the user-message suffix (10–50ms).

| Metric | No cache | 90% hit rate |
|--------|----------|-------------|
| Prefill per request | 1000 tokens | 100 tokens (suffix only) |
| TTFT | ~200 ms | ~20 ms |
| Prefill compute | ~200 TFLOP | ~20 TFLOP |
| Max prefills/sec | ~5 | ~50 |

A workload that's 60% prefill-bound can nearly double effective throughput
with a hot cache — same hardware, same price, 2× the requests served.

### Pricing

Pricing can incentivise user behaviour that improves your GPU utilisation.

**Cached token discounts.**  Anthropic charges cached input tokens at 90%
discount.  The provider's marginal cost for a cached token is near zero — the
KV already exists in GPU memory.  The discount nudges users to structure
prompts for cacheability (stable prefix first, variable content last), which
improves hit rates and GPU utilisation for the provider.  A virtuous cycle.

```mermaid
xychart-beta
    title "Effective Prefill Cost (1000-token prefix)"
    x-axis "Cache Hit Rate" ["0%", "25%", "50%", "75%", "90%", "100%"]
    y-axis "Relative Cost %" 0 --> 100
    bar [100, 77, 55, 32, 14, 5]
```

**Model size premium.**  A 7B fits on a consumer GPU; a 70B needs 4×H100s.
10–50× cost difference.  The large-model premium pays for the fleet.

**Why caching economics work.**  System prompts are simultaneously the
longest, most repetitive, and most expensive part of the input.  Caching the
thing that costs the most gives the biggest return.  At 100 req/s with a
4000-token prefix at 90% hit rate, that's 72 GPU-seconds of prefill compute
saved per wall-clock second.

### Tiers

Same model, same API, different hardware behind it.  Tiers are a mix of
hardware generation, quantization level, VRAM residency, and queue priority.

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

```mermaid
graph TD
    Req([API Request]) --> GW{Gateway<br/>checks user tier}

    GW -->|Pro| Pro["Dedicated H100 pool<br/>Low batch, full VRAM<br/>Hot prefix cache"]
    GW -->|Standard| Std["Shared A100/L40 pool<br/>Medium batch, quantized"]
    GW -->|Free| Free["Overflow pool<br/>High batch, disk-streamed<br/>Deprioritized queue"]

    Pro --> Resp([Response])
    Std --> Resp
    Free --> Resp
```

Pro users get dedicated GPU pools with low batch sizes and aggressive latency
SLOs.  Free users get overflow pools on older hardware, higher batch sizes,
disk-streamed models, and lower queue priority.  The GPU does the same work —
the difference is queue position and how much of the model lives in VRAM.

### Capacity planning

Default: 64 cached prefixes per instance.  First-party pool needs one slot.
API pattern: 64 covers your top customers by traffic (power-law
distribution).  A 1000-token prefix costs ~128MB; 64 entries = ~8GB —
significant on 24GB, negligible on 80GB.

### QA and eval

The gate between a model drop and a tier deployment: which models, at which
precisions, on which hardware, pass the quality bar?

```mermaid
graph LR
    Drop["New model drop"] --> Matrix["Test matrix<br/>precision × hardware"]
    Matrix --> Eval["Run evals<br/>accuracy, latency,<br/>throughput"]
    Eval --> Report["Ship / Hold"]
    Report -->|passes| Deploy[Deploy to tier]
    Report -->|fails| Tune["Try lower precision<br/>or different hardware"]
    Tune --> Matrix
```

The goal is CI/CD for models: spin up servers for each config, run evals,
produce reports.  "Llama 70B Q4 on 2×A100: passes quality bars, 35ms/tok
p50, $X/M tokens — ship it."

---

## Security Controls

> See [Threat Model](threat-model.md) for the full STRIDE analysis.

The inference fleet holds model weights and produces raw completions — two
assets worth protecting.  The security model treats it as a high-value,
low-surface-area zone.

```mermaid
graph LR
    Internet([Internet]) --> GW["Gateway<br/>(public VPC)"]

    subgraph Private["Private Network (no internet)"]
        Inf["Inference Servers<br/>(GPU fleet)"]
        Reg["Model Registry<br/>(encrypted weights)"]
        Logs["Audit Log<br/>Sink"]
    end

    GW -->|"signed inference<br/>token + prompt"| Inf
    Inf -->|"completion +<br/>token count"| GW
    Reg -->|"decrypt &<br/>load weights"| Inf
    Inf -->|"per-request<br/>audit events"| Logs
    GW -->|"billing<br/>events"| Logs
```

### Authentication and token exchange

The gateway validates the customer's API key, mints a short-lived,
customer-scoped inference token, and attaches it to the request.  The
inference server verifies the token before running a forward pass.

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant rLLM

    Client->>Gateway: API key + prompt
    Gateway->>Gateway: validate key, check access, rate limit
    Gateway->>Gateway: mint short-lived JWT (sub=customer, aud=rllm)
    Gateway->>rLLM: JWT + prompt
    rLLM->>rLLM: verify JWT (signature, exp, iss, aud)
    rLLM->>rLLM: forward pass (user.sub in audit log)
    rLLM->>Gateway: completion + token count
    Gateway->>Client: completion
```

Three properties:

1. **Identity at the inference layer** — every forward pass is tied to a
   customer, not just a gateway session
2. **Token theft detection** — inference activity without a matching gateway
   session is anomalous
3. **Blast radius** — a leaked inference token expires quickly and only
   authorises inference, not billing or account mutations

**rLLM implements this** via its pluggable auth hook system (`--auth-config`).
The built-in OIDC provider validates JWTs against the issuer's published
JWKS.  The `AuthProvider` trait can be extended to support an org's specific
auth — custom token formats, internal CAs, proprietary identity systems.
See [Authentication](authentication.md) for the full design.

### Audit logging

Every inference request is logged with customer identity, model, token
counts, latency, and timestamp.  The audit log is ground truth for:

- **Billing reconciliation** — compare inference-side and gateway-side token
  counts; discrepancies flag bugs or fraud
- **Abuse detection** — unusual volume, odd hours, pattern anomalies
- **Forensics** — trace harmful output to the exact request and customer

When auth is enabled, rLLM logs the authenticated user identity (JWT `sub`
claim) alongside token counts and latency in its per-request stderr output —
inference-side audit logging without additional infrastructure.

Both streams (inference-side and gateway-side) ship to a **log gateway**
within the private VPC, which forwards off-VPC to a central store.  Inference
servers never talk to the central store directly.

### Network isolation

```mermaid
graph TD
    Internet([Internet]) --> GW["Gateway<br/>(public VPC)"]

    subgraph Private["Private Network"]
        direction TB
        Inf["Inference Servers<br/>(GPU fleet)"]
        LogGW["Log Gateway<br/>(VPC-internal)"]
    end

    subgraph Isolated["Isolated Registry Network"]
        Reg["Model Registry<br/>(encrypted weights)"]
    end

    Central["Central Log Store<br/>(corporate infra)"]

    GW -->|"inference traffic<br/>(persistent)"| Inf
    Inf -->|"completions +<br/>token counts"| GW
    Inf -->|audit events| LogGW
    GW -->|billing events| LogGW
    LogGW -->|"forward off-VPC"| Central

    Reg -.->|"JIT route<br/>(provisioned per deploy,<br/>torn down after clone)"| Inf

    Internet x--x|"no outbound"| Private

    style Private fill:#f8d7da,stroke:#dc3545
    style Isolated fill:#fff3cd,stroke:#ffc107
```

Inference servers have **no standing outbound access**.  The only persistent
flow is inference to/from the gateway.

**JIT registry access.**  To pull weights, a short-lived route to the model
registry is provisioned, weights are cloned to local NVMe, and the route is
torn down.  Provisioning requires two-person approval with hardware-backed
(YubiKey) authentication.

### Weight protection

Weights are protected at the infrastructure layer — the inference server
never sees ciphertext.

- **Registry-level encryption** — weights encrypted at rest, decryption keys
  bound to inference server service accounts
- **Full-disk encryption** — NVMe encrypted via LUKS/FileVault/cloud KMS;
  rLLM calls `mmap`/`pread` and gets decrypted bytes transparently
- **No standing SSH** — shell access is break-glass: PR, peer approval,
  time-limited, auto-revoked

### Traffic volume monitoring

Outbound volume is monitored against audit-log token counts.  Unexpected
volume — more bytes than token counts warrant — triggers alerts.  This
catches weight exfiltration (a 70B Q4 is ~35GB), bulk inference abuse, or
a compromised server streaming data via an allowed egress path.

---

## Server-Side Tools

> See [Tool Calling](tool-calling.md) for rLLM's per-architecture prompt
> formatting, output parsing, and API surface.

The inference server produces tool-call JSON.  It never executes tools.

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

The gateway hands off tool-call requests to a **tool worker** — a stateful
process that owns the multi-round-trip loop.  The worker talks to the
inference server, dispatches tool calls to an isolated tool cluster, feeds
results back, and repeats until the model produces a final completion.

### Why isolate tool execution

- **Different compute profile** — tools are CPU-bound; running them on GPU
  machines wastes $2–3/hr hardware on $0.10/hr work
- **Different security profile** — tools may need network/filesystem/DB
  access; inference servers have none of these
- **Independent scaling** — tool traffic is bursty; tool cluster scales on
  CPU via Kubernetes, inference cluster scales on GPU availability
- **Blast radius** — a hanging tool affects one step of one request, not
  the GPU batch

### Worker orchestration

The tool worker manages the agent loop:

```
gateway → worker → inference → tool_call → tool cluster → tool_result →
inference → tool_call → tool cluster → tool_result →
inference → final completion → worker → gateway → client
```

Policy is split: the **gateway** enforces tool allow-lists and rate limits
before handoff.  The **worker** enforces per-request limits during the loop:
max tool calls, per-tool timeouts, and loop deadlines.  The inference server
just produces JSON.

---

## What This Means for rLLM

rLLM is the inference server in the diagram: model loading, Q4 quantization,
continuous batching, GPU dispatch, streaming generation, and an
OpenAI/Anthropic-compatible API.

Billing, routing, rate limiting, and tiering belong in the gateway.
Authentication is shared: the gateway authenticates the end user and mints a
scoped token; rLLM verifies it via its pluggable auth hook system
(`--auth-config`).  The built-in OIDC provider handles standard JWT
validation; the `AuthProvider` trait can be extended to support an org's
specific auth infrastructure without modifying rLLM's core.  When auth is
enabled, per-user identity is logged to stderr alongside token counts and
latency — audit logging at the inference layer for free.

---

## Related Documents

- [Authentication](authentication.md) — auth hook system, OIDC provider, custom providers
- [Inference Engine](inference-engine.md) — step loop, scheduler, continuous batching
- [KV Cache](kv-cache.md) — paged allocation, block tables, generational indices
- [Prompt Caching](prompt-caching.md) — prefix sharing, ref counting, eviction
- [Quantization](quantization.md) — Q4 format, pre-quantization, kernel dequantization
- [TurboQuant](turboquant.md) — KV cache vector quantization, ~4× compression, quality-neutral
- [Expert Streaming](expert-streaming.md) — SSD-backed MoE, LRU cache, pread I/O
- [API Server](api-server.md) — HTTP endpoints, worker thread, streaming
- [Tool Calling](tool-calling.md) — per-architecture formats, parsing, API surface
- [Threat Model](threat-model.md) — STRIDE analysis, weight theft, customer data, residual risks
