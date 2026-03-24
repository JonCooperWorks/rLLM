# Threat Model

STRIDE analysis of rLLM as a deployment component.  rLLM is an inference
engine — it turns prompts into completions on a GPU.  It pushes billing, rate
limiting, and authorization to infrastructure and an API gateway, but provides
optional pluggable authentication via an auth hook system (`--auth-config`).
This document explains what rLLM protects against, what it leaves to the
deployment environment, and why.

**Primary assets:**
- **Model weights** — proprietary or licensed, stored on local NVMe
- **Customer prompts and completions** — transit through the inference server

**Deployment assumption:** rLLM runs behind an API gateway on a private network
with no direct internet access.  See
[Production Considerations — Security Controls](production-considerations.md#security-controls)
for the reference deployment architecture.

---

## STRIDE Analysis

### Spoofing

| Threat | Who is spoofed | rLLM's stance |
|--------|---------------|---------------|
| Unauthenticated requests reach the inference server | The server treats any TCP connection as a legitimate caller | **Optional: auth hook validates tokens when configured.** Disabled by default. |
| Attacker impersonates the gateway | The server cannot distinguish gateway traffic from other traffic | **Optional: OIDC validates JWT signatures; static API key uses argon2id hash comparison (timing-safe).** |

**Two deployment modes for authentication:**

1. **Solo / dev (default)** — no auth.  rLLM binds to `127.0.0.1` and relies
   on network isolation (localhost, SSH tunnel to a rented GPU, trusted LAN).
   This is the right choice when you are the only user and don't want to
   configure an OAuth2 provider just to have auth.

2. **Gateway + rLLM auth (defense-in-depth)** — the gateway authenticates the
   end user, does token exchange, and mints a scoped JWT for rLLM.  rLLM
   validates the JWT via its auth hook (`--auth-config auth.json`), enforcing
   identity end-to-end.  Even if the network boundary is breached,
   unauthenticated requests are rejected before reaching the inference engine.
   The gateway handles authorization (rate limits, model access, token budgets);
   rLLM handles identity verification.

**Auth hook system.**  rLLM provides a pluggable `AuthProvider` trait with
three hooks:

- **Init** — called once at startup with the provider-specific config from
  `auth.json`.  This is where the provider fetches JWKs, loads keys, and sets
  up caching.
- **Request** — called on every HTTP request.  Receives the raw headers,
  returns `Allow(User)` or `Deny(Reason, StatusCode)`.
- **Background** — optional long-running task for maintenance work (JWKS
  rotation, cache refresh).  Spawned after init, runs for the server's lifetime.

Two built-in providers:
- **OIDC** — fetches the issuer's `/.well-known/openid-configuration`, caches
  the JWKS, and validates JWT Bearer tokens on each request.
- **Static API Key** — compares the Bearer token against an argon2id hash stored
  in the config file.  Constant-time via argon2id verify (immune to timing
  attacks).  Supports hot reload — the background task watches the config file
  every 30 seconds and swaps the hash if it changes (key rotation without restart).
  Suboptimal for production: single shared key, no per-user identity, no token expiry.

Custom providers can be added by implementing the trait and adding a variant to
`AuthProviderKind`.

**What OIDC auth does NOT provide:**
- Authorization — all authenticated users have equal access
- Rate limiting or token budgets — still a gateway concern
- Audit logging of prompt content — only caller identity is logged to stderr

**What the deployment must provide:**
- Network isolation — inference servers on a private network, only the gateway
  can reach the inference port
- API gateway — enforces rate limits, routes to backends, optionally does token
  exchange to mint scoped JWTs for rLLM
- Authorization policy — model access, token budgets, per-customer limits

**Auth without TLS.**  When auth is enabled but TLS is disabled
(auth enabled, no TLS), rLLM prints a startup warning.
Bearer tokens, prompts, and completions are sent in plaintext.  An attacker
with network access (man-in-the-middle) can:

- **Intercept tokens** — steal a valid JWT and impersonate the user
- **Read prompts and completions** — observe all inference traffic
- **Modify requests and responses** — alter prompts before they reach the
  server or tamper with completions before the client sees them

This is safe when rLLM is accessed over localhost or an SSH tunnel (the
transport is already encrypted), but dangerous on any network an attacker can
observe.  If binding to a non-localhost address with auth enabled, always
enable TLS (`--cert`/`--private-key` or `--letsencrypt`).

**Safety nets on external interfaces.**  When binding to a non-loopback
address (`--host 0.0.0.0`), rLLM requires explicit opt-in for running
without TLS (`--dangerous-no-tls`) and without auth (`--dangerous-no-auth`).
On localhost (the default), neither flag is needed — traffic never leaves
the machine.

---

### Tampering

| Threat | What is tampered | rLLM's stance |
|--------|-----------------|---------------|
| Modified weight files on disk | Model produces wrong or poisoned outputs | **rLLM does not verify weight integrity.** |
| Modified config/tokenizer files | Broken tokenization, wrong model behaviour | **No integrity checks.** |
| Man-in-the-middle on the inference API | Prompts or completions altered in transit | **TLS supported but optional.** Auth tokens also exposed without TLS. |
| Tampered safetensors index | Wrong shard loaded, potential crash | **Index trusted as-is.** |

**Weight integrity.**  rLLM loads weights from local safetensors files via
`mmap` and `pread`.  It does not verify checksums, signatures, or provenance.
The safetensors format itself is safe — unlike pickle, it cannot execute
arbitrary code — but rLLM trusts that the bytes on disk are correct.

**Why rLLM doesn't verify weights.**  Weight integrity is an infrastructure
concern:
- Weights are cloned from an internal model registry with checksums verified
  at transfer time
- Local NVMe is encrypted at the OS layer (LUKS, FileVault, cloud KMS)
- No standing SSH access — filesystem modifications require break-glass
  approval
- Adding signature verification to `mmap`/`pread` paths would add latency
  to every weight read, especially for [expert streaming](expert-streaming.md)
  where individual experts are loaded from SSD per token

**TLS.**  rLLM supports TLS via manual certificates (`--tls-cert`, `--tls-key`)
or automatic Let's Encrypt provisioning.  On a private network behind a
gateway, TLS between gateway and inference server may be unnecessary — the
network is already trusted.  For deployments where the gateway-to-inference
link crosses an untrusted boundary, enable TLS.

**What the deployment must provide:**
- Filesystem integrity — encrypted volumes, restricted access, immutable
  deployment images
- Weight provenance — verify checksums when cloning from the model registry
- TLS termination at the gateway for client-facing traffic

---

### Repudiation

| Threat | What is denied | rLLM's stance |
|--------|---------------|---------------|
| No record of who sent a request | Cannot attribute inference to a customer | **When auth is enabled, caller identity is logged to stderr per-request.** Without auth, no identity is logged. |
| No record of what was generated | Cannot investigate harmful outputs | **rLLM does not log prompts or completions.** |

**Logging.**  rLLM logs operational metrics to stderr: model loading progress,
per-request token counts, latency, and throughput.  When auth is enabled
(`--auth-config`), the authenticated user identity is included inline in the
per-request log line alongside token counts and latency.  rLLM does **not**
log prompt content or generated text.

**Why rLLM doesn't log prompts.**  Prompt and completion logging is a policy
decision with privacy and compliance implications.  The inference engine
shouldn't decide what to retain — that belongs in the gateway and audit
infrastructure.  rLLM reports token counts, timing, and (when auth is enabled)
caller identity; the gateway stores full audit records.

**What the deployment must provide:**
- Gateway-side audit logging — request details, billing
- Inference-side event forwarding — rLLM's stderr metrics shipped to a log
  aggregator for cross-referencing with gateway logs
- Log retention policy — how long to keep, what to redact, compliance
  requirements

---

### Information Disclosure

This is the most significant threat category.  Two assets are at risk:
**model weights** and **customer data** (prompts + completions).

#### Weight theft

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Read weights from disk | Filesystem access to safetensors files | **Weights stored unencrypted in application layer.** |
| Exfiltrate weights over the network | Compromised server sends weights to attacker | **rLLM makes no outbound connections.** |
| Extract weights via model extraction attack | Repeated API queries to reconstruct weights | **No protection — API gateway must rate-limit.** |
| Read weights from GPU memory | Side-channel or co-tenant attack | **No GPU memory isolation beyond OS defaults.** |

**rLLM stores weights as plaintext safetensors on local NVMe.** This is
deliberate — `mmap` and `pread` require unencrypted bytes for zero-copy and
direct-I/O weight loading.  Application-level encryption would break expert
streaming (which `pread`s individual expert tensors from arbitrary file
offsets) and add per-read decryption overhead to every forward pass.

**Encryption happens at the infrastructure layer:**
- Full-disk encryption (LUKS, FileVault, cloud KMS-managed volumes) makes
  weights unreadable without the decryption key bound to the server's service
  account
- Weights stored encrypted at rest in the model registry; decrypted only during
  the JIT clone window
- No standing network egress — a compromised server cannot send weights
  anywhere.  The only persistent outbound path is completions back to the
  gateway, monitored for anomalous volume

**Model extraction attacks** (querying the API to reconstruct weights) are
mitigated by the gateway: rate limiting, token budgets, and abuse detection.
rLLM has no visibility into request patterns across customers.

#### Customer data exposure

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Prompts logged to disk | stderr, log files | **rLLM does not log prompt content.** |
| Completions persisted in memory | KV cache, GPU buffers | **Freed after sequence completes; not zeroed.** |
| Cross-sequence data leakage via KV cache | Paged block reused by another sequence | **Blocks freed and reallocated; stale reads caught by generational indices.** |
| Prompt visible in error messages | Inference errors returned to client | **Error messages may include model-internal state; not sanitised.** |

**KV cache isolation.**  rLLM's paged KV cache reuses physical blocks across
sequences.  When a sequence finishes, its blocks return to the free list and
may be allocated to a new sequence.  The new sequence overwrites the blocks
during prefill — but between free and reallocation, stale K/V data remains
in GPU memory.  rLLM does not zero freed blocks.

[Generational indices](kv-cache.md#generational-indices) prevent a different
class of bug: a sequence reading blocks that were freed and reallocated to
someone else.  Each block handle carries a generation counter; stale handles
panic before reaching the GPU kernel.  This catches use-after-free bugs in
the block table, not data remanence in GPU memory.

**Prompt caching shares KV blocks** across sequences with identical prefixes.
This is by design — the shared blocks contain the same data (deterministic
KV from identical tokens).  There is no cross-customer data leakage because
shared blocks are read-only and only match on exact token equality.  See
[Prompt Caching — Correctness](prompt-caching.md#correctness).

**What the deployment must provide:**
- TLS for prompt/completion transit (gateway terminates client TLS; optionally
  TLS between gateway and inference)
- Prompt/completion retention policy enforced at the gateway layer
- GPU isolation if running multi-tenant on shared hardware (not an rLLM
  concern — use separate processes or VMs)

---

### Denial of Service

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Unbounded prompt length | Client sends millions of tokens | **No hard limit on prompt length.** KV block admission gates memory but not compute. |
| Unbounded generation length | `max_tokens` has no server-side cap | **Accepts any `max_tokens` value from the client.** |
| Concurrent request flood | Thousands of simultaneous requests | **Bounded by channel capacity and KV block availability.** No explicit rate limiting. |
| Large image uploads | Oversized images consume CPU during preprocessing | **Image pixel count bounded by model config** (typically 1008–262144 pixels). |
| Slowloris / connection exhaustion | Hold TCP connections open without sending | **Tokio's async accept loop handles this reasonably**, but no explicit connection timeout. |

**rLLM relies on the gateway for rate limiting and request validation.**
The inference server accepts whatever arrives on its port.  The scheduler's
KV block admission check provides natural backpressure — if GPU memory is
full, new sequences wait — but this doesn't prevent compute-bound attacks
like submitting prompts that are expensive to prefill.

**What the deployment must provide:**
- Request rate limiting at the gateway (per-customer, per-model)
- `max_tokens` cap at the gateway (enforce a server-side maximum regardless of
  client request)
- Prompt length limits at the gateway
- Connection timeouts and concurrent connection limits at the load balancer

---

### Elevation of Privilege

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Arbitrary code execution via weight files | Malicious model files | **Safetensors format cannot execute code.** No pickle, no ONNX, no arbitrary deserialization. |
| Path traversal via model directory | Attacker-controlled filenames in safetensors index | **Filenames joined to `model_dir`; constrained by `Path::join`.** |
| GPU kernel exploitation | Malformed tensors trigger GPU driver bugs | **Tensors validated by shape/dtype before upload.** GPU driver bugs are out of scope. |
| Escalation from inference to other services | Compromised inference server pivots to internal services | **rLLM makes no outbound connections.** Network isolation prevents lateral movement. |

**Safetensors is the key design choice.**  rLLM only loads safetensors files —
a flat binary format of tensor metadata (JSON header) and raw numeric data.
Unlike Python pickle (used by PyTorch `.bin` files), safetensors cannot
contain executable code.  This eliminates the most common model-supply-chain
attack vector.

**rLLM's process has no special privileges.**  It needs read access to weight
files and GPU device access.  It does not need root, network access, or write
access to anything beyond its own scratch space.

---

## What rLLM Does vs What the Deployment Does

| Concern | rLLM | Deployment / Gateway |
|---------|------|---------------------|
| Authentication | Optional auth hook via `--auth-config` (OIDC JWT validation); disabled by default, binds to localhost | Gateway authenticates callers, optionally mints scoped JWTs for rLLM |
| Authorisation | None — all authenticated users have equal access | Gateway enforces model access, rate limits, token budgets |
| Encryption in transit | TLS supported (optional) | Gateway terminates client TLS; internal TLS if needed |
| Encryption at rest | None (application layer) | Full-disk encryption, registry-level encryption |
| Weight integrity | Trusts local filesystem | Checksums at clone time, immutable deploys, no standing SSH |
| Audit logging | Per-user metrics to stderr when auth is enabled (identity, token counts, latency) | Gateway logs request details, billing |
| Rate limiting | None | Gateway enforces per-customer limits |
| Input validation | Image pixel bounds from model config | Gateway enforces prompt length, max_tokens caps |
| Network isolation | OIDC provider makes outbound HTTPS to issuer's JWKS endpoint (at startup + periodic refresh); no other outbound connections | Private network, no internet, JIT registry access |
| Prompt/completion privacy | Does not log content | Gateway controls retention policy |

**The design principle:** rLLM is a compute engine.  It should be fast, correct,
and small-surface-area.  Most security concerns are pushed to infrastructure.
Authentication is the exception — optional, pluggable auth hooks let rLLM
validate identity end-to-end without coupling it to a specific auth provider.
The hook system is designed so an org's existing auth infrastructure can be
integrated with minimal effort.

---

## Residual Risks

Things that network isolation and a gateway do not fully address:

1. **Insider threat.**  An engineer with break-glass access can read weights
   from disk.  Mitigated by two-person approval, session logging, time-limited
   access, and YubiKey-backed auth.  Not eliminated.

2. **GPU memory remanence.**  Freed KV cache blocks are not zeroed.  A
   co-tenant on shared GPU hardware could theoretically read stale data.
   Mitigated by running single-tenant (one model per GPU process).  Not
   relevant for dedicated hardware.

3. **Model extraction via API.**  Enough queries can approximate model weights.
   Mitigated by rate limiting, token budgets, and usage monitoring at the
   gateway.  Fundamentally unsolvable — any API that exposes model outputs
   leaks information about weights.

4. **Supply chain.**  Compromised dependencies (Rust crates, system libraries)
   could introduce backdoors.  Mitigated by `cargo audit`, pinned
   dependencies, and reproducible builds.  Note: the OIDC auth provider adds
   `reqwest` (HTTP client) and `jsonwebtoken` to the dependency tree.  When
   auth is disabled (default), these crates are compiled but never invoked —
   the binary makes no outbound connections.

5. **Error message leakage.**  Inference errors are returned to the client
   unsanitised.  Could expose internal tensor shapes, layer names, or memory
   state.  Low severity but worth sanitising in a production gateway.

---

See also: [Authentication](authentication.md) ·
[Production Considerations](production-considerations.md) ·
[KV Cache](kv-cache.md) · [Expert Streaming](expert-streaming.md) ·
[Prompt Caching](prompt-caching.md)
