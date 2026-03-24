# Threat Model

STRIDE analysis of rLLM's threat surface.  rLLM is a local inference
engine — it turns prompts into completions on a GPU.  This document
explains what rLLM protects against, what it doesn't, and why.

**Primary assets:**
- **Model weights** — stored on local disk
- **Prompts and completions** — transit through the inference server

---

## STRIDE Analysis

### Spoofing

| Threat | Who is spoofed | rLLM's stance |
|--------|---------------|---------------|
| Unauthenticated requests reach the server | The server treats any TCP connection as a legitimate caller | **rLLM does not authenticate requests.** |

**rLLM has no auth** and doesn't need it.  It's a local inference engine —
it binds to `127.0.0.1` by default and only serves requests from the local
machine.  If you use `--host 0.0.0.0` on a trusted network, anyone on that
network can use it.

---

### Tampering

| Threat | What is tampered | rLLM's stance |
|--------|-----------------|---------------|
| Modified weight files on disk | Model produces wrong or poisoned outputs | **rLLM does not verify weight integrity.** |
| Modified config/tokenizer files | Broken tokenization, wrong model behaviour | **No integrity checks.** |
| Man-in-the-middle on the inference API | Prompts or completions altered in transit | **TLS supported but optional.** |
| Tampered safetensors index | Wrong shard loaded, potential crash | **Index trusted as-is.** |

**Weight integrity.**  rLLM loads weights from local safetensors files via
`mmap` and `pread`.  It does not verify checksums, signatures, or provenance.
The safetensors format itself is safe — unlike pickle, it cannot execute
arbitrary code — but rLLM trusts that the bytes on disk are correct.

**Why rLLM doesn't verify weights.**  Adding signature verification to
`mmap`/`pread` paths would add latency to every weight read, especially for
[expert streaming](expert-streaming.md) where individual experts are loaded
from SSD per token.  For a local setup, you trust your own filesystem.

**TLS.**  rLLM supports TLS via manual certificates (`--tls-cert`, `--tls-key`)
or automatic Let's Encrypt provisioning.  For local use, plain HTTP on
localhost is fine (`--dangerous-no-tls`).

---

### Repudiation

| Threat | What is denied | rLLM's stance |
|--------|---------------|---------------|
| No record of who sent a request | Cannot attribute inference to a caller | **rLLM does not log request identity.** |
| No record of what was generated | Cannot investigate harmful outputs | **rLLM does not log prompts or completions.** |

**Logging.**  rLLM logs operational metrics to stderr: model loading progress,
per-request token counts, latency, and throughput.  It does **not** log prompt
content, generated text, or caller identity.

---

### Information Disclosure

Two assets are at risk: **model weights** and **prompts + completions**.

#### Weight theft

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Read weights from disk | Filesystem access to safetensors files | **Weights stored unencrypted in application layer.** |
| Exfiltrate weights over the network | Compromised server sends weights to attacker | **rLLM makes no outbound connections.** |
| Extract weights via model extraction attack | Repeated API queries to reconstruct weights | **No protection — no rate limiting.** |
| Read weights from GPU memory | Side-channel or co-tenant attack | **No GPU memory isolation beyond OS defaults.** |

**rLLM stores weights as plaintext safetensors on local disk.** This is
deliberate — `mmap` and `pread` require unencrypted bytes for zero-copy and
direct-I/O weight loading.  Application-level encryption would break expert
streaming (which `pread`s individual expert tensors from arbitrary file
offsets) and add per-read decryption overhead to every forward pass.

For local use, your OS-level disk encryption (FileVault, LUKS) is sufficient.

#### Prompt/completion exposure

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
KV from identical tokens).  There is no cross-sequence data leakage because
shared blocks are read-only and only match on exact token equality.  See
[Prompt Caching — Correctness](prompt-caching.md#correctness).


---

### Denial of Service

| Threat | Vector | rLLM's stance |
|--------|--------|---------------|
| Unbounded prompt length | Client sends millions of tokens | **No hard limit on prompt length.** KV block admission gates memory but not compute. |
| Unbounded generation length | `max_tokens` has no server-side cap | **Accepts any `max_tokens` value from the client.** |
| Concurrent request flood | Thousands of simultaneous requests | **Bounded by channel capacity and KV block availability.** No explicit rate limiting. |
| Large image uploads | Oversized images consume CPU during preprocessing | **Image pixel count bounded by model config** (typically 1008–262144 pixels). |
| Slowloris / connection exhaustion | Hold TCP connections open without sending | **Tokio's async accept loop handles this reasonably**, but no explicit connection timeout. |

rLLM accepts whatever arrives on its port.  The scheduler's KV block
admission check provides natural backpressure — if GPU memory is full, new
sequences wait — but there's no rate limiting or request validation.  For
local use this is fine; if you expose the port on a network, be aware that
anyone who can reach it can submit prompts.

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

## Summary

| Concern | rLLM's stance |
|---------|---------------|
| Authentication | None — binds to localhost by default |
| Encryption in transit | TLS supported but optional; localhost doesn't need it |
| Encryption at rest | Relies on OS-level disk encryption |
| Weight integrity | Trusts local filesystem |
| Logging | Metrics to stderr (token counts, latency); no prompt content |
| Rate limiting | None |
| Input validation | Image pixel bounds from model config |
| Network access | Makes no outbound connections |

**The design principle:** rLLM is a local compute engine.  It should be fast,
correct, and small-surface-area.  Security concerns like auth and rate limiting
don't apply to a local inference server.

---

## Residual Risks

1. **GPU memory remanence.**  Freed KV cache blocks are not zeroed.  Stale
   K/V data remains in GPU memory until overwritten by the next sequence.
   Not relevant for single-user local use.

2. **Supply chain.**  Compromised dependencies (Rust crates, system libraries)
   could introduce backdoors.  Mitigated by `cargo audit`, pinned
   dependencies, and reproducible builds.  rLLM's Cargo.toml has no outbound
   HTTP client dependencies — the binary cannot phone home.

3. **Error message leakage.**  Inference errors are returned to the client
   unsanitised.  Could expose internal tensor shapes, layer names, or memory
   state.  Low severity for local use.

---

See also: [KV Cache](kv-cache.md) · [Expert Streaming](expert-streaming.md) ·
[Prompt Caching](prompt-caching.md)
