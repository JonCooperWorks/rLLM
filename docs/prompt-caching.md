# Prompt Caching

Prompt caching avoids redundant prefill computation by reusing KV cache blocks
across requests that share a common prefix.  When ten users send messages with
the same system prompt, the KV for that system prompt is computed once and
shared by all ten sequences.

**Key files:**
- `src/model/kv_cache.rs` — `PrefixCache`, `CachedPrefix`, ref counting
- `src/engine/mod.rs` — cache lookup/register in the prefill flow
- `src/engine/dispatch.rs` — `Dispatch` trait prefix caching methods

---

## The Mechanism

### What gets cached

KV cache data is computed during **prefill** — the phase where the model
processes the prompt's token sequence through all transformer layers.  The
output is a set of K and V vectors stored in physical blocks (16 tokens per
block).  These blocks are the cached artifact.

Prompt caching works at the **block boundary**.  A 100-token system prompt
occupies `ceil(100/16) = 7` blocks, but only the first 6 full blocks (96
tokens) are cacheable.  The last 4 tokens sit in a partial block that can't be
shared — the next sequence might have different tokens in positions 97–112.

### How lookup works

When a new request arrives, the engine checks the prompt against the prefix
cache before running prefill:

```
1. Hash progressive block-aligned prefixes of the prompt
   (longest first: try 6 blocks, then 5, then 4, ...)
2. On match: verify full token equality (hash collision safety)
3. Link the cached blocks into the new sequence's block table
4. Set seq_len = prefix_token_count (skip those positions)
5. Only prefill the remaining suffix tokens
```

The sequence's block table ends up looking like:

```
block_table: [shared₀, shared₁, ..., sharedₙ, own₀, own₁, ...]
              ←── from cache ──→  ←── freshly allocated ──→
```

The attention kernel sees a contiguous logical sequence — it doesn't know or
care that some blocks are shared.  The block table indirection makes this
transparent.

### Reference counting

Each cache entry tracks how many active sequences are using its blocks:

```
Insert  → ref_count = 1
Lookup  → ref_count += 1
Release → ref_count -= 1  (when sequence finishes or aborts)
```

Blocks cannot be freed while `ref_count > 0`.  When a cache entry is evicted
(LRU, only if `ref_count == 0`), its blocks are returned to the pool's free
list.

### Registration after prefill

After a full prefill (no cache hit), the engine registers the block-aligned
portion of the prompt in the prefix cache:

```
tokens:       [t₀, t₁, ..., t₉₅, t₉₆, t₉₇, t₉₈, t₉₉]
blocks:       [  block 0  ] [  block 1  ] ... [block 5] [block 6 (partial)]
cached:       [←──────── 96 tokens, 6 blocks ────────→]
not cached:   ← these 4 tokens are in a partial block →
```

This means the first request pays full prefill cost, but every subsequent
request with the same prefix pays only the suffix cost.

---

## Block Lifecycle with Caching

```
                                    ┌──────────────────┐
                                    │   PrefixCache    │
                                    │   (hash → entry) │
                                    └──────┬───────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────┐
         │                                 │                     │
    insert (after prefill)           lookup (before prefill)   evict (LRU)
         │                                 │                     │
         ▼                                 ▼                     ▼
  blocks held out of             block indices copied      blocks returned
  free list, ref=1               to new seq, ref++         to free list
```

### Sequence free (prefix-aware)

When a sequence finishes, `free_sequence()` only returns blocks that the
sequence **owns** — blocks allocated after the shared prefix:

```rust
// Skip the first `shared_prefix_blocks` — those belong to the cache.
for &block_idx in &seq.block_table_cpu[seq.shared_prefix_blocks..] {
    pool.free_block(block_idx);
}
```

The shared prefix blocks stay allocated.  They're freed only when the
`PrefixCache` evicts the entry.

---

## What Gets Cached in Practice

The cache is most effective for **repetitive prefixes** — content that appears
identically at the start of many requests:

| Use case | Typical prefix | Tokens | Cache benefit |
|----------|---------------|--------|---------------|
| System prompt | "You are a helpful assistant..." | 50–500 | High (every request) |
| Few-shot examples | System + 5 examples | 500–2000 | High (same examples) |
| Tool definitions | System + tool schemas | 200–1000 | High (same tools) |
| RAG context | System + retrieved docs | 500–4000 | Medium (docs vary) |
| Multi-turn chat | System + conversation history | Varies | Low (each turn differs) |

Multi-turn chat is the weakest case: each new user message changes the token
sequence, so only the common prefix up to the divergence point is reusable.
But for API workloads dominated by system prompts and tool definitions, the
cache hit rate can exceed 90%.

---

## Eviction

The cache uses **LRU eviction** with a fixed entry count (default: 64 prefixes).
When the cache is full and a new prefix needs to be stored:

1. Find the entry with the lowest `last_used` timestamp AND `ref_count == 0`
2. Remove it and return its blocks to the free list
3. Insert the new entry

If no entry has `ref_count == 0`, the cache grows beyond capacity temporarily.
This is safe — it just means more blocks are held by the cache than intended.

---

## Admission Control Interaction

The scheduler's admission check counts blocks needed for the **full prompt**,
not just the suffix.  This is conservative — if the prefix is cached, the
sequence needs fewer new blocks than estimated.  The slack is harmless: the
sequence is admitted with room to spare, and the "saved" blocks are never
allocated (they're borrowed from the cache instead).

A future optimisation could subtract the prefix block count from the admission
check when a cache hit is likely, allowing more concurrent sequences.

---

## API Surface

The cache is transparent to API clients — no request changes needed.  Cached
tokens are reported in the response:

- **Server log:** `seq 42 | 1024 prompt (896 cached) + 256 gen | TTFT 12 ms | ...`
- **`FinishedSequence.cached_tokens`:** propagated through the API layer

Clients see lower TTFT (time to first token) on cache hits because the suffix
prefill is shorter.  For a 1000-token system prompt with a 50-token user
message, the cache turns a 1050-token prefill into a 50-token prefill — a
~20× reduction in prefill compute.

---

## Correctness

Prompt caching is correct because:

1. **KV is deterministic** — the same tokens always produce the same K/V vectors
   at the same positions (given the same model weights)
2. **Blocks are read-only after prefill** — no sequence ever writes to a shared
   block; new tokens go into newly-allocated blocks
3. **Hash + full token verification** — FNV-1a hash for fast lookup, with full
   `tokens == stored_tokens` check to prevent hash collision corruption
4. **Block alignment** — only full blocks are shared; partial blocks are never
   cached, avoiding position-mismatch bugs

---

See also: [KV Cache](kv-cache.md) · [Inference Engine](inference-engine.md) ·
[Production Considerations](production-considerations.md)
