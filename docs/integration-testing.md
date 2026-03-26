# Integration Testing

Why rLLM has GPU integration tests, what they cover, and when to run them.

For usage instructions, command reference, and file layout, see
[tests/README.md](../tests/README.md).

---

## The Problem

LLM inference engines have a long, fragile pipeline:

```
Weights on disk
  → safetensors parsing + dtype conversion
  → weight upload to GPU
  → tokenizer encodes prompt
  → chat template formatting
  → paged KV cache allocation
  → prefill (fused attention kernels)
  → decode loop (matmul, RoPE, attention, FFN, sampling)
  → detokenization
  → HTTP response serialization (JSON or SSE)
```

A bug at any stage can produce output that *looks* normal at a glance —
the server responds 200, tokens come back, latency is fine — but the text
is nonsensical.  Silent correctness failures are the hardest bugs to catch
and the most dangerous to ship.

Unit tests for individual kernels help, but they can't cover the interactions
between stages.  A correct RMSNorm kernel paired with a wrong weight layout
produces garbage.  A working attention kernel with a broken chat template
produces fluent text in response to the wrong prompt.  You need end-to-end
tests that validate the *output*, not just the components.

---

## What the Tests Prove

If the GPU integration tests pass, you know:

1. **Every model family loads and runs.**  Each of the 9 architectures
   (Llama, Qwen 2.5, Gemma 3, Mistral, Phi, Mixtral, Qwen3-MoE, Qwen 3.5,
   GPT-OSS) can load weights, process a chat prompt, and produce output
   through the full decode loop.

2. **The output is coherent English.**  Not just non-empty — actually
   detected as English by a language classifier, with no degenerate
   repetition or encoding corruption.

3. **The HTTP API works.**  Both OpenAI and Anthropic endpoints return
   correctly structured responses.  SSE streaming delivers incremental
   chunks.  Usage counts (prompt_tokens, completion_tokens) are populated.

4. **KV cache quantization doesn't break output.**  TurboQuant at 4-bit
   and 2-bit both produce coherent text, validating the rotation matrices,
   quantization codebooks, and dequantization kernels.

5. **Expert streaming works for MoE models.**  When a model is too large
   for VRAM, the SSD-backed expert loading path produces the same quality
   output as full-VRAM inference.

6. **Multi-GPU sharding works.**  On multi-GPU systems, tensor parallelism
   is exercised automatically.

---

## When to Run Them

**After kernel changes.**  Modified a Metal or CUDA shader?  Run the tests
for the affected model families.  `pytest tests/ -v -k llama` takes under
a minute with a 1B model.

**After weight loading changes.**  Touched `loader.rs`, `config.rs`, or
added a new model?  Run the full suite.  Weight layout bugs are the #1 cause
of silent correctness regressions.

**After chat template changes.**  Modified `chat.rs`?  The tests validate
that the model produces coherent responses to chat-formatted prompts, which
catches template formatting bugs.

**After KV cache or quantization changes.**  The turboquant tests specifically
exercise the 4-bit, 2-bit, and BF16 KV cache paths.

**Before releases.**  `tests/run.sh --medium` exercises all model families
including MoE.  This is the closest thing to "does the product work?"

**After merging PRs.**  A quick `tests/run.sh --small` on the small tier
takes 5–15 minutes depending on hardware and gives high confidence that
nothing is broken.

---

## Design Decisions

**Python, not Rust.**  The tests validate output quality, not internal
correctness.  Language detection, text heuristics, and HTTP client code
are easier and more readable in Python.  The Rust unit tests remain for
kernel-level validation.

**HTTP API, not CLI.**  The tests start `rllm serve` and hit the actual
API endpoints.  This is the closest to how users interact with rLLM in
production.  It also exercises the HTTP layer, JSON serialization,
SSE streaming, and the async worker architecture.

**One model per family.**  Each architecture has unique code paths (attention
patterns, normalization, weight formats, chat templates).  Testing the
smallest model from each family provides coverage without requiring
terabytes of weights.

**Coherence, not exact match.**  LLM output is non-deterministic across
hardware and quantization levels.  Instead of golden-file comparison, the
tests check that output is coherent English using `langdetect` plus simple
heuristics.  This is robust across platforms and model versions.

**512 tokens, not 10.**  Short outputs can be coherent by accident.
Generating 512 tokens (~400 words) exercises the KV cache across hundreds
of decode steps and surfaces issues that only appear in sustained generation
(cache overflow, attention drift, repetition collapse).

**Server reuse.**  Model loading is slow (10–60 seconds).  The
`ServerManager` caches running servers by configuration key so multiple
tests against the same model don't pay the startup cost repeatedly.

**Automatic expert streaming decision.**  MoE models are tested with or
without `--stream-experts` depending on available memory.  The harness
detects GPU memory and makes the call automatically — the same model
exercises different code paths on different machines.

---

## Benchmarking

The test infrastructure doubles as a benchmark harness.  Running
`tests/run.sh --bench` benchmarks every model in the models directory,
measuring TTFT and tok/s through the HTTP API.  Results are saved
as timestamped markdown files in `results/`.

This complements `scripts/benchmark.sh` (which measures via the CLI)
by capturing the full HTTP round-trip latency that real users experience.
See [tests/README.md](../tests/README.md#benchmarking) for usage.

---

## Relationship to Other Testing

| Layer | Tool | What It Validates |
|-------|------|-------------------|
| Kernel unit tests | `#[cfg(test)]` in Rust | Individual shader/op correctness |
| CPU reference backend | `src/gpu/cpu/` | Numerical agreement with Metal/CUDA |
| **Integration tests** | **`tests/`** | **End-to-end output coherence via HTTP API** |
| Benchmarks (CLI) | `scripts/benchmark.sh` | Performance (tok/s, TTFT) via CLI |
| Benchmarks (HTTP) | `tests/run.sh --bench` | Performance (tok/s, TTFT) via HTTP API |
