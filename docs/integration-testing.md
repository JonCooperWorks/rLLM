# Integration Testing and Benchmarking

Testing infrastructure for validating that rLLM produces correct, coherent output
across all model families, quantization levels, and API formats -- and for measuring
how fast it does so.

The test suite lives in `tests/` and exercises the full stack: weight loading, GPU
kernels, KV cache, the HTTP API layer, and streaming.  No mocking -- every test
starts `rllm serve`, sends real HTTP requests, and validates real model output.

**Files:**
```
tests/
  test_model_families.py  -- GPU integration tests (pytest)
  coherence.py            -- output quality validation (language, repetition, encoding)
  conftest.py             -- fixtures: server lifecycle, GPU detection, memory estimation
  bench.py                -- HTTP API benchmark (TTFT + tok/s + quality)
  run.sh                  -- all-in-one: build, download, quantize, test or bench
  requirements.txt        -- Python deps (requests, langdetect, psutil, pytest)

scripts/
  benchmark.sh            -- CLI benchmark (raw throughput, no HTTP overhead)
```

---

## GPU Integration Tests

### What they test

Every model family that rLLM supports gets tested end-to-end.  The test matrix covers:

- **All architectures** -- Llama, Qwen 2.5, Gemma 3, Mistral, Mixtral, Phi, Qwen 3 MoE,
  Qwen 3.5 (hybrid DeltaNet+GQA), GPT-OSS, NemotronH.  One representative model per
  family, smallest available, defined in the `BASE_MODELS` list.

- **Three quantization levels** -- bf16, Q4 (4-bit symmetric), and Q8 (8-bit).  Each
  variant gets a different prompt so we exercise distinct generation patterns per
  quantization level.

- **TurboQuant KV cache modes** -- turbo4 (default 4-bit), turbo2 (aggressive 2-bit),
  and none (bf16 baseline).  Tested on Llama 1B for fast iteration.  512 tokens
  exercises the quantized KV cache across many decode steps.

- **API formats** -- OpenAI `/v1/chat/completions` (streaming and non-streaming) and
  Anthropic `/v1/messages` (streaming and non-streaming).  Validates response structure,
  usage statistics, and SSE chunk delivery.

- **Vision (VLM)** -- Models with `has_vision=True` (Gemma 3, Qwen 3.5) receive a
  synthetic 128x128 PNG with red/green/blue/white quadrants.  The test validates the
  model identifies at least one colour, confirming the full pipeline works: image
  preprocessing, patch embedding, ViT encoder, spatial merge, projector, LLM generation.

- **Expert streaming** -- MoE models that exceed available memory automatically get
  `--stream-experts`, loading experts from SSD on demand.  The decision is made by
  `_should_stream_experts()` in conftest.py, which compares model size against 80% of
  available GPU/system memory.

- **Varied prompts** -- Ten different prompts (factual, expository, enumeration,
  reasoning, creative, technical, step-by-step, summary, multi-part, analysis) are
  rotated across models so we exercise different generation patterns rather than
  testing the same completion path repeatedly.

### How to run

Full suite (builds rLLM, downloads models if needed, quantizes to Q4, runs tests):

```bash
tests/run.sh
```

Just the tests (assumes binary and models are already in place):

```bash
tests/run.sh --skip-build --skip-download --skip-quantize
```

Single family:

```bash
pytest tests/ -v -k llama
```

Only Q4 variants:

```bash
pytest tests/ -v -k q4
```

TurboQuant tests only:

```bash
pytest tests/ -v -k turboquant
```

Vision tests only:

```bash
pytest tests/ -v -k vision
```

**Environment variables:**
- `RLLM_MODELS_DIR` -- model directory (default: `models/`)
- `RLLM_BIN` -- path to rllm binary (default: `target/release/rllm`)
- `HF_TOKEN` -- HuggingFace token for gated model downloads

### Server lifecycle

Tests share a session-scoped `ServerManager` (conftest.py) that keeps one `rllm serve`
process alive at a time.  When a test needs a different model, the manager stops the
current server and starts a new one.  If the same model+args are requested, the existing
server is reused with a `/health` check to confirm it is still alive.  This avoids
the overhead of restarting for every test while preventing OOM from running multiple
large models simultaneously.

The manager handles:
- Free port allocation (bind to port 0, let the OS assign)
- Startup health polling with configurable timeout (default 300s for large models)
- Graceful shutdown (SIGTERM with 15s grace, then SIGKILL)
- Automatic `--tp <gpu_count>` for multi-GPU tensor parallelism
- stderr capture on failure for debugging

### Coherence validation

Every test validates output quality using `coherence.py`, a lightweight local checker
that requires no external LLM.  Five checks run in sequence:

1. **Minimum length** -- at least 20 characters of content (catches silent failures
   where the model generates zero or near-zero tokens).

2. **ASCII ratio** -- at least 70% printable ASCII (catches encoding corruption, e.g.
   from broken bf16 dequantization producing garbage bytes).

3. **Degenerate repetition** -- two-tier trigram analysis:
   - *Stuttering* -- the same word repeated 3 times (e.g. "server server server")
     appearing 2+ times is always degenerate.
   - *General repetition* -- a length-scaled threshold (`max(6, word_count // 40)`)
     catches repetitive loops while allowing natural phrase reuse in long outputs.
   - Markdown formatting tokens (`|`, `---`, `*`, etc.) are filtered out before
     analysis so table-heavy output does not trigger false positives.

4. **Word salad** -- detects broken inference (e.g. corrupted Q4 weights) that produces
   concatenated word fragments without spaces.  Checks both the ratio of words exceeding
   20 characters (>15% is word salad) and overall space ratio (<10% for text longer than
   100 characters).

5. **Language detection** -- uses `langdetect` to confirm the output is English.  Catches
   models that degenerate into another language or produce statistically random tokens.

### Design decisions

**Python, not Rust.**  The tests validate output quality, not internal correctness.
Language detection, text heuristics, and HTTP client code are easier and more readable
in Python.  Rust unit tests remain for kernel-level validation.

**HTTP API, not CLI.**  Tests start `rllm serve` and hit the actual API endpoints.
This exercises the HTTP layer, JSON serialization, SSE streaming, and the async worker
architecture -- the same path real users exercise in production.

**One model per family.**  Each architecture has unique code paths (attention patterns,
normalization, weight formats, chat templates).  Testing the smallest model from each
family provides coverage without requiring terabytes of weights.

**Coherence, not exact match.**  LLM output is non-deterministic across hardware and
quantization levels.  Instead of golden-file comparison, the tests check that output is
coherent English using langdetect plus heuristics.  This is robust across platforms
and model versions.

**512 tokens, not 10.**  Short outputs can be coherent by accident.  Generating 512
tokens (~400 words) exercises the KV cache across hundreds of decode steps and surfaces
issues that only appear in sustained generation (cache overflow, attention drift,
repetition collapse).

---

## Benchmarks

Two complementary tools measure performance at different levels of the stack.

### scripts/benchmark.sh -- CLI throughput

Measures raw inference speed using `rllm run` (no HTTP server, no network overhead).
This isolates GPU kernel performance and weight loading from API/networking concerns.

**What it measures:**
- Generation throughput (tok/s) -- decode phase, the steady-state speed users experience
- Prefill throughput (tok/s) -- prompt processing speed
- TTFT (time to first token) -- parsed from Rust's timing output on stderr

**Usage:**

```bash
# Small models (1B-8B), bf16 + Q4
scripts/benchmark.sh --small

# All tiers
scripts/benchmark.sh --all

# Q4 only (skip bf16, useful when models don't fit in VRAM as bf16)
scripts/benchmark.sh --q4-only

# Multiple runs, averaged
scripts/benchmark.sh --runs 3

# Custom prompt and token count
RLLM_BENCH_PROMPT="Explain quantum computing" RLLM_BENCH_TOKENS=256 scripts/benchmark.sh
```

**Output:** Markdown table with bf16 and Q4 columns for both tok/s and TTFT, suitable
for pasting into the README.

**Model tiers:**
- `--small` -- 1B to 8B (Llama, Qwen, Gemma, Mistral)
- `--medium` -- adds Phi-4, Qwen 3.5, Mixtral 8x7B, GPT-OSS, NemotronH, DeepSeek
- `--big` -- 70B+ (Llama 70B, Qwen 72B, Mixtral 8x22B, Qwen 3.5 122B)
- `--massive` -- 300B+ (Qwen 3.5 397B)
- `--all` -- everything

### tests/bench.py -- HTTP API benchmark

Measures end-to-end performance through the HTTP API, including server startup, SSE
streaming, and response parsing.  This reflects what real users experience when calling
the OpenAI-compatible endpoint.

**What it measures:**
- TTFT (time to first token) -- measured via SSE streaming, from request to first
  `content` delta
- Generation throughput (tok/s) -- `completion_tokens / generation_duration`
- Quality -- coherence check on every generated response (same checks as the test suite)

**Usage:**

```bash
# Benchmark all discovered models
python tests/bench.py

# Filter to specific models
python tests/bench.py --filter qwen3.5

# Q4 only, 3 runs averaged, 256 tokens
python tests/bench.py --q4-only --runs 3 --max-tokens 256

# Custom prompt
python tests/bench.py --prompt "Write a poem about Rust"

# Save results to file (auto-generates results/bench-TIMESTAMP.md)
python tests/bench.py --output
```

**Output:** Markdown table with Model, Family, tok/s, TTFT, and Quality columns.
Results are always saved to a file (auto-named under `results/` by default).

**Key options:**
- `--models-dir PATH` -- where to find models (default: `models/`)
- `--binary PATH` -- rllm binary (default: `target/release/rllm`)
- `--max-tokens N` -- tokens to generate per run (default: 128)
- `--runs N` -- runs per model, results averaged (default: 1)
- `--q4-only` / `--bf16-only` -- filter quantization variants
- `--filter PATTERN` -- substring match on model directory names
- `--output [PATH]` -- write markdown to file (auto-generates if no path given)

**Model discovery:** bench.py scans the models directory and matches against a built-in
`KNOWN_MODELS` registry that maps directory names to (family, is_moe, bf16_size_gb,
supports_stream_experts).  Unknown models are still benchmarked -- the registry just
provides metadata for the results table and streaming decisions.

### Thinking models and pseudo-streaming

Models with extended thinking (e.g. Qwen 3.5) emit `reasoning_content` deltas before
`content` deltas.  The thinking-aware streaming path collects all tokens, then emits
content as a burst.  bench.py handles this by:

- Recording TTFT as the time to the first `reasoning_content` delta (not the first
  `content` delta, which would be misleadingly late)
- Computing generation throughput from total wall time minus TTFT, rather than from
  inter-token timing (which would give absurd numbers due to the burst delivery)

### Using run.sh for benchmarks

The all-in-one runner supports benchmarking mode:

```bash
# Build + benchmark all models
tests/run.sh --bench

# Benchmark with options
tests/run.sh --bench --bench-filter mixtral --bench-runs 3 --bench-tokens 256
```

---

## Economics

Why these measurements matter beyond raw numbers.

### Throughput (tok/s) and cost per query

Higher generation throughput means more concurrent users per GPU instance.  If a model
generates at 100 tok/s instead of 50, each request occupies the GPU for half the time,
roughly doubling the number of requests the server can handle per second.  At cloud GPU
prices ($2-4/hr for an A100, $1-2/hr for Apple Silicon Mac Studios), this translates
directly to cost per query.

### TTFT and user experience

Time to first token is the perceived latency -- how long a user stares at a blank screen
before text starts appearing.  Even if generation is fast, a 3-second TTFT feels slow.
Prefill speed (processing the input prompt) dominates TTFT for long prompts.  Optimizing
prefill throughput (batched GEMM, fused attention kernels) directly improves the
interactive experience.

### Q4 quantization and hardware requirements

4-bit quantization reduces weight memory by 3-4x compared to bf16 (18 bytes per 32
weights vs 64 bytes).  This has two economic impacts:

- **Serve larger models on cheaper hardware** -- a 70B bf16 model needs ~140 GB, which
  requires multiple high-end GPUs.  At Q4, it fits in ~35-40 GB, running on a single
  GPU or a Mac Studio with 64 GB unified memory.

- **More concurrent sequences** -- less memory per model means more room for KV cache
  entries, which means more users can have active conversations simultaneously.

### TurboQuant KV cache compression

The KV cache grows linearly with sequence length and batch size.  At bf16, each token
costs `2 * n_layers * n_kv_heads * head_dim * 2 bytes` per sequence.  For a 70B model
with 80 layers, that is ~1.3 MB per token per sequence.  At 4K context with 8 concurrent
sequences, KV cache alone consumes ~40 GB.

TurboQuant compresses the KV cache 4x (turbo4) or 8x (turbo2), reducing that to ~10 GB
or ~5 GB respectively.  This means 4-8x more concurrent sequences per instance at the
same hardware cost, with quality-neutral output (validated by the TurboQuant integration
tests).

### Expert streaming and consumer hardware

Traditional MoE inference requires all expert weights in GPU memory.  Mixtral 8x22B
needs ~175 GB at Q4; Qwen 3.5 397B needs ~200 GB.  This demands $50,000+ multi-GPU
clusters.

Expert streaming loads only the active experts from SSD on demand, trading bandwidth
for memory.  A 397B model runs on a Mac Studio with 192 GB unified memory and a fast
NVMe drive, at 4-5 tok/s with Q4 weights.  Not fast enough for real-time chat, but
usable for batch processing and dramatically cheaper than the GPU cluster alternative.

The bf16 scale in the Q4 block format (2 bytes instead of 4) was specifically chosen to
reduce per-expert I/O by 10% -- critical when expert loading is the bottleneck.
