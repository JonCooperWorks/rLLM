# GPU Integration Tests

End-to-end smoke tests that start `rllm serve`, send real HTTP requests, and
verify the model produces coherent English.  Designed to catch regressions fast
after code changes — if a kernel breaks, a weight-loading bug slips in, or a
chat template changes, these tests will surface it.

---

## Why

Unit tests validate individual functions.  Benchmarks measure speed.  Neither
tells you whether the full pipeline — loading weights, building KV caches,
running prefill and decode, formatting API responses — actually produces
correct output after a change.

These tests fill that gap:

1. **Catch silent correctness regressions.**  A shader typo might produce
   output that looks fast but is garbage.  These tests assert the output is
   coherent English, not just non-empty.

2. **Cover every model family.**  Each architecture (Llama, Qwen, Gemma,
   Mistral, Phi, Mixtral, Qwen3-MoE, Qwen3.5, GPT-OSS) has its own
   code paths — different attention patterns, normalization, weight formats,
   chat templates.  One test per family.

3. **Exercise critical features.**  TurboQuant KV cache quantization (4-bit,
   2-bit, none), expert streaming for MoE models, both OpenAI and Anthropic
   API formats, SSE streaming.

4. **Realistic workloads.**  512 max tokens with varied prompts (expository,
   enumeration, comparison, technical, creative) — not toy 10-token outputs.

5. **Multi-GPU automatic.**  On multi-GPU systems, tests auto-detect and pass
   `--tp <count>` to shard models across all available GPUs.

---

## Quick Start

```bash
# One command: build, download models, quantize Q4, install deps, run tests.
tests/run.sh

# Just the small models (1B–9B, fastest):
tests/run.sh --small

# Include MoE and larger models:
tests/run.sh --medium

# If models are already downloaded:
tests/run.sh --skip-download --skip-quantize
```

Or run pytest directly (assumes binary built and models present):

```bash
# Install dependencies.
pip install -r tests/requirements.txt

# Run all tests.
pytest tests/ -v

# Run just one model family.
pytest tests/ -v -k llama

# Run just turboquant tests.
pytest tests/ -v -k turboquant

# Run just streaming tests.
pytest tests/ -v -k streaming

# Stop on first failure (saves GPU time).
pytest tests/ -v -x
```

---

## What Gets Tested

### Model Families (bf16 + Q4)

Each family is tested with its smallest available model.  Q4 variants are
tested if the quantized directory exists (created by `scripts/quantize-models.sh`).

| Family | Model | Size | Tier | Notes |
|--------|-------|------|------|-------|
| Llama | llama-3.2-1b-instruct | 1B | small | Dense, head_dim=64 |
| Qwen2 | qwen2.5-3b-instruct | 3B | small | QKV bias |
| Gemma3 | gemma-3-4b-it | 4B | small | Sliding window, GeGLU, sandwich norms |
| Mistral | mistral-7b-instruct-v0.3 | 7B | small | SentencePiece tokenizer |
| Qwen3.5 | qwen3.5-9b | 9B | small | Hybrid DeltaNet + GQA |
| Phi | phi-4 | 14B | medium | Fused QKV/gate_up weights |
| Mixtral | mixtral-8x7b-instruct-v0.1 | 46.7B | medium | MoE, 8 experts |
| Qwen3-MoE | qwen3-coder-30b-a3b-instruct | 30B | medium | MoE, 128 experts |
| Qwen3.5-MoE | qwen3.5-35b-a3b | 35B | medium | MoE, 256 experts |
| GPT-OSS | gpt-oss-20b | 20B | medium | MoE, 160 experts, MXFP4 |

### TurboQuant KV Cache Modes

Tested on Llama 1B (fastest to load):

| Mode | Bits | Compression | What It Validates |
|------|------|-------------|-------------------|
| `turbo4` | 4-bit | ~4x | Default mode, quality-neutral |
| `turbo2` | 2-bit | ~7.5x | Aggressive quantization still coherent |
| `none` | 16-bit | 1x | BF16 baseline, no quantization path |

### API Formats

- **OpenAI** `/v1/chat/completions` — non-streaming and SSE streaming
- **Anthropic** `/v1/messages` — non-streaming and SSE streaming

### Expert Streaming

MoE models automatically use `--stream-experts` when the model's estimated
size exceeds 80% of available GPU memory.  On machines with enough memory,
experts load into VRAM directly (testing both code paths depending on
your hardware).

---

## Prompts

Tests use 10 varied prompts to exercise different generation patterns:

| Style | Example |
|-------|---------|
| Short factual | "Explain what a hash table is in two sentences." |
| Expository | "Write a short paragraph explaining how the internet works..." |
| Enumeration | "List five common sorting algorithms..." |
| Comparison | "Compare and contrast Python and Rust..." |
| Creative | "Write a short story in exactly three sentences..." |
| Technical | "Explain how public-key cryptography works..." |
| Step-by-step | "Walk me through the steps to deploy a web application..." |
| Summary | "Summarise the key ideas behind MapReduce..." |
| Multi-part | "What is gradient descent? Why is the learning rate important?..." |
| Analysis | "What are the trade-offs between microservices and monolithic..." |

Each model family gets a different prompt.  bf16 and Q4 variants of the same
model use different prompts (offset by 5).  All tests generate up to **512 tokens**
for realistic KV cache pressure.

---

## Coherence Validation

Output is validated without an external LLM using four checks:

1. **Language detection** (`langdetect`) — must detect English.
2. **Minimum length** — at least 20 characters.
3. **Repetition check** — no 3-word sequence repeated 4+ times (catches
   degenerate token loops).
4. **ASCII ratio** — at least 70% printable ASCII (catches encoding garbage).

---

## How It Works

```
tests/run.sh
  ├── cargo build --release          Build rllm binary
  ├── scripts/download-models.sh     Download model weights from HuggingFace
  ├── scripts/quantize-models.sh     Quantize to Q4 (optional)
  ├── pip install requirements.txt   Install pytest, requests, langdetect, psutil
  └── pytest test_model_families.py  Run tests
        │
        ├── conftest.py: ServerManager
        │     ├── Detect GPUs (nvidia-smi / Metal)
        │     ├── Detect available memory (sum across GPUs for TP)
        │     ├── Start `rllm serve --model X --port Y --tp Z`
        │     ├── Wait for GET /health → 200
        │     └── Cache servers by config (reuse across tests)
        │
        ├── POST /v1/chat/completions  (or /v1/messages)
        │     {"messages": [...], "max_tokens": 512, "temperature": 0}
        │
        └── coherence.py: check_coherence(response_text)
              ├── langdetect → "en"?
              ├── len ≥ 20?
              ├── no degenerate repetition?
              └── ASCII ratio ≥ 70%?
```

The `ServerManager` caches running servers by `(model_dir, extra_args)`.
Tests sharing the same server config reuse the process — critical since
model loading takes 10–60 seconds depending on model size.

---

## Multi-GPU

On systems with multiple NVIDIA GPUs, the test harness:

1. **Detects GPU count** via `nvidia-smi --query-gpu=name`.
2. **Passes `--tp <count>`** to `rllm serve` for tensor parallelism.
3. **Sums GPU memory** across all devices for the MoE streaming decision
   (since TP shards the model across all GPUs).

On macOS (Metal), single-GPU is used (TP > 1 requires CUDA + NCCL).

---

## Benchmarking

The same infrastructure supports benchmarking every model in the models
directory, measuring time-to-first-token (TTFT) and generation throughput
(tok/s) via the HTTP API.  Unlike `scripts/benchmark.sh` which uses
`rllm run` (CLI), this benchmark uses `rllm serve` and measures through the
actual HTTP endpoint — capturing the full end-to-end latency users experience.

### Quick Start

```bash
# Benchmark all models in models/ (bf16 + Q4):
tests/run.sh --bench
# → results/bench-20260326-143022.md

# 3 runs per model, averaged:
tests/run.sh --bench --bench-runs 3

# Longer generation (512 tokens):
tests/run.sh --bench --bench-tokens 512

# Only Q4 variants:
tests/run.sh --bench --q4-only

# Only models matching "llama":
tests/run.sh --bench --bench-filter llama

# Save to a specific file:
tests/run.sh --bench --bench-output my-results.md

# Skip build/download if already ready:
tests/run.sh --bench --skip-build --skip-download --skip-quantize
```

Or run the benchmark script directly:

```bash
python tests/bench.py --max-tokens 128 --runs 3
python tests/bench.py --filter qwen --q4-only --output custom.md
```

Results are always saved to a file — by default a timestamped markdown file
in `results/` (e.g. `results/bench-20260326-143022.md`).  Pass `--output PATH`
to override the location.

### What It Measures

| Metric | How |
|--------|-----|
| **TTFT** (ms) | Wall-clock time from HTTP request to first SSE content chunk |
| **tok/s** | `completion_tokens / (last_chunk_time - first_chunk_time)` |

Both use SSE streaming (`stream: true`) so TTFT reflects actual
time-to-first-token through the full stack (tokenization → prefill → first
decode → HTTP response).

### Output Format

Markdown table matching the style of `scripts/benchmark.sh`:

```
## Benchmark Results — Apple M4 Max
Max tokens: 128 | Runs: 3

| Model | Family | tok/s | TTFT |
|---|---|---|---|
| llama-3.2-1b-instruct | Llama | 142.3 tok/s | 45 ms |
| llama-3.2-1b-instruct-q4 | Llama | 148.1 tok/s | 38 ms |
| qwen2.5-3b-instruct | Qwen2 | 98.7 tok/s | 82 ms |
| ...
```

### Benchmark vs Tests

| | Tests (`run.sh`) | Benchmark (`run.sh --bench`) |
|---|---|---|
| **Purpose** | Correctness — is output coherent? | Performance — how fast? |
| **Models** | One per family (smallest) | Every model found in models/ |
| **Checks** | Coherence, usage, API format | TTFT, tok/s |
| **Max tokens** | 512 | 128 (configurable) |
| **Temperature** | 0 (deterministic) | 0 (deterministic) |

---

## Skip Logic

Tests skip gracefully when prerequisites are missing:

| Condition | Scope | Message |
|-----------|-------|---------|
| No GPU detected | All tests | "no GPU detected (need Metal on macOS or CUDA on Linux)" |
| Binary not built | All tests | "rllm binary not found (build with: cargo build --release)" |
| Model dir missing | Individual test | "model not found: llama-3.2-1b-instruct" |
| Q4 not quantized | Individual test | "Q4 model not found: llama-3.2-1b-instruct-q4" |

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RLLM_MODELS_DIR` | `models/` | Directory containing model subdirectories |
| `RLLM_BIN` | `target/release/rllm` | Path to rllm binary |
| `HF_TOKEN` | — | HuggingFace token for gated model downloads |

---

## Files

```
tests/
├── README.md               — this file
├── run.sh                  — one-command wrapper (build, download, quantize, test/bench)
├── requirements.txt        — Python dependencies
├── conftest.py             — fixtures: server lifecycle, GPU detection, memory
├── coherence.py            — English coherence validation (langdetect + heuristics)
├── bench.py                — benchmark script (TTFT + tok/s via HTTP API)
└── test_model_families.py  — parametrized tests for all families and features
```

---

## Adding a New Model Family

1. Add a `ModelConfig` entry to `BASE_MODELS` in `test_model_families.py`.
2. Add the model to `KNOWN_MODELS` in `bench.py` with size/family info.
3. Add the model to `scripts/download-models.sh` in the appropriate tier.
4. Run the tests — if the model supports chat, it should work with no other changes.

---

## Troubleshooting

**Server fails to start (timeout):**
The test prints the last 2000 characters of stderr.  Common causes:
- Model directory missing or incomplete (no `.safetensors` files).
- Not enough GPU memory — try a smaller model or add `--stream-experts`.
- Port conflict — the test picks a random free port, but firewalls may block it.

**Coherence check fails:**
The assertion message includes the prompt and first 500 chars of output.
If the model produces non-English (e.g., Chinese from a Qwen base model),
it may be a chat template issue.  Check `src/model/chat.rs`.

**Tests skip unexpectedly:**
Run `pytest -v` to see skip reasons.  Usually: binary not built, model not
downloaded, or no GPU detected.
