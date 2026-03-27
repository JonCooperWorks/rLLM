#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — convenience script to build rLLM, download models, optionally
# quantize to Q4, and run the GPU integration test suite or benchmarks.
#
# Usage:
#   tests/run.sh [options]
#
# Options:
#   --small       download small tier only (1B–8B, default)
#   --medium      download small + medium tier (adds MoE, Phi-4, etc.)
#   --skip-download  skip model download (assume models already present)
#   --skip-quantize  skip Q4 quantization step
#   --skip-build     skip cargo build step
#   --bench          run benchmarks instead of tests (all models)
#   --bench-runs N   number of benchmark runs per model (default: 1)
#   --bench-tokens N max tokens for benchmark (default: 128)
#   --bench-output F write benchmark markdown to file
#   --bench-filter P only benchmark models matching pattern
#   -k EXPR       pass pytest -k filter (e.g., -k llama)
#   -v            verbose pytest output
#   --            remaining args passed directly to pytest
#
# Environment:
#   RLLM_MODELS_DIR   model directory (default: models/)
#   RLLM_BIN          path to rllm binary (default: target/release/rllm)
#   HF_TOKEN          HuggingFace token for gated model downloads
#
# Related: conftest.py, test_model_families.py, bench.py,
#          scripts/download-models.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIER="small"
SKIP_DOWNLOAD=false
SKIP_QUANTIZE=false
SKIP_BUILD=false
MODE="test"
BENCH_ARGS=()
PYTEST_ARGS=()

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --small)          TIER="small"; shift ;;
    --medium)         TIER="medium"; shift ;;
    --skip-download)  SKIP_DOWNLOAD=true; shift ;;
    --skip-quantize)  SKIP_QUANTIZE=true; shift ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
    --bench)          MODE="bench"; shift ;;
    --bench-runs)     BENCH_ARGS+=("--runs" "$2"); shift 2 ;;
    --bench-tokens)   BENCH_ARGS+=("--max-tokens" "$2"); shift 2 ;;
    --bench-output)   BENCH_ARGS+=("--output" "$2"); shift 2 ;;
    --bench-filter)   BENCH_ARGS+=("--filter" "$2"); shift 2 ;;
    --q4-only)        BENCH_ARGS+=("--q4-only"); shift ;;
    --bf16-only)      BENCH_ARGS+=("--bf16-only"); shift ;;
    -k)               PYTEST_ARGS+=("-k" "$2"); shift 2 ;;
    -v|--verbose)     PYTEST_ARGS+=("-v"); shift ;;
    --)               shift; PYTEST_ARGS+=("$@"); break ;;
    *)                PYTEST_ARGS+=("$1"); shift ;;
  esac
done

MODELS_DIR="${RLLM_MODELS_DIR:-$REPO_ROOT/models}"

# ---- 1. Build rLLM in release mode ----------------------------------------
if [[ "$SKIP_BUILD" == "false" ]]; then
  echo "=== Building rLLM (release) ==="
  BUILD_FEATURES=""
  if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
    BUILD_FEATURES="--features cuda"
  fi
  cargo build --release $BUILD_FEATURES --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1 | tail -5
  echo ""
fi

RLLM_BIN="${RLLM_BIN:-$REPO_ROOT/target/release/rllm}"
if [[ ! -x "$RLLM_BIN" ]]; then
  echo "ERROR: rllm binary not found at $RLLM_BIN"
  echo "Build with: cargo build --release"
  exit 1
fi

# ---- 2. Download models if needed -----------------------------------------
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
  # Check if models dir has any safetensors.
  if ! find "$MODELS_DIR" -name "*.safetensors" -maxdepth 2 2>/dev/null | head -1 | grep -q .; then
    echo "=== Downloading models (tier: $TIER) ==="
    "$REPO_ROOT/scripts/download-models.sh" "--$TIER" "$MODELS_DIR"
    echo ""
  else
    echo "=== Models already present in $MODELS_DIR, skipping download ==="
    echo "    (use --skip-download to suppress this check)"
    echo ""
  fi
fi

# ---- 3. Quantize to Q4 if needed ------------------------------------------
if [[ "$SKIP_QUANTIZE" == "false" ]]; then
  NEEDS_QUANTIZE=false
  for model_dir in "$MODELS_DIR"/*/; do
    name="$(basename "${model_dir%/}")"
    # Skip dirs that are already Q4 or have no safetensors.
    [[ "$name" == *-q4 ]] && continue
    ls "$model_dir"*.safetensors &>/dev/null || continue
    # If Q4 sibling doesn't exist, we need to quantize.
    if [[ ! -d "$MODELS_DIR/${name}-q4" ]]; then
      NEEDS_QUANTIZE=true
      break
    fi
  done

  if [[ "$NEEDS_QUANTIZE" == "true" ]]; then
    echo "=== Quantizing models to Q4 ==="
    "$REPO_ROOT/scripts/quantize-models.sh" "$MODELS_DIR" || true
    echo ""
  else
    echo "=== Q4 models already present, skipping quantization ==="
    echo ""
  fi
fi

# ---- 4. Install uv if missing, then sync Python dependencies ----------------
if ! command -v uv &>/dev/null; then
  echo "=== Installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Installing Python dependencies ==="
uv pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo ""

# ---- 5. Run tests or benchmarks --------------------------------------------
export RLLM_MODELS_DIR="$MODELS_DIR"
export RLLM_BIN="$RLLM_BIN"

cd "$SCRIPT_DIR"

if [[ "$MODE" == "bench" ]]; then
  echo "=== Running benchmarks ==="
  exec uv run python bench.py "${BENCH_ARGS[@]}"
else
  echo "=== Running GPU integration tests ==="
  exec uv run pytest "${PYTEST_ARGS[@]:--v}" .
fi
