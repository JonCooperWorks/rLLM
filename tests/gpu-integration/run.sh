#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — convenience script to build rLLM, download models, optionally
# quantize to Q4, and run the GPU integration test suite.
#
# Usage:
#   tests/gpu-integration/run.sh [options]
#
# Options:
#   --small       download small tier only (1B–8B, default)
#   --medium      download small + medium tier (adds MoE, Phi-4, etc.)
#   --skip-download  skip model download (assume models already present)
#   --skip-quantize  skip Q4 quantization step
#   --skip-build     skip cargo build step
#   -k EXPR       pass pytest -k filter (e.g., -k llama)
#   -v            verbose pytest output
#   --            remaining args passed directly to pytest
#
# Environment:
#   RLLM_MODELS_DIR   model directory (default: models/)
#   RLLM_BIN          path to rllm binary (default: target/release/rllm)
#   HF_TOKEN          HuggingFace token for gated model downloads
#
# Related: conftest.py, test_model_families.py, scripts/download-models.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TIER="small"
SKIP_DOWNLOAD=false
SKIP_QUANTIZE=false
SKIP_BUILD=false
PYTEST_ARGS=()

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --small)          TIER="small"; shift ;;
    --medium)         TIER="medium"; shift ;;
    --skip-download)  SKIP_DOWNLOAD=true; shift ;;
    --skip-quantize)  SKIP_QUANTIZE=true; shift ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
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

# ---- 4. Install Python dependencies ---------------------------------------
echo "=== Installing Python dependencies ==="
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo ""

# ---- 5. Run tests ----------------------------------------------------------
echo "=== Running GPU integration tests ==="
export RLLM_MODELS_DIR="$MODELS_DIR"
export RLLM_BIN="$RLLM_BIN"

cd "$SCRIPT_DIR"
exec pytest "${PYTEST_ARGS[@]:--v}" .
