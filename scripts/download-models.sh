#!/bin/bash
# ---------------------------------------------------------------------------
# download-models.sh — one-shot setup: install HF CLI, authenticate, and
# pull rLLM model weights from HuggingFace.
#
# Usage:
#   scripts/download-models.sh [options] [dest]
#
# Options:
#   --small   1B–8B models only (fast iteration, ~100 GB)
#   --medium  default tier: all models ≤35B (~500 GB)
#   --big     add 70B+ models for multi-GPU / high-VRAM testing (~1 TB+)
#   dest      download directory (default: models/)
#
# Environment:
#   HF_TOKEN  HuggingFace access token (required for gated models like
#             Llama, Gemma, Mistral).  Get one at:
#             https://huggingface.co/settings/tokens
#
#             Set it before running:
#               export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
#
# Only downloads safetensors weights + config/tokenizer files — no
# pytorch bins, READMEs, or other artefacts.
# ---------------------------------------------------------------------------

set -euo pipefail

TIER="medium"
DEST="models"

for arg in "$@"; do
  case "$arg" in
    --small)  TIER="small" ;;
    --medium) TIER="medium" ;;
    --big)    TIER="big" ;;
    *)        DEST="$arg" ;;
  esac
done

# ---- 1. Install huggingface-cli if missing ------------------------------
if ! command -v huggingface-cli &>/dev/null; then
  echo "Installing huggingface_hub CLI..."
  pip install -q huggingface_hub[cli]
fi

# ---- 2. Authenticate with HuggingFace ----------------------------------
# Gated models (Llama, Gemma, Mistral) require an access token.
# Priority: HF_TOKEN env var > existing cached login > interactive prompt.
if ! huggingface-cli whoami &>/dev/null; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Logging in with HF_TOKEN..."
    huggingface-cli login --token "$HF_TOKEN"
  else
    echo "Not logged in to HuggingFace."
    echo "Gated models (Llama, Gemma, Mistral) require authentication."
    echo "Get a token at: https://huggingface.co/settings/tokens"
    echo ""
    read -rp "Paste your HF token (or press Enter to skip): " token
    if [[ -n "$token" ]]; then
      huggingface-cli login --token "$token"
    else
      echo "Skipping auth — gated model downloads may fail."
    fi
  fi
fi

echo "Authenticated as: $(huggingface-cli whoami 2>/dev/null | head -1)"
echo ""

mkdir -p "$DEST"

# ---- Small tier: 1B–8B, covers head_dim 64/96/128/256 -----------------
MODELS=(
  # Llama — head_dim=64 (1B/3B), head_dim=128 (8B)
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"

  # Qwen 2.5 — head_dim=128
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"

  # Gemma 3 — head_dim=256
  "google/gemma-3-4b-it"

  # Mistral — head_dim=128, SentencePiece tokenizer
  "mistralai/Mistral-7B-Instruct-v0.3"
)

# ---- Medium tier: add MoE, larger dense, Phi-4 -------------------------
if [[ "$TIER" == "medium" || "$TIER" == "big" ]]; then
  MODELS+=(
    # Phi-4 — head_dim=96 (odd one out)
    "microsoft/phi-4"

    # Gemma 27B — large dense, head_dim=128
    "google/gemma-3-27b-it"

    # Mixtral 8x7B — MoE, 46.7B total (12.9B active)
    "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Qwen 3.5 MoE — 35B total (3.3B active)
    "Qwen/Qwen3.5-35B-A3B"

    # Qwen 3.5 dense 27B
    "Qwen/Qwen3.5-27B"

    # Qwen 3.5 9B
    "Qwen/Qwen3.5-9B"

    # Qwen3 Coder MoE — 30B total (3.3B active)
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    # DeepSeek R1 distill
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  )
fi

# ---- Big tier: 70B+ for multi-GPU / high-VRAM --------------------------
if [[ "$TIER" == "big" ]]; then
  MODELS+=(
    # Llama 70B — the standard large-model benchmark
    "meta-llama/Llama-3.1-70B"
    "meta-llama/Llama-3.1-70B-Instruct"

    # Qwen 72B — largest dense Qwen 2.5
    "Qwen/Qwen2.5-72B-Instruct"

    # Qwen 3.5 73B MoE
    "Qwen/Qwen3.5-73B"

    # Mixtral 8x22B — 176B total MoE
    "mistralai/Mixtral-8x22B-Instruct-v0.1"

    # Qwen 3.5 122B MoE — 122B total (10B active), largest MoE we test
    "Qwen/Qwen3.5-122B-A10B"
  )
fi

echo "Tier: $TIER (${#MODELS[@]} models → $DEST/)"
echo ""

FAILED=()

for model in "${MODELS[@]}"; do
  name=$(echo "$model" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')
  echo "=== $model → $DEST/$name ==="

  if huggingface-cli download "$model" \
    --local-dir "$DEST/$name" \
    --include "*.safetensors" "config.json" "tokenizer.json" "tokenizer_config.json"; then
    echo "  ✓ done"
  else
    echo "  ✗ FAILED (gated model? run: huggingface-cli login)"
    FAILED+=("$model")
  fi
  echo ""
done

echo "=== Summary ==="
du -sh "$DEST"/*/ 2>/dev/null || true
echo ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed downloads (${#FAILED[@]}):"
  for m in "${FAILED[@]}"; do
    echo "  - $m"
  done
  exit 1
else
  echo "All ${#MODELS[@]} models downloaded."
fi
