#!/bin/bash
# download_models.sh — pull all rLLM test models from HuggingFace
# Usage: ./download_models.sh [--big] [dest]
#   --big  include 70B+ models for H100 testing
#   dest   download directory (default: /workspace/models)

BIG=false
DEST="/workspace/models"

for arg in "$@"; do
  case "$arg" in
    --big) BIG=true ;;
    *)     DEST="$arg" ;;
  esac
done

mkdir -p "$DEST"

MODELS=(
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-27b-it"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "microsoft/phi-4"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

if $BIG; then
  MODELS+=(
    "meta-llama/Llama-3.1-70B"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen3.5-73B"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
  )
fi

for model in "${MODELS[@]}"; do
  name=$(echo "$model" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')
  echo "=== $model → $DEST/$name ==="
  hf download "$model" \
    --local-dir "$DEST/$name" \
    --include "*.safetensors" "config.json" "tokenizer.json" "tokenizer_config.json"
  echo ""
done

echo "=== Done ==="
du -sh "$DEST"/*/
