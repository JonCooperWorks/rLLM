#!/bin/bash
# ---------------------------------------------------------------------------
# quantize-models.sh — batch-quantize every bf16 model in models/ to Q4.
#
# Iterates model directories, skips any that already end in -q4 or already
# have a -q4 sibling, and runs `rllm quantize` on the rest.
#
# Usage:
#   scripts/quantize-models.sh [models_dir]
#
# Arguments:
#   models_dir   directory containing model subdirectories (default: models/)
#
# Examples:
#   scripts/quantize-models.sh              # quantize everything in models/
#   scripts/quantize-models.sh /mnt/models  # custom model directory
#
# Related files:
#   src/commands/quantize.rs  — the rllm quantize CLI command
#   scripts/download-models.sh — downloads bf16 models from HuggingFace
# ---------------------------------------------------------------------------

set -euo pipefail

DEST="${1:-models}"

if [[ ! -d "$DEST" ]]; then
  echo "Error: model directory '$DEST' does not exist."
  exit 1
fi

# Build once in release mode before iterating models.
echo "Building rllm in release mode..."
cargo build --release
echo ""

BINARY="./target/release/rllm"
SKIPPED=0
QUANTIZED=0
FAILED=()

for model_dir in "$DEST"/*/; do
  # Strip trailing slash to get the directory name.
  model_dir="${model_dir%/}"
  name="$(basename "$model_dir")"

  # Skip directories that are already Q4.
  if [[ "$name" == *-q4 ]]; then
    echo "skip: $name (already Q4)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Skip if a -q4 sibling already exists.
  if [[ -d "$DEST/${name}-q4" ]]; then
    echo "skip: $name (${name}-q4 exists)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Skip directories with no safetensors files (incomplete downloads, etc.).
  if ! ls "$model_dir"/*.safetensors &>/dev/null; then
    echo "skip: $name (no .safetensors files)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "=== Quantizing $name → ${name}-q4 ==="
  if "$BINARY" quantize --model "$model_dir" --output "$DEST/${name}-q4"; then
    echo "  ✓ done"
    QUANTIZED=$((QUANTIZED + 1))
  else
    echo "  ✗ FAILED"
    FAILED+=("$name")
  fi
  echo ""
done

echo "=== Summary ==="
echo "Quantized: $QUANTIZED"
echo "Skipped:   $SKIPPED"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed (${#FAILED[@]}):"
  for m in "${FAILED[@]}"; do
    echo "  - $m"
  done
  exit 1
else
  echo "All done."
fi
