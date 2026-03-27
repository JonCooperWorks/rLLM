#!/bin/bash
# ---------------------------------------------------------------------------
# quantize-models.sh — batch-quantize every bf16 model in models/ to Q4/Q8.
#
# Iterates model directories, skips any that already end in -q4/-q8 or already
# have the target sibling, and runs `rllm quantize` on the rest.
#
# Usage:
#   scripts/quantize-models.sh [--format q4|q8] [models_dir]
#
# Arguments:
#   --format FMT  quantization format: q4 (default) or q8
#   models_dir    directory containing model subdirectories (default: models/)
#
# Examples:
#   scripts/quantize-models.sh                     # quantize everything to Q4
#   scripts/quantize-models.sh --format q8         # quantize everything to Q8
#   scripts/quantize-models.sh --format q8 /mnt/models  # Q8, custom directory
#
# Related files:
#   src/commands/quantize.rs  — the rllm quantize CLI command
#   scripts/download-models.sh — downloads bf16 models from HuggingFace
# ---------------------------------------------------------------------------

set -euo pipefail

FORMAT="q4"

# Parse optional --format flag.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      FORMAT="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

DEST="${1:-models}"

if [[ ! -d "$DEST" ]]; then
  echo "Error: model directory '$DEST' does not exist."
  exit 1
fi

if [[ "$FORMAT" != "q4" && "$FORMAT" != "q8" ]]; then
  echo "Error: unsupported format '$FORMAT' (use q4 or q8)"
  exit 1
fi

SUFFIX="-${FORMAT}"

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

  # Skip directories that are already quantized (any format).
  if [[ "$name" == *-q4 ]] || [[ "$name" == *-q8 ]]; then
    echo "skip: $name (already quantized)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Skip if the target sibling already exists.
  if [[ -d "$DEST/${name}${SUFFIX}" ]]; then
    echo "skip: $name (${name}${SUFFIX} exists)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Skip directories with no safetensors files (incomplete downloads, etc.).
  if ! ls "$model_dir"/*.safetensors &>/dev/null; then
    echo "skip: $name (no .safetensors files)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "=== Quantizing $name → ${name}${SUFFIX} (${FORMAT}) ==="
  if "$BINARY" quantize --model "$model_dir" --output "$DEST/${name}${SUFFIX}" --format "$FORMAT"; then
    echo "  ✓ done"
    QUANTIZED=$((QUANTIZED + 1))
  else
    echo "  ✗ FAILED"
    FAILED+=("$name")
  fi
  echo ""
done

echo "=== Summary ==="
echo "Format:    ${FORMAT}"
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
