#!/bin/bash
# ---------------------------------------------------------------------------
# quantize-models.sh — batch-quantize every bf16 model in models/ to Q4/Q8.
#
# Iterates model directories, skips any that already end in -q4/-q8 or already
# have the target sibling, and runs `rllm quantize` on the rest.
#
# Usage:
#   scripts/quantize-models.sh [--format q4|q8|all] [models_dir]
#
# Arguments:
#   --format FMT  quantization format: q4, q8, or all (default: all)
#                 "all" runs both q4 and q8 for every model
#   models_dir    directory containing model subdirectories (default: models/)
#
# Examples:
#   scripts/quantize-models.sh                     # quantize everything to Q4 + Q8
#   scripts/quantize-models.sh --format q4         # quantize everything to Q4 only
#   scripts/quantize-models.sh --format q8         # quantize everything to Q8 only
#   scripts/quantize-models.sh --format all /mnt/models  # both, custom directory
#
# Related files:
#   src/commands/quantize.rs  — the rllm quantize CLI command
#   scripts/download-models.sh — downloads bf16 models from HuggingFace
# ---------------------------------------------------------------------------

set -euo pipefail

FORMAT="all"

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

if [[ "$FORMAT" != "q4" && "$FORMAT" != "q8" && "$FORMAT" != "all" ]]; then
  echo "Error: unsupported format '$FORMAT' (use q4, q8, or all)"
  exit 1
fi

# Build the list of formats to run.
if [[ "$FORMAT" == "all" ]]; then
  FORMATS=(q4 q8)
else
  FORMATS=("$FORMAT")
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

  # Skip directories that are already quantized (any format).
  if [[ "$name" == *-q4 ]] || [[ "$name" == *-q8 ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Skip directories with no safetensors files (incomplete downloads, etc.).
  if ! ls "$model_dir"/*.safetensors &>/dev/null; then
    echo "skip: $name (no .safetensors files)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  for fmt in "${FORMATS[@]}"; do
    suffix="-${fmt}"

    # Skip if the target sibling already exists.
    if [[ -d "$DEST/${name}${suffix}" ]]; then
      echo "skip: $name → ${name}${suffix} (exists)"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo "=== Quantizing $name → ${name}${suffix} (${fmt}) ==="
    if "$BINARY" quantize --model "$model_dir" --output "$DEST/${name}${suffix}" --format "$fmt"; then
      echo "  done"
      QUANTIZED=$((QUANTIZED + 1))
    else
      echo "  FAILED"
      FAILED+=("${name} (${fmt})")
    fi
    echo ""
  done
done

echo "=== Summary ==="
echo "Formats:   ${FORMATS[*]}"
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
