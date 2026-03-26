#!/usr/bin/env bash
# ===========================================================================
# TurboQuant KV cache quantization benchmark — turbo4 vs none (BF16).
#
# Runs `rllm bench` for every model in models/ with both --kv-quant turbo4
# and --kv-quant none, capturing full output for reproducibility.
#
# Usage:
#   ./scripts/bench-turboquant.sh                  # all models
#   ./scripts/bench-turboquant.sh llama-3.2-1b-instruct qwen2.5-3b-instruct
#
# Output:  results/turboquant-bench-YYYY-MM-DD-HHMMSS/
#   ├── <model>__turbo4.txt      — full bench output (turbo4)
#   ├── <model>__none.txt        — full bench output (no KV quant)
#   └── summary.txt              — side-by-side comparison table
#
# The individual .txt files contain the full stderr+stdout so Claude can
# reproduce the exact numbers from a future conversation.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Build once up front.
echo "Building rllm (release)..."
cargo build --release 2>&1 | tail -1
RLLM="./target/release/rllm"

# Timestamp for this run.
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
OUTDIR="results/turboquant-bench-$TIMESTAMP"
mkdir -p "$OUTDIR"

# Models to bench — either from args or auto-discover.
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for d in models/*/config.json; do
        name="$(basename "$(dirname "$d")")"
        MODELS+=("$name")
    done
fi

# Models that require --stream-experts (too large to fit in GPU memory).
STREAM_EXPERTS_MODELS=(
    "qwen3.5-122b"
    "qwen3.5-122b-q4"
    "qwen3.5-397b-q4"
)

# Skip base (non-instruct) models — bench uses --chat which requires instruct.
SKIP_MODELS=(
    "llama-3.2-1b"
    "llama-3.2-1b-q4"
    "llama-3.2-3b"
)

should_skip() {
    local model="$1"
    for s in "${SKIP_MODELS[@]}"; do
        [ "$model" = "$s" ] && return 0
    done
    return 1
}

needs_stream_experts() {
    local model="$1"
    for s in "${STREAM_EXPERTS_MODELS[@]}"; do
        [ "$model" = "$s" ] && return 0
    done
    return 1
}

MAX_TOKENS=32
NUM_PROMPTS=3

echo "=== TurboQuant Benchmark ==="
echo "Timestamp: $TIMESTAMP"
echo "Output:    $OUTDIR/"
echo "Models:    ${#MODELS[@]} found"
echo ""

# Collect summary lines for the final table.
declare -a SUMMARY_LINES=()
SUMMARY_LINES+=("$(printf '%-40s %8s %8s %8s %8s %6s' 'Model' 'T4 gen' 'BF16 gen' 'T4 TTFT' 'BF16 TTFT' 'KV MB')")
SUMMARY_LINES+=("$(printf '%0.s-' {1..86})")

run_bench() {
    local model="$1"
    local kv_quant="$2"
    local outfile="$3"

    local extra_args=()
    if needs_stream_experts "$model"; then
        extra_args+=(--stream-experts)
    fi

    $RLLM bench --model "models/$model" --chat --kv-quant "$kv_quant" \
        --max-tokens "$MAX_TOKENS" --prompts "$NUM_PROMPTS" \
        ${extra_args[@]+"${extra_args[@]}"} > "$outfile" 2>&1
}

for model in "${MODELS[@]}"; do
    if should_skip "$model"; then
        echo "SKIP: $model (base model, no chat template)"
        continue
    fi

    MODEL_DIR="models/$model"
    if [ ! -f "$MODEL_DIR/config.json" ]; then
        echo "SKIP: $model (no config.json)"
        continue
    fi

    STREAM_TAG=""
    if needs_stream_experts "$model"; then
        STREAM_TAG=" [stream-experts]"
    fi

    echo ""
    echo ">>> $model$STREAM_TAG"

    # --- TurboQuant 4-bit ---
    OUTFILE_T4="$OUTDIR/${model}__turbo4.txt"
    echo "  turbo4..."
    if run_bench "$model" turbo4 "$OUTFILE_T4"; then
        T4_GEN=$(grep "decode throughput" "$OUTFILE_T4" | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -1 || echo "N/A")
        T4_TTFT_COLD=$(grep "avg TTFT cold" "$OUTFILE_T4" | grep -oE '[0-9]+\.[0-9]+ms' || echo "N/A")
        T4_TTFT_WARM=$(grep "avg TTFT warm" "$OUTFILE_T4" | grep -oE '[0-9]+\.[0-9]+ms' || echo "N/A")
        T4_KV_MB=$(grep "memory:" "$OUTFILE_T4" | grep -oE '[0-9]+ MB KV' | grep -oE '[0-9]+' || echo "N/A")
        echo "    decode: $T4_GEN | TTFT cold: $T4_TTFT_COLD warm: $T4_TTFT_WARM | KV: ${T4_KV_MB} MB"
    else
        echo "    FAILED (see $OUTFILE_T4)"
        T4_GEN="FAIL"
        T4_TTFT_COLD="FAIL"
        T4_KV_MB="N/A"
    fi

    # --- No KV quantization (BF16) ---
    OUTFILE_NONE="$OUTDIR/${model}__none.txt"
    echo "  none (BF16)..."
    if run_bench "$model" none "$OUTFILE_NONE"; then
        NONE_GEN=$(grep "decode throughput" "$OUTFILE_NONE" | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -1 || echo "N/A")
        NONE_TTFT_COLD=$(grep "avg TTFT cold" "$OUTFILE_NONE" | grep -oE '[0-9]+\.[0-9]+ms' || echo "N/A")
        NONE_TTFT_WARM=$(grep "avg TTFT warm" "$OUTFILE_NONE" | grep -oE '[0-9]+\.[0-9]+ms' || echo "N/A")
        NONE_KV_MB=$(grep "memory:" "$OUTFILE_NONE" | grep -oE '[0-9]+ MB KV' | grep -oE '[0-9]+' || echo "N/A")
        echo "    decode: $NONE_GEN | TTFT cold: $NONE_TTFT_COLD warm: $NONE_TTFT_WARM | KV: ${NONE_KV_MB} MB"
    else
        echo "    FAILED (see $OUTFILE_NONE)"
        NONE_GEN="FAIL"
        NONE_TTFT_COLD="FAIL"
        NONE_KV_MB="N/A"
    fi

    # Summary line: strip " tok/s" for the table.
    T4_GEN_NUM="${T4_GEN/ tok\/s/}"
    NONE_GEN_NUM="${NONE_GEN/ tok\/s/}"
    SUMMARY_LINES+=("$(printf '%-40s %8s %8s %8s %8s %6s' \
        "$model" "$T4_GEN_NUM" "$NONE_GEN_NUM" "$T4_TTFT_COLD" "$NONE_TTFT_COLD" "$T4_KV_MB")")
done

# Write summary table.
SUMMARY_FILE="$OUTDIR/summary.txt"
{
    echo "TurboQuant Benchmark Summary"
    echo "Date: $(date)"
    echo "System: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
    echo "GPU: Metal (Apple Silicon)"
    echo "RAM: $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1024/1024/1024}')"
    echo "Max tokens per request: $MAX_TOKENS"
    echo ""
    for line in "${SUMMARY_LINES[@]}"; do
        echo "$line"
    done
    echo ""
    echo "T4 gen   = decode tok/s with TurboQuant 4-bit KV cache"
    echo "BF16 gen = decode tok/s with BF16 KV cache (no quantization)"
    echo "T4 TTFT  = avg time-to-first-token (cold) with TurboQuant"
    echo "BF16 TTFT = avg time-to-first-token (cold) without TurboQuant"
    echo "KV MB    = KV cache memory with TurboQuant"
} > "$SUMMARY_FILE"

echo ""
echo "=== Summary ==="
cat "$SUMMARY_FILE"
echo ""
echo "Full results: $OUTDIR/"
