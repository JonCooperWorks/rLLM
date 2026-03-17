#!/bin/bash
# ---------------------------------------------------------------------------
# benchmark.sh — run each downloaded model in bf16 and Q4, collect tok/s and
# TTFT, and print a Markdown table matching the README format.
#
# Usage:
#   scripts/benchmark.sh [options] [models_dir]
#
# Options:
#   --small    only benchmark 1B–8B models
#   --medium   add medium-tier models (default)
#   --big      add 70B+ models
#   --quantize only run Q4 for each model (compact Q4-only table)
#   --q4-only  skip bf16 runs (useful for models that don't fit in VRAM)
#   --bf16-only skip Q4 runs
#   --runs N   number of runs per config (default: 1)
#
# Environment:
#   RLLM_BENCH_PROMPT   override the benchmark prompt
#   RLLM_BENCH_TOKENS   override --max-tokens (default: 128)
# ---------------------------------------------------------------------------

set -euo pipefail

TIER="medium"
DEST="models"
RUNS=1
SKIP_BF16=false
SKIP_Q4=false
QUANTIZE_ONLY=false
MAX_TOKENS="${RLLM_BENCH_TOKENS:-128}"
PROMPT="${RLLM_BENCH_PROMPT:-The meaning of life is}"

for arg in "$@"; do
  case "$arg" in
    --small)     TIER="small" ;;
    --medium)    TIER="medium" ;;
    --big)       TIER="big" ;;
    --quantize)  SKIP_BF16=true; QUANTIZE_ONLY=true ;;
    --q4-only)   SKIP_BF16=true ;;
    --bf16-only) SKIP_Q4=true ;;
    --runs)      shift_next=runs ;;
    *)
      if [[ "${shift_next:-}" == "runs" ]]; then
        RUNS="$arg"
        shift_next=""
      else
        DEST="$arg"
      fi
      ;;
  esac
done

# ---- Build rLLM in release mode -----------------------------------------
echo "Building rLLM (release)..."
BUILD_FEATURES=""
if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
  BUILD_FEATURES="--features cuda"
fi
cargo build --release $BUILD_FEATURES --manifest-path "$(dirname "$0")/../Cargo.toml" 2>&1 | tail -1
RLLM="$(dirname "$0")/../target/release/rllm"

if [[ ! -x "$RLLM" ]]; then
  echo "ERROR: failed to build rLLM"
  exit 1
fi

# ---- Detect GPU ----------------------------------------------------------
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
elif [[ "$(uname)" == "Darwin" ]]; then
  GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chip" | head -1 | sed 's/.*: //' | xargs)
fi
echo "GPU: $GPU_NAME"
echo ""

# ---- Model lists by tier -------------------------------------------------
SMALL_MODELS=(
  "llama-3.2-1b-instruct"
  "llama-3.2-3b-instruct"
  "qwen2.5-3b-instruct"
  "qwen2.5-7b-instruct"
  "mistral-7b-instruct-v0.3"
  "llama-3.1-8b-instruct"
  "gemma-3-4b-it"
)

MEDIUM_MODELS=(
  "phi-4"
  "qwen3.5-9b"
  "gemma-3-27b-it"
  "qwen3.5-27b"
  "mixtral-8x7b-instruct-v0.1"
  "qwen3-coder-30b-a3b-instruct"
  "qwen3.5-35b-a3b"
  "deepseek-r1-distill-qwen-32b"
)

BIG_MODELS=(
  "llama-3.1-70b-instruct"
  "qwen2.5-72b-instruct"
  "qwen3.5-122b-a10b"
  "mixtral-8x22b-instruct-v0.1"
)

MODELS=("${SMALL_MODELS[@]}")
if [[ "$TIER" == "medium" || "$TIER" == "big" ]]; then
  MODELS+=("${MEDIUM_MODELS[@]}")
fi
if [[ "$TIER" == "big" ]]; then
  MODELS+=("${BIG_MODELS[@]}")
fi

# ---- Helper: run one benchmark and extract metrics -----------------------
# Outputs: gen_tps prefill_tps ttft_ms
run_one() {
  local model_dir="$1"
  local quantize_flag="${2:-}"
  local tmpfile
  tmpfile=$(mktemp)

  local cmd=("$RLLM" run --model "$model_dir" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0)
  if [[ -n "$quantize_flag" ]]; then
    cmd+=(--quantize)
  fi

  # Run inference, capture stderr (metrics) to tmpfile, discard stdout (tokens)
  if ! "${cmd[@]}" >/dev/null 2>"$tmpfile"; then
    echo "FAIL 0 0"
    rm -f "$tmpfile"
    return
  fi

  # Parse prefill line: "prefill: N tokens in DURATION (X.X tok/s)"
  local prefill_line gen_line
  prefill_line=$(grep "^prefill:" "$tmpfile" || true)
  gen_line=$(grep "^generation:" "$tmpfile" || true)

  local prefill_tps=0 gen_tps=0 ttft_ms=0

  if [[ -n "$prefill_line" ]]; then
    prefill_tps=$(echo "$prefill_line" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
    # Extract duration — handles "1.2s", "345.6ms", "1.2µs"
    local duration_str
    duration_str=$(echo "$prefill_line" | sed -n 's/.* in \([^ ]*\) .*/\1/p')
    ttft_ms=$(parse_duration "$duration_str")
  fi

  if [[ -n "$gen_line" ]]; then
    gen_tps=$(echo "$gen_line" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')
  fi

  echo "$gen_tps $prefill_tps $ttft_ms"
  rm -f "$tmpfile"
}

# ---- Helper: parse Rust Debug duration to milliseconds -------------------
parse_duration() {
  local d="$1"
  if [[ "$d" == *ms ]]; then
    echo "${d%ms}"
  elif [[ "$d" == *µs ]]; then
    local us="${d%µs}"
    echo "scale=1; $us / 1000" | bc
  elif [[ "$d" == *s ]]; then
    local s="${d%s}"
    echo "scale=0; $s * 1000 / 1" | bc
  else
    echo "0"
  fi
}

# ---- Run benchmarks ------------------------------------------------------
echo "Tier: $TIER | Models: ${#MODELS[@]} | Runs: $RUNS | Max tokens: $MAX_TOKENS"
echo "Prompt: \"$PROMPT\""
echo ""

# Results arrays (parallel indexed with MODELS)
declare -a BF16_TPS Q4_TPS BF16_TTFT Q4_TTFT

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  model_dir="$DEST/$model"

  if [[ ! -d "$model_dir" ]]; then
    echo "SKIP $model (not found in $DEST/)"
    BF16_TPS[$i]="—"
    Q4_TPS[$i]="—"
    BF16_TTFT[$i]="—"
    Q4_TTFT[$i]="—"
    continue
  fi

  echo "=== $model ==="

  # bf16 run
  if [[ "$SKIP_BF16" == "false" ]]; then
    echo -n "  bf16: "
    local_tps=0 local_ttft=0
    for _ in $(seq "$RUNS"); do
      result=$(run_one "$model_dir" "")
      tps=$(echo "$result" | awk '{print $1}')
      ttft=$(echo "$result" | awk '{print $3}')
      if [[ "$tps" == "FAIL" ]]; then
        echo "FAILED"
        tps=0; ttft=0
        break
      fi
      local_tps=$(echo "$local_tps + $tps" | bc)
      local_ttft=$(echo "$local_ttft + $ttft" | bc)
    done
    if [[ "$tps" == "0" && "$RUNS" == "1" ]]; then
      BF16_TPS[$i]="—"
      BF16_TTFT[$i]="—"
    else
      BF16_TPS[$i]=$(echo "scale=1; $local_tps / $RUNS" | bc)
      BF16_TTFT[$i]=$(echo "scale=0; $local_ttft / $RUNS" | bc)
      echo "${BF16_TPS[$i]} tok/s, TTFT ${BF16_TTFT[$i]} ms"
    fi
  else
    BF16_TPS[$i]="—"
    BF16_TTFT[$i]="—"
  fi

  # Q4 run
  if [[ "$SKIP_Q4" == "false" ]]; then
    echo -n "  Q4:   "
    local_tps=0 local_ttft=0
    for _ in $(seq "$RUNS"); do
      result=$(run_one "$model_dir" "--quantize")
      tps=$(echo "$result" | awk '{print $1}')
      ttft=$(echo "$result" | awk '{print $3}')
      if [[ "$tps" == "FAIL" ]]; then
        echo "FAILED"
        tps=0; ttft=0
        break
      fi
      local_tps=$(echo "$local_tps + $tps" | bc)
      local_ttft=$(echo "$local_ttft + $ttft" | bc)
    done
    if [[ "$tps" == "0" && "$RUNS" == "1" ]]; then
      Q4_TPS[$i]="—"
      Q4_TTFT[$i]="—"
    else
      Q4_TPS[$i]=$(echo "scale=1; $local_tps / $RUNS" | bc)
      Q4_TTFT[$i]=$(echo "scale=0; $local_ttft / $RUNS" | bc)
      echo "${Q4_TPS[$i]} tok/s, TTFT ${Q4_TTFT[$i]} ms"
    fi
  else
    Q4_TPS[$i]="—"
    Q4_TTFT[$i]="—"
  fi

  echo ""
done

# ---- Print Markdown table ------------------------------------------------
echo ""
echo "## Results — $GPU_NAME"
echo ""

# Helper: format TTFT value for table display.
fmt_ttft() {
  local val="$1"
  if [[ "$val" == "—" ]]; then
    echo "—"
  elif (( $(echo "$val >= 1000" | bc -l) )); then
    echo "$(echo "scale=0; $val / 1000" | bc),$(printf '%03.0f' "$(echo "$val % 1000" | bc)") ms"
  else
    echo "${val} ms"
  fi
}

if [[ "$QUANTIZE_ONLY" == "true" ]]; then
  echo "| Model | Q4 tok/s | TTFT |"
  echo "|---|---|---|"
else
  echo "| Model | bf16 | Q4 | TTFT (bf16) | TTFT (Q4) |"
  echo "|---|---|---|---|---|"
fi

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"

  bf16="${BF16_TPS[$i]}"
  q4="${Q4_TPS[$i]}"
  bf16_ttft="${BF16_TTFT[$i]}"
  q4_ttft="${Q4_TTFT[$i]}"

  # Format tok/s
  [[ "$bf16" != "—" ]] && bf16="${bf16} tok/s"
  [[ "$q4" != "—" ]] && q4="${q4} tok/s"

  # Format TTFT
  bf16_ttft=$(fmt_ttft "$bf16_ttft")
  q4_ttft=$(fmt_ttft "$q4_ttft")

  if [[ "$QUANTIZE_ONLY" == "true" ]]; then
    echo "| $model | $q4 | $q4_ttft |"
  else
    echo "| $model | $bf16 | $q4 | $bf16_ttft | $q4_ttft |"
  fi
done

echo ""
echo "Benchmark complete."
