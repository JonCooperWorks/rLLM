#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# bench.py — HTTP API benchmark for rLLM, measuring TTFT and generation
# throughput via the serve endpoint.
#
# Runs every model found in the models directory (bf16 + Q4), starts
# rllm serve per model, measures time-to-first-token via SSE streaming,
# and computes generation tok/s from wall-clock time + usage counts.
#
# Usage:
#   python tests/gpu-integration/bench.py [options]
#
# Options:
#   --models-dir PATH   model directory (default: models/)
#   --binary PATH       rllm binary (default: target/release/rllm)
#   --max-tokens N      tokens to generate (default: 128)
#   --runs N            runs per model, results averaged (default: 1)
#   --prompt TEXT        override benchmark prompt
#   --q4-only           skip bf16, only bench Q4 variants
#   --bf16-only         skip Q4 variants
#   --filter PATTERN    only bench models matching pattern (substring)
#   --output PATH       write markdown results to file
#
# Output: Markdown table with TTFT (ms) and generation throughput (tok/s)
# for every model, matching the format of scripts/benchmark.sh.
#
# Related: conftest.py (ServerManager reused here), scripts/benchmark.sh
# ---------------------------------------------------------------------------

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add this directory to path so we can import conftest helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import ServerManager, _find_rllm_binary, _get_gpu_count, _get_available_memory_gb, _should_stream_experts


# ---------------------------------------------------------------------------
# Model registry — every model the benchmark knows about, ordered by size.
# Maps directory name → (family, is_moe, bf16_size_gb, supports_stream_experts).
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    # Small tier
    "llama-3.2-1b-instruct":           ("Llama",     False,  2, True),
    "llama-3.2-3b-instruct":           ("Llama",     False,  6, True),
    "qwen2.5-3b-instruct":             ("Qwen2",     False,  6, True),
    "gemma-3-4b-it":                   ("Gemma3",    False,  8, True),
    "qwen2.5-7b-instruct":             ("Qwen2",     False, 14, True),
    "mistral-7b-instruct-v0.3":        ("Mistral",   False, 14, True),
    "llama-3.1-8b-instruct":           ("Llama",     False, 16, True),
    "qwen3.5-9b":                      ("Qwen3_5",   False, 18, True),
    # Medium tier
    "phi-4":                           ("Phi",       False, 28, True),
    "gemma-3-27b-it":                  ("Gemma3",    False, 54, True),
    "qwen3.5-27b":                     ("Qwen3_5",   False, 54, True),
    "gpt-oss-20b":                     ("GptOss",    True,  40, False),
    "qwen3-coder-30b-a3b-instruct":    ("Qwen3Moe",  True,  60, True),
    "qwen3.5-35b-a3b":                 ("Qwen3_5M",  True,  70, True),
    "deepseek-r1-distill-qwen-32b":    ("Qwen2",     False, 64, True),
    "mixtral-8x7b-instruct-v0.1":      ("Mixtral",   True,  93, True),
    # Big tier
    "llama-3.1-70b-instruct":          ("Llama",     False, 140, True),
    "qwen2.5-72b-instruct":            ("Qwen2",     False, 144, True),
    "qwen3.5-122b-a10b":              ("Qwen3_5M",  True,  244, True),
    "mixtral-8x22b-instruct-v0.1":     ("Mixtral",   True,  352, True),
    "gpt-oss-120b":                    ("GptOss",    True,  240, False),
    # Massive tier
    "qwen3.5-397b-a27b":              ("Qwen3_5M",  True,  794, True),
}

DEFAULT_PROMPT = "The meaning of life is"


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def detect_gpu_name() -> str:
    """Return the GPU name for the results header."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            # Try to get the chip name instead.
            sp = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            for line in sp.stdout.splitlines():
                if "Chip" in line:
                    return line.split(":")[-1].strip()
            return result.stdout.strip()
        except Exception:
            return "Apple Silicon"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def bench_one_streaming(base_url: str, prompt: str, max_tokens: int) -> dict:
    """Run one streaming benchmark request, measuring TTFT and generation speed.

    Returns dict with: ttft_ms, gen_tps, prompt_tokens, completion_tokens, total_ms.
    Returns None on failure.
    """
    import requests

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.monotonic()
    t_first_token = None
    completion_tokens = 0
    prompt_tokens = 0

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=600,
            stream=True,
        )
        if resp.status_code != 200:
            return None

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
                # Record TTFT on first content delta.
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") and t_first_token is None:
                    t_first_token = time.monotonic()
                # Check for usage in final chunk.
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
            except json.JSONDecodeError:
                continue

        t_end = time.monotonic()

    except Exception:
        return None

    if t_first_token is None:
        return None

    ttft_ms = (t_first_token - t_start) * 1000
    gen_duration = t_end - t_first_token
    total_ms = (t_end - t_start) * 1000

    # If we didn't get usage from streaming, make a non-streaming call to get it.
    if completion_tokens == 0:
        try:
            payload["stream"] = False
            resp2 = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            if resp2.status_code == 200:
                body = resp2.json()
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_ms = (time.monotonic() - t_start) * 1000
        except Exception:
            pass

    gen_tps = completion_tokens / gen_duration if gen_duration > 0 and completion_tokens > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "gen_tps": gen_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_ms": total_ms,
    }


def bench_one_nonstreaming(base_url: str, prompt: str, max_tokens: int) -> dict:
    """Run one non-streaming benchmark, measuring total time and throughput.

    Fallback for when streaming metrics aren't needed — simpler and more
    reliable for just getting tok/s.
    """
    import requests

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    t_start = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=600,
        )
        t_end = time.monotonic()
        if resp.status_code != 200:
            return None

        body = resp.json()
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_ms = (t_end - t_start) * 1000

        gen_tps = completion_tokens / (total_ms / 1000) if total_ms > 0 and completion_tokens > 0 else 0

        return {
            "ttft_ms": total_ms,  # Approximation — non-streaming can't measure true TTFT.
            "gen_tps": gen_tps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_ms": total_ms,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark rLLM models via HTTP API")
    parser.add_argument("--models-dir", default=os.environ.get("RLLM_MODELS_DIR", "models"))
    parser.add_argument("--binary", default=os.environ.get("RLLM_BIN"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--q4-only", action="store_true")
    parser.add_argument("--bf16-only", action="store_true")
    parser.add_argument("--filter", default="", help="Only bench models matching this substring")
    parser.add_argument("--output", nargs="?", const="auto",
                        help="Write markdown results to file.  Without a path, "
                        "auto-generates results/bench-TIMESTAMP.md")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for auto-named result files (default: results/)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        print(f"Error: models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Find binary.
    if args.binary:
        binary = Path(args.binary)
    else:
        binary = _find_rllm_binary()
    if binary is None or not binary.is_file():
        print("Error: rllm binary not found. Build with: cargo build --release", file=sys.stderr)
        sys.exit(1)

    gpu_count = _get_gpu_count()
    gpu_name = detect_gpu_name()
    available_mem = _get_available_memory_gb()

    print(f"GPU: {gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''})")
    print(f"Available memory: {available_mem:.0f} GB")
    print(f"Max tokens: {args.max_tokens} | Runs: {args.runs}")
    print(f"Prompt: \"{args.prompt}\"")
    print()

    # Discover models — scan the directory and match against known models.
    discovered = []
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if args.filter and args.filter not in name:
            continue
        # Determine if this is a Q4 variant.
        is_q4 = name.endswith("-q4")
        base_name = name[:-3] if is_q4 else name

        if args.q4_only and not is_q4:
            continue
        if args.bf16_only and is_q4:
            continue

        # Look up model info.
        if base_name in KNOWN_MODELS:
            family, is_moe, bf16_size_gb, supports_stream = KNOWN_MODELS[base_name]
        else:
            # Unknown model — still try to bench it.
            family, is_moe, bf16_size_gb, supports_stream = ("Unknown", False, 0, False)

        # Check it has safetensors.
        if not any(entry.glob("*.safetensors")):
            continue

        discovered.append({
            "name": name,
            "path": str(entry),
            "family": family,
            "is_moe": is_moe,
            "is_q4": is_q4,
            "bf16_size_gb": bf16_size_gb,
            "supports_stream_experts": supports_stream,
        })

    if not discovered:
        print("No models found to benchmark.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(discovered)} models to benchmark")
    print()

    # Run benchmarks.
    mgr = ServerManager(binary, gpu_count=gpu_count)
    results = []

    try:
        for model in discovered:
            name = model["name"]
            print(f"=== {name} ===")

            # Build extra args.
            extra_args = []
            if model["is_moe"] and model["supports_stream_experts"]:
                if _should_stream_experts(model["bf16_size_gb"], model["is_q4"]):
                    extra_args.append("--stream-experts")

            try:
                base_url = mgr.get_or_start(model["path"], extra_args)
            except RuntimeError as e:
                print(f"  FAILED to start server: {e}", file=sys.stderr)
                results.append({
                    "name": name, "family": model["family"],
                    "gen_tps": None, "ttft_ms": None,
                })
                continue

            # Run multiple iterations and average.
            run_results = []
            for run_idx in range(args.runs):
                r = bench_one_streaming(base_url, args.prompt, args.max_tokens)
                if r is None:
                    print(f"  Run {run_idx + 1}: FAILED")
                    continue
                run_results.append(r)
                print(
                    f"  Run {run_idx + 1}: {r['gen_tps']:.1f} tok/s, "
                    f"TTFT {r['ttft_ms']:.0f} ms, "
                    f"{r['completion_tokens']} tokens in {r['total_ms']:.0f} ms"
                )

            if run_results:
                avg_tps = sum(r["gen_tps"] for r in run_results) / len(run_results)
                avg_ttft = sum(r["ttft_ms"] for r in run_results) / len(run_results)
                results.append({
                    "name": name, "family": model["family"],
                    "gen_tps": avg_tps, "ttft_ms": avg_ttft,
                })
            else:
                results.append({
                    "name": name, "family": model["family"],
                    "gen_tps": None, "ttft_ms": None,
                })

            print()

    finally:
        mgr.stop_all()

    # Resolve output path — always save results to a file.
    if args.output == "auto" or args.output is None:
        # Auto-generate: results/bench-YYYYMMDD-HHMMSS.md
        # Use the repo root's results/ dir if we can find it, otherwise cwd.
        repo_root = Path(__file__).resolve().parent.parent.parent
        results_dir = repo_root / args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = results_dir / f"bench-{timestamp}.md"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print and save markdown table.
    print()
    table = format_markdown_table(results, gpu_name, gpu_count, args)
    print(table)

    output_path.write_text(table)
    print(f"Results saved to {output_path}")


def format_markdown_table(results: list[dict], gpu_name: str, gpu_count: int, args) -> str:
    """Format benchmark results as a markdown table."""
    lines = []
    lines.append(f"## Benchmark Results — {gpu_name}")
    if gpu_count > 1:
        lines.append(f"Tensor parallelism: {gpu_count} GPUs")
    lines.append(f"Max tokens: {args.max_tokens} | Runs: {args.runs}")
    lines.append("")
    lines.append("| Model | Family | tok/s | TTFT |")
    lines.append("|---|---|---|---|")

    for r in results:
        name = r["name"]
        family = r["family"]
        if r["gen_tps"] is not None:
            tps = f"{r['gen_tps']:.1f} tok/s"
            ttft_ms = r["ttft_ms"]
            if ttft_ms >= 1000:
                secs = int(ttft_ms) // 1000
                ms = int(ttft_ms) % 1000
                ttft = f"{secs},{ms:03d} ms"
            else:
                ttft = f"{ttft_ms:.0f} ms"
        else:
            tps = "FAIL"
            ttft = "—"
        lines.append(f"| {name} | {family} | {tps} | {ttft} |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
