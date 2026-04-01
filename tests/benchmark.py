# ---------------------------------------------------------------------------
# benchmark.py — measurement engine for rLLM HTTP API benchmarks.
#
# Provides streaming and non-streaming benchmark functions that measure
# TTFT (time to first token) and generation throughput (tok/s), plus
# markdown table formatting for results output.
#
# Used by conftest.py's --bench pytest integration.  Not meant to be run
# standalone — use `pytest --bench` instead.
#
# Related: conftest.py (--bench hooks), models.py (model registry),
#          quality.py (output validation)
# ---------------------------------------------------------------------------

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of benchmarking one model configuration."""
    model_name: str
    family: str
    gen_tps: float | None = None
    ttft_ms: float | None = None
    quality: str = ""
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu_name() -> str:
    """Return the GPU name for the results header."""
    if platform.system() == "Darwin":
        try:
            sp = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            for line in sp.stdout.splitlines():
                if "Chip" in line:
                    return line.split(":")[-1].strip()
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
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


# ---------------------------------------------------------------------------
# Streaming benchmark — /v1/chat/completions (SSE)
# ---------------------------------------------------------------------------

def bench_one_streaming(base_url: str, prompt: str, max_tokens: int) -> dict | None:
    """Run one streaming benchmark request, measuring TTFT and generation speed.

    Returns dict with: ttft_ms, gen_tps, prompt_tokens, completion_tokens,
    total_ms, content.  Returns None on failure.
    """
    import requests

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.monotonic()
    t_first_token = None
    t_first_reasoning = None
    completion_tokens = 0
    prompt_tokens = 0
    content_pieces = []

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload, timeout=600, stream=True,
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
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("reasoning_content") and t_first_reasoning is None:
                    t_first_reasoning = time.monotonic()
                if delta.get("content"):
                    content_pieces.append(delta["content"])
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
            except json.JSONDecodeError:
                continue

        t_end = time.monotonic()
    except Exception:
        return None

    if t_first_token is None and t_first_reasoning is None:
        return None

    total_ms = (t_end - t_start) * 1000

    if t_first_token is not None:
        ttft_ms = (t_first_token - t_start) * 1000
    elif t_first_reasoning is not None:
        ttft_ms = (t_first_reasoning - t_start) * 1000
    else:
        ttft_ms = total_ms

    # Fallback: get usage from non-streaming call if streaming didn't include it.
    if completion_tokens == 0:
        try:
            payload["stream"] = False
            resp2 = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload, timeout=600,
            )
            if resp2.status_code == 200:
                body = resp2.json()
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_ms = (time.monotonic() - t_start) * 1000
        except Exception:
            pass

    gen_duration = total_ms / 1000
    gen_tps = (completion_tokens / gen_duration
               if gen_duration > 0 and completion_tokens > 0 else 0)

    return {
        "ttft_ms": ttft_ms,
        "gen_tps": gen_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_ms": total_ms,
        "content": "".join(content_pieces),
    }


# ---------------------------------------------------------------------------
# Streaming benchmark — /v1/completions (raw text, for base models)
# ---------------------------------------------------------------------------

def bench_one_streaming_completions(base_url: str, prompt: str,
                                     max_tokens: int) -> dict | None:
    """Run one streaming benchmark via /v1/completions (raw text, no chat template).

    Used for base (pretrained) models that don't understand chat template tokens.
    Returns dict with: ttft_ms, gen_tps, prompt_tokens, completion_tokens,
    total_ms, content.  Returns None on failure.
    """
    import requests

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.monotonic()
    t_first_token = None
    completion_tokens = 0
    prompt_tokens = 0
    content_pieces = []

    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json=payload, timeout=600, stream=True,
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
                choices = obj.get("choices", [{}])
                text = choices[0].get("text", "") if choices else ""
                if text:
                    content_pieces.append(text)
                    if t_first_token is None:
                        t_first_token = time.monotonic()
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

    total_ms = (t_end - t_start) * 1000
    ttft_ms = (t_first_token - t_start) * 1000

    if completion_tokens == 0:
        try:
            payload["stream"] = False
            resp2 = requests.post(
                f"{base_url}/v1/completions",
                json=payload, timeout=600,
            )
            if resp2.status_code == 200:
                body = resp2.json()
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
        except Exception:
            pass

    gen_duration = total_ms / 1000
    gen_tps = (completion_tokens / gen_duration
               if gen_duration > 0 and completion_tokens > 0 else 0)

    return {
        "ttft_ms": ttft_ms,
        "gen_tps": gen_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_ms": total_ms,
        "content": "".join(content_pieces),
    }


# ---------------------------------------------------------------------------
# Markdown table formatting
# ---------------------------------------------------------------------------

def format_markdown_table(results: list[BenchResult], gpu_name: str,
                          gpu_count: int, max_tokens: int, runs: int) -> str:
    """Format benchmark results as a markdown table."""
    lines = []
    lines.append(f"## Benchmark Results -- {gpu_name}")
    if gpu_count > 1:
        lines.append(f"Tensor parallelism: {gpu_count} GPUs")
    lines.append(f"Max tokens: {max_tokens} | Runs: {runs}")
    lines.append("")

    # Include TTR and Flesch columns if any result has scores.
    has_scores = any(r.scores for r in results)
    if has_scores:
        lines.append("| Model | Family | tok/s | TTFT | Quality | Flesch | TTR |")
        lines.append("|---|---|---|---|---|---|---|")
    else:
        lines.append("| Model | Family | tok/s | TTFT | Quality |")
        lines.append("|---|---|---|---|---|")

    for r in results:
        name = r.model_name
        family = r.family
        quality = r.quality or "--"

        if r.gen_tps is not None:
            tps = f"{r.gen_tps:.1f} tok/s"
            ttft_ms = r.ttft_ms
            if ttft_ms >= 1000:
                secs = int(ttft_ms) // 1000
                ms = int(ttft_ms) % 1000
                ttft = f"{secs},{ms:03d} ms"
            else:
                ttft = f"{ttft_ms:.0f} ms"
        else:
            tps = "FAIL"
            ttft = "--"

        if has_scores:
            flesch = f"{r.scores['flesch']:.0f}" if "flesch" in r.scores else "--"
            ttr = f"{r.scores['ttr']:.2f}" if "ttr" in r.scores else "--"
            lines.append(f"| {name} | {family} | {tps} | {ttft} | {quality} | {flesch} | {ttr} |")
        else:
            lines.append(f"| {name} | {family} | {tps} | {ttft} | {quality} |")

    lines.append("")
    return "\n".join(lines)


def _resolve_output_path(output_path: Path | None = None,
                         results_dir: str = "results") -> Path:
    """Resolve the output path, creating directories as needed."""
    if output_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        rdir = repo_root / results_dir
        rdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = rdir / f"bench-{timestamp}.md"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def append_result_line(result: BenchResult, output_path: Path) -> None:
    """Append a single result row to the live output file.

    Creates the file with a markdown header on first call, then appends
    one table row per result so you can `tail -f` the file during a run.
    """
    if result.gen_tps is not None:
        tps = f"{result.gen_tps:.1f} tok/s"
        ttft_ms = result.ttft_ms
        if ttft_ms >= 1000:
            secs = int(ttft_ms) // 1000
            ms = int(ttft_ms) % 1000
            ttft = f"{secs},{ms:03d} ms"
        else:
            ttft = f"{ttft_ms:.0f} ms"
    else:
        tps = "FAIL"
        ttft = "--"

    quality = result.quality or "--"
    has_scores = bool(result.scores)

    if not output_path.exists():
        # Write header on first result.
        gpu_name = detect_gpu_name()
        lines = [
            f"## Benchmark Results -- {gpu_name}",
            f"_(live — updated after each test)_",
            "",
        ]
        if has_scores:
            lines.append("| Model | Family | tok/s | TTFT | Quality | Flesch | TTR |")
            lines.append("|---|---|---|---|---|---|---|")
        else:
            lines.append("| Model | Family | tok/s | TTFT | Quality |")
            lines.append("|---|---|---|---|---|")
        output_path.write_text("\n".join(lines) + "\n")

    if has_scores:
        flesch = f"{result.scores['flesch']:.0f}" if "flesch" in result.scores else "--"
        ttr = f"{result.scores['ttr']:.2f}" if "ttr" in result.scores else "--"
        row = f"| {result.model_name} | {result.family} | {tps} | {ttft} | {quality} | {flesch} | {ttr} |"
    else:
        row = f"| {result.model_name} | {result.family} | {tps} | {ttft} | {quality} |"

    with open(output_path, "a") as f:
        f.write(row + "\n")


def write_results(results: list[BenchResult], gpu_name: str, gpu_count: int,
                  max_tokens: int, runs: int, output_path: Path | None = None,
                  results_dir: str = "results") -> Path:
    """Write benchmark results to a markdown file.

    If output_path is None, auto-generates results/bench-TIMESTAMP.md.
    Returns the path written to.
    """
    output_path = _resolve_output_path(output_path, results_dir)

    table = format_markdown_table(results, gpu_name, gpu_count, max_tokens, runs)
    output_path.write_text(table)
    return output_path
