# ---------------------------------------------------------------------------
# test_model_families.py — GPU integration tests for every rLLM model family.
#
# Starts `rllm serve` for each model configuration, sends requests via the
# OpenAI / Anthropic HTTP API, and validates that outputs pass quality checks
# (coherence, grammar structure, vocabulary diversity, readability).
#
# When run with `--bench`, also records performance measurements (TTFT, tok/s)
# alongside correctness validation.
#
# Tests are skipped when:
#   - No GPU is available (Metal on macOS, CUDA on Linux)
#   - The rllm binary hasn't been built
#   - A model's directory doesn't exist
#
# Run:  pytest tests/ -v
# Run one family:  pytest tests/ -v -k llama
# Run + bench:  pytest tests/ -v --bench
#
# Related: models.py (model registry), quality.py (validation),
#          benchmark.py (measurement), conftest.py (fixtures)
# ---------------------------------------------------------------------------

import base64
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import requests

from models import (
    MODEL_REGISTRY, TURBOQUANT_CONFIGS, PROMPTS, MAX_TOKENS,
    get_test_models, get_vision_models, prompt_for_index,
)
from quality import check_quality, QualityResult
from conftest import _should_stream_experts


# ---------------------------------------------------------------------------
# Derived model lists
# ---------------------------------------------------------------------------

BASE_MODELS = get_test_models()
VISION_MODELS = get_vision_models()


# ---------------------------------------------------------------------------
# Streaming result container
# ---------------------------------------------------------------------------

@dataclass
class StreamingResult:
    """Parsed result from a streaming SSE request."""
    content: str
    chunk_count: int
    completion_tokens: int
    prompt_tokens: int
    ttft_ms: float       # Time to first content token (from request start).
    total_ms: float      # Total request duration.
    gen_tps: float       # Generation throughput (tokens / generation time).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(models_dir: Path, config, q4: bool = False,
                       q8: bool = False, tq3: bool = False) -> Path | None:
    """Return the model directory path, or None if it doesn't exist."""
    suffix = "-q4" if q4 else ("-q8" if q8 else ("-tq3" if tq3 else ""))
    name = config.model_name + suffix
    d = models_dir / name
    return d if d.is_dir() else None


def _build_extra_args(config, is_q4: bool, model_dir: str = "") -> list[str]:
    """Build CLI extra args, including --stream-experts decision for MoE."""
    args = list(config.extra_args)
    if config.is_moe and config.supports_stream_experts:
        if _should_stream_experts(config.bf16_size_gb, is_q4, model_dir):
            args.extend(["--stream-experts"])
    return args


def _estimate_memory_gb(config, is_q4: bool, model_dir: str = "") -> float:
    """Estimate GPU memory required for a model (weights + KV cache overhead)."""
    weight_gb = config.bf16_size_gb * 0.5 if is_q4 else config.bf16_size_gb
    if config.is_moe and config.supports_stream_experts:
        if _should_stream_experts(config.bf16_size_gb, is_q4, model_dir):
            weight_gb *= 0.25
    return weight_gb * 1.2


def _streaming_chat_completion(base_url: str, prompt: str,
                               max_tokens: int = MAX_TOKENS,
                               temperature: float = 0) -> StreamingResult:
    """Send a streaming OpenAI chat completion and measure real TTFT + tok/s.

    TTFT = time from HTTP request to first content token in the SSE stream.
    tok/s = completion_tokens / (total_time - TTFT), i.e. pure generation speed.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.monotonic()
    t_first_token = None
    t_first_reasoning = None
    completion_tokens = 0
    prompt_tokens = 0
    content_pieces = []
    chunk_count = 0

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload, timeout=600, stream=True,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

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
                chunk_count += 1
                if t_first_token is None:
                    t_first_token = time.monotonic()
            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
        except json.JSONDecodeError:
            continue

    t_end = time.monotonic()

    # TTFT: time to first output of any kind (reasoning or content).
    # For thinking models, the first reasoning token is the meaningful TTFT.
    t_first_output = t_first_reasoning or t_first_token
    if t_first_output is not None:
        ttft_ms = (t_first_output - t_start) * 1000
    else:
        ttft_ms = (t_end - t_start) * 1000

    total_ms = (t_end - t_start) * 1000

    # tok/s = tokens / generation_time (from first output to end).
    # For thinking models this includes reasoning + content tokens.
    gen_ms = total_ms - ttft_ms
    gen_tps = (completion_tokens / (gen_ms / 1000)
               if gen_ms > 0 and completion_tokens > 0 else 0)

    return StreamingResult(
        content="".join(content_pieces),
        chunk_count=chunk_count,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        gen_tps=gen_tps,
    )


def _anthropic_message(base_url: str, prompt: str, max_tokens: int = MAX_TOKENS,
                       temperature: float = 0, stream: bool = False) -> requests.Response:
    """Send an Anthropic-format message request."""
    return requests.post(
        f"{base_url}/v1/messages",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
        timeout=300, stream=stream,
    )


def _collect_anthropic_sse_content(response: requests.Response) -> tuple[str, int]:
    """Parse Anthropic SSE stream, return (concatenated content, chunk_count)."""
    content_parts = []
    chunk_count = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        try:
            obj = json.loads(data)
            if obj.get("type") == "content_block_delta":
                delta = obj.get("delta", {})
                text = delta.get("text", "")
                if text:
                    content_parts.append(text)
                    chunk_count += 1
        except json.JSONDecodeError:
            continue
    return "".join(content_parts), chunk_count


def _assert_quality(content: str, test_id: str, prompt: str) -> QualityResult:
    """Run quality checks, assert the output passes, return the result."""
    result = check_quality(content)
    assert result.passed, (
        f"quality check failed for {test_id}: {result.reason}\n"
        f"Prompt: {prompt}\nOutput: {content[:500]}"
    )
    if result.warnings:
        print(f"  Quality warnings for {test_id}: {'; '.join(result.warnings)}")
    return result


def _record_bench(bench_context, test_id: str, family: str,
                  quality_result: QualityResult,
                  gen_tps: float, ttft_ms: float) -> None:
    """Record benchmark measurements if --bench is active."""
    if not bench_context.enabled:
        return

    quality = "PASS" if quality_result.passed else f"FAIL: {quality_result.reason}"

    bench_context.record(
        model_name=test_id,
        family=family,
        gen_tps=gen_tps,
        ttft_ms=ttft_ms,
        quality=quality,
        scores=quality_result.scores,
    )


# ---------------------------------------------------------------------------
# Tests — Basic model inference (one per family, bf16 + optional Q4/Q8/TQ3)
#
# All tests use streaming requests so we get real TTFT (time to first token)
# and accurate tok/s (completion_tokens / generation_time, excluding prefill).
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[c.test_id for c in BASE_MODELS],
)
def test_model_bf16(config_index, server_manager, models_dir, bench_context):
    """Each model family produces coherent English via streaming OpenAI API."""
    config = BASE_MODELS[config_index]

    if config.is_moe and _should_stream_experts(config.bf16_size_gb, is_q4=False):
        pytest.skip(f"bf16 MoE too large for memory ({config.bf16_size_gb}GB), use Q4")

    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    extra_args = _build_extra_args(config, is_q4=False)
    mem = _estimate_memory_gb(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    prompt = prompt_for_index(config_index)
    sr = _streaming_chat_completion(base_url, prompt, temperature=config.temperature)

    assert sr.completion_tokens > 0, "no tokens generated"
    assert len(sr.content.strip()) > 0, "empty response"

    qr = _assert_quality(sr.content, config.test_id, prompt)
    _record_bench(bench_context, config.test_id, config.family, qr,
                  sr.gen_tps, sr.ttft_ms)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[f"{c.test_id}-q4" for c in BASE_MODELS],
)
def test_model_q4(config_index, server_manager, models_dir, bench_context):
    """Q4-quantized variant of each model family."""
    config = BASE_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config, q4=True)
    if model_dir is None:
        pytest.skip(f"Q4 model not found: {config.model_name}-q4")

    extra_args = _build_extra_args(config, is_q4=True, model_dir=str(model_dir))
    mem = _estimate_memory_gb(config, is_q4=True, model_dir=str(model_dir))
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    temp = 0 if config.quality_sensitive else config.temperature

    prompt = prompt_for_index(config_index + 5)
    test_id = f"{config.test_id}-q4"
    sr = _streaming_chat_completion(base_url, prompt, temperature=temp)

    assert sr.completion_tokens > 0, "no tokens generated"

    qr = _assert_quality(sr.content, test_id, prompt)
    _record_bench(bench_context, test_id, config.family, qr,
                  sr.gen_tps, sr.ttft_ms)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[f"{c.test_id}-q8" for c in BASE_MODELS],
)
def test_model_q8(config_index, server_manager, models_dir, bench_context):
    """Q8-quantized variant of each model family."""
    config = BASE_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config, q8=True)
    if model_dir is None:
        pytest.skip(f"Q8 model not found: {config.model_name}-q8")

    extra_args = _build_extra_args(config, is_q4=False, model_dir=str(model_dir))
    mem = _estimate_memory_gb(config, is_q4=False, model_dir=str(model_dir))
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    prompt = prompt_for_index(config_index + 3)
    test_id = f"{config.test_id}-q8"
    sr = _streaming_chat_completion(base_url, prompt, temperature=config.temperature)

    assert sr.completion_tokens > 0, "no tokens generated"

    qr = _assert_quality(sr.content, test_id, prompt)
    _record_bench(bench_context, test_id, config.family, qr,
                  sr.gen_tps, sr.ttft_ms)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[f"{c.test_id}-tq3" for c in BASE_MODELS],
)
def test_model_tq3(config_index, server_manager, models_dir, bench_context):
    """TQ3-quantized variant of each model family (TurboQuant 3-bit weights)."""
    config = BASE_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config, tq3=True)
    if model_dir is None:
        pytest.skip(f"TQ3 model not found: {config.model_name}-tq3")

    extra_args = _build_extra_args(config, is_q4=True, model_dir=str(model_dir))
    mem = _estimate_memory_gb(config, is_q4=True, model_dir=str(model_dir))
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    temp = 0 if config.quality_sensitive else config.temperature

    prompt = prompt_for_index(config_index + 7)
    test_id = f"{config.test_id}-tq3"
    sr = _streaming_chat_completion(base_url, prompt, temperature=temp)

    assert sr.completion_tokens > 0, "no tokens generated"

    qr = _assert_quality(sr.content, test_id, prompt)
    _record_bench(bench_context, test_id, config.family, qr,
                  sr.gen_tps, sr.ttft_ms)


# ---------------------------------------------------------------------------
# Tests — TurboQuant variations
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(TURBOQUANT_CONFIGS)),
    ids=[c.test_id for c in TURBOQUANT_CONFIGS],
)
def test_turboquant(config_index, server_manager, models_dir, bench_context):
    """Different KV cache quantization modes produce coherent output."""
    config = TURBOQUANT_CONFIGS[config_index]
    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    mem = _estimate_memory_gb(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), list(config.extra_args), memory_gb=mem)

    prompt = prompt_for_index(config_index + 3)
    sr = _streaming_chat_completion(base_url, prompt)

    assert sr.completion_tokens > 0, "no tokens generated"

    qr = _assert_quality(sr.content, config.test_id, prompt)
    _record_bench(bench_context, config.test_id, config.family, qr,
                  sr.gen_tps, sr.ttft_ms)


# ---------------------------------------------------------------------------
# Tests — Streaming protocol validation (SSE format, chunk delivery)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_openai_streaming(server_manager, models_dir, bench_context):
    """OpenAI-format SSE streaming returns incremental chunks of coherent text."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])
    if model_dir is None:
        pytest.skip("llama-3.2-3b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=7.2)

    prompt = PROMPTS[1]
    sr = _streaming_chat_completion(base_url, prompt)

    assert sr.chunk_count > 1, f"expected multiple SSE chunks, got {sr.chunk_count}"
    assert len(sr.content.strip()) > 0, "empty streamed response"

    _assert_quality(sr.content, "openai-streaming", prompt)


@pytest.mark.gpu
def test_anthropic_streaming(server_manager, models_dir, bench_context):
    """Anthropic-format SSE streaming returns incremental chunks."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])
    if model_dir is None:
        pytest.skip("llama-3.2-3b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=7.2)

    prompt = PROMPTS[2]
    resp = _anthropic_message(base_url, prompt, stream=True)
    assert resp.status_code == 200

    content, chunk_count = _collect_anthropic_sse_content(resp)

    assert chunk_count > 1, f"expected multiple SSE chunks, got {chunk_count}"
    assert len(content.strip()) > 0, "empty streamed response"

    _assert_quality(content, "anthropic-streaming", prompt)


# ---------------------------------------------------------------------------
# Tests — Anthropic API (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_anthropic_api(server_manager, models_dir, bench_context):
    """Anthropic /v1/messages endpoint returns coherent response."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])
    if model_dir is None:
        pytest.skip("llama-3.2-3b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=7.2)

    prompt = PROMPTS[5]
    resp = _anthropic_message(base_url, prompt)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    assert body.get("type") == "message"
    assert body.get("role") == "assistant"

    content_blocks = body.get("content", [])
    assert len(content_blocks) > 0, "no content blocks"

    text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text += block.get("text", "")

    assert len(text.strip()) > 0, "empty response"

    _assert_quality(text, "anthropic-api", prompt)

    usage = body.get("usage", {})
    assert usage.get("output_tokens", 0) > 0, "no output tokens"


# ---------------------------------------------------------------------------
# Tests — Vision (VLM) models
# ---------------------------------------------------------------------------

def _make_test_image() -> str:
    """Generate a simple 128x128 test image as a base64-encoded PNG.

    Creates a 2x2 grid: red (top-left), green (top-right),
    blue (bottom-left), white (bottom-right).
    """
    import struct
    import zlib

    width, height = 128, 128
    half = 64

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if y < half:
                if x < half:
                    row.extend([255, 0, 0, 255])
                else:
                    row.extend([0, 255, 0, 255])
            else:
                if x < half:
                    row.extend([0, 0, 255, 255])
                else:
                    row.extend([255, 255, 255, 255])
        rows.append(bytes(row))

    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")

    return base64.b64encode(png).decode("ascii")


def _vision_chat_completion(base_url: str, image_b64: str, prompt: str,
                            max_tokens: int = 256) -> requests.Response:
    """Send an OpenAI-format multimodal chat completion with an image."""
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=300,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(VISION_MODELS)),
    ids=[f"{c.test_id}-vision" for c in VISION_MODELS],
)
def test_vision(config_index, server_manager, models_dir, bench_context):
    """Vision models describe a simple coloured-quadrant image coherently."""
    config = VISION_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    extra_args = _build_extra_args(config, is_q4=False)
    mem = _estimate_memory_gb(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    image_b64 = _make_test_image()
    prompt = "Describe the colours you see in this image.  Be specific."

    resp = _vision_chat_completion(base_url, image_b64, prompt)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"
    assert len(content.strip()) > 0, "empty response"

    content_lower = content.lower()
    colours_found = [c for c in ["red", "green", "blue", "white"]
                     if c in content_lower]
    assert len(colours_found) >= 1, (
        f"vision model didn't mention any expected colours\n"
        f"Output: {content[:500]}"
    )

    _assert_quality(content, f"{config.test_id}-vision", prompt)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(VISION_MODELS)),
    ids=[f"{c.test_id}-vision-q4" for c in VISION_MODELS],
)
def test_vision_q4(config_index, server_manager, models_dir, bench_context):
    """Q4-quantized vision model variant describes images correctly."""
    config = VISION_MODELS[config_index]

    if config.family == "Gemma3":
        pytest.xfail("Gemma3 Q4 vision weights are U8 (unsupported) -- vision encoder skipped")

    model_dir = _resolve_model_dir(models_dir, config, q4=True)
    if model_dir is None:
        pytest.skip(f"Q4 model not found: {config.model_name}-q4")

    extra_args = _build_extra_args(config, is_q4=True)
    mem = _estimate_memory_gb(config, is_q4=True)
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    image_b64 = _make_test_image()
    prompt = "What colours are in this image?  List them."

    resp = _vision_chat_completion(base_url, image_b64, prompt)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]

    assert len(content.strip()) > 0, "empty response"

    content_lower = content.lower()
    colours_found = [c for c in ["red", "green", "blue", "white"]
                     if c in content_lower]
    assert len(colours_found) >= 1, (
        f"Q4 vision model didn't mention any expected colours\n"
        f"Output: {content[:500]}"
    )
