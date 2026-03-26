# ---------------------------------------------------------------------------
# test_model_families.py — GPU integration tests for every rLLM model family.
#
# Starts `rllm serve` for each model configuration, sends requests via the
# OpenAI / Anthropic HTTP API, and validates that outputs are coherent English.
#
# Tests are skipped when:
#   - No GPU is available (Metal on macOS, CUDA on Linux)
#   - The rllm binary hasn't been built
#   - A model's directory doesn't exist
#
# Run:  pytest tests/gpu-integration/ -v
# Run one family:  pytest tests/gpu-integration/ -v -k llama
#
# Related: conftest.py (fixtures), coherence.py (validation)
# ---------------------------------------------------------------------------

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests

from coherence import check_coherence
from conftest import _should_stream_experts


# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Describes one model configuration to test."""

    test_id: str
    model_name: str        # directory name under models/
    family: str            # architecture family
    is_moe: bool = False   # requires --stream-experts decision
    bf16_size_gb: float = 0  # estimated bf16 weight size for streaming decision
    extra_args: tuple[str, ...] = ()  # additional CLI args (e.g., --kv-quant)
    supports_stream_experts: bool = True  # some MoE models don't support it yet


# One model per architecture family — smallest available.
BASE_MODELS = [
    ModelConfig("llama-1b",    "llama-3.2-1b-instruct",           "Llama",     bf16_size_gb=2),
    ModelConfig("qwen2-3b",    "qwen2.5-3b-instruct",             "Qwen2",     bf16_size_gb=6),
    ModelConfig("gemma3-4b",   "gemma-3-4b-it",                   "Gemma3",    bf16_size_gb=8),
    ModelConfig("mistral-7b",  "mistral-7b-instruct-v0.3",        "Mistral",   bf16_size_gb=14),
    ModelConfig("qwen3.5-9b",  "qwen3.5-9b",                      "Qwen3_5",   bf16_size_gb=18),
    ModelConfig("phi-4",       "phi-4",                            "Phi",       bf16_size_gb=28),
    ModelConfig("mixtral-8x7b","mixtral-8x7b-instruct-v0.1",      "Mixtral",   is_moe=True, bf16_size_gb=93),
    ModelConfig("qwen3moe-30b","qwen3-coder-30b-a3b-instruct",    "Qwen3Moe",  is_moe=True, bf16_size_gb=60),
    ModelConfig("qwen3.5-35b", "qwen3.5-35b-a3b",                 "Qwen3_5M",  is_moe=True, bf16_size_gb=70),
    ModelConfig("gpt-oss-20b", "gpt-oss-20b",                     "GptOss",    is_moe=True, bf16_size_gb=40, supports_stream_experts=False),
]

# TurboQuant variations (tested on Llama 1B — fastest to load).
TURBOQUANT_CONFIGS = [
    ModelConfig("turbo4-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "turbo4")),
    ModelConfig("turbo2-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "turbo2")),
    ModelConfig("no-kv-quant-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "none")),
]

# The prompt used for all inference tests.  Factual, short, easy for any model.
PROMPT = "Explain what a hash table is in two sentences."
MAX_TOKENS = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(models_dir: Path, config: ModelConfig, q4: bool = False) -> Path | None:
    """Return the model directory path, or None if it doesn't exist."""
    name = config.model_name + ("-q4" if q4 else "")
    d = models_dir / name
    return d if d.is_dir() else None


def _build_extra_args(config: ModelConfig, is_q4: bool) -> list[str]:
    """Build CLI extra args, including --stream-experts decision for MoE."""
    args = list(config.extra_args)
    if config.is_moe and config.supports_stream_experts:
        if _should_stream_experts(config.bf16_size_gb, is_q4):
            args.extend(["--stream-experts"])
    return args


def _chat_completion(base_url: str, prompt: str, max_tokens: int = MAX_TOKENS,
                     temperature: float = 0, stream: bool = False) -> requests.Response:
    """Send an OpenAI-format chat completion request."""
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        },
        timeout=120,
        stream=stream,
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
        timeout=120,
        stream=stream,
    )


def _collect_sse_content(response: requests.Response) -> tuple[str, int]:
    """Parse SSE stream, return (concatenated content, chunk_count)."""
    content_parts = []
    chunk_count = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                content_parts.append(text)
                chunk_count += 1
        except json.JSONDecodeError:
            continue
    return "".join(content_parts), chunk_count


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


# ---------------------------------------------------------------------------
# Tests — Basic model inference (one per family, bf16 + optional Q4)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("config", BASE_MODELS, ids=lambda c: c.test_id)
def test_model_bf16(config, server_manager, models_dir):
    """Each model family produces coherent English via the OpenAI API."""
    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    extra_args = _build_extra_args(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), extra_args)

    resp = _chat_completion(base_url, PROMPT)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"
    assert len(content.strip()) > 0, "empty response"

    ok, reason = check_coherence(content)
    assert ok, f"coherence check failed for {config.test_id}: {reason}\nOutput: {content[:500]}"


@pytest.mark.gpu
@pytest.mark.parametrize("config", BASE_MODELS, ids=lambda c: f"{c.test_id}-q4")
def test_model_q4(config, server_manager, models_dir):
    """Q4-quantized variant of each model family (skipped if not quantized)."""
    model_dir = _resolve_model_dir(models_dir, config, q4=True)
    if model_dir is None:
        pytest.skip(f"Q4 model not found: {config.model_name}-q4")

    extra_args = _build_extra_args(config, is_q4=True)
    base_url = server_manager.get_or_start(str(model_dir), extra_args)

    resp = _chat_completion(base_url, PROMPT)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"

    ok, reason = check_coherence(content)
    assert ok, f"coherence check failed for {config.test_id}-q4: {reason}\nOutput: {content[:500]}"


# ---------------------------------------------------------------------------
# Tests — TurboQuant variations
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("config", TURBOQUANT_CONFIGS, ids=lambda c: c.test_id)
def test_turboquant(config, server_manager, models_dir):
    """Different KV cache quantization modes produce coherent output."""
    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    base_url = server_manager.get_or_start(str(model_dir), list(config.extra_args))

    resp = _chat_completion(base_url, PROMPT)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]

    ok, reason = check_coherence(content)
    assert ok, f"coherence check failed for {config.test_id}: {reason}\nOutput: {content[:500]}"


# ---------------------------------------------------------------------------
# Tests — Streaming (SSE)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_openai_streaming(server_manager, models_dir):
    """OpenAI-format SSE streaming returns incremental chunks of coherent text."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [])

    resp = _chat_completion(base_url, PROMPT, stream=True)
    assert resp.status_code == 200

    content, chunk_count = _collect_sse_content(resp)

    assert chunk_count > 1, f"expected multiple SSE chunks, got {chunk_count}"
    assert len(content.strip()) > 0, "empty streamed response"

    ok, reason = check_coherence(content)
    assert ok, f"streaming coherence failed: {reason}\nOutput: {content[:500]}"


@pytest.mark.gpu
def test_anthropic_streaming(server_manager, models_dir):
    """Anthropic-format SSE streaming returns incremental chunks."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [])

    resp = _anthropic_message(base_url, PROMPT, stream=True)
    assert resp.status_code == 200

    content, chunk_count = _collect_anthropic_sse_content(resp)

    assert chunk_count > 1, f"expected multiple SSE chunks, got {chunk_count}"
    assert len(content.strip()) > 0, "empty streamed response"

    ok, reason = check_coherence(content)
    assert ok, f"anthropic streaming coherence failed: {reason}\nOutput: {content[:500]}"


# ---------------------------------------------------------------------------
# Tests — Anthropic API (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_anthropic_api(server_manager, models_dir):
    """Anthropic /v1/messages endpoint returns coherent response."""
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [])

    resp = _anthropic_message(base_url, PROMPT)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    assert body.get("type") == "message"
    assert body.get("role") == "assistant"

    # Content is an array of blocks.
    content_blocks = body.get("content", [])
    assert len(content_blocks) > 0, "no content blocks"

    text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text += block.get("text", "")

    assert len(text.strip()) > 0, "empty response"

    ok, reason = check_coherence(text)
    assert ok, f"anthropic coherence failed: {reason}\nOutput: {text[:500]}"

    # Validate usage.
    usage = body.get("usage", {})
    assert usage.get("output_tokens", 0) > 0, "no output tokens"
