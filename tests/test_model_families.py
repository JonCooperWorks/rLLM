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
# Run:  pytest tests/ -v
# Run one family:  pytest tests/ -v -k llama
#
# Related: conftest.py (fixtures), coherence.py (validation)
# ---------------------------------------------------------------------------

import base64
import io
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
    temperature: float = 0  # sampling temperature (0 = greedy)
    has_vision: bool = False  # model supports image input


# One model per architecture family — smallest available.
BASE_MODELS = [
    ModelConfig("llama-1b",    "llama-3.2-1b-instruct",           "Llama",     bf16_size_gb=2),
    ModelConfig("qwen2-3b",    "qwen2.5-3b-instruct",             "Qwen2",     bf16_size_gb=6),
    ModelConfig("gemma3-4b",   "gemma-3-4b-it",                   "Gemma3",    bf16_size_gb=8, has_vision=True),
    ModelConfig("mistral-7b",  "mistral-7b-instruct-v0.3",        "Mistral",   bf16_size_gb=14),
    # Qwen3.5 Q4 text generates 0 tokens (thinking mode interaction). bf16 works.
    ModelConfig("qwen3.5-9b",  "qwen3.5-9b",                      "Qwen3_5",   bf16_size_gb=18, has_vision=True),
    ModelConfig("phi-4",       "phi-4",                            "Phi",       bf16_size_gb=28),
    ModelConfig("mixtral-8x7b","mixtral-8x7b-instruct-v0.1",      "Mixtral",   is_moe=True, bf16_size_gb=93),
    ModelConfig("qwen3moe-30b","qwen3-coder-30b-a3b-instruct",    "Qwen3Moe",  is_moe=True, bf16_size_gb=60),
    ModelConfig("qwen3.5-35b", "qwen3.5-35b-a3b",                 "Qwen3_5M",  is_moe=True, bf16_size_gb=70, has_vision=True),
    # GPT-OSS Q4 degenerates into self-evaluation loops at low temperature due to
    # reduced logit precision.  temperature=0.6 adds enough noise to break the loop.
    ModelConfig("gpt-oss-20b", "gpt-oss-20b",                     "GptOss",    is_moe=True, bf16_size_gb=40, supports_stream_experts=False, temperature=0.6),
    ModelConfig("nemotron-h-30b", "nemotron-3-30b",               "NemotronH", is_moe=True, bf16_size_gb=63),
]

# TurboQuant variations (tested on Llama 1B — fastest to load).
TURBOQUANT_CONFIGS = [
    ModelConfig("turbo4-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "turbo4")),
    ModelConfig("turbo2-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "turbo2")),
    ModelConfig("no-kv-quant-llama", "llama-3.2-1b-instruct", "Llama", extra_args=("--kv-quant", "none")),
]

# Varied prompts that exercise different generation patterns.  Each model
# family test uses a different prompt so we're not just testing one pattern.
# All are factual / instructional — any instruction-tuned model should handle
# them, and the output is easy to validate for English coherence.
PROMPTS = [
    # Short factual — tests basic instruction following.
    "Explain what a hash table is in two sentences.",
    # Longer expository — tests sustained multi-paragraph generation.
    "Write a short paragraph explaining how the internet works, from typing "
    "a URL in a browser to seeing the webpage.  Include DNS, TCP, and HTTP.",
    # Enumeration / structured — tests list generation and formatting.
    "List five common sorting algorithms and give a one-sentence description "
    "of how each one works.",
    # Reasoning / comparison — tests coherent argumentation.
    "Compare and contrast Python and Rust.  What are the strengths and "
    "weaknesses of each language?  Give concrete examples.",
    # Creative but bounded — tests fluency without drifting into gibberish.
    "Write a short story in exactly three sentences about a robot that "
    "discovers it can dream.",
    # Technical explanation — tests domain vocabulary coherence.
    "Explain how public-key cryptography works.  Include the roles of the "
    "public key, private key, and why it is difficult to reverse.",
    # Step-by-step — tests sequential reasoning.
    "Walk me through the steps to deploy a web application to a cloud "
    "provider.  Cover DNS setup, containerisation, and monitoring.",
    # Concise summary — tests compression and accuracy.
    "Summarise the key ideas behind MapReduce in a few sentences.",
    # Multi-part question — tests handling compound prompts.
    "What is gradient descent?  Why is the learning rate important?  "
    "What happens if it is set too high or too low?",
    # Opinion / analysis — tests balanced generation.
    "What are the trade-offs between microservices and monolithic "
    "architectures?  When would you choose one over the other?",
]

# Realistic max_tokens — long enough to exercise sustained generation and
# KV cache behavior (prefill + many decode steps), but not so long that
# tests take forever.  512 tokens is ~400 words, a solid paragraph or two.
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(models_dir: Path, config: ModelConfig, q4: bool = False,
                       q8: bool = False) -> Path | None:
    """Return the model directory path, or None if it doesn't exist."""
    suffix = "-q4" if q4 else ("-q8" if q8 else "")
    name = config.model_name + suffix
    d = models_dir / name
    return d if d.is_dir() else None


def _build_extra_args(config: ModelConfig, is_q4: bool) -> list[str]:
    """Build CLI extra args, including --stream-experts decision for MoE."""
    args = list(config.extra_args)
    if config.is_moe and config.supports_stream_experts:
        if _should_stream_experts(config.bf16_size_gb, is_q4):
            args.extend(["--stream-experts"])
    return args


def _estimate_memory_gb(config: ModelConfig, is_q4: bool) -> float:
    """Estimate GPU memory required for a model (weights + KV cache overhead).

    Used by ServerManager to evict servers before OOM.  Streamed-expert MoE
    models only load active experts, so we estimate ~25% of total weight size.
    """
    weight_gb = config.bf16_size_gb * 0.5 if is_q4 else config.bf16_size_gb
    if config.is_moe and config.supports_stream_experts:
        if _should_stream_experts(config.bf16_size_gb, is_q4):
            weight_gb *= 0.25  # only active experts in memory
    return weight_gb * 1.2  # +20% overhead for KV cache, activations, scratch


def _prompt_for_index(index: int) -> str:
    """Return a prompt from the PROMPTS list, cycling if index exceeds length."""
    return PROMPTS[index % len(PROMPTS)]


def _chat_completion(base_url: str, prompt: str, max_tokens: int = MAX_TOKENS,
                     temperature: float = 0, stream: bool = False,
                     thinking: bool = False) -> requests.Response:
    """Send an OpenAI-format chat completion request.

    thinking=False (default) disables extended thinking so models like Qwen3.5
    don't spend their entire token budget on chain-of-thought reasoning.
    """
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "thinking": thinking,
    }
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=300,
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
        timeout=300,
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
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[c.test_id for c in BASE_MODELS],
)
def test_model_bf16(config_index, server_manager, models_dir):
    """Each model family produces coherent English via the OpenAI API.

    Each model gets a different prompt from the PROMPTS list so we exercise
    varied generation patterns across the suite.
    """
    config = BASE_MODELS[config_index]

    # BF16 MoE models that require expert streaming are too slow (0.4 tok/s for
    # 336 MB bf16 experts per token) — skip in favour of the Q4 variant.
    if config.is_moe and _should_stream_experts(config.bf16_size_gb, is_q4=False):
        pytest.skip(f"bf16 MoE too large for memory ({config.bf16_size_gb}GB), use Q4")

    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    extra_args = _build_extra_args(config, is_q4=False)
    mem = _estimate_memory_gb(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    prompt = _prompt_for_index(config_index)
    resp = _chat_completion(base_url, prompt, temperature=config.temperature)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"
    assert len(content.strip()) > 0, "empty response"

    ok, reason = check_coherence(content)
    assert ok, (
        f"coherence check failed for {config.test_id}: {reason}\n"
        f"Prompt: {prompt}\nOutput: {content[:500]}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[f"{c.test_id}-q4" for c in BASE_MODELS],
)
def test_model_q4(config_index, server_manager, models_dir):
    """Q4-quantized variant of each model family (skipped if not quantized).

    Uses a different prompt than the bf16 test (offset by 5) so each model
    is tested with two distinct prompts across bf16 and Q4.
    """
    config = BASE_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config, q4=True)
    if model_dir is None:
        pytest.skip(f"Q4 model not found: {config.model_name}-q4")

    extra_args = _build_extra_args(config, is_q4=True)
    mem = _estimate_memory_gb(config, is_q4=True)
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    prompt = _prompt_for_index(config_index + 5)
    resp = _chat_completion(base_url, prompt, temperature=config.temperature)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"

    ok, reason = check_coherence(content)
    assert ok, (
        f"coherence check failed for {config.test_id}-q4: {reason}\n"
        f"Prompt: {prompt}\nOutput: {content[:500]}"
    )


# ---------------------------------------------------------------------------
# Tests — Q8-quantized model inference
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(BASE_MODELS)),
    ids=[f"{c.test_id}-q8" for c in BASE_MODELS],
)
def test_model_q8(config_index, server_manager, models_dir):
    """Q8-quantized variant of each model family (skipped if not quantized).

    Uses a different prompt than bf16 and Q4 tests (offset by 3) so each model
    is tested with three distinct prompts across bf16, Q4, and Q8.
    """
    config = BASE_MODELS[config_index]

    model_dir = _resolve_model_dir(models_dir, config, q8=True)
    if model_dir is None:
        pytest.skip(f"Q8 model not found: {config.model_name}-q8")

    extra_args = _build_extra_args(config, is_q4=False)
    # Q8 is ~53% of bf16 size (34/64 bytes per weight)
    mem = _estimate_memory_gb(config, is_q4=False) * 0.53
    base_url = server_manager.get_or_start(str(model_dir), extra_args, memory_gb=mem)

    prompt = _prompt_for_index(config_index + 3)
    resp = _chat_completion(base_url, prompt, temperature=config.temperature)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    assert usage.get("completion_tokens", 0) > 0, "no tokens generated"

    ok, reason = check_coherence(content)
    assert ok, (
        f"coherence check failed for {config.test_id}-q8: {reason}\n"
        f"Prompt: {prompt}\nOutput: {content[:500]}"
    )


# ---------------------------------------------------------------------------
# Tests — TurboQuant variations
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(TURBOQUANT_CONFIGS)),
    ids=[c.test_id for c in TURBOQUANT_CONFIGS],
)
def test_turboquant(config_index, server_manager, models_dir):
    """Different KV cache quantization modes produce coherent output.

    Tests turbo4 (default 4-bit), turbo2 (aggressive 2-bit), and none
    (BF16 baseline) on the same model with different prompts.  512 tokens
    exercises the KV cache across many decode steps.
    """
    config = TURBOQUANT_CONFIGS[config_index]
    model_dir = _resolve_model_dir(models_dir, config)
    if model_dir is None:
        pytest.skip(f"model not found: {config.model_name}")

    mem = _estimate_memory_gb(config, is_q4=False)
    base_url = server_manager.get_or_start(str(model_dir), list(config.extra_args), memory_gb=mem)

    prompt = _prompt_for_index(config_index + 3)
    resp = _chat_completion(base_url, prompt)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    body = resp.json()
    content = body["choices"][0]["message"]["content"]

    ok, reason = check_coherence(content)
    assert ok, (
        f"coherence check failed for {config.test_id}: {reason}\n"
        f"Prompt: {prompt}\nOutput: {content[:500]}"
    )


# ---------------------------------------------------------------------------
# Tests — Streaming (SSE)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_openai_streaming(server_manager, models_dir):
    """OpenAI-format SSE streaming returns incremental chunks of coherent text.

    Uses the expository internet prompt at 512 tokens to exercise sustained
    streaming over many decode steps.
    """
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=2.4)

    prompt = PROMPTS[1]  # Internet / DNS / TCP / HTTP — longer response expected.
    resp = _chat_completion(base_url, prompt, stream=True)
    assert resp.status_code == 200

    content, chunk_count = _collect_sse_content(resp)

    assert chunk_count > 1, f"expected multiple SSE chunks, got {chunk_count}"
    assert len(content.strip()) > 0, "empty streamed response"

    ok, reason = check_coherence(content)
    assert ok, f"streaming coherence failed: {reason}\nOutput: {content[:500]}"


@pytest.mark.gpu
def test_anthropic_streaming(server_manager, models_dir):
    """Anthropic-format SSE streaming returns incremental chunks.

    Uses the sorting algorithms prompt to exercise list-style generation.
    """
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=2.4)

    prompt = PROMPTS[2]  # Sorting algorithms enumeration.
    resp = _anthropic_message(base_url, prompt, stream=True)
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
    """Anthropic /v1/messages endpoint returns coherent response.

    Uses the public-key cryptography prompt — a technical explanation that
    any model should handle coherently at 512 tokens.
    """
    model_dir = _resolve_model_dir(models_dir, BASE_MODELS[0])  # Llama 1B
    if model_dir is None:
        pytest.skip("llama-3.2-1b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=2.4)

    prompt = PROMPTS[5]  # Public-key cryptography.
    resp = _anthropic_message(base_url, prompt)
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


# ---------------------------------------------------------------------------
# Tests — Vision (VLM) models
# ---------------------------------------------------------------------------

# Filter BASE_MODELS to those with vision support.
VISION_MODELS = [c for c in BASE_MODELS if c.has_vision]


def _make_test_image() -> str:
    """Generate a simple 128×128 test image as a base64-encoded PNG.

    Creates a 2×2 grid: red (top-left), green (top-right),
    blue (bottom-left), white (bottom-right).  This is trivially
    recognisable by any VLM, and tiny enough to not slow down tests.
    Returns a base64 string (no data URL prefix).
    """
    # Minimal PNG via raw pixel data — no PIL dependency needed.
    import struct
    import zlib

    width, height = 128, 128
    half = 64

    # Build raw RGBA pixel rows (filter byte 0 = None at start of each row).
    rows = []
    for y in range(height):
        row = bytearray([0])  # PNG filter: None
        for x in range(width):
            if y < half:
                if x < half:
                    row.extend([255, 0, 0, 255])      # red
                else:
                    row.extend([0, 255, 0, 255])      # green
            else:
                if x < half:
                    row.extend([0, 0, 255, 255])      # blue
                else:
                    row.extend([255, 255, 255, 255])  # white
        rows.append(bytes(row))

    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    png = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=6 (RGBA)
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
def test_vision(config_index, server_manager, models_dir):
    """Vision models describe a simple coloured-quadrant image coherently.

    Sends a 128×128 PNG with red/green/blue/white quadrants and asks the
    model to describe the colours.  Validates that the response mentions at
    least one colour, confirming the vision encoder + projector + LLM pipeline
    works end-to-end.
    """
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

    # The model should mention at least one colour from the test image.
    content_lower = content.lower()
    colours_found = [c for c in ["red", "green", "blue", "white"]
                     if c in content_lower]
    assert len(colours_found) >= 1, (
        f"vision model didn't mention any expected colours\n"
        f"Output: {content[:500]}"
    )

    # Also check general coherence (no word salad).
    ok, reason = check_coherence(content)
    assert ok, (
        f"vision coherence failed for {config.test_id}: {reason}\n"
        f"Output: {content[:500]}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "config_index", range(len(VISION_MODELS)),
    ids=[f"{c.test_id}-vision-q4" for c in VISION_MODELS],
)
def test_vision_q4(config_index, server_manager, models_dir):
    """Q4-quantized vision model variant describes images correctly."""
    config = VISION_MODELS[config_index]

    # Gemma3 Q4 distributions quantize vision weights to U8 (unsupported).
    # The loader skips the vision encoder entirely, so the model can't see images.
    # This is a dataset/distribution issue, not an inference bug.
    if config.family == "Gemma3":
        pytest.xfail("Gemma3 Q4 vision weights are U8 (unsupported) — vision encoder skipped")

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
