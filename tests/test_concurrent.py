# ---------------------------------------------------------------------------
# test_concurrent.py — Integration tests for concurrent request handling.
#
# Verifies that the continuous batching engine correctly interleaves prefill
# across sequences (no head-of-line blocking) and that timed-out requests
# free resources for subsequent requests.
#
# Requires a GPU and the rllm binary.  Uses llama-3.2-3b-instruct as the
# test model (same as the streaming API tests).
#
# Run:  uv run pytest tests/test_concurrent.py -v
# ---------------------------------------------------------------------------

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests

from models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_dir(models_dir: Path) -> Path | None:
    """Return the llama-3.2-3b-instruct model directory, or None."""
    d = models_dir / "llama-3.2-3b-instruct"
    return d if d.is_dir() else None


def _streaming_chat(base_url: str, prompt: str, max_tokens: int = 64,
                    timeout: float = 300) -> tuple[str, float]:
    """Send a streaming chat completion and return (content, ttft_seconds).

    TTFT = time from request start to first content token received.
    """
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=body, timeout=timeout, stream=True,
    )
    resp.raise_for_status()

    ttft = None
    content_parts = []
    for line in resp.iter_lines(decode_unicode=True):
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
                if ttft is None:
                    ttft = time.monotonic() - start
                content_parts.append(text)
        except json.JSONDecodeError:
            continue

    content = "".join(content_parts)
    if ttft is None:
        ttft = time.monotonic() - start
    return content, ttft


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_short_not_blocked_by_long_prefill(server_manager, models_dir):
    """A short request should start generating quickly even when a long
    request is prefilling concurrently.

    We send a request with a ~4k token prompt and a request with a ~20 token
    prompt at the same time.  The short request's TTFT should be much less
    than the long request's TTFT, proving interleaved prefill works.
    """
    model_dir = _resolve_model_dir(models_dir)
    if model_dir is None:
        pytest.skip("llama-3.2-3b-instruct not found")

    base_url = server_manager.get_or_start(str(model_dir), [], memory_gb=7.2)

    # Long prompt: ~1000 tokens of filler text to force multiple prefill chunks.
    long_prompt = "Summarize: " + ("The quick brown fox jumps over the lazy dog. " * 100)
    # Short prompt: trivial
    short_prompt = "What is 2+2?"

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Submit long request first, then short request shortly after.
        long_future = pool.submit(_streaming_chat, base_url, long_prompt, 16)
        time.sleep(0.1)  # small delay so long request is registered first
        short_future = pool.submit(_streaming_chat, base_url, short_prompt, 16)

        long_content, long_ttft = long_future.result(timeout=120)
        short_content, short_ttft = short_future.result(timeout=120)

    print(f"\n  Long TTFT:  {long_ttft:.2f}s")
    print(f"  Short TTFT: {short_ttft:.2f}s")

    # The short request should not be gated by the long request's full prefill.
    # Allow generous margin — the key property is short_ttft << long_ttft.
    assert len(short_content) > 0, "short request should produce output"
    assert len(long_content) > 0, "long request should produce output"
    assert short_ttft < long_ttft, (
        f"short request TTFT ({short_ttft:.2f}s) should be less than "
        f"long request TTFT ({long_ttft:.2f}s)"
    )


@pytest.mark.gpu
def test_timeout_then_normal_request(server_manager, models_dir):
    """After a request times out, subsequent requests should succeed without
    delay — the timed-out request's KV blocks must be reclaimed.

    Start the server with a short request timeout, send a long request that
    will timeout, then verify a normal request succeeds promptly.
    """
    model_dir = _resolve_model_dir(models_dir)
    if model_dir is None:
        pytest.skip("llama-3.2-3b-instruct not found")

    # Start with a 3-second request timeout.
    base_url = server_manager.get_or_start(
        str(model_dir), ["--request-timeout", "3"], memory_gb=7.2,
    )

    # Send a long-prompt request that should timeout.
    long_prompt = "Summarize: " + ("word " * 3000)
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": long_prompt}],
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=30,
        )
        # The server may return an error due to timeout, or generate partial
        # output.  Either way, we just want it to complete/fail.
    except requests.exceptions.RequestException:
        pass  # Timeout or connection error is expected

    # Wait a moment for cleanup.
    time.sleep(1)

    # Now send a normal, short request.  It should succeed quickly.
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=30,
    )
    elapsed = time.monotonic() - start

    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert len(content.strip()) > 0, "response should not be empty"
    assert elapsed < 10, (
        f"normal request after timeout took {elapsed:.1f}s — "
        f"blocks may not have been freed"
    )
    print(f"\n  Normal request after timeout: {elapsed:.2f}s, content: {content[:60]}")
