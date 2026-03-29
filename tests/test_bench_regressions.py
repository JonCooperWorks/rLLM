# ---------------------------------------------------------------------------
# test_bench_regressions.py — Regression tests for benchmark failure modes.
#
# Covers four specific failure patterns found in bench-20260329-121552.md:
#   1. GPT-OSS Q4 degenerate repetition (coherence check)
#   2. Base model detection for /v1/completions routing
#   3. Shard completeness validation (missing safetensors files)
#   4. Nemotron Q8 word salad (low space ratio coherence check)
#
# Related: bench.py, coherence.py, conftest.py
# ---------------------------------------------------------------------------

import json
import os
import tempfile
from pathlib import Path

import pytest

# Add test directory to path so we can import helpers.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coherence import check_coherence


# ---------------------------------------------------------------------------
# Issue #1: GPT-OSS Q4 repetition — coherence detects it correctly
# ---------------------------------------------------------------------------

class TestCoherenceRepetition:
    """Verify that check_coherence catches degenerate repetition patterns."""

    def test_catches_gpt_oss_repetition_pattern(self):
        """The 'the user says' repeated 10 times should fail coherence."""
        text = (
            "The meaning of life is something that the user says "
            "the user says the user says the user says the user says "
            "the user says the user says the user says the user says "
            "the user says the user says and then continues."
        )
        ok, reason = check_coherence(text)
        assert not ok, "should detect degenerate repetition"
        assert "repetition" in reason.lower()

    def test_catches_base_model_repetition_pattern(self):
        """The 'the meaning of' repeated 15 times should fail coherence."""
        text = "The meaning of " * 15 + "life is a beautiful thing."
        ok, reason = check_coherence(text)
        assert not ok, "should detect degenerate repetition"
        assert "repetition" in reason.lower()

    def test_passes_normal_text(self):
        """Normal English text should pass coherence."""
        text = (
            "The meaning of life is a deeply philosophical question that "
            "has been debated by thinkers for thousands of years. Some "
            "argue it lies in personal fulfillment, while others find "
            "meaning through relationships and service to others."
        )
        ok, reason = check_coherence(text)
        assert ok, f"normal text should pass: {reason}"


# ---------------------------------------------------------------------------
# Issue #4: Nemotron Q8 word salad — coherence detects low space ratio
# ---------------------------------------------------------------------------

class TestCoherenceWordSalad:
    """Verify that check_coherence catches word salad (low space ratio)."""

    def test_catches_low_space_ratio(self):
        """Text with <10% spaces should fail coherence."""
        # Simulate word salad: long concatenated fragments with few spaces.
        text = "abcdefghij" * 20 + " " + "klmnopqrst" * 20
        ok, reason = check_coherence(text)
        assert not ok, "should detect low space ratio"
        assert "space ratio" in reason.lower() or "word salad" in reason.lower()

    def test_passes_normal_space_ratio(self):
        """Normal text has ~15-20% spaces and should pass."""
        text = "This is a perfectly normal sentence with proper spacing between words. " * 5
        ok, reason = check_coherence(text)
        assert ok, f"normal spacing should pass: {reason}"


# ---------------------------------------------------------------------------
# Issue #2 + #3: Base model detection and shard validation
# ---------------------------------------------------------------------------

class TestBaseModelDetection:
    """Verify that bench.py can distinguish base from instruct models."""

    def test_instruct_models_detected(self):
        """Models with '-instruct' suffix are not base models."""
        from bench import is_base_model
        assert not is_base_model("llama-3.2-3b-instruct")
        assert not is_base_model("llama-3.1-8b-instruct")
        assert not is_base_model("qwen2.5-3b-instruct")

    def test_base_models_detected(self):
        """Models without '-instruct' or known chat suffix are base models."""
        from bench import is_base_model
        assert is_base_model("llama-3.2-3b")
        assert is_base_model("llama-3.2-1b")

    def test_chat_models_not_base(self):
        """Models with '-it' or known chat capability are not base models."""
        from bench import is_base_model
        assert not is_base_model("gemma-3-4b-it")
        assert not is_base_model("phi-4")  # Phi-4 is a chat model


class TestShardValidation:
    """Verify that missing safetensors shards are detected."""

    def test_detects_missing_shard(self):
        """Should detect when index.json references a shard that doesn't exist."""
        from bench import check_model_shards
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an index file referencing 2 shards.
            index = {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
            index_path = os.path.join(tmpdir, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump(index, f)

            # Only create shard 2.
            Path(os.path.join(tmpdir, "model-00002-of-00002.safetensors")).touch()

            ok, missing = check_model_shards(tmpdir)
            assert not ok, "should detect missing shard"
            assert "model-00001-of-00002.safetensors" in missing

    def test_passes_complete_model(self):
        """Should pass when all referenced shards exist."""
        from bench import check_model_shards
        with tempfile.TemporaryDirectory() as tmpdir:
            index = {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
            index_path = os.path.join(tmpdir, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump(index, f)

            # Create both shards.
            Path(os.path.join(tmpdir, "model-00001-of-00002.safetensors")).touch()
            Path(os.path.join(tmpdir, "model-00002-of-00002.safetensors")).touch()

            ok, missing = check_model_shards(tmpdir)
            assert ok, f"complete model should pass: missing={missing}"

    def test_single_file_model(self):
        """Single-file models (no index.json) should always pass."""
        from bench import check_model_shards
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "model.safetensors")).touch()

            ok, missing = check_model_shards(tmpdir)
            assert ok, "single-file model should pass"
