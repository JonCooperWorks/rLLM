# ---------------------------------------------------------------------------
# test_bench_regressions.py — Regression tests for quality checks and model
# registry helpers.
#
# Covers failure patterns found in real benchmark runs plus regression tests
# for the new grammar/structure quality checks (checks 8-11).
#
# Run:  pytest tests/test_bench_regressions.py -v
#
# Related: quality.py (validation), models.py (registry helpers)
# ---------------------------------------------------------------------------

import json
import os
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality import check_quality, check_coherence


# ---------------------------------------------------------------------------
# Original coherence regression tests (checks 1-7)
# ---------------------------------------------------------------------------

class TestCoherenceRepetition:
    """Verify that check_quality catches degenerate repetition patterns."""

    def test_catches_gpt_oss_repetition_pattern(self):
        """The 'the user says' repeated 10 times should fail."""
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
        """The 'the meaning of' repeated 15 times should fail."""
        text = "The meaning of " * 15 + "life is a beautiful thing."
        ok, reason = check_coherence(text)
        assert not ok, "should detect degenerate repetition"
        assert "repetition" in reason.lower()

    def test_passes_normal_text(self):
        """Normal English text should pass."""
        text = (
            "The meaning of life is a deeply philosophical question that "
            "has been debated by thinkers for thousands of years. Some "
            "argue it lies in personal fulfillment, while others find "
            "meaning through relationships and service to others."
        )
        ok, reason = check_coherence(text)
        assert ok, f"normal text should pass: {reason}"


class TestCoherenceWordSalad:
    """Verify that check_quality catches word salad (low space ratio)."""

    def test_catches_low_space_ratio(self):
        text = "abcdefghij" * 20 + " " + "klmnopqrst" * 20
        ok, reason = check_coherence(text)
        assert not ok, "should detect low space ratio"
        assert "space ratio" in reason.lower() or "word salad" in reason.lower()

    def test_passes_normal_space_ratio(self):
        text = "This is a perfectly normal sentence with proper spacing between words. " * 5
        ok, reason = check_coherence(text)
        assert ok, f"normal spacing should pass: {reason}"


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

class TestBaseModelDetection:
    """Verify that is_base_model distinguishes base from instruct models."""

    def test_instruct_models_detected(self):
        from models import is_base_model
        assert not is_base_model("llama-3.2-3b-instruct")
        assert not is_base_model("llama-3.1-8b-instruct")
        assert not is_base_model("qwen2.5-3b-instruct")

    def test_base_models_detected(self):
        from models import is_base_model
        assert is_base_model("llama-3.2-3b")
        assert is_base_model("llama-3.2-1b")

    def test_chat_models_not_base(self):
        from models import is_base_model
        assert not is_base_model("gemma-3-4b-it")
        assert not is_base_model("phi-4")


class TestShardValidation:
    """Verify that missing safetensors shards are detected."""

    def test_detects_missing_shard(self):
        from models import check_model_shards
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
            Path(os.path.join(tmpdir, "model-00002-of-00002.safetensors")).touch()

            ok, missing = check_model_shards(tmpdir)
            assert not ok, "should detect missing shard"
            assert "model-00001-of-00002.safetensors" in missing

    def test_passes_complete_model(self):
        from models import check_model_shards
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
            Path(os.path.join(tmpdir, "model-00001-of-00002.safetensors")).touch()
            Path(os.path.join(tmpdir, "model-00002-of-00002.safetensors")).touch()

            ok, missing = check_model_shards(tmpdir)
            assert ok, f"complete model should pass: missing={missing}"

    def test_single_file_model(self):
        from models import check_model_shards
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "model.safetensors")).touch()
            ok, missing = check_model_shards(tmpdir)
            assert ok, "single-file model should pass"


# ---------------------------------------------------------------------------
# New quality check regression tests (checks 8-11)
# ---------------------------------------------------------------------------

class TestSentenceStructure:
    """Verify sentence structure checks (check 8)."""

    def test_catches_missing_capitalization(self):
        """Many lowercase sentence starts should fail."""
        text = (
            "This is fine. but this is not. and neither is this. "
            "or this one. but wait there is more. and even more. "
            "still going. yet another one."
        )
        result = check_quality(text)
        assert not result.passed, "should detect capitalization issues"
        assert "capitalization" in result.reason.lower()

    def test_passes_proper_capitalization(self):
        """Properly capitalized text should pass."""
        text = (
            "The first sentence is here. The second follows naturally. "
            "A third sentence completes the paragraph. Each one starts "
            "with a capital letter. This is how English works."
        )
        result = check_quality(text)
        assert result.passed, f"proper capitalization should pass: {result.reason}"

    def test_catches_fragmented_sentences(self):
        """Median sentence length under 3 words should fail."""
        text = "Yes. No. Maybe. Sure. Ok. Fine. Right. Good. Yes. No. Maybe. True."
        result = check_quality(text)
        assert not result.passed, "should detect fragmented sentences"
        assert "fragmented" in result.reason.lower()

    def test_allows_truncated_ending(self):
        """Text truncated at max_tokens mid-sentence should still pass.

        Output cut at max_tokens naturally ends mid-sentence — that's
        expected behavior, not broken inference.
        """
        text = (
            "A hash table is a data structure that maps keys to values. "
            "It uses a hash function to compute an index. "
            "The key advantage of hash tables is the ability to provide "
            "constant time lookup for data that is stored in the"
        )
        result = check_quality(text)
        assert result.passed, f"truncated text should pass: {result.reason}"

    def test_passes_complete_ending(self):
        """Text ending with a complete sentence should pass."""
        text = (
            "A hash table is a data structure that maps keys to values. "
            "It uses a hash function to compute an index into an array "
            "of buckets, from which the desired value can be found."
        )
        result = check_quality(text)
        assert result.passed, f"complete text should pass: {result.reason}"


class TestPunctuationPatterns:
    """Verify punctuation pattern checks (check 9)."""

    def test_catches_excessive_punctuation(self):
        """5+ consecutive punctuation chars should fail."""
        text = "This is great!!!!! Really amazing content here with proper length and structure."
        result = check_quality(text)
        assert not result.passed, "should detect excessive punctuation"
        assert "punctuation run" in result.reason.lower()

    def test_allows_markdown_separators(self):
        """Markdown horizontal rules and table separators should pass."""
        text = (
            "This is the first section.\n\n"
            "---\n\n"
            "This is the second section with proper formatting. "
            "It has multiple sentences and reads naturally."
        )
        result = check_quality(text)
        assert result.passed, f"markdown separators should pass: {result.reason}"

    def test_allows_markdown_headings(self):
        """Markdown heading markers (####) should pass."""
        text = (
            "Here is a guide to deployment.\n\n"
            "#### Step 1: Set Up DNS\n"
            "Configure your domain name. "
            "#### Step 2: Containerize\n"
            "Build a Docker image for your application."
        )
        result = check_quality(text)
        assert result.passed, f"markdown headings should pass: {result.reason}"

    def test_allows_markdown_bold_with_punctuation(self):
        """Bold markers mixed with parens/commas should pass."""
        text = (
            "Public-key cryptography (also known as **asymmetric cryptography**), "
            "uses two keys. The public key encrypts data and the private key "
            "decrypts it. This is fundamental to modern secure communications."
        )
        result = check_quality(text)
        assert result.passed, f"bold+punctuation should pass: {result.reason}"

    def test_allows_latex_notation(self):
        r"""LaTeX math notation like \(\theta\) should pass."""
        text = (
            r"Gradient descent updates parameters using the formula "
            r"\(\theta = \theta - \alpha \nabla J(\theta)\) where alpha "
            "is the learning rate. This iterative process converges to "
            "a local minimum of the cost function."
        )
        result = check_quality(text)
        assert result.passed, f"LaTeX notation should pass: {result.reason}"

    def test_strips_thinking_tags(self):
        """<think> tags should be stripped before quality checks."""
        text = (
            "The meaning of life is a philosophical question.</think> "
            "Many thinkers have pondered this throughout history. "
            "Some argue it lies in personal fulfillment and growth."
        )
        result = check_quality(text)
        assert result.passed, f"thinking tags should be stripped: {result.reason}"

    def test_allows_ellipsis(self):
        """Ellipsis (3-4 dots) should pass."""
        text = (
            "The meaning of life is... well, that depends on who you ask. "
            "Philosophers have debated this for centuries. "
            "Some say it is about happiness, others about purpose."
        )
        result = check_quality(text)
        assert result.passed, f"ellipsis should pass: {result.reason}"

    def test_catches_unmatched_brackets(self):
        """More than 3 unmatched brackets should fail."""
        text = (
            "The function takes (first value (second value (third value "
            "(fourth value and processes them to produce the output. "
            "This is a long enough sentence with enough content."
        )
        result = check_quality(text)
        assert not result.passed, "should detect unmatched brackets"
        assert "unmatched" in result.reason.lower()

    def test_passes_balanced_brackets(self):
        """Balanced brackets should pass."""
        text = (
            "The function f(x) returns (a + b) when called with "
            "the parameter [0, 1, 2]. This is documented in the "
            "API reference (see section 3)."
        )
        result = check_quality(text)
        assert result.passed, f"balanced brackets should pass: {result.reason}"

    def test_allows_markdown_tables(self):
        """Markdown tables with mixed punctuation should pass."""
        text = (
            "Public-key cryptography uses two mathematically linked keys.\n\n"
            "| Algorithm | Key Size | Use Case |\n"
            "|-----------|----------|----------|\n"
            '| RSA | 2048-bit | Encryption ("standard"). |\n'
            "| ECDSA | 256-bit | Digital signatures |\n"
            "| Ed25519 | 256-bit | SSH, TLS |\n\n"
            "These algorithms form the basis of modern secure communications."
        )
        result = check_quality(text)
        assert result.passed, f"markdown tables should pass: {result.reason}"

    def test_allows_smart_quotes(self):
        """Smart/curly quotes in prose should pass."""
        text = (
            "The professor said \u201cquantum computing will change everything.\u201d "
            "Her colleague\u2019s response was measured \u2014 he noted that "
            "practical applications remain years away. The field continues "
            "to attract significant research investment worldwide."
        )
        result = check_quality(text)
        assert result.passed, f"smart quotes should pass: {result.reason}"


class TestVocabularyDiversity:
    """Verify vocabulary diversity check (check 11)."""

    def test_catches_low_diversity(self):
        """Text with extremely low vocabulary diversity should fail.

        Note: with very low TTR the text almost always also triggers
        trigram repetition (check 3), which runs first.  Both checks
        catch the same underlying problem — repetitive output.
        """
        phrases = [
            "the large canine chased the tiny feline around ",
            "the tiny feline fled from the large canine near ",
            "the large feline watched the tiny canine below ",
            "the tiny canine followed the large feline above ",
            "the large canine found the tiny feline beside ",
            "the tiny feline spotted the large canine behind ",
            "the large feline greeted the tiny canine under ",
            "the tiny canine approached the large feline past ",
        ]
        text = "".join(phrases * 2) + "the large canine rested near the tiny feline."
        result = check_quality(text)
        assert not result.passed, "should detect repetitive/low-diversity text"
        assert ("vocabulary" in result.reason.lower() or
                "repetition" in result.reason.lower())

    def test_passes_normal_diversity(self):
        """Normal English text should have adequate vocabulary diversity."""
        text = (
            "Machine learning is a subset of artificial intelligence that "
            "focuses on building systems which learn from data. Instead of "
            "being explicitly programmed with rules, these algorithms identify "
            "patterns in training datasets and use those patterns to make "
            "predictions on new, unseen information. Common approaches include "
            "supervised learning, unsupervised clustering, and reinforcement "
            "learning techniques."
        )
        result = check_quality(text)
        assert result.passed, f"normal text should pass: {result.reason}"
        assert "ttr" in result.scores


class TestQualityResultScores:
    """Verify that QualityResult includes numeric scores for bench reporting."""

    def test_includes_ttr_score(self):
        """Passing text with 20+ words should include TTR score."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "A wonderful serenity has taken possession of my entire soul. "
            "I am so happy, my dear friend, so absorbed in the exquisite "
            "sense of mere tranquil existence."
        )
        result = check_quality(text)
        assert result.passed, f"should pass: {result.reason}"
        assert "ttr" in result.scores
        assert 0 < result.scores["ttr"] <= 1.0

    def test_backward_compat_check_coherence(self):
        """check_coherence() wrapper returns (bool, str) as before."""
        text = (
            "The meaning of life is a deeply philosophical question. "
            "Many thinkers have pondered this throughout history."
        )
        ok, reason = check_coherence(text)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        assert ok
        assert reason == ""
