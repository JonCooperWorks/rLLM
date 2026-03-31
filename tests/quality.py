# ---------------------------------------------------------------------------
# quality.py — output quality validation for rLLM inference tests and
# benchmarks.  11 checks covering structural integrity, grammar heuristics,
# vocabulary diversity, and readability metrics.
#
# Checks 1-7 are the original coherence checks (min length, ASCII ratio,
# repetition, word salad, unicode garbage, token fragments, language detect).
# Checks 8-11 add sentence structure, punctuation patterns, readability
# (via textstat), and vocabulary diversity.
#
# Hard checks (1-9, 11) set passed=False on failure — these indicate broken
# inference.  Soft checks (10) produce warnings + scores for bench reporting
# but never fail the test — they track quality degradation trends.
#
# No external LLM or Java needed — runs entirely locally with minimal deps.
#
# Related: models.py (model registry), coherence.py (backward-compat wrapper)
# ---------------------------------------------------------------------------

import re
import string
import statistics
from dataclasses import dataclass, field

from langdetect import detect, LangDetectException


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Result of quality validation on generated text.

    passed     — True if all hard checks passed (checks 1-9, 11).
    reason     — first hard failure reason (empty string if passed).
    warnings   — soft warnings (readability anomalies from check 10).
    scores     — numeric scores for benchmark reporting:
                   "flesch" (Flesch Reading Ease), "ttr" (type-token ratio).
    """
    passed: bool = True
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common short English words — used by token fragment detection (check 6).
_COMMON_SHORT = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go",
    "he", "if", "in", "is", "it", "me", "my", "no", "of", "on",
    "or", "so", "to", "up", "us", "we",
}

# Words that signal a dangling/truncated sentence ending (check 8c).
_DANGLING_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
    "of", "with", "in", "on", "to", "at", "by", "from", "that", "which",
    "this", "these", "those", "is", "are", "was", "were", "be",
}

# Markdown patterns to exclude from punctuation run detection (check 9a).
_MARKDOWN_PUNCT_RE = re.compile(r'(?:---+|===+|\*\*\*+|```)')


def _strip_code_blocks(text: str) -> str:
    """Remove markdown fenced code blocks from text.

    Code has different structural rules (no capitalization, no sentence
    endings), so grammar-oriented checks should skip it.
    """
    return re.sub(r'```[\s\S]*?```', ' ', text)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries.

    Simple heuristic: split on sentence-ending punctuation followed by
    whitespace or end-of-string.  Handles common abbreviations (Mr., Dr.,
    etc.) and decimal numbers to avoid false splits.
    """
    # Use a placeholder that won't appear in normal text.
    placeholder = "\x00"

    # Protect common abbreviations and decimals from splitting.
    protected = text
    for abbr in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.",
                 "vs.", "etc.", "i.e.", "e.g.", "approx.", "dept."]:
        protected = protected.replace(abbr, abbr.replace(".", placeholder))

    # Protect decimal numbers (e.g., "3.14").
    protected = re.sub(r'(\d)\.(\d)', lambda m: m.group(1) + placeholder + m.group(2), protected)

    # Split on sentence-ending punctuation.
    parts = re.split(r'(?<=[.!?])\s+', protected)

    # Restore protected periods.
    return [p.replace(placeholder, ".") for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main quality check
# ---------------------------------------------------------------------------

def check_quality(text: str) -> QualityResult:
    """Validate that `text` is coherent, well-structured English.

    Returns a QualityResult with pass/fail status, failure reason,
    soft warnings, and numeric quality scores.
    """
    result = QualityResult()

    # ---- 1. Minimum length --------------------------------------------------
    if len(text.strip()) < 20:
        result.passed = False
        result.reason = f"too short ({len(text.strip())} chars, need >=20)"
        return result

    # ---- 2. ASCII ratio (catches encoding garbage) --------------------------
    printable = set(string.printable)
    n_printable = sum(1 for ch in text if ch in printable)
    ratio = n_printable / len(text) if text else 0
    if ratio < 0.70:
        result.passed = False
        result.reason = f"low ASCII ratio ({ratio:.0%}, need >=70%)"
        return result

    # ---- 3. Degenerate repetition -------------------------------------------
    words = [w for w in text.lower().split()
             if not re.fullmatch(r'[-|:*#>_=~`]+', w)]
    if len(words) >= 6:
        trigram_counts: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - 2):
            tri = (words[i], words[i + 1], words[i + 2])
            trigram_counts[tri] = trigram_counts.get(tri, 0) + 1

        # (a) Stuttering: same word repeated 3x, occurring 2+ times.
        for tri, count in trigram_counts.items():
            if tri[0] == tri[1] == tri[2] and count >= 2:
                result.passed = False
                result.reason = (
                    f"degenerate repetition (stuttering): "
                    f"\"{tri[0]}\" repeated 3+ times, occurring {count} times"
                )
                return result

        # (b) General repetition: any trigram exceeding a length-scaled threshold.
        worst = max(trigram_counts.values()) if trigram_counts else 0
        threshold = max(6, len(words) // 40)
        if worst >= threshold:
            repeated = max(trigram_counts, key=trigram_counts.get)
            result.passed = False
            result.reason = (
                f"degenerate repetition: \"{' '.join(repeated)}\" "
                f"appears {worst} times (threshold {threshold})"
            )
            return result

    # ---- 4. Word salad detection --------------------------------------------
    if len(words) >= 10:
        long_words = sum(1 for w in words if len(w) > 20)
        long_ratio = long_words / len(words)
        if long_ratio > 0.15:
            result.passed = False
            result.reason = (
                f"word salad: {long_ratio:.0%} of words exceed 20 chars "
                f"({long_words}/{len(words)})"
            )
            return result

    space_count = text.count(" ")
    space_ratio = space_count / len(text) if text else 0
    if len(text) > 100 and space_ratio < 0.10:
        result.passed = False
        result.reason = (
            f"word salad: low space ratio ({space_ratio:.0%}, need >=10%)"
        )
        return result

    # ---- 5. Unicode garbage detection ---------------------------------------
    if len(text) >= 30:
        exotic_count = sum(1 for ch in text if ord(ch) > 0x00FF)
        exotic_ratio = exotic_count / len(text)
        if exotic_ratio > 0.05:
            result.passed = False
            result.reason = (
                f"unicode garbage: {exotic_ratio:.0%} exotic chars "
                f"({exotic_count}/{len(text)}), need <=5%"
            )
            return result

    # ---- 6. Consecutive non-word fragments ----------------------------------
    if len(words) >= 8:
        run = 0
        max_run = 0
        for w in words:
            if len(w) <= 2 and w not in _COMMON_SHORT:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run >= 4:
            result.passed = False
            result.reason = (
                f"token fragment run: {max_run} consecutive non-word "
                f"fragments (<=2 chars each)"
            )
            return result

    # ---- 7. Language detection -----------------------------------------------
    try:
        lang = detect(text)
    except LangDetectException:
        result.passed = False
        result.reason = "langdetect could not identify language"
        return result

    if lang != "en":
        result.passed = False
        result.reason = f"detected language '{lang}', expected 'en'"
        return result

    # ---- 8. Sentence structure validation -----------------------------------
    # Operate on text with code blocks stripped — code has different rules.
    prose = _strip_code_blocks(text)

    if len(prose.strip()) >= 50:
        sentences = _split_sentences(prose)

        # 8a. Capitalization after sentence boundaries.
        # After .!? + space, the next alpha char should be uppercase.
        boundaries = re.finditer(r'[.!?]\s+([a-zA-Z])', prose)
        boundary_list = list(boundaries)
        if len(boundary_list) >= 3:
            violations = sum(1 for m in boundary_list if m.group(1).islower())
            violation_ratio = violations / len(boundary_list)
            if violation_ratio > 0.40:
                result.passed = False
                result.reason = (
                    f"capitalization: {violation_ratio:.0%} of sentence starts "
                    f"are lowercase ({violations}/{len(boundary_list)})"
                )
                return result

        # 8b. Sentence length distribution.
        if len(sentences) >= 3:
            sent_lengths = [len(s.split()) for s in sentences]
            median_len = statistics.median(sent_lengths)
            if median_len < 3:
                result.passed = False
                result.reason = (
                    f"fragmented: median sentence length {median_len:.0f} words "
                    f"(need >=3)"
                )
                return result
            if median_len > 80:
                result.passed = False
                result.reason = (
                    f"run-on: median sentence length {median_len:.0f} words "
                    f"(need <=80)"
                )
                return result

        # 8c. Dangling sentence endings — truncated mid-sentence.
        # Check if the text after the last sentence-ending punctuation ends
        # with a function word (conjunction, preposition, article).
        last_punct = max(
            (prose.rfind("."), prose.rfind("!"), prose.rfind("?")),
            default=-1,
        )
        if last_punct >= 0 and last_punct < len(prose) - 1:
            trailing = prose[last_punct + 1:].strip()
            trailing_words = trailing.split()
            if len(trailing_words) > 5:
                last_word = trailing_words[-1].lower().rstrip(".,;:!?")
                if last_word in _DANGLING_WORDS:
                    result.passed = False
                    result.reason = (
                        f"truncated: text ends with dangling word \"{last_word}\" "
                        f"after {len(trailing_words)} words"
                    )
                    return result

    # ---- 9. Punctuation pattern analysis ------------------------------------

    # 9a. Excessive punctuation runs (4+ consecutive non-markdown).
    for m in re.finditer(r'[^\w\s]{4,}', text):
        span = m.group()
        # Allow common markdown patterns.
        if _MARKDOWN_PUNCT_RE.fullmatch(span):
            continue
        # Allow ellipsis variants and common decorators.
        if re.fullmatch(r'\.{3,4}', span):
            continue
        # Allow horizontal rules and table separators.
        if re.fullmatch(r'[-=|:+]+', span):
            continue
        result.passed = False
        result.reason = (
            f"punctuation run: \"{span[:20]}\" ({len(span)} consecutive "
            f"punctuation chars)"
        )
        return result

    # 9b. Unmatched brackets/quotes.
    for open_ch, close_ch, name in [("(", ")", "parentheses"),
                                     ("[", "]", "brackets"),
                                     ("{", "}", "braces")]:
        opens = text.count(open_ch)
        closes = text.count(close_ch)
        if abs(opens - closes) > 3:
            result.passed = False
            result.reason = (
                f"unmatched {name}: {opens} open vs {closes} close "
                f"(difference {abs(opens - closes)})"
            )
            return result

    # ---- 10. Readability metrics (soft warnings) ----------------------------
    try:
        import textstat
        if len(prose.split()) >= 20:
            flesch = textstat.flesch_reading_ease(prose)
            result.scores["flesch"] = flesch
            if flesch < 0:
                result.warnings.append(
                    f"very low readability (Flesch {flesch:.0f}) — "
                    f"may indicate word salad"
                )
            elif flesch > 100:
                result.warnings.append(
                    f"extremely simple text (Flesch {flesch:.0f}) — "
                    f"may indicate repetitive output"
                )

            wps = textstat.words_per_sentence(prose)
            if wps > 50:
                result.warnings.append(
                    f"very long sentences (avg {wps:.0f} words/sentence)"
                )

            if len(prose) > 200:
                sent_count = textstat.sentence_count(prose)
                if sent_count < 2:
                    result.warnings.append(
                        f"only {sent_count} sentence in {len(prose)} chars"
                    )
    except ImportError:
        # textstat not installed — skip readability checks.
        pass

    # ---- 11. Vocabulary diversity (type-token ratio) ------------------------
    if len(words) >= 100:
        unique = len(set(words))
        ttr = unique / len(words)
        result.scores["ttr"] = ttr
        if ttr < 0.15:
            result.passed = False
            result.reason = (
                f"low vocabulary diversity: TTR {ttr:.2f} "
                f"({unique} unique words / {len(words)} total, need >=0.15)"
            )
            return result
    elif len(words) >= 20:
        # Still compute TTR for reporting even if below threshold for hard fail.
        unique = len(set(words))
        result.scores["ttr"] = unique / len(words)

    return result


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

def check_coherence(text: str) -> tuple[bool, str]:
    """Validate that `text` is coherent English.

    Returns (is_coherent, reason).  reason is empty on success, otherwise
    describes why the check failed.

    This is a backward-compatible wrapper around check_quality() — new code
    should use check_quality() directly for richer results.
    """
    result = check_quality(text)
    return result.passed, result.reason
