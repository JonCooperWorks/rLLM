# ---------------------------------------------------------------------------
# coherence.py — lightweight English coherence validation for GPU integration
# tests.  Uses langdetect for language identification plus simple heuristics
# (minimum length, ASCII ratio, degenerate repetition) to confirm model
# output is coherent English rather than gibberish.
#
# No external LLM needed — this runs entirely locally with minimal deps.
# ---------------------------------------------------------------------------

import re
import string

from langdetect import detect, LangDetectException


def check_coherence(text: str) -> tuple[bool, str]:
    """Validate that `text` is coherent English.

    Returns (is_coherent, reason).  reason is empty on success, otherwise
    describes why the check failed.
    """

    # --- 1. Minimum length ---------------------------------------------------
    if len(text.strip()) < 20:
        return False, f"too short ({len(text.strip())} chars, need ≥20)"

    # --- 2. ASCII ratio (catches encoding garbage) ---------------------------
    printable = set(string.printable)
    n_printable = sum(1 for ch in text if ch in printable)
    ratio = n_printable / len(text) if text else 0
    if ratio < 0.70:
        return False, f"low ASCII ratio ({ratio:.0%}, need ≥70%)"

    # --- 3. Degenerate repetition --------------------------------------------
    # Two-tier detection on 3-word (trigram) sequences:
    #
    #   a) Stuttering — the same word repeated 3 times (e.g. "server server
    #      server").  Any stuttering trigram appearing 2+ times is degenerate
    #      regardless of text length.
    #
    #   b) General repetition — normal phrases (e.g. "a pair of") may appear
    #      several times in long text without being degenerate.  The threshold
    #      scales with text length: max(6, word_count // 40).  For a typical
    #      512-token response (~400 words), the threshold is 10.
    # Filter out markdown table separators (|, |---|, etc.) and lone
    # punctuation before repetition analysis — these are formatting, not
    # degenerate generation.
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
                return False, (
                    f"degenerate repetition (stuttering): "
                    f"\"{tri[0]}\" repeated 3+ times, occurring {count} times"
                )

        # (b) General repetition: any trigram exceeding a length-scaled threshold.
        worst = max(trigram_counts.values()) if trigram_counts else 0
        threshold = max(6, len(words) // 40)
        if worst >= threshold:
            repeated = max(trigram_counts, key=trigram_counts.get)
            return False, (
                f"degenerate repetition: \"{' '.join(repeated)}\" "
                f"appears {worst} times (threshold {threshold})"
            )

    # --- 4. Word salad detection -----------------------------------------------
    # Broken inference (e.g. corrupted Q4 dequantization) often produces random
    # word fragments concatenated without spaces.  Detect this by checking:
    #   a) Long "words" — normal English rarely has words > 20 chars.  If more
    #      than 15% of words exceed 20 chars, the output is likely word salad.
    #   b) Space ratio — coherent text has roughly 1 space per 5-6 characters.
    #      Word salad with concatenated fragments has far fewer spaces.
    if len(words) >= 10:
        long_words = sum(1 for w in words if len(w) > 20)
        long_ratio = long_words / len(words)
        if long_ratio > 0.15:
            return False, (
                f"word salad: {long_ratio:.0%} of words exceed 20 chars "
                f"({long_words}/{len(words)})"
            )

    space_count = text.count(" ")
    space_ratio = space_count / len(text) if text else 0
    if len(text) > 100 and space_ratio < 0.10:
        return False, (
            f"word salad: low space ratio ({space_ratio:.0%}, need ≥10%)"
        )

    # --- 5. Language detection ------------------------------------------------
    try:
        lang = detect(text)
    except LangDetectException:
        return False, "langdetect could not identify language"

    if lang != "en":
        return False, f"detected language '{lang}', expected 'en'"

    return True, ""
