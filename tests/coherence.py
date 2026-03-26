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
    # Split into words and look for any 3-word sequence repeated 4+ times.
    words = text.lower().split()
    if len(words) >= 6:
        trigram_counts: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - 2):
            tri = (words[i], words[i + 1], words[i + 2])
            trigram_counts[tri] = trigram_counts.get(tri, 0) + 1
        worst = max(trigram_counts.values()) if trigram_counts else 0
        if worst >= 4:
            repeated = max(trigram_counts, key=trigram_counts.get)
            return False, (
                f"degenerate repetition: \"{' '.join(repeated)}\" "
                f"appears {worst} times"
            )

    # --- 4. Language detection ------------------------------------------------
    try:
        lang = detect(text)
    except LangDetectException:
        return False, "langdetect could not identify language"

    if lang != "en":
        return False, f"detected language '{lang}', expected 'en'"

    return True, ""
