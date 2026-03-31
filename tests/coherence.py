# ---------------------------------------------------------------------------
# coherence.py — backward-compatible re-export of quality validation.
#
# All quality checks have moved to quality.py, which provides a richer
# QualityResult type with scores and warnings.  This file re-exports
# check_coherence() for existing code that imports from here.
#
# New code should import from quality.py directly:
#   from quality import check_quality, check_coherence, QualityResult
#
# Related: quality.py (full implementation)
# ---------------------------------------------------------------------------

from quality import check_coherence, check_quality, QualityResult

__all__ = ["check_coherence", "check_quality", "QualityResult"]
