"""Versioned rubric artifacts ported from StoryDaemon, with provenance headers.

Each rubric module carries a ``PROVENANCE`` dict recording the source repo and
commit, the reliability numbers as measured in the source harness, and the
re-verification caveat: those reliability numbers were measured with a specific
annotator model, prompt framing, and corpus in the SOURCE harness. They do not
transfer automatically. Re-verify reliability in THIS harness (a double-pass
agreement check on a stratified sample) before trusting findings built on these
rubrics here.
"""
from __future__ import annotations

REVERIFICATION_CAVEAT = (
    "Reliability numbers in PROVENANCE were measured in the source repo's harness "
    "(its annotator model, prompt framing, and corpus). They must be re-verified "
    "in this harness (double-pass agreement on a stratified sample) before "
    "findings that depend on this rubric are trusted here."
)
