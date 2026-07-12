"""Text-structure profile (paragraphs, sentences, words) per text. Pure stdlib.

The single-text (st1) equivalent of the frozen v1 structure analysis: v1's
structure numbers are produced by the legacy N-run pipeline in
``text_analysis.py`` and cannot be reused per-text, so this module recomputes
the same style of profile as a library metric. It is a *parallel* gauge, not a
replacement: sentence segmentation here is a deliberate regex approximation
(terminal punctuation followed by whitespace, closing quotes honored) so the
module stays stdlib-only, which means its absolute counts are not identical to
v1's spaCy-segmented numbers.

Under st1 each response is one segmentation unit (chapter/window), so
``per_run`` reads as per-unit structure and ``aggregate`` as the book profile.
Under a vN-style run it profiles each generation, which is also valid.
"""
from __future__ import annotations

import re
from typing import Optional

NAME = "text_structure"

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
# Sentence boundary: terminal punctuation (with optional closing quote/paren)
# followed by whitespace, or end of text. An approximation; uncommon
# abbreviations still split (common title/name abbreviations are guarded
# below).
_SENT_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*(?:\s+|$)")
_PARA_SPLIT = re.compile(r"\n\s*\n")

# Title/name abbreviations whose trailing period does not end a sentence.
# Conservative frozen list: honorifics and name suffixes, matched
# case-insensitively. Two-book shakedown evidence: splitting at "Dr." et al.
# inflated Dracula's sentence counts and truncated its chapter openings.
# Kept in sync with the twin list in opening_formula.py; the metric modules
# stay self-contained and file-path loadable (see tests/conftest.py), so the
# small guard is duplicated rather than shared.
_ABBREVIATIONS = frozenset((
    "dr", "mr", "mrs", "ms", "st", "prof", "capt", "col", "lieut", "sgt",
    "rev", "hon", "jr", "sr",
))
_WORD_BEFORE = re.compile(r"[A-Za-z]+$")


def _is_abbreviation_break(text: str, m: "re.Match[str]") -> bool:
    """True when a boundary match is just the period of a title/name
    abbreviation (``Dr.``, ``Mr.``, a single initial), not a sentence end.

    Conservative on both sides: only a bare period can belong to an
    abbreviation, so any other terminal punctuation, or a period wrapped in a
    closing quote or bracket (a sentence that legitimately ends with ``"Dr."``),
    is still a boundary. Single capital initials (``J. S. Fletcher``) count as
    abbreviations, except ``A`` and ``I``, which are common English words that
    legitimately end sentences.
    """
    if m.group(0).rstrip() != ".":
        return False
    # the word immediately before the period (12 chars covers the longest
    # abbreviation; a longer word's tail cannot match the list or an initial)
    w = _WORD_BEFORE.search(text[max(0, m.start() - 12):m.start()])
    if not w:
        return False
    word = w.group(0)
    if len(word) == 1:
        return word.isupper() and word not in ("A", "I")
    return word.lower() in _ABBREVIATIONS


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return round(x, nd) if x is not None else None


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def count_sentences(text: str) -> int:
    """Approximate sentence count: number of terminal-punctuation boundaries
    (abbreviation periods excluded), with a floor of 1 for any text
    containing words."""
    if not _WORD.search(text):
        return 0
    n = sum(1 for m in _SENT_BOUNDARY.finditer(text)
            if not _is_abbreviation_break(text, m))
    return max(1, n)


def profile(text: str) -> dict:
    paragraphs = split_paragraphs(text)
    words = len(_WORD.findall(text))
    sentences = sum(count_sentences(p) for p in paragraphs)
    return {
        "words": words,
        "sentences": sentences,
        "paragraphs": len(paragraphs),
        "words_per_sentence": _round(words / sentences) if sentences else None,
        "sentences_per_paragraph": _round(sentences / len(paragraphs)) if paragraphs else None,
        "words_per_paragraph": _round(words / len(paragraphs)) if paragraphs else None,
        "paragraphs_per_1k_words": _round(len(paragraphs) / words * 1000, 1) if words else None,
    }


def _agg(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "std": None}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"n": len(values), "mean": _round(mean), "min": _round(min(values)),
            "max": _round(max(values)), "std": _round(var ** 0.5)}


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    per_run = [profile(t) for t in responses]
    aggregate = {}
    for key in ("words", "sentences", "paragraphs", "words_per_sentence",
                "sentences_per_paragraph", "words_per_paragraph",
                "paragraphs_per_1k_words"):
        aggregate[key] = _agg([r[key] for r in per_run if r[key] is not None])
    return {
        "schema": "text_structure/1",
        "method": "regex word/sentence/paragraph profile (stdlib approximation)",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": aggregate,
        "note": (
            "Structural profile per text. Sentence segmentation is a regex "
            "approximation (not v1's spaCy segmentation), so compare within this "
            "metric only. Under st1 each run is one segmentation unit of a "
            "single text."
        ),
    }
