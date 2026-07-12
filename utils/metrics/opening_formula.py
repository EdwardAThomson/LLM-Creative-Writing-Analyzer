"""Chapter-opening formula detection within one text. Pure stdlib.

The single-text (st1) adaptation of the cross-run opening-line metric: instead
of asking "do N runs of one prompt open alike" it asks "do this book's own
chapters open alike". Formulaic generated long-form tends to re-enter every
chapter the same way (same POV-name-plus-verb opener, same weather beat); human
chapters vary their entries.

Measured on the first sentence of each unit: all-pairs token-level
``SequenceMatcher`` similarity (openings are short, so all pairs stay cheap),
plus a first-words census (how many units open with the same first three
words). ``high_pairs`` lists opening pairs at or above 0.5 similarity. No
embedding, no LLM: this is the lexical-formula tell; semantic variety is a vN
concern (``opening_lines``), not replicated here.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

NAME = "opening_formula"

HIGH_SIMILARITY = 0.5    # an opening pair at or above this is reported
OPENER_WORDS = 3         # first-N-words census window
MAX_OPENING_CHARS = 300  # fallback cap when no sentence boundary is found

_SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s")
# Unicode-aware word token ([^\W\d_] is "letter" plus \d for digits, in re's
# unicode mode). The old ASCII class [a-z0-9] split accented names apart
# (War and Peace shakedown: "Natásha" tokenized as "nat", "sha"); tokens are
# NFKD-folded afterwards so Kutúzov and Kutuzov compare equal.
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")


def _fold_marks(text: str) -> str:
    """NFKD-decompose and drop combining marks: ``kutúzov`` -> ``kutuzov``."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return [_fold_marks(t) for t in _WORD.findall(text.lower())]

# Title/name abbreviations whose trailing period does not end a sentence.
# Conservative frozen list: honorifics and name suffixes, matched
# case-insensitively. Two-book shakedown evidence: Dracula's "DR. SEWARD'S
# DIARY" chapter headers were truncated to "DR.", scoring eleven chapter
# openings as identical 1.0-similarity pairs (a pure splitter artifact).
# Kept in sync with the twin list in text_structure.py; the metric modules
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


def first_sentence(text: str) -> str:
    """First sentence of a unit (or its first MAX_OPENING_CHARS as fallback).

    Boundaries that are only an abbreviation's period are skipped (see
    ``_is_abbreviation_break``), so ``DR. SEWARD'S DIARY ...`` headers are no
    longer cut to ``DR.``.
    """
    t = text.strip()
    probe = t + " "
    for m in _SENT_END.finditer(probe):
        if _is_abbreviation_break(probe, m):
            continue
        return t[: m.end()].strip()
    return t[:MAX_OPENING_CHARS].strip()


def _opener_key(opening: str) -> Optional[str]:
    words = _tokens(opening)
    if len(words) < OPENER_WORDS:
        return None
    return " ".join(words[:OPENER_WORDS])


def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return round(x, nd) if x is not None else None


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    openings = [first_sentence(t) for t in responses]
    tokens = [_tokens(o) for o in openings]

    pair_sims = []
    for i in range(len(openings)):
        for j in range(i + 1, len(openings)):
            if not tokens[i] or not tokens[j]:
                continue
            sim = SequenceMatcher(None, tokens[i], tokens[j], autojunk=False).ratio()
            pair_sims.append({"pair": [i, j], "similarity": _round(sim)})

    sims = [p["similarity"] for p in pair_sims]
    high_pairs = [p for p in pair_sims if p["similarity"] >= HIGH_SIMILARITY]

    opener_counts = Counter(k for o in openings if (k := _opener_key(o)))
    repeated_openers = [{"opener": k, "count": c}
                        for k, c in opener_counts.most_common() if c >= 2]

    if sims:
        max_i = max(range(len(pair_sims)), key=lambda k: sims[k])
        aggregate = {
            "n_units": len(responses),
            "n_pairs": len(pair_sims),
            "mean": _round(sum(sims) / len(sims)),
            "max": sims[max_i],
            "max_pair": pair_sims[max_i]["pair"],
            "high_pair_rate": _round(len(high_pairs) / len(pair_sims)),
        }
    else:
        aggregate = {"n_units": len(responses), "n_pairs": 0, "mean": None,
                     "max": None, "max_pair": None, "high_pair_rate": None}

    return {
        "schema": "opening_formula/1",
        "method": (f"all-pairs token SequenceMatcher over unit first sentences; "
                   f"first-{OPENER_WORDS}-words census"),
        "runs": len(responses),
        "openings": openings,
        "per_pair_high": high_pairs,
        "repeated_openers": repeated_openers,
        "aggregate": aggregate,
        "note": (
            "Do this text's own units open alike (the within-text formulaic-"
            "opening tell)? high_pair_rate is the share of opening pairs at or "
            f"above {HIGH_SIMILARITY} similarity; repeated_openers counts units "
            "sharing their first words."
        ),
    }
