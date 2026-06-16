"""Intra-text repetition — word/phrase overuse *within* a single story.

The v1 metrics and the other v2 diversity metrics all measure variation *across*
the N runs. None of them catches the failure where one story leans on the same
word or phrase over and over ("word obsession") — a genuine per-text quality defect
(METRICS_ROADMAP Phase 2). This metric scores each story on its own.

Per run, three repetition rates (fraction of n-gram instances that are repeats of
something seen earlier in the same text, ``1 - distinct/total``; higher = more
repetitive):
  * ``unigram`` — over **content words only** (stopwords filtered), so "the/and"
    repetition isn't mistaken for a defect; this is the "word obsession" signal.
  * ``bigram`` / ``trigram`` — over **all word tokens** (function words included),
    since repeated phrases like "she said" / "he could feel" are the phrase-level
    tell.

Each also lists its top repeated items (count ≥ 2) for eyeballing. Runs under
``MIN_RELIABLE_TOKENS`` word tokens are flagged ``reliable: false`` (rates over a
handful of tokens are noisy). Pure stdlib (regex + Counter + a built-in stopword
set).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

NAME = "intra_text_repetition"
MIN_RELIABLE_TOKENS = 30
_TOP_K = 5

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Compact built-in English stopword set — enough to keep function-word repetition
# from masquerading as a lexical-obsession defect. Stdlib-only (no nltk).
_STOPWORDS = frozenset(
    """
    a an the and or but nor so yet for of to in on at by from with into onto over
    under as is am are was were be been being do does did doing have has had having
    will would shall should can could may might must this that these those it its
    he she they them him her his hers their theirs we us our ours you your yours i
    me my mine not no yes if then than there here when where which who whom whose what
    why how all any both each few more most other some such only own same too very
    s t just up down out off again once about above below between through during
    """.split()
)


def _ngram_rep_rate(tokens: list[str], n: int) -> tuple[Optional[float], list[tuple[str, int]]]:
    """(repeat-rate, top repeated n-grams). Rate = 1 - distinct/total, in [0,1)."""
    if len(tokens) < n:
        return None, []
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(grams)
    counts = Counter(grams)
    rate = (total - len(counts)) / total if total else None
    top = [(g, c) for g, c in counts.most_common(_TOP_K) if c >= 2]
    return (round(rate, 3) if rate is not None else None), top


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    per_run = []
    for text in responses:
        tokens = _WORD.findall(text.lower())
        content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

        uni_rate, uni_top = _ngram_rep_rate(content, 1)
        bi_rate, bi_top = _ngram_rep_rate(tokens, 2)
        tri_rate, tri_top = _ngram_rep_rate(tokens, 3)

        per_run.append(
            {
                "word_tokens": len(tokens),
                "content_tokens": len(content),
                "unigram": {"rep_rate": uni_rate, "top": uni_top},
                "bigram": {"rep_rate": bi_rate, "top": bi_top},
                "trigram": {"rep_rate": tri_rate, "top": tri_top},
                "reliable": len(tokens) >= MIN_RELIABLE_TOKENS,
            }
        )

    def _agg(gram: str, reliable_only: bool) -> dict:
        vals = [
            r[gram]["rep_rate"]
            for r in per_run
            if r[gram]["rep_rate"] is not None and (r["reliable"] or not reliable_only)
        ]
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        m = sum(vals) / len(vals)
        return {
            "n": len(vals),
            "mean": round(m, 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    grams = ("unigram", "bigram", "trigram")
    return {
        "schema": "intra_text_repetition/1",
        "method": "per-text n-gram repeat rate (1 - distinct/total); unigram=content words",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": {g: _agg(g, False) for g in grams},
        "aggregate_reliable_only": {g: _agg(g, True) for g in grams},
        "unreliable_runs": sum(1 for r in per_run if not r["reliable"]),
        "note": (
            "Repetition WITHIN each story (complements the across-run diversity "
            "metrics). rep_rate = 1 - distinct/total n-grams; higher = more "
            "repetitive. unigram is over content words only (the 'word obsession' "
            "defect); bigram/trigram over all tokens. Runs under "
            f"{MIN_RELIABLE_TOKENS} tokens flagged reliable=false."
        ),
    }
