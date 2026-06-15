"""Sentence-length burstiness — the variation in sentence rhythm.

The v1 structure metrics track the *mean* sentence length but not its variance.
Yet rhythm — long sentences next to short ones — is one of the more robust
human-vs-AI tells: human prose varies sentence length freely, while models tend to
flatten toward a uniform mid-length cadence (METRICS_ROADMAP Phase 2). This metric
measures that spread.

Two complementary numbers per run:
  * ``cv`` — coefficient of variation (std / mean of sentence length). This is the
    headline, length-robust figure: because it normalizes std by the mean, it does
    not drift just because one model writes longer sentences than another, so it's
    comparable across models (the roadmap's length-robustness principle).
  * ``burstiness`` — the Goh–Barabási coefficient B = (std − mean)/(std + mean),
    bounded [−1, 1]. B = 0 is Poisson-random spacing; B > 0 is "bursty" (uneven,
    more human-like rhythm); B < 0 is regular/metronomic. Reported alongside CV as
    the named burstiness measure.

Higher ``cv`` / higher ``burstiness`` = more varied rhythm. Sentences come from the
shared spaCy segmenter (``_base.sentences_by_run``); word counts use a simple
self-contained tokenizer so the unit ("word") is explicit and independent of the
NER pipeline. Runs with fewer than ``MIN_RELIABLE_SENTENCES`` sentences are flagged
(``reliable: false``) — variance over one or two sentences is meaningless.
"""
from __future__ import annotations

import re
from typing import Optional

from ._base import sentences_by_run

NAME = "burstiness"
MIN_RELIABLE_SENTENCES = 3  # variance over 1–2 sentences is not meaningful

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _word_count(sentence: str) -> int:
    return len(_WORD.findall(sentence))


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 3) if x is not None else None


def _stats(lengths: list[int]) -> dict:
    """mean / population-std / CV / burstiness-B for one run's sentence lengths."""
    n = len(lengths)
    if n == 0:
        return {"mean": None, "std": None, "cv": None, "burstiness": None}
    mean = sum(lengths) / n
    var = sum((x - mean) ** 2 for x in lengths) / n
    std = var ** 0.5
    cv = std / mean if mean else None
    b = (std - mean) / (std + mean) if (std + mean) else None
    return {"mean": _round(mean), "std": _round(std), "cv": _round(cv), "burstiness": _round(b)}


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    sents_by_run = sentences_by_run(responses, ctx)

    per_run = []
    for sents in sents_by_run:
        lengths = [_word_count(s) for s in sents]
        lengths = [n for n in lengths if n > 0]  # drop sentences with no words
        s = _stats(lengths)
        per_run.append(
            {
                "sentences": len(lengths),
                "words": sum(lengths),
                **s,
                "reliable": len(lengths) >= MIN_RELIABLE_SENTENCES,
            }
        )

    def _agg(key: str, reliable_only: bool) -> dict:
        vals = [
            r[key]
            for r in per_run
            if r[key] is not None and (r["reliable"] or not reliable_only)
        ]
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return {
            "n": len(vals),
            "mean": _round(m),
            "min": _round(min(vals)),
            "max": _round(max(vals)),
            "std": _round(var ** 0.5),
        }

    return {
        "schema": "burstiness/1",
        "method": "per-run sentence-length CV and Goh-Barabasi B (spaCy segmentation)",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": {key: _agg(key, False) for key in ("cv", "burstiness", "mean", "std")},
        "aggregate_reliable_only": {
            key: _agg(key, True) for key in ("cv", "burstiness", "mean", "std")
        },
        "unreliable_runs": sum(1 for r in per_run if not r["reliable"]),
        "note": (
            "Variation in sentence length (rhythm). cv = std/mean is the "
            "length-robust headline; burstiness = (std-mean)/(std+mean) in [-1,1] is "
            "the Goh-Barabasi coefficient (>0 bursty/human-like, <0 metronomic). "
            "Higher = more varied rhythm. Runs under "
            f"{MIN_RELIABLE_SENTENCES} sentences are flagged reliable=false."
        ),
    }
