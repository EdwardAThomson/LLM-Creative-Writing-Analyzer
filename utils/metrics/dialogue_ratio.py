"""Dialogue-to-narration ratio — share of the text spoken inside quotation marks.

A pacing/craft signal the v1 metrics don't capture: dialogue-heavy prose reads
faster and more scene-like; narration-heavy prose is denser and more summary-like.
Models differ markedly in this mix, so it's a useful fingerprint dimension
(METRICS_ROADMAP Phase 2).

Measured by words: ``dialogue_ratio = words inside quotes / total words``. Both
straight (``"..."``) and curly (``“...”``) double quotes are paired; single quotes
are deliberately ignored because they collide with apostrophes (``don't``,
``Kepler's``). Unbalanced/leftover quotes are simply not matched. Reported per run
plus the count of distinct quoted passages (a finer pacing tell than the ratio
alone — many short exchanges vs one long speech). Pure stdlib (regex only).
"""
from __future__ import annotations

import re
from typing import Optional

NAME = "dialogue_ratio"

# Paired double quotes: straight "..." OR curly “...”. Non-greedy, no nesting.
_QUOTED = re.compile(r'"([^"]*)"|“([^”]*)”')
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 3) if x is not None else None


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    per_run = []
    for text in responses:
        total_words = len(_WORD.findall(text))
        spans = _QUOTED.findall(text)
        # findall yields (straight, curly) tuples; exactly one group is non-empty.
        passages = [s or c for s, c in spans]
        dialogue_words = sum(len(_WORD.findall(p)) for p in passages)
        per_run.append(
            {
                "total_words": total_words,
                "dialogue_words": dialogue_words,
                "dialogue_ratio": _round(dialogue_words / total_words) if total_words else None,
                "quoted_passages": len(passages),
            }
        )

    ratios = [r["dialogue_ratio"] for r in per_run if r["dialogue_ratio"] is not None]
    if ratios:
        mean = sum(ratios) / len(ratios)
        var = sum((v - mean) ** 2 for v in ratios) / len(ratios)
        aggregate = {
            "n": len(ratios),
            "mean": _round(mean),
            "min": _round(min(ratios)),
            "max": _round(max(ratios)),
            "std": _round(var ** 0.5),
        }
    else:
        aggregate = {"n": 0, "mean": None, "min": None, "max": None, "std": None}

    return {
        "schema": "dialogue_ratio/1",
        "method": "words inside paired double quotes / total words",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": aggregate,
        "note": (
            "Fraction of words inside double quotes (dialogue) vs narration; a "
            "pacing fingerprint. Straight and curly double quotes are paired; single "
            "quotes are ignored (apostrophe collision). quoted_passages counts "
            "distinct quoted spans."
        ),
    }
