"""Within-text self-similarity of adjacent units: the duplication detector.

The single-text (st1) adaptation of the cross-run text-similarity idea: instead
of comparing N runs of one prompt, compare each ADJACENT pair of one text's own
segmentation units (chapters/windows). Healthy long-form prose has very low
adjacent-unit similarity; near-duplicate or partially recycled units are a real
generation defect class. The motivating incident (StoryDaemon, 2026): a shipped
novel carried ~9,200 verbatim characters across two adjacent scenes, adjacent
similarity 0.668 against a ~0.02 book baseline. That is exactly the signature
this metric must expose: a similarity spike on one pair plus a long verbatim
match, against a near-zero median.

Measurement (stdlib difflib, token-level):

* ``similarity``: ``SequenceMatcher.ratio()`` over word tokens (autojunk left
  on: with book-sized units it ignores very common tokens, which speeds the
  ratio and biases it toward content overlap).
* ``longest_match_words`` / ``longest_match_chars``: the longest common verbatim
  token run (``find_longest_match`` with ``autojunk=False`` so common words do
  not break a genuine verbatim block).

A pair is flagged when ``similarity >= 0.3`` or the verbatim match reaches 500
characters; both thresholds sit far above healthy baselines and well below the
observed defect.

Semantic adjacent similarity (embedding cosine) is reported additionally when
the local sentence-embedding dependency is importable (it is in this repo's
requirements); otherwise ``semantic.available`` is false and the lexical result
stands alone. No LLM calls either way.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import median
from typing import Optional

NAME = "self_similarity"

FLAG_SIMILARITY = 0.3     # adjacent-pair ratio at or above this is flagged
FLAG_MATCH_CHARS = 500    # a verbatim run this long is flagged regardless of ratio

_WORD = re.compile(r"\S+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return round(x, nd) if x is not None else None


def adjacent_pairs(responses: list[str]) -> list[dict]:
    """Per adjacent pair: token-level ratio plus the longest verbatim token run."""
    pairs = []
    tok = [_tokens(t) for t in responses]
    for i in range(len(responses) - 1):
        a, b = tok[i], tok[i + 1]
        ratio = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
        if a and b:
            m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
                0, len(a), 0, len(b))
            match_words = m.size
            match_chars = len(" ".join(a[m.a : m.a + m.size])) if m.size else 0
        else:
            match_words = match_chars = 0
        pairs.append({
            "pair": [i, i + 1],
            "similarity": _round(ratio),
            "longest_match_words": match_words,
            "longest_match_chars": match_chars,
        })
    return pairs


def _flagged(pairs: list[dict]) -> list[dict]:
    return [p for p in pairs
            if p["similarity"] >= FLAG_SIMILARITY
            or p["longest_match_chars"] >= FLAG_MATCH_CHARS]


def _semantic(responses: list[str], ctx: Optional[dict]) -> dict:
    """Adjacent-pair embedding cosine, when the local model is importable.

    Never raises: in an environment without the embedding dependency (or under
    standalone file-path loading) it reports ``available: false``.
    """
    try:
        from ._base import get_sentence_model  # lazy; heavy; package-only

        model = get_sentence_model(ctx)
        embeddings = model.encode(responses)
        sims = []
        for i in range(len(responses) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            num = float(sum(x * y for x, y in zip(a, b)))
            da = float(sum(x * x for x in a)) ** 0.5
            db = float(sum(x * x for x in b)) ** 0.5
            sims.append(_round(num / (da * db)) if da and db else None)
        scored = [s for s in sims if s is not None]
        return {
            "available": True,
            "series": sims,
            "mean": _round(sum(scored) / len(scored)) if scored else None,
            "max": max(scored) if scored else None,
        }
    except Exception as e:  # missing dep, no package context, model failure
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    pairs = adjacent_pairs(responses)
    sims = [p["similarity"] for p in pairs]
    if sims:
        max_i = max(range(len(pairs)), key=lambda k: sims[k])
        aggregate = {
            "n_pairs": len(pairs),
            "mean": _round(sum(sims) / len(sims)),
            "median": _round(median(sims)),
            "max": sims[max_i],
            "max_pair": pairs[max_i]["pair"],
            "max_verbatim_chars": max(p["longest_match_chars"] for p in pairs),
        }
    else:
        aggregate = {"n_pairs": 0, "mean": None, "median": None, "max": None,
                     "max_pair": None, "max_verbatim_chars": None}
    flagged = _flagged(pairs)
    return {
        "schema": "self_similarity/1",
        "method": ("token-level SequenceMatcher ratio + longest verbatim token "
                   "run over adjacent unit pairs"),
        "runs": len(responses),
        "thresholds": {"similarity": FLAG_SIMILARITY,
                       "verbatim_chars": FLAG_MATCH_CHARS},
        "per_pair": pairs,
        "aggregate": aggregate,
        "flagged": flagged,
        "duplication_suspected": bool(flagged),
        "semantic": _semantic(responses, ctx) if len(responses) > 1 else
                    {"available": False, "reason": "fewer than 2 units"},
        "note": (
            "Adjacent-unit self-similarity of ONE text (st1 reading; under a vN "
            "run it compares consecutive generations). A spike over a near-zero "
            "median plus a long verbatim match is the recycled-content defect "
            "class (observed in the wild: 0.668 adjacent similarity vs ~0.02 "
            "baseline, ~9,200 verbatim chars)."
        ),
    }
