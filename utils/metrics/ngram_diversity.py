"""Corpus-level n-gram diversity of the N runs — distinct-n and Self-BLEU.

These score how varied the *set* of runs is, sitting between the v1 exact-string
similarity and the v1 semantic similarity: they catch runs that are neither
byte-identical nor paraphrases but lean on the same recurring words/phrases
(METRICS_ROADMAP Phase 1). Both are standard NLG-diversity measures, so they make
the study's diversity claims legible to the wider literature.

  * ``distinct_n`` = distinct n-grams / total n-grams, pooled across all runs.
    Higher = more diverse. Blunt but interpretable (distinct-1 = vocabulary
    breadth, distinct-2/3 = phrase variety).
  * ``self_bleu`` = for each run, BLEU(run, references = all other runs), averaged.
    Higher = runs more similar to each other (LESS diverse) — i.e. it runs OPPOSITE
    to distinct-n. Weights 1- to 4-grams with a brevity penalty, so it sees
    phrase-overlap structure, not just a unique/total ratio. Zero-match orders are
    floored (smoothing) so a single missing 4-gram doesn't collapse a run to 0.

Both are **length-sensitive** (longer corpora push distinct-n down and can move
Self-BLEU), so they're most comparable across run sets of similar size; the output
reports token counts and a length-spread flag. Pure stdlib (regex + Counter + math).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

NAME = "ngram_diversity"
DISTINCT_ORDERS = (1, 2, 3)
BLEU_MAX_N = 4  # standard BLEU-4

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def _distinct(token_runs: list[list[str]], n: int) -> dict:
    total = 0
    seen: set[tuple] = set()
    for tokens in token_runs:
        grams = _ngrams(tokens, n)
        total += len(grams)
        seen.update(grams)
    return {"distinct": len(seen), "total": total, "ratio": _round(len(seen) / total) if total else None}


def _modified_precision(cand_counts: Counter, ref_counts_list: list[Counter]) -> tuple[int, int]:
    """Clipped n-gram precision of a candidate against multiple references."""
    total = sum(cand_counts.values())
    if total == 0:
        return 0, 0
    clipped = 0
    for ng, c in cand_counts.items():
        max_ref = 0
        for ref in ref_counts_list:
            rc = ref.get(ng, 0)
            if rc > max_ref:
                max_ref = rc
        clipped += min(c, max_ref)
    return clipped, total


def _sentence_bleu(cand_tokens: list[str], cand_counts_by_n: dict, ref_runs: list[dict]) -> float:
    c = len(cand_tokens)
    if c == 0 or not ref_runs:
        return 0.0
    # brevity penalty against the closest reference length
    ref_lens = [r["len"] for r in ref_runs]
    closest = min(ref_lens, key=lambda r: (abs(r - c), r))
    bp = 1.0 if c > closest else math.exp(1 - closest / c)

    log_sum = 0.0
    weight = 1.0 / BLEU_MAX_N
    for n in range(1, BLEU_MAX_N + 1):
        matches, total = _modified_precision(
            cand_counts_by_n[n], [r["counts"][n] for r in ref_runs]
        )
        if total == 0:
            p = 1.0 / (2 ** n)  # candidate too short for this order — smoothed
        elif matches == 0:
            p = 1.0 / (2 * total)  # smoothing: floor zero-match orders
        else:
            p = matches / total
        log_sum += weight * math.log(p)
    return bp * math.exp(log_sum)


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    token_runs = [_tokenize(t) for t in responses]
    n_runs = len(token_runs)
    lengths = [len(t) for t in token_runs]

    distinct = {f"distinct_{n}": _distinct(token_runs, n) for n in DISTINCT_ORDERS}

    # Precompute each run's n-gram counters once (reused as both candidate & reference).
    runs_meta = [
        {"len": len(tokens), "counts": {n: Counter(_ngrams(tokens, n)) for n in range(1, BLEU_MAX_N + 1)}}
        for tokens in token_runs
    ]

    if n_runs >= 2:
        per_run_bleu = []
        for i, tokens in enumerate(token_runs):
            refs = [runs_meta[j] for j in range(n_runs) if j != i]
            per_run_bleu.append(_round(_sentence_bleu(tokens, runs_meta[i]["counts"], refs)))
        valid = [b for b in per_run_bleu if b is not None]
        mean = sum(valid) / len(valid) if valid else None
        self_bleu = {
            "mean": _round(mean),
            "min": _round(min(valid)) if valid else None,
            "max": _round(max(valid)) if valid else None,
            "per_run": per_run_bleu,
        }
    else:
        self_bleu = {"mean": None, "min": None, "max": None, "per_run": [],
                     "note": "Self-BLEU needs >=2 runs"}

    mean_len = sum(lengths) / n_runs if n_runs else 0
    # length spread: high spread weakens cross-run comparability of these metrics
    length_spread = (max(lengths) - min(lengths)) / mean_len if mean_len else None

    return {
        "schema": "ngram_diversity/1",
        "method": f"distinct-n (n={','.join(map(str, DISTINCT_ORDERS))}) + Self-BLEU-{BLEU_MAX_N}",
        "runs": n_runs,
        "tokens": {"total": sum(lengths), "mean_per_run": round(mean_len, 1),
                   "min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0,
                   "length_spread": _round(length_spread)},
        "distinct": distinct,
        "self_bleu": self_bleu,
        "note": (
            "Corpus-level diversity of the N runs. distinct_n (distinct/total n-grams) "
            "is HIGHER for more diverse sets; self_bleu is HIGHER for more SIMILAR "
            "(less diverse) sets — they move in opposite directions. Both are "
            "length-sensitive: compare across run sets of similar size (see "
            "tokens.length_spread; a large spread weakens comparability)."
        ),
    }
