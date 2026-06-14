"""Length-robust lexical diversity (MTLD).

The v1 metric reports raw TTR (type/token ratio), which falls *automatically* as a
text lengthens — so longer outputs look less diverse purely for being longer, and
the longitudinal notes keep caveating it with an equal-length-pair workaround
(report §… length caveat; METRICS_ROADMAP Phase 1).

MTLD (Measure of Textual Lexical Diversity, McCarthy & Jarvis 2010) is designed to
be length-stable: it measures the *mean number of tokens it takes for the running
TTR to fall to a threshold* (0.72, the value validated in the original paper),
averaged over a forward and a reverse pass. A higher MTLD = more lexically varied.
Because it's a rate (tokens-per-factor), not a ratio over the whole text, it does
not drift with length the way TTR does — which is the whole point: it makes the
longer Codex/Opus outputs directly comparable to shorter ones.

Tokenization is deliberately simple and self-contained (lowercased alphabetic word
runs, apostrophes kept) so this metric stays stdlib-only — no spaCy, no new dep.
That means it is *not* identical to the v1 vocabulary tokenization; MTLD is a new,
parallel diversity axis, not a replacement for the frozen v1 number.

Caveat: MTLD is unreliable on very short texts — the original paper recommends
~100+ tokens, and values from runs below ~50 tokens are flagged (``reliable:
false``) rather than dropped.
"""
from __future__ import annotations

import re
from typing import Optional

NAME = "mtld"
THRESHOLD = 0.72          # McCarthy & Jarvis (2010) validated factor threshold
MIN_RELIABLE_TOKENS = 50  # below this, MTLD is noisy — flag, don't trust

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _mtld_pass(tokens: list[str], threshold: float = THRESHOLD) -> float:
    """One directional MTLD pass: total tokens / number of TTR-decline factors."""
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        token_count += 1
        types.add(tok)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types = set()
            token_count = 0
    if token_count > 0:  # trailing partial factor
        ttr = len(types) / token_count
        # how far this remnant got toward a full factor (1.0 == would-be complete)
        factors += (1.0 - ttr) / (1.0 - threshold)
    if factors == 0:
        return float(len(tokens))  # never crossed the threshold (very diverse/short)
    return len(tokens) / factors


def _mtld(tokens: list[str], threshold: float = THRESHOLD) -> Optional[float]:
    """Bidirectional MTLD (mean of forward and reverse passes)."""
    if not tokens:
        return None
    forward = _mtld_pass(tokens, threshold)
    backward = _mtld_pass(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 2) if x is not None else None


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    per_run = []
    for text in responses:
        tokens = _tokenize(text)
        n = len(tokens)
        per_run.append(
            {
                "tokens": n,
                "types": len(set(tokens)),
                "ttr": _round(len(set(tokens)) / n) if n else None,
                "mtld": _round(_mtld(tokens)),
                "reliable": n >= MIN_RELIABLE_TOKENS,
            }
        )

    scored = [r["mtld"] for r in per_run if r["mtld"] is not None]
    reliable = [r["mtld"] for r in per_run if r["mtld"] is not None and r["reliable"]]
    n_unreliable = sum(1 for r in per_run if not r["reliable"])

    def _agg(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {
            "n": len(vals),
            "mean": _round(mean),
            "min": _round(min(vals)),
            "max": _round(max(vals)),
            "std": _round(var ** 0.5),
        }

    return {
        "schema": "mtld/1",
        "method": f"bidirectional MTLD, threshold={THRESHOLD}",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": _agg(scored),
        "aggregate_reliable_only": _agg(reliable),
        "unreliable_runs": n_unreliable,
        "note": (
            "Length-robust lexical diversity (MTLD, McCarthy & Jarvis 2010); higher "
            "= more varied vocabulary. Unlike v1 TTR it does not fall merely because "
            "a text is longer. Runs under "
            f"{MIN_RELIABLE_TOKENS} tokens are flagged reliable=false; prefer "
            "aggregate_reliable_only when comparing models."
        ),
    }
