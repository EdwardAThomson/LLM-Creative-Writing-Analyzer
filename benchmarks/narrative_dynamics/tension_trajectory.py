"""Tension trajectory: per-unit 0-10 dramatic-tension scoring plus curve-shape aggregations.

Each unit is scored by an LLM judge against the anchored TENSION_ANCHORS rubric
(``rubrics/tension_anchors.py``, ported from StoryDaemon with provenance). Long
units are truncated head+tail per the source study's protocol. A unit whose
score cannot be parsed after a re-ask is recorded as a hole (``tension: null``),
never a hard failure, so one bad call cannot kill a long document.

Aggregations (the masters-study battery):
  mean register, std, min/max, volatility (mean absolute successive difference),
  decile table (unit midpoints), peak height/position, tail behavior (tail mean,
  final value, ending-mode classification), calm/high shares.

The per-unit scores are stashed in ``ctx["unit_tensions"]`` so thread_architecture
can compute tension deltas at thread switches when both metrics run.
"""
from __future__ import annotations

from typing import Optional

from . import segmentation
from .judge import JudgeError, ask_json, extract_json_object, require_judge
from .rubrics import tension_anchors

NAME = "tension_trajectory"
REQUIRES_LLM = True
SCHEMA = "tension_trajectory/1"

HEAD_WORDS = 2000  # source-study long-unit policy: first 2000 + last 2000 words
TAIL_WORDS = 2000
TAIL_FRACTION = 0.1        # tail = final 10 percent of units (at least one)
CALM_MAX = 3               # study convention: a "calm" unit scores <= 3
HIGH_MIN = 8
ENDING_WIND_DOWN_MAX = 3.5  # tail mean at or below: wind-down ending
ENDING_CLIMAX_MIN = 7.0     # tail mean at or above: climax-hold ending


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return round(x, nd) if x is not None else None


def _parse(raw: str) -> dict:
    obj = extract_json_object(raw)
    lvl = int(obj["tension_level"])
    return {"tension": max(0, min(10, lvl)), "rationale": str(obj.get("rationale", ""))}


def score_units(units: list[dict], ctx: dict, title: str) -> list[dict]:
    """Per-unit judge scores; holes (with the error) where parsing failed twice."""
    judge = require_judge(ctx)
    out = []
    for u in units:
        prompt = tension_anchors.render_tension_prompt(
            title=title, label=u["label"],
            text=segmentation.truncate_middle(u["text"], HEAD_WORDS, TAIL_WORDS))
        try:
            rec = ask_json(judge, prompt, _parse, ctx=ctx)
        except JudgeError as e:
            rec = {"tension": None, "rationale": None, "error": str(e)}
        rec.update({"index": u["index"], "label": u["label"], "words": u["words"]})
        out.append(rec)
    return out


# --- pure aggregations ----------------------------------------------------------------


def volatility(series: list[Optional[int]]) -> Optional[float]:
    """Mean absolute successive difference over adjacent pairs where both scored."""
    deltas = [abs(b - a) for a, b in zip(series, series[1:])
              if a is not None and b is not None]
    return sum(deltas) / len(deltas) if deltas else None


def decile_table(series: list[Optional[int]]) -> dict:
    """Mean tension by tenth of the document (unit midpoints), keys "0".."9"."""
    n = len(series)
    buckets: dict[int, list[int]] = {}
    for i, t in enumerate(series):
        if t is None:
            continue
        d = min(9, int((i + 0.5) / n * 10))
        buckets.setdefault(d, []).append(t)
    return {str(d): _round(sum(v) / len(v), 1) for d, v in sorted(buckets.items())}


def peak(series: list[Optional[int]]) -> dict:
    scored = [(i, t) for i, t in enumerate(series) if t is not None]
    if not scored:
        return {"height": None, "first_position": None, "last_position": None}
    n = len(series)
    height = max(t for _, t in scored)
    positions = [(i + 0.5) / n for i, t in scored if t == height]
    return {"height": height,
            "first_position": _round(positions[0], 3),
            "last_position": _round(positions[-1], 3)}


def tail_behavior(series: list[Optional[int]]) -> dict:
    n = len(series)
    if n == 0:
        return {"tail_units": 0, "tail_mean": None, "final_tension": None, "ending_mode": None}
    k = max(1, round(n * TAIL_FRACTION))
    tail = [t for t in series[-k:] if t is not None]
    final = next((t for t in reversed(series) if t is not None), None)
    tail_mean = sum(tail) / len(tail) if tail else None
    if tail_mean is None:
        mode = None
    elif tail_mean <= ENDING_WIND_DOWN_MAX:
        mode = "wind_down"          # masters' denouement endings land at 1-3
    elif tail_mean >= ENDING_CLIMAX_MIN:
        mode = "climax_hold"        # thrillers hold 8-9 to the final page
    else:
        mode = "moderate"           # the shape the masters study found matches nobody
    return {"tail_units": k, "tail_mean": _round(tail_mean),
            "final_tension": final, "ending_mode": mode}


def aggregate(series: list[Optional[int]]) -> dict:
    scored = [t for t in series if t is not None]
    n = len(scored)
    mean = sum(scored) / n if n else None
    std = (sum((t - mean) ** 2 for t in scored) / n) ** 0.5 if n else None
    agg = {
        "n_units": len(series),
        "n_scored": n,
        "mean_register": _round(mean),
        "std": _round(std),
        "min": min(scored) if scored else None,
        "max": max(scored) if scored else None,
        "volatility": _round(volatility(series)),
        "calm_share": _round(sum(1 for t in scored if t <= CALM_MAX) / n) if n else None,
        "high_share": _round(sum(1 for t in scored if t >= HIGH_MIN) / n) if n else None,
    }
    pk = peak(series)
    agg["peak_height"] = pk["height"]
    agg["peak_position"] = pk["first_position"]
    agg["peak_last_position"] = pk["last_position"]
    agg.update(tail_behavior(series))
    return agg


def compute(units: list[dict], ctx: Optional[dict] = None) -> dict:
    ctx = ctx if ctx is not None else {}
    title = ctx.get("title", "the document")
    per_unit = score_units(units, ctx, title)
    series = [r["tension"] for r in per_unit]
    ctx["unit_tensions"] = series  # downstream: thread_architecture switch deltas
    return {
        "schema": SCHEMA,
        "rubric": {"version": tension_anchors.RUBRIC_VERSION,
                   **tension_anchors.PROVENANCE},
        "method": (f"per-unit LLM judge on the anchored 0-10 rubric; long units "
                   f"truncated to first {HEAD_WORDS} + last {TAIL_WORDS} words"),
        "per_unit": per_unit,
        "deciles": decile_table(series),
        "aggregate": aggregate(series),
        "note": ("Volatility is the mean absolute unit-to-unit change. Masters "
                 "calibration (as measured in the source harness): registers 4.3-7.1, "
                 "volatility 0.9-1.7, no chapter scored 10 in 149; endings are either "
                 "wind_down (1-3) or climax_hold (8-9)."),
    }
