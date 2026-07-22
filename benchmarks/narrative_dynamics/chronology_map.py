"""Chronology map: per-unit temporal-placement annotation plus discontinuity aggregations.

The StoryScope-informed temporal-structure metric (see METRICS_ROADMAP nd2/v3
section and ``rubrics/chronology_scale.py`` for provenance). Each unit is annotated
by an LLM judge with its dominant temporal placement relative to the story's
narrative present (present / analepsis / prolepsis / mixed), whether it contains an
internal time jump, and its flashback-nesting depth. A unit whose annotation can't
be parsed after a re-ask is recorded as a hole (``placement: null``), never a hard
failure, so one bad call cannot kill a long document.

Everything downstream is deterministic over the placement sequence:
  placement distribution and shares, present/flashback shares, adjacent-unit
  placement transitions, within-unit jump rate, max nesting depth, and a
  discontinuity index.

``storyscope_projection`` reconstructs StoryScope's whole-story temporal scales
(TMP_ORD_010 / TMP_ORD_001 / TMP_ORD_004 / TMP_ORD_007 / EVT_TYP_009) from those
quantities so a run can be validated against their released gold annotations
(join on prompt_id + source). Those projection bands are HEURISTIC and provisional
-- calibrate them on the StoryScope dev split before trusting the projected labels;
the measured quantities in ``aggregate`` are the primary, non-heuristic output.
"""
from __future__ import annotations

from typing import Optional

from . import segmentation
from .judge import JudgeError, ask_json, extract_json_object, require_judge
from .rubrics import chronology_scale

NAME = "chronology_map"
REQUIRES_LLM = True
SCHEMA = "chronology_map/1"

HEAD_WORDS = 2000  # same long-unit policy as tension_trajectory
TAIL_WORDS = 2000

# storyscope_projection band edges (PROVISIONAL: calibrate on the dev split).
# Discontinuity index d = transition_rate + internal_jump_rate, range [0, 2].
_DISC_BANDS = (0.15, 0.40, 0.70)   # -> scale 2 / 3 / 4 / 5 above each edge (1 if d==0 & no flashback)
_FLASHBACK_LOW = 0.20              # EVT_TYP_009 present-vs-past thresholds
_FLASHBACK_HIGH = 0.50
_RARE_FLASHBACK = 0.15            # TMP_ORD_001 "rare flashbacks" ceiling
_STRAND_SHARE = 0.20             # a non-present placement this common counts as a sustained strand
_STRAND_MIN_UNITS = 2            # ...or appearing in at least this many units


def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return round(x, nd) if x is not None else None


def _parse(raw: str) -> dict:
    obj = extract_json_object(raw)
    placement = str(obj["placement"]).strip().lower()
    if placement not in chronology_scale.PLACEMENT_KEYS:
        raise ValueError(f"unknown placement: {placement!r}")
    nesting = str(obj.get("nesting", "none")).strip().lower()
    if nesting not in chronology_scale.NESTING_LEVELS:
        nesting = "none"
    return {"placement": placement,
            "internal_jump": bool(obj.get("internal_jump", False)),
            "nesting": nesting,
            "rationale": str(obj.get("rationale", ""))}


def score_units(units: list[dict], ctx: dict, title: str) -> list[dict]:
    """Per-unit judge annotations; holes (with the error) where parsing failed twice."""
    judge = require_judge(ctx)
    out = []
    for u in units:
        prompt = chronology_scale.render_chronology_prompt(
            title=title, label=u["label"],
            text=segmentation.truncate_middle(u["text"], HEAD_WORDS, TAIL_WORDS))
        try:
            rec = ask_json(judge, prompt, _parse, ctx=ctx)
        except JudgeError as e:
            rec = {"placement": None, "internal_jump": None, "nesting": None,
                   "rationale": None, "error": str(e)}
        rec.update({"index": u["index"], "label": u["label"], "words": u["words"]})
        out.append(rec)
    return out


# --- pure aggregations ----------------------------------------------------------------


def distribution(placements: list[Optional[str]]) -> dict:
    """Share of scored units in each placement, keys = the four placement types."""
    scored = [p for p in placements if p is not None]
    n = len(scored)
    return {k: _round(sum(1 for p in scored if p == k) / n) if n else None
            for k in chronology_scale.PLACEMENT_KEYS}


def transition_rate(placements: list[Optional[str]]) -> Optional[float]:
    """Fraction of adjacent scored-unit pairs whose placement changes."""
    pairs = [(a, b) for a, b in zip(placements, placements[1:])
             if a is not None and b is not None]
    return _round(sum(1 for a, b in pairs if a != b) / len(pairs)) if pairs else None


def internal_jump_rate(records: list[dict]) -> Optional[float]:
    flags = [r["internal_jump"] for r in records if r.get("internal_jump") is not None]
    return _round(sum(1 for f in flags if f) / len(flags)) if flags else None


def max_nesting(records: list[dict]) -> str:
    """Deepest observed flashback nesting across scored units (none < single < multi)."""
    order = {lvl: i for i, lvl in enumerate(chronology_scale.NESTING_LEVELS)}
    seen = [order[r["nesting"]] for r in records if r.get("nesting") in order]
    return chronology_scale.NESTING_LEVELS[max(seen)] if seen else "none"


def discontinuity_index(placements: list[Optional[str]], records: list[dict]) -> Optional[float]:
    """transition_rate + internal_jump_rate, range [0, 2]; None if nothing scored."""
    tr = transition_rate(placements)
    ij = internal_jump_rate(records)
    if tr is None and ij is None:
        return None
    return _round((tr or 0.0) + (ij or 0.0))


def _sustained_others(dist: dict) -> list[str]:
    """Non-present placements common enough to count as a sustained temporal strand."""
    return [k for k in ("analepsis", "prolepsis", "mixed")
            if (dist.get(k) or 0.0) >= _STRAND_SHARE]


def storyscope_projection(dist: dict, placements: list[Optional[str]],
                          records: list[dict]) -> dict:
    """Reconstruct StoryScope whole-story scales from the measured quantities.

    HEURISTIC / provisional bands (see module docstring): the join key for
    validation is (prompt_id, source); score these projected labels against the
    gold TMP_ORD_* / EVT_TYP_009 columns and calibrate the band edges before use.
    """
    n_scored = sum(1 for p in placements if p is not None)
    if n_scored == 0:
        return {k: None for k in
                ("TMP_ORD_010_est", "TMP_ORD_001_est", "TMP_ORD_004_est",
                 "TMP_ORD_007_est", "EVT_TYP_009_est")}

    present = dist.get("present") or 0.0
    analepsis = dist.get("analepsis") or 0.0
    prolepsis = dist.get("prolepsis") or 0.0
    mixed = dist.get("mixed") or 0.0
    past_heavy = analepsis + mixed          # flashback-weighted share
    nest = max_nesting(records)
    d = discontinuity_index(placements, records) or 0.0
    any_flashback = analepsis > 0 or mixed > 0 or nest != "none"

    # TMP_ORD_010: degree of chronological discontinuity, 1-5
    if d == 0.0 and not any_flashback:
        disc = 1
    elif d <= _DISC_BANDS[0]:
        disc = 2
    elif d <= _DISC_BANDS[1]:
        disc = 3
    elif d <= _DISC_BANDS[2]:
        disc = 4
    else:
        disc = 5

    # EVT_TYP_009: balance of present-time vs flashback events
    if past_heavy < _FLASHBACK_LOW:
        evt = "mostly_present_progression"
    elif past_heavy <= _FLASHBACK_HIGH:
        evt = "balanced_present_and_past"
    else:
        evt = "mostly_past_or_recollected_events"

    # TMP_ORD_004: depth of flashback nesting
    if not any_flashback:
        nest_cat = "no_flashbacks"
    elif nest == "multi":
        nest_cat = "multi_level_nested_flashbacks"
    else:
        nest_cat = "single_level_flashbacks_only"

    # TMP_ORD_001: global chronological structure
    tr = transition_rate(placements) or 0.0
    if prolepsis > present and present < 0.25:
        glob = "reverse_chronology_dominant"   # weakly detected; see note
    elif present >= 0.999 and tr == 0.0:
        glob = "strictly_chronological"
    elif past_heavy <= _RARE_FLASHBACK:
        glob = "mostly_chronological_with_rare_flashbacks"
    elif present >= 0.50:
        glob = "chronological_spine_with_frequent_flashbacks"
    else:
        glob = "strongly_non_linear_or_fragmented"

    # TMP_ORD_007: number of interwoven temporal strands
    others = _sustained_others(dist)
    if not others:
        strands = "single_strand"
    elif len(others) == 1:
        strands = "two_main_strands_braided"
    else:
        strands = "three_or_more_strands_braided"

    return {"TMP_ORD_010_est": disc, "TMP_ORD_001_est": glob,
            "TMP_ORD_004_est": nest_cat, "TMP_ORD_007_est": strands,
            "EVT_TYP_009_est": evt}


def aggregate(placements: list[Optional[str]], records: list[dict]) -> dict:
    scored = [p for p in placements if p is not None]
    n = len(scored)
    dist = distribution(placements)
    counts = {k: sum(1 for p in scored if p == k) for k in chronology_scale.PLACEMENT_KEYS}
    return {
        "n_units": len(placements),
        "n_scored": n,
        "distribution": dist,
        "present_share": dist["present"],
        "flashback_share": _round((counts["analepsis"] + counts["mixed"]) / n) if n else None,
        "transition_rate": transition_rate(placements),
        "internal_jump_rate": internal_jump_rate(records),
        "max_nesting": max_nesting(records),
        "discontinuity_index": discontinuity_index(placements, records),
    }


def compute(units: list[dict], ctx: Optional[dict] = None) -> dict:
    ctx = ctx if ctx is not None else {}
    title = ctx.get("title", "the document")
    per_unit = score_units(units, ctx, title)
    placements = [r["placement"] for r in per_unit]
    agg = aggregate(placements, per_unit)
    return {
        "schema": SCHEMA,
        "rubric": {"version": chronology_scale.RUBRIC_VERSION,
                   **chronology_scale.PROVENANCE},
        "method": (f"per-unit LLM temporal-placement annotation (one re-ask; long "
                   f"units truncated to first {HEAD_WORDS} + last {TAIL_WORDS} "
                   f"words); deterministic discontinuity aggregations over the "
                   f"placement sequence"),
        "per_unit": per_unit,
        "aggregate": agg,
        "storyscope_projection": storyscope_projection(
            agg["distribution"], placements, per_unit),
        "note": ("storyscope_projection reconstructs StoryScope's whole-story "
                 "temporal scales (TMP_ORD_010/001/004/007, EVT_TYP_009) from the "
                 "measured per-unit quantities; its band edges are provisional and "
                 "meant to be calibrated against their gold labels on the dev split "
                 "(see rubric provenance and the re-verification caveat). "
                 "reverse_chronology_dominant is weakly detected from placement "
                 "shares alone."),
    }
