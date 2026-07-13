"""Thread architecture: LLM cast extraction plus deterministic thread clustering.

Per unit, an LLM judge extracts the POV, principal cast, and a one-line strand
summary (``rubrics/cast_extraction.py``, ported with provenance). Everything
after that is pure code (``clustering.py``): majority-cast Jaccard clustering at
threshold 0.3, run/hand-off structure, convergence events, and threshold
sensitivity. A unit whose extraction fails twice gets an empty signature (it
opens or stays in its own singleton thread) and is flagged.

If ``tension_trajectory`` ran earlier in the same ctx (``ctx["unit_tensions"]``),
per-thread tension registers and the tension deltas at thread switches are also
reported (the masters study's cut-away analysis); otherwise those fields are null
and the metric stays independently runnable.
"""
from __future__ import annotations

from typing import Optional

from . import clustering, segmentation
from .judge import JudgeError, ask_json, extract_json_object, require_judge
from .rubrics import cast_extraction

NAME = "thread_architecture"
REQUIRES_LLM = True
SCHEMA = "thread_architecture/1"

HEAD_WORDS = 4000  # source-study long-unit policy for cast extraction
TAIL_WORDS = 4000


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return round(x, nd) if x is not None else None


def _parse(raw: str) -> dict:
    obj = extract_json_object(raw)
    return {"pov": str(obj.get("pov", "")),
            "principal_cast": [str(x) for x in obj.get("principal_cast", [])],
            "strand": str(obj.get("strand", ""))}


def extract_casts(units: list[dict], ctx: dict, title: str) -> list[dict]:
    judge = require_judge(ctx)
    out = []
    for u in units:
        prompt = cast_extraction.render_cast_prompt(
            title=title, label=u["label"],
            text=segmentation.truncate_middle(u["text"], HEAD_WORDS, TAIL_WORDS))
        try:
            rec = ask_json(judge, prompt, _parse, ctx=ctx)
        except JudgeError as e:
            rec = {"pov": "", "principal_cast": [], "strand": "", "error": str(e)}
        out.append(rec)
    return out


# --- pure aggregations ----------------------------------------------------------------

def switch_deltas(assign: list[int], tensions: list[Optional[int]]) -> dict:
    """Tension deltas at thread switches vs same-thread transitions (cut-away analysis)."""
    sw, same = [], []
    for i in range(1, len(assign)):
        if tensions[i] is None or tensions[i - 1] is None:
            continue
        d = tensions[i] - tensions[i - 1]
        (sw if assign[i] != assign[i - 1] else same).append(d)

    def summary(ds: list[int]) -> dict:
        if not ds:
            return {"n": 0, "cooler": 0, "same": 0, "hotter": 0,
                    "mean": None, "mean_abs": None}
        return {"n": len(ds),
                "cooler": sum(1 for d in ds if d < 0),
                "same": sum(1 for d in ds if d == 0),
                "hotter": sum(1 for d in ds if d > 0),
                "mean": _round(sum(ds) / len(ds)),
                "mean_abs": _round(sum(abs(d) for d in ds) / len(ds))}

    return {"switch": summary(sw), "same_thread": summary(same),
            "switch_deltas": sw}


def compute(units: list[dict], ctx: Optional[dict] = None) -> dict:
    ctx = ctx if ctx is not None else {}
    title = ctx.get("title", "the document")
    aliases = ctx.get("aliases")
    tensions = ctx.get("unit_tensions")

    raw = extract_casts(units, ctx, title)
    sigs = clustering.build_signatures(raw, aliases)
    threads, assign, merges = clustering.cluster(sigs)
    runs = clustering.run_lengths(assign)
    n = len(units)

    per_unit = []
    for u, r, s, a in zip(units, raw, sigs, assign):
        rec = {"index": u["index"], "label": u["label"], "thread": a,
               "pov": s["pov"], "cast": s["cast"], "strand": r["strand"]}
        if "error" in r:
            rec["error"] = r["error"]
        per_unit.append(rec)

    thread_info = []
    for t in threads:
        members = t["members"]
        info = {
            "id": t["id"],
            "n_units": len(members),
            "span": [min(members), max(members)],
            "profile": sorted(clustering.majority_profile([sigs[i] for i in members], "sig")),
            "strands": [raw[i]["strand"] for i in members[:2]],
        }
        if tensions:
            ts = [tensions[i] for i in members if tensions[i] is not None]
            info["mean_tension"] = _round(sum(ts) / len(ts)) if ts else None
            info["tension_range"] = [min(ts), max(ts)] if ts else None
        else:
            info["mean_tension"] = None
            info["tension_range"] = None
        thread_info.append(info)

    switches = sum(1 for a, b in zip(assign, assign[1:]) if a != b)
    merge_events = [{"unit": i, "position": _round((i + 0.5) / n),
                     "threads": ids} for i, ids in merges]
    first_convergence = merge_events[0]["position"] if merge_events else None
    merged_ids = {tid for m in merge_events for tid in m["threads"]}
    never_merged = [t["id"] for t in thread_info
                    if t["n_units"] >= 2 and t["id"] not in merged_ids]

    aggregate = {
        "n_units": n,
        "n_extraction_failures": sum(1 for r in raw if "error" in r),
        "n_threads": len(threads),
        "n_threads_2plus": sum(1 for t in thread_info if t["n_units"] >= 2),
        "switch_rate": _round(switches / (n - 1), 3) if n > 1 else None,
        "mean_run": _round(sum(runs) / len(runs)) if runs else None,
        "max_run": max(runs) if runs else None,
        "n_convergence_events": len(merge_events),
        "first_convergence": first_convergence,
    }

    deltas = (switch_deltas(assign, tensions)
              if tensions and len(tensions) == n else None)
    if deltas:
        aggregate["switch_delta_mean"] = deltas["switch"]["mean"]
        aggregate["switch_delta_mean_abs"] = deltas["switch"]["mean_abs"]

    return {
        "schema": SCHEMA,
        "rubric": {"version": cast_extraction.RUBRIC_VERSION,
                   **cast_extraction.PROVENANCE},
        "method": (f"LLM cast/POV extraction per unit (long units truncated to "
                   f"first {HEAD_WORDS} + last {TAIL_WORDS} words); deterministic "
                   f"majority-cast Jaccard clustering at theta {clustering.DEFAULT_THETA}"),
        "per_unit": per_unit,
        "threads": thread_info,
        "runs": runs,
        "merges": merge_events,
        "never_merged_threads": never_merged,
        "theta_sensitivity": clustering.threshold_sensitivity(sigs),
        "switch_tension": deltas,
        "aggregate": aggregate,
        "note": ("Thread counts are threshold-dependent (see theta_sensitivity; "
                 "the source study's known edge: single-POV books with rotating "
                 "supporting casts fragment at 0.3 and unify at 0.2). switch_tension "
                 "is present only when tension_trajectory ran in the same pass."),
    }
