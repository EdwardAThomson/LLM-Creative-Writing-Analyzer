"""Text report rendering for narrative-dynamics results.

The JSON side of the output is the result dict itself (written by the CLI, the
same self-describing-sidecar convention as ``utils/metrics``). This module
renders the human-readable text report in the repo's existing style: uppercase
section headers, two-space indents, plain aligned tables (see
``utils/prompt_io.write_summary``).
"""
from __future__ import annotations

from typing import Optional


def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return str(x)


def _kv(lines: list[str], label: str, value, indent: str = "  ") -> None:
    lines.append(f"{indent}{label}: {_fmt(value)}")


def _tension_section(lines: list[str], res: dict) -> None:
    lines.append("TENSION TRAJECTORY (0-10, anchored rubric):")
    if "error" in res:
        lines.append(f"  ERROR: {res['error']}")
        return
    agg = res["aggregate"]
    _kv(lines, "Units scored", f"{agg['n_scored']}/{agg['n_units']}")
    _kv(lines, "Mean register", agg["mean_register"])
    _kv(lines, "Std / min / max",
        f"{_fmt(agg['std'])} / {_fmt(agg['min'])} / {_fmt(agg['max'])}")
    _kv(lines, "Volatility (mean |delta|)", agg["volatility"])
    _kv(lines, "Calm share (<=3)", agg["calm_share"])
    _kv(lines, "High share (>=8)", agg["high_share"])
    _kv(lines, "Peak", f"{_fmt(agg['peak_height'])} at position {_fmt(agg['peak_position'])}")
    _kv(lines, "Tail",
        f"mean {_fmt(agg['tail_mean'])} over final {agg['tail_units']} unit(s), "
        f"final {_fmt(agg['final_tension'])}, mode: {_fmt(agg['ending_mode'])}")
    deciles = res.get("deciles", {})
    if deciles:
        lines.append("  Decile table (mean tension by tenth of the document):")
        keys = [str(d) for d in range(10)]
        lines.append("    decile : " + " ".join(f"{k:>5}" for k in keys))
        lines.append("    tension: " + " ".join(f"{_fmt(deciles.get(k)):>5}" for k in keys))


def _blocks_section(lines: list[str], res: dict) -> None:
    lines.append("BLOCK RHYTHM (7-type prose modes):")
    if "error" in res:
        lines.append(f"  ERROR: {res['error']}")
        return
    agg = res["aggregate"]
    _kv(lines, "Paragraphs annotated",
        f"{agg['n_paragraphs'] - agg['n_unlabeled']}/{agg['n_paragraphs']}")
    dist = agg["distribution"]
    lines.append("  Type distribution (share of paragraphs):")
    for lab, v in dist.items():
        lines.append(f"    {lab:<15} {_fmt(v, 3)}")
    lines.append("  Structural gauge (the four validated signals):")
    gauge = res.get("structural_gauge", {})
    bands = gauge.get("master_bands", {})
    for key in ("words_per_mode_segment", "interiority_self_transition",
                "secondary_shading_rate", "setting_touch_rate"):
        band = bands.get(key, {}).get("master_range")
        ctx = f"  (masters {band[0]}-{band[1]}, source harness)" if band else ""
        lines.append(f"    {key:<30} {_fmt(gauge.get(key)):>8}{ctx}")
    _kv(lines, "Switch rate (per paragraph)", agg["switch_rate"])
    _kv(lines, "Max segment (words)", agg["max_segment_words"])
    _kv(lines, "Interiority share", agg["interiority_share"])
    _kv(lines, "Interiority exits (action / dialogue)",
        f"{_fmt(agg['interiority_exit_to_action'])} / {_fmt(agg['interiority_exit_to_dialogue'])}")


def _threads_section(lines: list[str], res: dict) -> None:
    lines.append("THREAD ARCHITECTURE (cast-based clustering):")
    if "error" in res:
        lines.append(f"  ERROR: {res['error']}")
        return
    agg = res["aggregate"]
    _kv(lines, "Threads (total / 2+ units)",
        f"{agg['n_threads']} / {agg['n_threads_2plus']}")
    _kv(lines, "Switch rate", agg["switch_rate"])
    _kv(lines, "Run lengths (mean / max)", f"{_fmt(agg['mean_run'])} / {_fmt(agg['max_run'])}")
    _kv(lines, "Convergence events", agg["n_convergence_events"])
    _kv(lines, "First convergence (position)", agg["first_convergence"])
    _kv(lines, "Theta sensitivity (threads at 0.2/0.3/0.4)",
        "/".join(str(v) for v in res.get("theta_sensitivity", {}).values()))
    for t in res.get("threads", []):
        if t["n_units"] < 2:
            continue
        lines.append(
            f"    T{t['id']}: {t['n_units']} units, span {t['span'][0]}-{t['span'][1]}, "
            f"mean tension {_fmt(t['mean_tension'])}, "
            f"profile: {', '.join(t['profile']) or '-'}")
    st = res.get("switch_tension")
    if st:
        sw = st["switch"]
        lines.append(
            f"  Tension at switches: n={sw['n']}, cooler {sw['cooler']} / same "
            f"{sw['same']} / hotter {sw['hotter']}, mean {_fmt(sw['mean'])}, "
            f"mean |delta| {_fmt(sw['mean_abs'])}")


def _comparison_section(lines: list[str], comparison: dict) -> None:
    lines.append("MASTERS COMPARISON (against reference distribution):")
    for metric, rows in comparison.items():
        lines.append(f"  {metric}:")
        lines.append(f"    {'key':<30}{'value':>10}{'ref mean':>10}"
                     f"{'ref range':>16}  in range")
        for key, r in rows.items():
            rng = f"{_fmt(r['ref_min'])}..{_fmt(r['ref_max'])}"
            mark = "yes" if r["within_range"] else "NO"
            lines.append(f"    {key:<30}{_fmt(r['value']):>10}"
                         f"{_fmt(r['ref_mean']):>10}{rng:>16}  {mark}")


_SECTIONS = {
    "tension_trajectory": _tension_section,
    "block_rhythm": _blocks_section,
    "thread_architecture": _threads_section,
}


def render_text(source: str, result: dict, comparison: Optional[dict] = None) -> str:
    """Render one document's narrative-dynamics result as a text report."""
    lines = ["NARRATIVE DYNAMICS ANALYSIS"]
    lines.append(f"Source: {source}")
    seg = result.get("segmentation", {})
    lines.append(f"Segmentation: {seg.get('strategy_used', '-')} "
                 f"({seg.get('n_units', '-')} units, {seg.get('total_words', '-')} words)")
    fm = seg.get("front_matter")
    if fm and fm.get("excluded"):
        lines.append(f"Front matter: {len(fm['front_units'])} \"(front)\" unit(s) "
                     f"excluded from scoring ({fm['n_units_scored']}/"
                     f"{fm['n_units_segmented']} units scored; --include-front to keep)")
    tt = seg.get("tail_trim")
    if tt:
        lines.append(f"Tail trim: {tt['trimmed_words']} words of back-matter "
                     f"removed after \"{tt['marker']}\"")
    lines.append(f"Judge: {result.get('judge', '-')}")
    lines.append(f"Benchmark: {result.get('benchmark') or 'ad hoc'} "
                 f"(metrics: {', '.join(result.get('metrics_run', []))})")
    if result.get("judge") == "dry-run":
        lines.append("NOTE: dry-run judge; all LLM-derived numbers are placeholders.")
    lines.append("")
    for name, res in result.get("metrics", {}).items():
        renderer = _SECTIONS.get(name)
        if renderer:
            renderer(lines, res)
        else:
            lines.append(f"{name.upper()}: (no text renderer; see JSON)")
        lines.append("")
    if comparison:
        _comparison_section(lines, comparison)
        lines.append("")
    return "\n".join(lines)
