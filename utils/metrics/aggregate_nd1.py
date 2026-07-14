#!/usr/bin/env python3
"""Cross-corpus aggregator for nd1 (Narrative Dynamics) benchmark result sidecars.

Loads every ``*.nd.json`` sidecar in an input directory (one per book),
flattens each into a single comparable row of key metrics (tension
trajectory, block rhythm, thread architecture), self-checks each book's
internal segmentation/paragraph bookkeeping, and computes cross-corpus
outlier statistics. Mirrors ``aggregate_corpus.py`` (the st1 aggregator) in
structure and conventions.

This is a QA / correctness SURFACING tool, not a bug-decider: it never
concludes that a flagged value is wrong. It reports arithmetic, distribution
outliers, and hard-coded "always worth a look" conditions, tags each against
a short list of pre-declared, known-and-accepted quirks of this corpus, and
leaves every judgment call to the human triaging the report.

Handles a variable / partial set of books (the nd1 run is ongoing — some
titles are pending) rather than assuming a fixed count.

Outputs (written under --outdir):
  - nd1_corpus_table.csv    one row per book, flat metric columns
  - nd1_corpus_table.md     the same table, as Markdown
  - nd1_corpus_flags.md     integrity findings + outlier/hard flags +
                             known-limitations section + distribution summary

Optionally, if --st1-table is given, also writes a combined dataset that left
joins the st1 table with this nd1 table on book name:
  - corpus_dataset.csv
  - corpus_dataset.md

Usage:
    .venv/bin/python utils/metrics/aggregate_nd1.py \\
        --input-dir /path/to/scores/nd1_ab/deepseek \\
        --st1-table /path/to/scores/st1_corpus_table.csv \\
        --outdir /path/to/scores
"""
from __future__ import annotations

import argparse
import csv
import math
import json
import statistics
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

EXPECTED_METRICS = [
    "tension_trajectory",
    "block_rhythm",
    "thread_architecture",
]

BLOCK_TYPES = [
    "SETTING",
    "CHARACTER_DESC",
    "LORE",
    "DIALOGUE",
    "ACTION",
    "INTERIORITY",
    "TRANSITION",
]

# Flat row columns, in output order. "book" and "judge" are identity/string
# columns; everything else is numeric (bool "complete" handled separately).
IDENTITY_COLUMNS = [
    "book",
    "judge",
    "units_scored",
    "units_segmented",
    "total_words",
    "paragraphs_annotated",
    "paragraphs_total",
    "complete",
]

TENSION_COLUMNS = [
    "tension_mean",
    "tension_std",
    "tension_min",
    "tension_max",
    "tension_peak",
    "tension_peak_position",
    "tension_calm_share",
    "tension_high_share",
    "tension_volatility",
    "tension_tail_mean",
    "tension_tail_final",
]

BLOCK_RHYTHM_COLUMNS = (
    [f"br_{t.lower()}_share" for t in BLOCK_TYPES]
    + [
        "br_switch_rate",
        "br_words_per_mode_segment",
        "br_interiority_self_transition",
        "br_secondary_shading_rate",
        "br_setting_touch_rate",
    ]
)

THREAD_COLUMNS = [
    "th_threads_total",
    "th_threads_2plus",
    "th_switch_rate",
    "th_run_length_mean",
    "th_run_length_max",
    "th_convergence_events",
    "th_first_convergence_position",
]

ALL_ROW_COLUMNS = IDENTITY_COLUMNS + TENSION_COLUMNS + BLOCK_RHYTHM_COLUMNS + THREAD_COLUMNS

# Numeric columns that participate in cross-corpus outlier (z-score) analysis.
# Excludes "book"/"judge" (strings) and "complete" (bool, handled as a hard
# flag instead).
NUMERIC_METRIC_COLUMNS = [
    c for c in ALL_ROW_COLUMNS if c not in ("book", "judge", "complete")
]

# Columns expected to be a share/rate/position in [0, 1].
UNIT_INTERVAL_COLUMNS = (
    [f"br_{t.lower()}_share" for t in BLOCK_TYPES]
    + [
        "tension_calm_share",
        "tension_high_share",
        "br_switch_rate",
        "br_interiority_self_transition",
        "br_secondary_shading_rate",
        "br_setting_touch_rate",
        "th_switch_rate",
        "th_first_convergence_position",
        "tension_peak_position",
    ]
)

# Columns expected to be on the 0-10 tension scale.
TENSION_SCALE_COLUMNS = ["tension_mean", "tension_min", "tension_max", "tension_peak", "tension_tail_mean", "tension_tail_final"]


# ---------------------------------------------------------------------------
# Pre-declared known limitations. These do not suppress a flag, they just get
# it re-tagged EXPECTED instead of REVIEW when the (book, metric) pair
# matches. `metrics=None` means "any metric column, for this book". `books=None`
# means "any book, for this metric column".
# ---------------------------------------------------------------------------

KNOWN_LIMITATIONS = [
    {
        "id": "conrad-heartofdarkness-3units",
        "books": {"conrad-heartofdarkness"},
        "metrics": None,
        "description": (
            "conrad-heartofdarkness is segmented into only 3 units (its 3 long "
            "parts), so its per-unit stats (tension std/min/max, thread run "
            "lengths, block-rhythm switch rate, etc.) are extreme-but-expected, "
            "not bugs."
        ),
    },
    {
        "id": "zero-convergence-null-position",
        "books": None,
        "metrics": {"th_first_convergence_position"},
        "description": (
            "th_first_convergence_position is null exactly when "
            "n_convergence_events == 0 (verified across the corpus: no thread "
            "ever converges, so 'position of first convergence' is undefined) "
            "— a well-defined structural null, not missing data."
        ),
    },
]


def classify_known_limitation(book: str, metric: str) -> dict | None:
    """Return the first matching known-limitation rule for (book, metric), if any."""
    for rule in KNOWN_LIMITATIONS:
        book_ok = rule["books"] is None or book in rule["books"]
        metric_ok = rule["metrics"] is None or metric in rule["metrics"]
        if book_ok and metric_ok:
            return rule
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sidecars(input_dir: Path) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Load all *.nd.json in input_dir. Returns (book -> parsed, failures).

    failures is a list of (filename, error message) for files that existed but
    failed to parse/load — reported, never silently skipped.
    """
    sidecars: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for path in sorted(input_dir.glob("*.nd.json")):
        book = path.name[: -len(".nd.json")]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                sidecars[book] = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - report, don't swallow
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
    return sidecars, failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_bad_number(value: Any) -> bool:
    """True if value is missing/None/NaN where a real number was expected."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return isinstance(value, float) and math.isnan(value)
    return True  # wrong type entirely (e.g. a string) counts as bad


def _get(d: Any, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# Row extraction (Step 1) + per-book integrity checks (Step 2)
# ---------------------------------------------------------------------------

def extract_row(book: str, sidecar: dict) -> tuple[dict, list[dict]]:
    """Build the flat metrics row for one book, plus a list of integrity findings.

    Never raises: any structural surprise becomes an integrity finding and the
    corresponding row field is left as None rather than crashing the run.
    """
    findings: list[dict] = []

    metrics = sidecar.get("metrics", {}) or {}
    for metric in EXPECTED_METRICS:
        if metric not in metrics:
            findings.append({
                "book": book, "metric": metric, "kind": "missing_metric",
                "detail": f"metric {metric!r} absent from sidecar['metrics']",
            })
            continue
        if not isinstance(metrics[metric], dict) or "aggregate" not in metrics[metric] or not metrics[metric]["aggregate"]:
            findings.append({
                "book": book, "metric": metric, "kind": "missing_aggregate",
                "detail": f"expected key 'aggregate' missing/empty in metrics[{metric!r}]",
            })

    tension = metrics.get("tension_trajectory", {}) or {}
    block_rhythm = metrics.get("block_rhythm", {}) or {}
    threads = metrics.get("thread_architecture", {}) or {}

    tt_agg = tension.get("aggregate", {}) or {}
    br_agg = block_rhythm.get("aggregate", {}) or {}
    th_agg = threads.get("aggregate", {}) or {}

    seg = sidecar.get("segmentation", {}) or {}
    front_matter = seg.get("front_matter") or {}
    apparatus = seg.get("apparatus") or {}

    n_paragraphs = br_agg.get("n_paragraphs")
    n_unlabeled = br_agg.get("n_unlabeled")
    paragraphs_annotated = (
        n_paragraphs - n_unlabeled
        if isinstance(n_paragraphs, (int, float)) and isinstance(n_unlabeled, (int, float))
        else None
    )

    complete = (
        paragraphs_annotated == n_paragraphs
        if paragraphs_annotated is not None and n_paragraphs is not None
        else None
    )

    dist = br_agg.get("distribution", {}) or {}

    row: dict[str, Any] = {
        "book": book,
        "judge": sidecar.get("judge"),
        "units_scored": tt_agg.get("n_units"),
        "units_segmented": seg.get("n_units"),
        "total_words": seg.get("total_words"),
        "paragraphs_annotated": paragraphs_annotated,
        "paragraphs_total": n_paragraphs,
        "complete": complete,
        # tension trajectory
        "tension_mean": tt_agg.get("mean_register"),
        "tension_std": tt_agg.get("std"),
        "tension_min": tt_agg.get("min"),
        "tension_max": tt_agg.get("max"),
        "tension_peak": tt_agg.get("peak_height"),
        "tension_peak_position": tt_agg.get("peak_position"),
        "tension_calm_share": tt_agg.get("calm_share"),
        "tension_high_share": tt_agg.get("high_share"),
        "tension_volatility": tt_agg.get("volatility"),
        "tension_tail_mean": tt_agg.get("tail_mean"),
        "tension_tail_final": tt_agg.get("final_tension"),
        # block rhythm
        "br_switch_rate": br_agg.get("switch_rate"),
        "br_words_per_mode_segment": br_agg.get("words_per_mode_segment"),
        "br_interiority_self_transition": br_agg.get("interiority_self_transition"),
        "br_secondary_shading_rate": br_agg.get("secondary_shading_rate"),
        "br_setting_touch_rate": br_agg.get("setting_touch_rate"),
        # thread architecture
        "th_threads_total": th_agg.get("n_threads"),
        "th_threads_2plus": th_agg.get("n_threads_2plus"),
        "th_switch_rate": th_agg.get("switch_rate"),
        "th_run_length_mean": th_agg.get("mean_run"),
        "th_run_length_max": th_agg.get("max_run"),
        "th_convergence_events": th_agg.get("n_convergence_events"),
        "th_first_convergence_position": th_agg.get("first_convergence"),
    }
    for t in BLOCK_TYPES:
        row[f"br_{t.lower()}_share"] = dist.get(t)

    # --- null/NaN check on numeric fields ---
    for col in NUMERIC_METRIC_COLUMNS:
        if _is_bad_number(row.get(col)):
            findings.append({
                "book": book, "metric": col, "kind": "null_or_nan",
                "detail": f"{col} is missing/None/NaN (value={row.get(col)!r})",
            })

    # --- incomplete paragraph annotation ---
    if complete is False:
        findings.append({
            "book": book, "metric": "paragraphs_annotated", "kind": "incomplete_paragraph_annotation",
            "detail": f"paragraphs_annotated({paragraphs_annotated}) < paragraphs_total({n_paragraphs}); "
                      f"n_unlabeled={n_unlabeled}",
        })

    # --- units_scored arithmetic cross-check against segmentation exclusions ---
    units_segmented = row["units_segmented"]
    units_scored = row["units_scored"]
    excluded_front = len(front_matter.get("front_units") or [])
    excluded_apparatus = len(apparatus.get("apparatus_units") or [])
    excluded_total = excluded_front + excluded_apparatus
    if units_segmented is not None and units_scored is not None:
        expected_scored = units_segmented - excluded_total
        if expected_scored != units_scored:
            findings.append({
                "book": book, "metric": "units_scored", "kind": "arithmetic_mismatch",
                "detail": (
                    f"units_segmented({units_segmented}) - excluded(front={excluded_front}"
                    f"+apparatus={excluded_apparatus}={excluded_total}) = {expected_scored}"
                    f" != units_scored({units_scored})"
                ),
            })
    else:
        findings.append({
            "book": book, "metric": "units_scored", "kind": "arithmetic_uncheckable",
            "detail": f"units_segmented={units_segmented!r}, units_scored={units_scored!r}",
        })

    # --- block_rhythm distribution should sum to ~1.0 ---
    if dist:
        dist_sum = sum(v for v in dist.values() if isinstance(v, (int, float)))
        if abs(dist_sum - 1.0) > 0.02:
            findings.append({
                "book": book, "metric": "br_distribution_sum", "kind": "out_of_range",
                "detail": f"block_rhythm distribution shares sum to {dist_sum:.4f}, expected ~1.0",
            })

    # --- range checks: [0, 1] shares/rates/positions ---
    for col in UNIT_INTERVAL_COLUMNS:
        v = row.get(col)
        if isinstance(v, (int, float)) and not (0.0 <= v <= 1.0):
            findings.append({
                "book": book, "metric": col, "kind": "out_of_range",
                "detail": f"{col}={v} not in [0,1]",
            })

    # --- range checks: [0, 10] tension scale ---
    for col in TENSION_SCALE_COLUMNS:
        v = row.get(col)
        if isinstance(v, (int, float)) and not (0.0 <= v <= 10.0):
            findings.append({
                "book": book, "metric": col, "kind": "out_of_range",
                "detail": f"{col}={v} not in [0,10]",
            })

    # --- thread extraction failures (hard-ish, surfaced as a finding too) ---
    n_extraction_failures = th_agg.get("n_extraction_failures")
    if isinstance(n_extraction_failures, (int, float)) and n_extraction_failures > 0:
        findings.append({
            "book": book, "metric": "n_extraction_failures", "kind": "extraction_failures",
            "detail": f"thread_architecture.aggregate.n_extraction_failures={n_extraction_failures}",
        })

    return row, findings


# ---------------------------------------------------------------------------
# Step 3: cross-corpus outlier + hard flags
# ---------------------------------------------------------------------------

def compute_stats(rows: list[dict]) -> dict[str, dict]:
    stats = {}
    for col in NUMERIC_METRIC_COLUMNS:
        values = [r[col] for r in rows if isinstance(r.get(col), (int, float)) and not _is_bad_number(r.get(col))]
        if len(values) < 2:
            stats[col] = {"mean": (values[0] if values else None), "std": 0.0, "n": len(values)}
            continue
        stats[col] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "n": len(values),
        }
    return stats


def outlier_flags(rows: list[dict], stats: dict[str, dict]) -> list[dict]:
    flags = []
    for col in NUMERIC_METRIC_COLUMNS:
        s = stats[col]
        if not s["std"]:
            continue
        for r in rows:
            v = r.get(col)
            if not isinstance(v, (int, float)) or _is_bad_number(v):
                continue
            z = (v - s["mean"]) / s["std"]
            if abs(z) > 2.5:
                flags.append({
                    "book": r["book"], "metric": col, "value": v, "z": z,
                    "kind": "outlier_zscore",
                })
    return flags


def hard_flags(rows: list[dict]) -> list[dict]:
    flags = []
    for r in rows:
        book = r["book"]
        if r.get("complete") is False:
            flags.append({"book": book, "metric": "complete", "value": False, "kind": "hard_incomplete_annotation"})
    return flags


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_ROW_COLUMNS)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["book"]):
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in ALL_ROW_COLUMNS})


def write_md_table(rows: list[dict], out_path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda x: x["book"])
    lines = []
    lines.append("# nd1 cross-corpus metrics table")
    lines.append("")
    lines.append("| " + " | ".join(ALL_ROW_COLUMNS) + " |")
    lines.append("|" + "|".join(["---"] * len(ALL_ROW_COLUMNS)) + "|")
    for r in rows_sorted:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in ALL_ROW_COLUMNS) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _distribution_summary(rows: list[dict], stats: dict[str, dict]) -> list[str]:
    lines = []
    for col in NUMERIC_METRIC_COLUMNS:
        pairs = [(r["book"], r.get(col)) for r in rows if isinstance(r.get(col), (int, float)) and not _is_bad_number(r.get(col))]
        if not pairs:
            lines.append(f"- **{col}**: no valid values")
            continue
        pairs_sorted = sorted(pairs, key=lambda p: p[1])
        min_book, min_v = pairs_sorted[0]
        max_book, max_v = pairs_sorted[-1]
        vals = [v for _, v in pairs]
        med = statistics.median(vals)
        s = stats[col]
        mean_s = f"{s['mean']:.4g}" if s["mean"] is not None else "n/a"
        std_s = f"{s['std']:.4g}" if s["std"] is not None else "n/a"
        lines.append(
            f"- **{col}**: min={_fmt(min_v)} ({min_book}), median={_fmt(med)}, "
            f"max={_fmt(max_v)} ({max_book}); mean={mean_s}, sd={std_s}, n={s['n']}"
        )
    return lines


def write_flags_report(
    out_path: Path,
    books_loaded: list[str],
    load_failures: list[tuple[str, str]],
    all_findings: list[dict],
    all_outlier_flags: list[dict],
    all_hard_flags: list[dict],
    rows: list[dict],
    stats: dict[str, dict],
) -> tuple[int, int, list[str]]:
    """Write the flags report. Returns (n_integrity_findings, n_review_flags, review_one_liners)."""

    def tag_line(book: str, metric: str) -> tuple[str, dict | None]:
        rule = classify_known_limitation(book, metric)
        if rule:
            return f"EXPECTED [{rule['id']}]", rule
        return "REVIEW", None

    lines: list[str] = []
    lines.append("# nd1 corpus anomaly report")
    lines.append("")
    lines.append(
        "QA surfacing report only. Nothing here is asserted to be a bug — each "
        "item is tagged EXPECTED (matches a pre-declared known limitation) or "
        "REVIEW (novel, needs a human look)."
    )
    lines.append("")

    # --- load status ---
    lines.append("## Load status")
    lines.append("")
    lines.append(f"- Sidecars loaded: {len(books_loaded)}")
    if load_failures:
        lines.append(f"- **Load failures: {len(load_failures)}**")
        for fname, err in load_failures:
            lines.append(f"  - `{fname}`: {err}")
    else:
        lines.append("- Load failures: 0")
    judges = sorted({r.get("judge") for r in rows if r.get("judge")})
    lines.append(f"- Judge model(s) observed: {', '.join(judges) if judges else 'none'}")
    if len(judges) <= 1:
        lines.append(
            "  - Single judge across the corpus (expected at this stage — nd1 is "
            "currently a DeepSeek-only run; no cross-judge comparison possible yet)."
        )
    else:
        lines.append("  - REVIEW: more than one judge model present — cross-judge comparability not yet established.")
    lines.append("")

    # --- integrity findings ---
    lines.append("## Integrity findings (per book)")
    lines.append("")
    if not all_findings:
        lines.append("None. All books passed the arithmetic/null/range checks.")
    else:
        lines.append(f"Total: {len(all_findings)}")
        lines.append("")
        lines.append("| book | metric | kind | tag | detail |")
        lines.append("|---|---|---|---|---|")
        for f in sorted(all_findings, key=lambda x: (x["book"], x["metric"])):
            tag, _ = tag_line(f["book"], f["metric"])
            lines.append(f"| {f['book']} | {f['metric']} | {f['kind']} | {tag} | {f['detail']} |")
    lines.append("")

    # --- outlier flags ---
    lines.append("## Cross-corpus outlier flags (|z| > 2.5)")
    lines.append("")
    if not all_outlier_flags:
        lines.append("None.")
    else:
        lines.append("| book | metric | value | z-score | tag |")
        lines.append("|---|---|---|---|---|")
        for f in sorted(all_outlier_flags, key=lambda x: (x["metric"], -abs(x["z"]))):
            tag, _ = tag_line(f["book"], f["metric"])
            lines.append(f"| {f['book']} | {f['metric']} | {_fmt(f['value'])} | {f['z']:.2f} | {tag} |")
    lines.append("")

    # --- hard flags ---
    lines.append("## Hard flags (always reported, regardless of SD)")
    lines.append("")
    if not all_hard_flags:
        lines.append("None.")
    else:
        lines.append("| book | metric | value | tag |")
        lines.append("|---|---|---|---|")
        for f in sorted(all_hard_flags, key=lambda x: (x["metric"], x["book"])):
            tag, _ = tag_line(f["book"], f["metric"])
            lines.append(f"| {f['book']} | {f['metric']} | {_fmt(f['value'])} | {tag} |")
    lines.append("")

    # --- known limitations section ---
    lines.append("## Known, pre-declared limitations (not bugs)")
    lines.append("")
    for rule in KNOWN_LIMITATIONS:
        scope = "any book" if rule["books"] is None else ", ".join(sorted(rule["books"]))
        metrics_scope = "any metric" if rule["metrics"] is None else ", ".join(sorted(rule["metrics"]))
        lines.append(f"- **[{rule['id']}]** (scope: {scope}; metrics: {metrics_scope}) — {rule['description']}")
    lines.append(
        "- **[partial-corpus]** (scope: any book; metrics: none — corpus-level) — "
        "this nd1 run is in progress; 5 'giant' books (very long, expensive to "
        "judge) are pending and simply absent from this table. Their absence is "
        "expected scope, not a load failure."
    )
    lines.append("")

    # --- distribution summary ---
    lines.append("## Distribution summary (one line per metric)")
    lines.append("")
    lines.extend(_distribution_summary(rows, stats))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_integrity = len(all_findings)
    review_items = []
    for f in all_findings:
        tag, _ = tag_line(f["book"], f["metric"])
        if tag == "REVIEW":
            review_items.append(f"integrity: {f['book']}/{f['metric']} ({f['kind']})")
    for f in all_outlier_flags:
        tag, _ = tag_line(f["book"], f["metric"])
        if tag == "REVIEW":
            review_items.append(f"outlier: {f['book']}/{f['metric']} (z={f['z']:.2f})")
    for f in all_hard_flags:
        tag, _ = tag_line(f["book"], f["metric"])
        if tag == "REVIEW":
            review_items.append(f"hard: {f['book']}/{f['metric']} ({f['kind']})")
    return n_integrity, len(review_items), review_items


# ---------------------------------------------------------------------------
# Step 3 (combined dataset): join with st1 table
# ---------------------------------------------------------------------------

def load_st1_table(path: Path) -> tuple[list[str], dict[str, dict]]:
    """Load the st1 corpus table CSV. Returns (fieldnames, book -> row dict)."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = {row["book"]: row for row in reader}
    return fieldnames, rows


def build_combined_dataset(
    st1_fieldnames: list[str],
    st1_rows: dict[str, dict],
    nd1_rows: dict[str, dict],
) -> tuple[list[str], list[dict]]:
    """LEFT JOIN st1 (base) with nd1 on book. Prefixes every non-key column.

    Every st1 book gets a row; books without a matching nd1 sidecar get blank
    nd1 columns (expected — nd1 coverage is a strict subset while the run is
    in progress).
    """
    st1_value_cols = [c for c in st1_fieldnames if c != "book"]
    nd1_value_cols = [c for c in ALL_ROW_COLUMNS if c != "book"]

    combined_fieldnames = (
        ["book"]
        + [f"st1_{c}" for c in st1_value_cols]
        + [f"nd1_{c}" for c in nd1_value_cols]
    )

    combined_rows: list[dict] = []
    for book in sorted(st1_rows.keys()):
        st1_row = st1_rows[book]
        nd1_row = nd1_rows.get(book)
        out: dict[str, Any] = {"book": book}
        for c in st1_value_cols:
            out[f"st1_{c}"] = st1_row.get(c, "")
        for c in nd1_value_cols:
            val = nd1_row.get(c) if nd1_row is not None else None
            out[f"nd1_{c}"] = "" if val is None else val
        combined_rows.append(out)

    return combined_fieldnames, combined_rows


def write_combined_csv(fieldnames: list[str], rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_combined_note(
    out_path: Path,
    n_total: int,
    n_with_nd1: int,
    n_st1_only: int,
) -> None:
    lines = [
        "# combined corpus dataset",
        "",
        f"- Row count: {n_total} (one row per book in the st1 table)",
        f"- Join key: `book`",
        f"- Books with both st1 and nd1 results: {n_with_nd1}",
        f"- Books with st1 only (nd1 pending): {n_st1_only}",
        "",
        "Every st1 column is prefixed `st1_`, every nd1 column `nd1_` (join key "
        "`book` is unprefixed). Rows for books without an nd1 sidecar yet have "
        "blank `nd1_*` columns — this is expected (nd1 is a partial, in-progress "
        "run) and shows current benchmark scope at a glance, not missing data to "
        "chase down.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing *.nd.json sidecars")
    parser.add_argument("--st1-table", required=False, type=Path, default=None,
                         help="Path to st1_corpus_table.csv; if given, also writes the combined corpus_dataset")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write the CSV/Markdown outputs to")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    sidecars, load_failures = load_sidecars(args.input_dir)

    rows = []
    all_findings = []
    for book, sidecar in sidecars.items():
        row, findings = extract_row(book, sidecar)
        rows.append(row)
        all_findings.extend(findings)

    stats = compute_stats(rows)
    o_flags = outlier_flags(rows, stats)
    h_flags = hard_flags(rows)

    csv_path = args.outdir / "nd1_corpus_table.csv"
    md_table_path = args.outdir / "nd1_corpus_table.md"
    flags_path = args.outdir / "nd1_corpus_flags.md"

    write_csv(rows, csv_path)
    write_md_table(rows, md_table_path)
    n_integrity, n_review, review_items = write_flags_report(
        flags_path, sorted(sidecars.keys()), load_failures, all_findings, o_flags, h_flags, rows, stats
    )

    # --- stdout summary ---
    print(f"Books loaded: {len(sidecars)}")
    if load_failures:
        print(f"Load failures: {len(load_failures)}")
        for fname, err in load_failures:
            print(f"  - {fname}: {err}")
    else:
        print("Load failures: 0")
    print(f"Integrity findings: {n_integrity}")
    print(f"Outlier flags: {len(o_flags)}  |  Hard flags: {len(h_flags)}")
    print(f"REVIEW-tagged items: {n_review}")
    for item in review_items:
        print(f"  - {item}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_table_path}")
    print(f"Wrote: {flags_path}")

    # --- combined dataset (Step 3) ---
    if args.st1_table is not None:
        st1_fieldnames, st1_rows = load_st1_table(args.st1_table)
        nd1_rows_by_book = {r["book"]: r for r in rows}
        combined_fieldnames, combined_rows = build_combined_dataset(st1_fieldnames, st1_rows, nd1_rows_by_book)

        n_total = len(combined_rows)
        n_with_nd1 = sum(1 for r in combined_rows if r["book"] in nd1_rows_by_book)
        n_st1_only = n_total - n_with_nd1

        dataset_csv_path = args.outdir / "corpus_dataset.csv"
        dataset_md_path = args.outdir / "corpus_dataset.md"
        write_combined_csv(combined_fieldnames, combined_rows, dataset_csv_path)
        write_combined_note(dataset_md_path, n_total, n_with_nd1, n_st1_only)

        print(f"Combined dataset rows: {n_total} (st1+nd1: {n_with_nd1}, st1-only: {n_st1_only})")
        print(f"Wrote: {dataset_csv_path}")
        print(f"Wrote: {dataset_md_path}")


if __name__ == "__main__":
    main()
