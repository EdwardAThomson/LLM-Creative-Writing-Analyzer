#!/usr/bin/env python3
"""Cross-corpus aggregator for st1 benchmark result sidecars.

Loads every ``*.st1.metrics.json`` sidecar in an input directory (one per
book), flattens each into a single comparable row of key metrics, cross
checks each book's internal bookkeeping against its frozen extract sidecar,
and computes cross-corpus outlier statistics.

This is a QA / correctness SURFACING tool, not a bug-decider: it never
concludes that a flagged value is wrong. It reports arithmetic, distribution
outliers, and hard-coded "always worth a look" conditions, tags each against
a short list of pre-declared, known-and-accepted quirks of this corpus, and
leaves every judgment call to the human triaging the report.

Outputs (written under --outdir):
  - st1_corpus_table.csv    one row per book, flat metric columns
  - st1_corpus_table.md     the same table, as Markdown
  - st1_corpus_flags.md     integrity findings + outlier/hard flags +
                             known-limitations section + distribution summary

Usage:
    .venv/bin/python utils/metrics/aggregate_corpus.py \\
        --input-dir /path/to/scores/st1 \\
        --extract-dir /path/to/extracted \\
        --outdir /path/to/scores
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

EXPECTED_METRICS = [
    "text_structure",
    "mtld",
    "burstiness",
    "dialogue_ratio",
    "intra_text_repetition",
    "cliche_density",
    "ngram_diversity",
    "phonetic_names",
    "self_similarity",
    "opening_formula",
    "entity_census",
]

# Per-metric key(s) that stand in for "this metric block has a real summary".
# Not all 11 metrics use a literal top-level "aggregate" key (ngram_diversity
# and phonetic_names are structured differently), so this is metric-specific.
SUMMARY_KEYS = {
    "text_structure": ["aggregate"],
    "mtld": ["aggregate"],
    "burstiness": ["aggregate"],
    "dialogue_ratio": ["aggregate"],
    "intra_text_repetition": ["aggregate"],
    "cliche_density": ["aggregate"],
    "ngram_diversity": ["distinct", "self_bleu"],
    "phonetic_names": ["repeated_sounds"],
    "self_similarity": ["aggregate"],
    "opening_formula": ["aggregate"],
    "entity_census": ["aggregate"],
}

# Flat row columns that are numeric and should participate in cross-corpus
# outlier (z-score) analysis. Booleans / small integer flags handled
# separately as hard flags.
NUMERIC_METRIC_COLUMNS = [
    "units_scored",
    "units_segmented",
    "total_words",
    "mean_chapter_words",
    "mtld_mean",
    "mtld_unreliable_runs",
    "cliche_per_1k",
    "slop_per_1k",
    "em_dash_per_1k",
    "dialogue_ratio_mean",
    "distinct_3_ratio",
    "self_bleu_mean",
    "intra_unigram",
    "intra_bigram",
    "intra_trigram",
    "burstiness_mean",
    "max_pairwise_sim",
    "max_verbatim_chars",
    "n_flagged_pairs",
    "opening_mean_pairwise",
    "opening_high_pair_rate",
    "cast_size",
    "recurring_cast_size",
    "person_mentions_per_1k",
]

ALL_ROW_COLUMNS = [
    "book",
    "units_scored",
    "units_segmented",
    "total_words",
    "mean_chapter_words",
    "mtld_mean",
    "mtld_unreliable_runs",
    "cliche_per_1k",
    "slop_per_1k",
    "em_dash_per_1k",
    "dialogue_ratio_mean",
    "distinct_3_ratio",
    "self_bleu_mean",
    "intra_unigram",
    "intra_bigram",
    "intra_trigram",
    "burstiness_mean",
    "duplication_suspected",
    "max_pairwise_sim",
    "max_verbatim_chars",
    "n_flagged_pairs",
    "opening_mean_pairwise",
    "opening_high_pair_rate",
    "cast_size",
    "recurring_cast_size",
    "person_mentions_per_1k",
]


# ---------------------------------------------------------------------------
# Pre-declared known limitations (Step 4). These do not suppress a flag, they
# just get it re-tagged EXPECTED instead of REVIEW when the (book, metric)
# pair matches. `metrics=None` means "any metric column, for this book".
# `books=None` means "any book, for this metric column".
# ---------------------------------------------------------------------------

KNOWN_LIMITATIONS = [
    {
        "id": "ner-name-splitting",
        "books": None,
        "metrics": {"cast_size", "recurring_cast_size", "person_mentions_per_1k"},
        "description": (
            "NER name-splitting and title conflation inflate/split cast entries "
            "(\"van\"/\"helsing\", \"monte\"/\"cristo\", \"prince\" as a title, "
            "first-name vs surname not coreferenced). cast_size / mentions "
            "anomalies driven by naming are known, not a bug."
        ),
    },
    {
        "id": "conrad-heartofdarkness-3units",
        "books": {"conrad-heartofdarkness"},
        "metrics": None,
        "description": (
            "conrad-heartofdarkness has only 3 units (structured in 3 long "
            "parts), so its per-chapter structure metrics (mean_chapter_words "
            "very high, tiny n) are expected outliers, not bugs."
        ),
    },
    {
        "id": "collins-moonstone-signpost-calibration",
        "books": {"collins-moonstone"},
        "metrics": None,
        "description": (
            "collins-moonstone intentionally drops 9 of 12 short narrator-"
            "signpost headings (known calibration, baked in at extraction "
            "time); its structure is otherwise fine."
        ),
    },
    {
        "id": "short-chapter-mtld-unreliable",
        "books": None,
        "metrics": {"mtld_unreliable_runs"},
        "description": (
            "Short chapters (e.g. War and Peace's ~146-word chapters) can make "
            "MTLD \"unreliable\" for those units — expected, not a failure; "
            "just report the count."
        ),
    },
    {
        "id": "size-format-extremes",
        "books": {"tolstoy-warandpeace", "eddison-ouroboros"},
        "metrics": None,
        "description": (
            "tolstoy-warandpeace (365 scored units) and eddison-ouroboros "
            "(roman-colon headings) are the corpus's size/format extremes, so "
            "their unit-count and structure-shaped outliers are expected."
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
    """Load all *.st1.metrics.json in input_dir. Returns (book -> parsed, failures).

    failures is a list of (filename, error message) for files that existed but
    failed to parse/load — reported, never silently skipped.
    """
    sidecars: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for path in sorted(input_dir.glob("*.st1.metrics.json")):
        book = path.name[: -len(".st1.metrics.json")]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                sidecars[book] = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - report, don't swallow
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
    return sidecars, failures


def load_extract(extract_dir: Path, book: str) -> dict | None:
    path = extract_dir / f"{book}.extract.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


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

def extract_row(book: str, sidecar: dict, extract_json: dict | None) -> tuple[dict, list[dict]]:
    """Build the flat metrics row for one book, plus a list of integrity findings.

    Never raises: any structural surprise becomes an integrity finding and the
    corresponding row field is left as None rather than crashing the run.
    """
    findings: list[dict] = []
    models = sidecar.get("models", {})
    book_models = models.get(book)
    if book_models is None:
        # models keyed by something other than the filename stem — try the
        # sole key if there's exactly one, else bail with a hard finding.
        if isinstance(models, dict) and len(models) == 1:
            book_models = next(iter(models.values()))
        else:
            findings.append({
                "book": book, "metric": "models", "kind": "missing_metric",
                "detail": f"could not find models[{book!r}] (keys: {list(models.keys())})",
            })
            book_models = {}

    # --- metric / aggregate presence ---
    for metric in EXPECTED_METRICS:
        if metric not in book_models:
            findings.append({
                "book": book, "metric": metric, "kind": "missing_metric",
                "detail": f"metric {metric!r} absent from models[{book!r}]",
            })
            continue
        block = book_models[metric]
        for key in SUMMARY_KEYS[metric]:
            if not isinstance(block, dict) or key not in block or block[key] in (None, {}, []):
                findings.append({
                    "book": book, "metric": metric, "kind": "missing_aggregate",
                    "detail": f"expected summary key {key!r} missing/empty in models[{book!r}][{metric!r}]",
                })

    def m(metric):
        return book_models.get(metric, {}) if isinstance(book_models, dict) else {}

    text_structure = m("text_structure")
    mtld = m("mtld")
    burstiness = m("burstiness")
    dialogue_ratio = m("dialogue_ratio")
    intra_rep = m("intra_text_repetition")
    cliche = m("cliche_density")
    ngram = m("ngram_diversity")
    self_sim = m("self_similarity")
    opening = m("opening_formula")
    entity = m("entity_census")

    seg = sidecar.get("segmentation", {}) or {}

    units_segmented = seg.get("n_units")
    units_scored = _get(text_structure, "runs")

    row = {
        "book": book,
        "units_scored": units_scored,
        "units_segmented": units_segmented,
        "total_words": seg.get("total_words"),
        "mean_chapter_words": _get(text_structure, "aggregate", "words", "mean"),
        "mtld_mean": (
            _get(mtld, "aggregate_reliable_only", "mean")
            if _get(mtld, "aggregate_reliable_only") is not None
            else _get(mtld, "aggregate", "mean")
        ),
        "mtld_unreliable_runs": mtld.get("unreliable_runs"),
        "cliche_per_1k": _get(cliche, "aggregate", "cliche_per_1k"),
        "slop_per_1k": _get(cliche, "aggregate", "slop_words_per_1k"),
        "em_dash_per_1k": _get(cliche, "aggregate", "em_dash_per_1k"),
        "dialogue_ratio_mean": _get(dialogue_ratio, "aggregate", "mean"),
        "distinct_3_ratio": _get(ngram, "distinct", "distinct_3", "ratio"),
        "self_bleu_mean": _get(ngram, "self_bleu", "mean"),
        "intra_unigram": (
            _get(intra_rep, "aggregate_reliable_only", "unigram", "mean")
            if _get(intra_rep, "aggregate_reliable_only") is not None
            else _get(intra_rep, "aggregate", "unigram", "mean")
        ),
        "intra_bigram": (
            _get(intra_rep, "aggregate_reliable_only", "bigram", "mean")
            if _get(intra_rep, "aggregate_reliable_only") is not None
            else _get(intra_rep, "aggregate", "bigram", "mean")
        ),
        "intra_trigram": (
            _get(intra_rep, "aggregate_reliable_only", "trigram", "mean")
            if _get(intra_rep, "aggregate_reliable_only") is not None
            else _get(intra_rep, "aggregate", "trigram", "mean")
        ),
        "burstiness_mean": (
            _get(burstiness, "aggregate_reliable_only", "burstiness", "mean")
            if _get(burstiness, "aggregate_reliable_only") is not None
            else _get(burstiness, "aggregate", "burstiness", "mean")
        ),
        "duplication_suspected": self_sim.get("duplication_suspected"),
        "max_pairwise_sim": _get(self_sim, "aggregate", "max"),
        "max_verbatim_chars": _get(self_sim, "aggregate", "max_verbatim_chars"),
        "n_flagged_pairs": len(self_sim.get("flagged") or []),
        "opening_mean_pairwise": _get(opening, "aggregate", "mean"),
        "opening_high_pair_rate": _get(opening, "aggregate", "high_pair_rate"),
        "cast_size": _get(entity, "aggregate", "cast_size"),
        "recurring_cast_size": _get(entity, "aggregate", "recurring_cast_size"),
        "person_mentions_per_1k": _get(entity, "aggregate", "person_mentions_per_1k"),
    }

    # --- null/NaN check on numeric fields ---
    for col in NUMERIC_METRIC_COLUMNS:
        if _is_bad_number(row.get(col)):
            findings.append({
                "book": book, "metric": col, "kind": "null_or_nan",
                "detail": f"{col} is missing/None/NaN (value={row.get(col)!r})",
            })

    # --- units_scored arithmetic cross-check against frozen extract ---
    if extract_json is None:
        findings.append({
            "book": book, "metric": "units_scored", "kind": "missing_extract",
            "detail": "no matching *.extract.json found; cannot cross-check units_scored",
        })
    else:
        ext_n_units = extract_json.get("n_units")
        front_matter = seg.get("front_matter") or {}
        apparatus = seg.get("apparatus") or {}
        excluded_front = len(front_matter.get("front_units") or [])
        excluded_apparatus = len(apparatus.get("apparatus_units") or [])
        excluded_total = excluded_front + excluded_apparatus
        if ext_n_units is None or units_scored is None:
            findings.append({
                "book": book, "metric": "units_scored", "kind": "arithmetic_uncheckable",
                "detail": f"extract.n_units={ext_n_units!r}, units_scored={units_scored!r}",
            })
        else:
            expected_scored = ext_n_units - excluded_total
            if expected_scored != units_scored:
                findings.append({
                    "book": book, "metric": "units_scored", "kind": "arithmetic_mismatch",
                    "detail": (
                        f"extract.n_units({ext_n_units}) - excluded(front={excluded_front}"
                        f"+apparatus={excluded_apparatus}={excluded_total}) = {expected_scored}"
                        f" != units_scored({units_scored})"
                    ),
                })
        # segmentation.n_units should also equal the extract's n_units
        if ext_n_units is not None and units_segmented is not None and ext_n_units != units_segmented:
            findings.append({
                "book": book, "metric": "units_segmented", "kind": "arithmetic_mismatch",
                "detail": f"extract.n_units({ext_n_units}) != segmentation.n_units({units_segmented})",
            })

    # --- zero-word scored unit check ---
    ts_min_words = _get(text_structure, "aggregate", "words", "min")
    if ts_min_words == 0:
        findings.append({
            "book": book, "metric": "text_structure.words.min", "kind": "zero_word_unit",
            "detail": "at least one scored unit has 0 words (text_structure aggregate min == 0)",
        })

    # --- hard sanity ranges ---
    dr = row.get("dialogue_ratio_mean")
    if isinstance(dr, (int, float)) and not (0.0 <= dr <= 1.0):
        findings.append({
            "book": book, "metric": "dialogue_ratio_mean", "kind": "out_of_range",
            "detail": f"dialogue_ratio_mean={dr} not in [0,1]",
        })
    for col in ("cliche_per_1k", "slop_per_1k", "em_dash_per_1k", "person_mentions_per_1k"):
        v = row.get(col)
        if isinstance(v, (int, float)) and v < 0:
            findings.append({
                "book": book, "metric": col, "kind": "out_of_range",
                "detail": f"{col}={v} < 0",
            })
    mtld_mean = row.get("mtld_mean")
    if isinstance(mtld_mean, (int, float)) and not mtld_mean > 0:
        findings.append({
            "book": book, "metric": "mtld_mean", "kind": "out_of_range",
            "detail": f"mtld_mean={mtld_mean} not > 0",
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
        if r.get("duplication_suspected") is True:
            flags.append({"book": book, "metric": "duplication_suspected", "value": True, "kind": "hard_duplication_suspected"})
        n_flagged = r.get("n_flagged_pairs") or 0
        if isinstance(n_flagged, (int, float)) and n_flagged > 0:
            flags.append({"book": book, "metric": "n_flagged_pairs", "value": n_flagged, "kind": "hard_flagged_pairs"})
        mvc = r.get("max_verbatim_chars")
        if isinstance(mvc, (int, float)) and mvc > 200:
            flags.append({"book": book, "metric": "max_verbatim_chars", "value": mvc, "kind": "hard_high_verbatim_chars"})
        hpr = r.get("opening_high_pair_rate")
        if isinstance(hpr, (int, float)) and hpr > 0:
            flags.append({"book": book, "metric": "opening_high_pair_rate", "value": hpr, "kind": "hard_opening_high_pair_rate"})
        mur = r.get("mtld_unreliable_runs")
        if isinstance(mur, (int, float)) and mur > 0:
            flags.append({"book": book, "metric": "mtld_unreliable_runs", "value": mur, "kind": "hard_mtld_unreliable_runs"})
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
    lines.append("# st1 cross-corpus metrics table")
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
    lines.append("# st1 corpus anomaly report")
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
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing *.st1.metrics.json sidecars")
    parser.add_argument("--extract-dir", required=True, type=Path, help="Directory containing *.extract.json sidecars")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write the CSV/Markdown outputs to")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    sidecars, load_failures = load_sidecars(args.input_dir)

    rows = []
    all_findings = []
    for book, sidecar in sidecars.items():
        extract_json = load_extract(args.extract_dir, book)
        row, findings = extract_row(book, sidecar, extract_json)
        rows.append(row)
        all_findings.extend(findings)

    stats = compute_stats(rows)
    o_flags = outlier_flags(rows, stats)
    h_flags = hard_flags(rows)

    csv_path = args.outdir / "st1_corpus_table.csv"
    md_table_path = args.outdir / "st1_corpus_table.md"
    flags_path = args.outdir / "st1_corpus_flags.md"

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


if __name__ == "__main__":
    main()
