"""Tests for the nd1 cross-corpus aggregator (utils/metrics/aggregate_nd1.py).

Covers row flattening from a sidecar, the incomplete-annotation flag firing,
and the st1<->nd1 join producing st1-only rows with blank nd1 columns. No
network, no real corpus data — tiny synthetic fixtures only.
"""
from __future__ import annotations

import csv
import json

from utils.metrics import aggregate_nd1 as agg


def _make_sidecar(
    book: str,
    *,
    n_units=4,
    n_scored=4,
    n_paragraphs=10,
    n_unlabeled=0,
    excluded_front=0,
) -> dict:
    """Build a minimal-but-schema-shaped nd1 sidecar dict."""
    dist = {
        "SETTING": 0.1,
        "CHARACTER_DESC": 0.1,
        "LORE": 0.1,
        "DIALOGUE": 0.4,
        "ACTION": 0.2,
        "INTERIORITY": 0.05,
        "TRANSITION": 0.05,
    }
    front_matter = None
    if excluded_front:
        front_matter = {"front_units": [{"index": 0, "label": "(front)", "words": 100}] * excluded_front}
    return {
        "schema": "narrative_dynamics/1",
        "source": f"{book}.md",
        "benchmark": "nd1",
        "judge": "ai_helper:openrouter:deepseek/deepseek-chat",
        "segmentation": {
            "n_units": n_units,
            "total_words": 12345,
            "front_matter": front_matter,
            "apparatus": None,
        },
        "metrics": {
            "tension_trajectory": {
                "aggregate": {
                    "n_units": n_scored,
                    "n_scored": n_scored,
                    "mean_register": 5.5,
                    "std": 1.5,
                    "min": 2,
                    "max": 9,
                    "volatility": 1.8,
                    "calm_share": 0.2,
                    "high_share": 0.3,
                    "peak_height": 9,
                    "peak_position": 0.6,
                    "tail_mean": 4.0,
                    "final_tension": 3,
                }
            },
            "block_rhythm": {
                "aggregate": {
                    "n_paragraphs": n_paragraphs,
                    "n_unlabeled": n_unlabeled,
                    "distribution": dist,
                    "switch_rate": 0.5,
                    "words_per_mode_segment": 120.0,
                    "interiority_self_transition": 0.15,
                    "secondary_shading_rate": 0.28,
                    "setting_touch_rate": 0.09,
                }
            },
            "thread_architecture": {
                "aggregate": {
                    "n_units": n_scored,
                    "n_extraction_failures": 0,
                    "n_threads": 3,
                    "n_threads_2plus": 2,
                    "switch_rate": 0.4,
                    "mean_run": 2.0,
                    "max_run": 3,
                    "n_convergence_events": 1,
                    "first_convergence": 0.5,
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Row flattening
# ---------------------------------------------------------------------------

def test_extract_row_flattens_expected_fields():
    sidecar = _make_sidecar("alpha-book")
    row, findings = agg.extract_row("alpha-book", sidecar)

    assert row["book"] == "alpha-book"
    assert row["judge"] == "ai_helper:openrouter:deepseek/deepseek-chat"
    assert row["units_scored"] == 4
    assert row["units_segmented"] == 4
    assert row["total_words"] == 12345
    assert row["paragraphs_annotated"] == 10
    assert row["paragraphs_total"] == 10
    assert row["complete"] is True

    assert row["tension_mean"] == 5.5
    assert row["tension_peak"] == 9
    assert row["tension_tail_final"] == 3

    assert row["br_dialogue_share"] == 0.4
    assert row["br_switch_rate"] == 0.5
    assert row["br_words_per_mode_segment"] == 120.0

    assert row["th_threads_total"] == 3
    assert row["th_run_length_max"] == 3

    # a clean, fully-consistent sidecar should raise no integrity findings
    assert findings == []


def test_extract_row_arithmetic_mismatch_flagged():
    # segmented=4, one front unit excluded -> expected scored = 3, but sidecar
    # (inconsistently) reports 4 scored.
    sidecar = _make_sidecar("mismatch-book", n_units=4, n_scored=4, excluded_front=1)
    row, findings = agg.extract_row("mismatch-book", sidecar)
    kinds = {f["kind"] for f in findings}
    assert "arithmetic_mismatch" in kinds


# ---------------------------------------------------------------------------
# Incomplete paragraph annotation flag
# ---------------------------------------------------------------------------

def test_incomplete_paragraph_annotation_flagged():
    sidecar = _make_sidecar("partial-book", n_paragraphs=100, n_unlabeled=7)
    row, findings = agg.extract_row("partial-book", sidecar)

    assert row["paragraphs_annotated"] == 93
    assert row["paragraphs_total"] == 100
    assert row["complete"] is False

    kinds = {f["kind"] for f in findings}
    assert "incomplete_paragraph_annotation" in kinds

    # and it should also produce a hard flag
    h_flags = agg.hard_flags([row])
    assert any(f["kind"] == "hard_incomplete_annotation" and f["book"] == "partial-book" for f in h_flags)


def test_complete_annotation_not_flagged():
    sidecar = _make_sidecar("complete-book", n_paragraphs=50, n_unlabeled=0)
    row, findings = agg.extract_row("complete-book", sidecar)
    assert row["complete"] is True
    kinds = {f["kind"] for f in findings}
    assert "incomplete_paragraph_annotation" not in kinds
    assert agg.hard_flags([row]) == []


# ---------------------------------------------------------------------------
# Load from disk (glob) + report writers don't crash on tiny input
# ---------------------------------------------------------------------------

def test_load_sidecars_and_write_reports(tmp_path):
    input_dir = tmp_path / "nd1"
    input_dir.mkdir()
    for book in ("book-one", "book-two"):
        (input_dir / f"{book}.nd.json").write_text(json.dumps(_make_sidecar(book)), encoding="utf-8")
    # a malformed file should be reported as a load failure, not crash the run
    (input_dir / "broken.nd.json").write_text("{not valid json", encoding="utf-8")

    sidecars, failures = agg.load_sidecars(input_dir)
    assert set(sidecars.keys()) == {"book-one", "book-two"}
    assert len(failures) == 1
    assert failures[0][0] == "broken.nd.json"

    rows = []
    all_findings = []
    for book, sidecar in sidecars.items():
        row, findings = agg.extract_row(book, sidecar)
        rows.append(row)
        all_findings.extend(findings)

    stats = agg.compute_stats(rows)
    o_flags = agg.outlier_flags(rows, stats)
    h_flags = agg.hard_flags(rows)

    outdir = tmp_path / "out"
    outdir.mkdir()
    csv_path = outdir / "nd1_corpus_table.csv"
    md_path = outdir / "nd1_corpus_table.md"
    flags_path = outdir / "nd1_corpus_flags.md"

    agg.write_csv(rows, csv_path)
    agg.write_md_table(rows, md_path)
    agg.write_flags_report(flags_path, sorted(sidecars.keys()), failures, all_findings, o_flags, h_flags, rows, stats)

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    flags_text = flags_path.read_text(encoding="utf-8")
    assert "Load failures: 1" in flags_text
    assert "broken.nd.json" in flags_text


# ---------------------------------------------------------------------------
# st1 <-> nd1 join
# ---------------------------------------------------------------------------

def _write_st1_csv(path):
    fieldnames = ["book", "mtld_mean", "cast_size"]
    rows = [
        {"book": "alpha-book", "mtld_mean": "90.1", "cast_size": "50"},
        {"book": "st1-only-book", "mtld_mean": "80.0", "cast_size": "30"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_combined_dataset_left_join_st1_only_rows_blank(tmp_path):
    st1_path = tmp_path / "st1_corpus_table.csv"
    _write_st1_csv(st1_path)
    st1_fieldnames, st1_rows = agg.load_st1_table(st1_path)

    nd1_sidecar = _make_sidecar("alpha-book")
    nd1_row, _ = agg.extract_row("alpha-book", nd1_sidecar)
    nd1_rows_by_book = {"alpha-book": nd1_row}

    fieldnames, combined_rows = agg.build_combined_dataset(st1_fieldnames, st1_rows, nd1_rows_by_book)

    assert fieldnames[0] == "book"
    assert "st1_mtld_mean" in fieldnames
    assert "nd1_tension_mean" in fieldnames

    by_book = {r["book"]: r for r in combined_rows}
    assert set(by_book.keys()) == {"alpha-book", "st1-only-book"}

    # matched row: nd1 columns populated
    matched = by_book["alpha-book"]
    assert matched["st1_mtld_mean"] == "90.1"
    assert matched["nd1_tension_mean"] == 5.5
    assert matched["nd1_judge"] == "ai_helper:openrouter:deepseek/deepseek-chat"

    # st1-only row: nd1 columns blank, st1 columns present
    unmatched = by_book["st1-only-book"]
    assert unmatched["st1_mtld_mean"] == "80.0"
    assert unmatched["nd1_tension_mean"] == ""
    assert unmatched["nd1_judge"] == ""
