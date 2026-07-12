"""Tests for the masters-comparison reference format, loader, and comparison."""
from __future__ import annotations

import json

import pytest

from benchmarks.narrative_dynamics import reference as rf


def _doc(mean_register, volatility, wps):
    return {
        "metrics": {
            "tension_trajectory": {
                "aggregate": {"mean_register": mean_register, "volatility": volatility},
            },
            "block_rhythm": {
                "aggregate": {"words_per_mode_segment": wps},
            },
        }
    }


def test_make_reference_summarizes_values():
    ref = rf.make_reference({"a": _doc(5.0, 1.0, 90.0), "b": _doc(7.0, 1.6, 130.0)},
                            description="test corpus", benchmark="nd1")
    assert ref["schema"] == "nd_reference/1"
    assert ref["documents"] == ["a", "b"]
    assert ref["benchmark"] == "nd1"
    mr = ref["metrics"]["tension_trajectory"]["mean_register"]
    assert mr["values"] == [5.0, 7.0]
    assert mr["mean"] == 6.0
    assert mr["min"] == 5.0 and mr["max"] == 7.0
    assert ref["metrics"]["block_rhythm"]["words_per_mode_segment"]["mean"] == 110.0


def test_make_reference_skips_missing_values():
    doc_missing = {"metrics": {"tension_trajectory": {"aggregate": {"mean_register": None}}}}
    ref = rf.make_reference({"a": _doc(5.0, 1.0, 90.0), "b": doc_missing})
    assert ref["metrics"]["tension_trajectory"]["mean_register"]["values"] == [5.0]
    # thread_architecture had no values anywhere: omitted entirely
    assert "thread_architecture" not in ref["metrics"]


def test_load_reference_roundtrip(tmp_path):
    ref = rf.make_reference({"a": _doc(5.0, 1.0, 90.0)})
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    assert rf.load_reference(str(p)) == ref


def test_load_reference_rejects_wrong_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "something/else", "metrics": {}}))
    with pytest.raises(ValueError):
        rf.load_reference(str(p))


def test_load_reference_rejects_missing_metrics(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "nd_reference/1"}))
    with pytest.raises(ValueError):
        rf.load_reference(str(p))


def test_compare_flags_in_and_out_of_range():
    ref = rf.make_reference({"a": _doc(5.0, 1.0, 90.0), "b": _doc(7.0, 1.6, 130.0)})
    result = _doc(6.0, 3.5, 100.0)["metrics"]
    cmp = rf.compare(result, ref)
    assert cmp["tension_trajectory"]["mean_register"]["within_range"] is True
    assert cmp["tension_trajectory"]["volatility"]["within_range"] is False
    assert cmp["tension_trajectory"]["volatility"]["ref_min"] == 1.0
    assert cmp["tension_trajectory"]["volatility"]["ref_max"] == 1.6
    assert cmp["block_rhythm"]["words_per_mode_segment"]["value"] == 100.0


def test_compare_skips_keys_absent_from_result():
    ref = rf.make_reference({"a": _doc(5.0, 1.0, 90.0)})
    cmp = rf.compare({"tension_trajectory": {"aggregate": {"mean_register": 5.5}}}, ref)
    assert "volatility" not in cmp["tension_trajectory"]
    assert "block_rhythm" not in cmp


def test_reference_keys_cover_the_three_metrics():
    assert set(rf.REFERENCE_KEYS) == {"tension_trajectory", "block_rhythm",
                                      "thread_architecture"}
