"""Tests for the chronology_map metric (fake judge; pure aggregations)."""
from __future__ import annotations

import json

from benchmarks.narrative_dynamics import chronology_map as cm
from benchmarks.narrative_dynamics.judge import FakeJudge


def _units(n, words=300):
    return [{"index": i, "label": f"Ch {i+1}",
             "text": " ".join(f"u{i}w{k}" for k in range(words)), "words": words}
            for i in range(n)]


def _resp(placement, internal_jump=False, nesting="none", rationale="r"):
    return json.dumps({"placement": placement, "internal_jump": internal_jump,
                       "nesting": nesting, "rationale": rationale})


def _compute(responses, n=None):
    n = n if n is not None else len(responses)
    ctx = {"judge": FakeJudge(list(responses))}
    return cm.compute(_units(n), ctx), ctx


# --- scoring path ---------------------------------------------------------------------

def test_per_unit_placements_and_schema():
    out, _ = _compute([_resp("present"), _resp("analepsis"), _resp("present")])
    assert [r["placement"] for r in out["per_unit"]] == ["present", "analepsis", "present"]
    assert out["per_unit"][0]["label"] == "Ch 1"
    assert out["schema"] == "chronology_map/1"


def test_unknown_placement_retried_then_hole():
    ctx = {"judge": FakeJudge([_resp("sideways"), _resp("nowhere")])}
    out = cm.compute(_units(1), ctx)
    assert out["per_unit"][0]["placement"] is None
    assert "error" in out["per_unit"][0]
    assert ctx["judge"].calls == 2


def test_malformed_response_retried_once():
    ctx = {"judge": FakeJudge(["not json", _resp("present")])}
    out = cm.compute(_units(1), ctx)
    assert out["per_unit"][0]["placement"] == "present"
    assert ctx["judge"].calls == 2


def test_hole_does_not_crash_aggregate():
    ctx = {"judge": FakeJudge(["bad", "bad again", _resp("present")])}
    out = cm.compute(_units(2), ctx)
    assert out["per_unit"][0]["placement"] is None
    assert out["aggregate"]["n_scored"] == 1


def test_rubric_provenance_carried_in_output():
    out, _ = _compute([_resp("present")])
    rub = out["rubric"]
    assert rub["version"] == "chronology_scale/1"
    assert rub["source_project"] == "StoryScope"
    assert "TMP_ORD_010" in rub["source_features"]
    assert "re-verify" in rub["reverification_note"] or "re-verified" in rub["reverification_note"]


def test_long_unit_truncated_in_prompt():
    units = [{"index": 0, "label": "Ch 1",
              "text": " ".join(f"w{k}" for k in range(6000)), "words": 6000}]
    ctx = {"judge": FakeJudge([_resp("present")])}
    cm.compute(units, ctx)
    assert "[... middle omitted:" in ctx["judge"].prompts[0]


# --- aggregations ---------------------------------------------------------------------

def test_distribution_and_shares():
    out, _ = _compute([_resp("present"), _resp("present"),
                       _resp("analepsis"), _resp("mixed")])
    agg = out["aggregate"]
    assert agg["distribution"]["present"] == 0.5
    assert agg["present_share"] == 0.5
    assert agg["flashback_share"] == 0.5   # analepsis + mixed


def test_transition_rate_counts_placement_changes():
    # present -> analepsis -> analepsis -> present : 2 changes over 3 pairs
    assert cm.transition_rate(["present", "analepsis", "analepsis", "present"]) == round(2 / 3, 3)


def test_transition_rate_skips_holes():
    assert cm.transition_rate(["present", None, "present"]) is None  # no adjacent scored pair
    assert cm.transition_rate(["present", "analepsis", None, "present"]) == 1.0


def test_internal_jump_rate():
    recs = [{"internal_jump": True}, {"internal_jump": False}, {"internal_jump": None}]
    assert cm.internal_jump_rate(recs) == 0.5   # None (a hole) excluded


def test_max_nesting_orders_levels():
    assert cm.max_nesting([{"nesting": "none"}, {"nesting": "multi"},
                           {"nesting": "single"}]) == "multi"
    assert cm.max_nesting([{"nesting": "none"}]) == "none"


def test_discontinuity_index_is_transition_plus_jump():
    out, _ = _compute([_resp("present", internal_jump=True),
                       _resp("analepsis", internal_jump=False)])
    # 1 change over 1 pair (1.0) + 1 of 2 units jumped (0.5) = 1.5
    assert out["aggregate"]["discontinuity_index"] == 1.5


# --- storyscope_projection ------------------------------------------------------------

def test_projection_strictly_chronological():
    out, _ = _compute([_resp("present"), _resp("present"), _resp("present")])
    proj = out["storyscope_projection"]
    assert proj["TMP_ORD_010_est"] == 1
    assert proj["TMP_ORD_001_est"] == "strictly_chronological"
    assert proj["TMP_ORD_004_est"] == "no_flashbacks"
    assert proj["EVT_TYP_009_est"] == "mostly_present_progression"
    assert proj["TMP_ORD_007_est"] == "single_strand"


def test_projection_flashback_heavy_nonlinear():
    out, _ = _compute([_resp("analepsis", internal_jump=True, nesting="multi"),
                       _resp("present"),
                       _resp("analepsis", nesting="single"),
                       _resp("mixed", internal_jump=True)])
    proj = out["storyscope_projection"]
    assert proj["TMP_ORD_010_est"] == 5                     # high discontinuity index
    assert proj["TMP_ORD_004_est"] == "multi_level_nested_flashbacks"
    assert proj["EVT_TYP_009_est"] == "mostly_past_or_recollected_events"
    assert proj["TMP_ORD_007_est"] == "three_or_more_strands_braided"


def test_projection_all_holes_is_none():
    out = cm.compute(_units(1), {"judge": FakeJudge(["bad", "bad again"])})
    assert out["storyscope_projection"]["TMP_ORD_010_est"] is None
