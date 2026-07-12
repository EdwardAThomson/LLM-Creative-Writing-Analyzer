"""Tests for the tension_trajectory metric (fake judge; pure aggregations)."""
from __future__ import annotations

import json

from benchmarks.narrative_dynamics import tension_trajectory as tt
from benchmarks.narrative_dynamics.judge import FakeJudge


def _units(n, words=300):
    return [{"index": i, "label": f"Ch {i+1}",
             "text": " ".join(f"u{i}w{k}" for k in range(words)), "words": words}
            for i in range(n)]


def _resp(level, rationale="r"):
    return json.dumps({"tension_level": level, "rationale": rationale})


def _compute(levels, n=None):
    n = n if n is not None else len(levels)
    ctx = {"judge": FakeJudge([_resp(v) for v in levels])}
    return tt.compute(_units(n), ctx), ctx


# --- scoring path ---------------------------------------------------------------------

def test_per_unit_scores_and_series():
    out, ctx = _compute([2, 5, 8, 3])
    assert [r["tension"] for r in out["per_unit"]] == [2, 5, 8, 3]
    assert ctx["unit_tensions"] == [2, 5, 8, 3]
    assert out["per_unit"][0]["label"] == "Ch 1"
    assert out["schema"] == "tension_trajectory/1"


def test_scores_clamped_to_0_10():
    out, _ = _compute([15])
    assert out["per_unit"][0]["tension"] == 10


def test_malformed_response_retried_once():
    ctx = {"judge": FakeJudge(["not json", _resp(7)])}
    out = tt.compute(_units(1), ctx)
    assert out["per_unit"][0]["tension"] == 7
    assert ctx["judge"].calls == 2


def test_persistent_failure_becomes_hole_not_crash():
    ctx = {"judge": FakeJudge(["bad", "bad again", _resp(6)])}
    out = tt.compute(_units(2), ctx)
    assert out["per_unit"][0]["tension"] is None
    assert "error" in out["per_unit"][0]
    assert out["per_unit"][1]["tension"] == 6
    assert out["aggregate"]["n_scored"] == 1


def test_rubric_provenance_carried_in_output():
    out, _ = _compute([5])
    rub = out["rubric"]
    assert rub["version"] == "tension_anchors/1"
    assert rub["source_repo"] == "StoryDaemon"
    assert "re-verified" in rub["reverification_note"] or "re-verify" in rub["reverification_note"]


def test_long_unit_truncated_in_prompt():
    units = [{"index": 0, "label": "Ch 1",
              "text": " ".join(f"w{k}" for k in range(6000)), "words": 6000}]
    ctx = {"judge": FakeJudge([_resp(4)])}
    tt.compute(units, ctx)
    assert "[... middle omitted:" in ctx["judge"].prompts[0]


# --- aggregations ---------------------------------------------------------------------

def test_mean_std_min_max():
    agg = _compute([2, 5, 8, 3])[0]["aggregate"]
    assert agg["mean_register"] == 4.5
    assert agg["min"] == 2 and agg["max"] == 8
    assert agg["n_units"] == 4 and agg["n_scored"] == 4


def test_volatility_mean_abs_successive_difference():
    agg = _compute([2, 5, 8, 3])[0]["aggregate"]
    assert agg["volatility"] == round((3 + 3 + 5) / 3, 2)


def test_volatility_skips_holes():
    assert tt.volatility([2, None, 4]) is None       # no adjacent scored pair
    assert tt.volatility([2, 4, None, 9]) == 2.0     # only the 2-4 pair counts


def test_decile_table_positions():
    out, _ = _compute([2, 5, 8, 3])
    # 4 units at midpoints 0.125/0.375/0.625/0.875 -> deciles 1, 3, 6, 8
    assert out["deciles"] == {"1": 2, "3": 5, "6": 8, "8": 3}


def test_decile_table_averages_within_bucket():
    series = [4, 6] * 10  # 20 units: two per decile
    assert tt.decile_table(series) == {str(d): 5.0 for d in range(10)}


def test_peak_first_and_last_position():
    pk = tt.peak([1, 9, 3, 9])
    assert pk["height"] == 9
    assert pk["first_position"] == round(1.5 / 4, 3)
    assert pk["last_position"] == round(3.5 / 4, 3)


def test_tail_wind_down_classification():
    agg = _compute([5, 6, 7, 2])[0]["aggregate"]
    assert agg["tail_units"] == 1
    assert agg["tail_mean"] == 2
    assert agg["final_tension"] == 2
    assert agg["ending_mode"] == "wind_down"


def test_tail_climax_hold_classification():
    agg = _compute([4, 5, 8, 9])[0]["aggregate"]
    assert agg["ending_mode"] == "climax_hold"


def test_tail_moderate_classification():
    agg = _compute([4, 5, 8, 5])[0]["aggregate"]
    assert agg["ending_mode"] == "moderate"


def test_tail_uses_final_ten_percent():
    series = [5] * 18 + [1, 1]
    tb = tt.tail_behavior(series)
    assert tb["tail_units"] == 2
    assert tb["tail_mean"] == 1


def test_calm_and_high_shares():
    agg = _compute([2, 3, 8, 9])[0]["aggregate"]
    assert agg["calm_share"] == 0.5
    assert agg["high_share"] == 0.5


def test_all_holes_yield_none_aggregates():
    agg = tt.aggregate([None, None])
    assert agg["mean_register"] is None
    assert agg["volatility"] is None
    assert agg["ending_mode"] is None
    assert agg["n_scored"] == 0
