"""Tests for the thread_architecture metric (fake judge; deterministic clustering)."""
from __future__ import annotations

import json

from benchmarks.narrative_dynamics import thread_architecture as ta
from benchmarks.narrative_dynamics.judge import FakeJudge


def _units(n):
    return [{"index": i, "label": f"Ch {i+1}", "text": f"chapter {i} text", "words": 3}
            for i in range(n)]


def _cast_resp(pov, cast, strand="S doing X toward Y"):
    return json.dumps({"pov": pov, "principal_cast": cast, "strand": strand})


def _two_strand_responses():
    a = [_cast_resp("Jonathan Harker", ["Jonathan Harker", "Count Dracula"],
                    "Jonathan is trapped in the castle")] * 3
    b = [_cast_resp("Mina Murray", ["Mina Murray", "Lucy Westenra"],
                    "Mina worries about Lucy")] * 3
    conv = [_cast_resp("Mina Murray",
                       ["Jonathan Harker", "Mina Murray", "Lucy Westenra"],
                       "the strands reunite")]
    return a + b + conv


def _compute(responses, n=None, ctx_extra=None):
    ctx = {"judge": FakeJudge(responses)}
    ctx.update(ctx_extra or {})
    return ta.compute(_units(n if n is not None else len(responses)), ctx), ctx


# --- extraction + clustering ------------------------------------------------------------

def test_two_strands_detected():
    out, _ = _compute(_two_strand_responses())
    agg = out["aggregate"]
    assert agg["n_threads"] == 2
    assert agg["n_threads_2plus"] == 2
    assert [u["thread"] for u in out["per_unit"]] == [0, 0, 0, 1, 1, 1, 1]
    assert out["schema"] == "thread_architecture/1"


def test_run_and_switch_structure():
    out, _ = _compute(_two_strand_responses())
    assert out["runs"] == [3, 4]
    assert out["aggregate"]["switch_rate"] == round(1 / 6, 3)
    assert out["aggregate"]["mean_run"] == 3.5
    assert out["aggregate"]["max_run"] == 4


def test_convergence_event_reported():
    out, _ = _compute(_two_strand_responses())
    assert out["aggregate"]["n_convergence_events"] == 1
    m = out["merges"][0]
    assert m["unit"] == 6
    assert m["threads"] == [0, 1]
    assert m["position"] == round(6.5 / 7, 2)
    assert out["aggregate"]["first_convergence"] == m["position"]
    assert out["never_merged_threads"] == []


def test_per_unit_records_canonical_cast_and_pov():
    out, _ = _compute(_two_strand_responses())
    u0 = out["per_unit"][0]
    assert u0["pov"] == "jonathan harker"
    # 'Count' is an honorific and gets stripped by the normalization rules
    assert u0["cast"] == ["dracula", "jonathan harker"]
    assert u0["strand"] == "Jonathan is trapped in the castle"


def test_aliases_from_ctx_are_applied():
    out, _ = _compute(_two_strand_responses(),
                      ctx_extra={"aliases": {"mina murray": "mina harker"}})
    assert out["per_unit"][3]["pov"] == "mina harker"


def test_theta_sensitivity_present():
    out, _ = _compute(_two_strand_responses())
    assert set(out["theta_sensitivity"]) == {"0.2", "0.3", "0.4"}


def test_extraction_failure_flagged_and_isolated():
    responses = ["garbage", "more garbage"] + _two_strand_responses()[1:]
    out, _ = _compute(responses, n=7)
    assert out["aggregate"]["n_extraction_failures"] == 1
    assert "error" in out["per_unit"][0]
    assert out["per_unit"][0]["cast"] == []


def test_long_unit_truncated_for_extraction():
    units = [{"index": 0, "label": "Ch 1",
              "text": " ".join(f"w{k}" for k in range(9000)), "words": 9000}]
    ctx = {"judge": FakeJudge([_cast_resp("A", ["A"])])}
    ta.compute(units, ctx)
    assert "[... middle omitted:" in ctx["judge"].prompts[0]


# --- tension coupling -------------------------------------------------------------------

def test_thread_tension_registers_when_tensions_present():
    tensions = [8, 7, 9, 3, 4, 3, 5]
    out, _ = _compute(_two_strand_responses(), ctx_extra={"unit_tensions": tensions})
    t0, t1 = out["threads"]
    assert t0["mean_tension"] == 8.0
    assert t0["tension_range"] == [7, 9]
    assert t1["mean_tension"] == 3.75
    assert out["switch_tension"] is not None


def test_switch_deltas_cut_away_analysis():
    tensions = [8, 7, 9, 3, 4, 3, 5]
    out, _ = _compute(_two_strand_responses(), ctx_extra={"unit_tensions": tensions})
    sw = out["switch_tension"]["switch"]
    assert sw["n"] == 1                       # one thread switch (unit 2 -> 3)
    assert sw["cooler"] == 1 and sw["hotter"] == 0
    assert sw["mean"] == -6.0
    assert out["switch_tension"]["switch_deltas"] == [-6]
    same = out["switch_tension"]["same_thread"]
    assert same["n"] == 5
    assert out["aggregate"]["switch_delta_mean"] == -6.0


def test_no_tensions_means_null_coupling():
    out, _ = _compute(_two_strand_responses())
    assert out["switch_tension"] is None
    assert out["threads"][0]["mean_tension"] is None
    assert "switch_delta_mean" not in out["aggregate"]


def test_switch_deltas_pure_function_skips_holes():
    d = ta.switch_deltas([0, 0, 1, 1], [5, None, 2, 4])
    assert d["switch"]["n"] == 0      # the switch pair had a hole
    assert d["same_thread"]["n"] == 1  # only the 2->4 pair
