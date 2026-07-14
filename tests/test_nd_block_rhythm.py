"""Tests for the block_rhythm metric (fake judge; pure mode-dynamics aggregations)."""
from __future__ import annotations

import json
import re

from benchmarks.narrative_dynamics import block_rhythm as br
from benchmarks.narrative_dynamics.judge import FakeJudge


def _para(words, tag="w"):
    return " ".join(f"{tag}{k}" for k in range(words))


def _unit(paragraphs, index=0):
    text = "\n\n".join(paragraphs)
    return {"index": index, "label": f"u{index}", "text": text,
            "words": sum(len(p.split()) for p in paragraphs)}


def _labels_resp(labels):
    """labels: list of (primary, secondary) -> a valid judge response."""
    return json.dumps([
        {"n": i + 1, "primary": p, "secondary": s}
        for i, (p, s) in enumerate(labels)
    ])


FOUR_PARA_LABELS = [("DIALOGUE", None), ("DIALOGUE", None),
                    ("INTERIORITY", "SETTING"), ("ACTION", None)]


def _four_para_out():
    unit = _unit([_para(10), _para(5), _para(8), _para(6)])
    ctx = {"judge": FakeJudge([_labels_resp(FOUR_PARA_LABELS)])}
    return br.compute([unit], ctx)


# --- annotation path ------------------------------------------------------------------

def test_per_unit_labels_recorded():
    out = _four_para_out()
    pu = out["per_unit"][0]
    assert pu["n_paragraphs"] == 4
    assert pu["n_unlabeled"] == 0
    assert pu["labels"] == [["DIALOGUE", None], ["DIALOGUE", None],
                            ["INTERIORITY", "SETTING"], ["ACTION", None]]
    assert out["schema"] == "block_rhythm/1"
    assert out["rubric"]["version"] == "block_types/1"


def test_batching_splits_at_20():
    paras = [_para(4, f"p{i}") for i in range(25)]

    def fake(prompt):
        n = len(re.findall(r"^\[\d+\] ", prompt, re.M))
        return _labels_resp([("ACTION", None)] * n)

    ctx = {"judge": FakeJudge(fake)}
    out = br.compute([_unit(paras)], ctx)
    assert ctx["judge"].calls == 2  # 20 + 5
    assert out["aggregate"]["n_paragraphs"] == 25
    assert out["aggregate"]["n_unlabeled"] == 0


def test_failed_batch_becomes_hole_not_crash():
    ctx = {"judge": FakeJudge(["garbage", "still garbage"])}
    out = br.compute([_unit([_para(5), _para(5)])], ctx)
    assert out["aggregate"]["n_unlabeled"] == 2
    assert out["per_unit"][0]["labels"] == [[None, None], [None, None]]
    assert out["aggregate"]["words_per_mode_segment"] is None


def test_bad_label_rejected_then_retried():
    bad = json.dumps([{"n": 1, "primary": "MUSING", "secondary": None}])
    good = _labels_resp([("INTERIORITY", None)])
    ctx = {"judge": FakeJudge([bad, good])}
    out = br.compute([_unit([_para(5)])], ctx)
    assert out["per_unit"][0]["labels"] == [["INTERIORITY", None]]
    assert ctx["judge"].calls == 2


def test_wrong_count_recovers_partial_no_retry():
    # 1 item for a 2-paragraph batch: the lenient parser now recovers the one
    # valid label instead of discarding the whole batch, and does NOT re-ask
    # (a partial recovery is accepted as success, per the return-vs-raise
    # contract in judge.ask_json) -- this is the behavior change under test.
    short = _labels_resp([("ACTION", None)])
    good = _labels_resp([("ACTION", None), ("ACTION", None)])
    ctx = {"judge": FakeJudge([short, good])}
    out = br.compute([_unit([_para(5), _para(5)])], ctx)
    assert out["aggregate"]["n_unlabeled"] == 1
    assert out["per_unit"][0]["labels"] == [["ACTION", None], [None, None]]
    assert ctx["judge"].calls == 1  # accepted first try; "good" left unconsumed


# --- _parse_batch leniency (unit-level) ------------------------------------------------

def test_parse_batch_fully_valid_matches_strict_output():
    """Regression guard: a clean, complete, correct-length array must produce
    exactly the same labels as before this change -- a strict superset."""
    raw = _labels_resp(FOUR_PARA_LABELS)
    out = br._parse_batch(raw, 4)
    assert out == [
        {"primary": "DIALOGUE", "secondary": None},
        {"primary": "DIALOGUE", "secondary": None},
        {"primary": "INTERIORITY", "secondary": "SETTING"},
        {"primary": "ACTION", "secondary": None},
    ]


def test_parse_batch_wrong_length_recovers_present_items():
    # 17 well-formed objects for a 20-paragraph batch.
    raw = json.dumps([
        {"n": i + 1, "primary": "ACTION", "secondary": None} for i in range(17)
    ])
    out = br._parse_batch(raw, 20)
    assert len(out) == 20
    assert out[:17] == [{"primary": "ACTION", "secondary": None}] * 17
    assert out[17:] == [None, None, None]


def test_parse_batch_mixed_malformed_and_valid():
    raw = json.dumps([
        {"n": 1, "primary": "ACTION", "secondary": None},
        {"n": 2, "primary": "ACTION"},  # missing secondary key entirely: fine, treated as null
        "not an object",
        {"n": 4, "primary": "DIALOGUE", "secondary": None},
        {"n": 5},  # missing primary
    ])
    out = br._parse_batch(raw, 5)
    assert out[0] == {"primary": "ACTION", "secondary": None}
    assert out[1] == {"primary": "ACTION", "secondary": None}
    assert out[2] is None
    assert out[3] == {"primary": "DIALOGUE", "secondary": None}
    assert out[4] is None


def test_parse_batch_invalid_primary_label_becomes_hole():
    raw = json.dumps([
        {"n": 1, "primary": "MUSING", "secondary": None},  # not in LABELS
        {"n": 2, "primary": "ACTION", "secondary": None},
    ])
    out = br._parse_batch(raw, 2)
    assert out == [None, {"primary": "ACTION", "secondary": None}]


def test_parse_batch_invalid_secondary_label_becomes_hole():
    raw = json.dumps([
        {"n": 1, "primary": "ACTION", "secondary": "NONSENSE"},
        {"n": 2, "primary": "ACTION", "secondary": None},
    ])
    out = br._parse_batch(raw, 2)
    assert out == [None, {"primary": "ACTION", "secondary": None}]


def test_parse_batch_json_fenced_valid_array():
    inner = _labels_resp([("ACTION", None), ("DIALOGUE", None)])
    raw = f"Here are the labels:\n```json\n{inner}\n```\nThanks."
    out = br._parse_batch(raw, 2)
    assert out == [{"primary": "ACTION", "secondary": None},
                   {"primary": "DIALOGUE", "secondary": None}]


def test_parse_batch_tolerates_trailing_commas():
    raw = ('[{"n": 1, "primary": "ACTION", "secondary": null,}, '
           '{"n": 2, "primary": "DIALOGUE", "secondary": null},]')
    out = br._parse_batch(raw, 2)
    assert out == [{"primary": "ACTION", "secondary": None},
                   {"primary": "DIALOGUE", "secondary": None}]


def test_parse_batch_total_garbage_raises():
    import pytest
    with pytest.raises(ValueError):
        br._parse_batch("I'm sorry, I can't help with that.", 3)


def test_parse_batch_no_recoverable_objects_raises():
    import pytest
    # a syntactically valid array, but every element is unusable
    raw = json.dumps([{"n": 1, "primary": "NOT_A_LABEL"}, "garbage", {}])
    with pytest.raises(ValueError):
        br._parse_batch(raw, 3)


def test_parse_batch_out_of_range_n_falls_back_positionally():
    raw = json.dumps([
        {"n": 99, "primary": "ACTION", "secondary": None},
        {"n": 2, "primary": "DIALOGUE", "secondary": None},
    ])
    out = br._parse_batch(raw, 2)
    # item 0's n is out of range -> falls back to its array position (0)
    assert out == [{"primary": "ACTION", "secondary": None},
                   {"primary": "DIALOGUE", "secondary": None}]


# --- return-vs-raise contract with ask_json (integration-ish) -------------------------

def test_ask_json_accepts_partial_recovery_as_success_no_retry():
    """A partial (some-None) return from the parse callback must be treated by
    ask_json as a SUCCESS: cached, no retry -- confirming the design contract
    (return = accepted outcome, raise = retry) holds end to end for the
    lenient batch parser, not just in isolation."""
    from benchmarks.narrative_dynamics.judge import ask_json

    short = _labels_resp([("ACTION", None)])  # 1 of 2 expected
    calls = {"n": 0}

    def counting_judge(prompt):
        calls["n"] += 1
        return short

    result = ask_json(counting_judge, "prompt", lambda raw: br._parse_batch(raw, 2))
    assert result == [{"primary": "ACTION", "secondary": None}, None]
    assert calls["n"] == 1  # no retry triggered by the partial result


# --- aggregations ---------------------------------------------------------------------

def test_distribution_and_shading():
    agg = _four_para_out()["aggregate"]
    assert agg["distribution"]["DIALOGUE"] == 0.5
    assert agg["distribution"]["INTERIORITY"] == 0.25
    assert agg["distribution"]["ACTION"] == 0.25
    assert agg["distribution"]["SETTING"] == 0.0
    assert agg["secondary_shading_rate"] == 0.25
    assert agg["setting_touch_rate"] == 0.25  # via the secondary label


def test_words_per_mode_segment():
    agg = _four_para_out()["aggregate"]
    # segments: DIALOGUE 15w, INTERIORITY 8w, ACTION 6w
    assert agg["n_segments"] == 3
    assert agg["words_per_mode_segment"] == round(29 / 3, 1)
    assert agg["max_segment_words"] == 15


def test_switch_rate():
    agg = _four_para_out()["aggregate"]
    assert agg["switch_rate"] == round(2 / 3, 3)  # D->D, D->I, I->A


def test_interiority_exit_dynamics():
    agg = _four_para_out()["aggregate"]
    assert agg["interiority_share"] == 0.25
    assert agg["interiority_self_transition"] == 0.0
    assert agg["interiority_exit_to_action"] == 1.0
    assert agg["interiority_exit_to_dialogue"] == 0.0


def test_interiority_self_loop_counted():
    labels = [("INTERIORITY", None), ("INTERIORITY", None), ("ACTION", None)]
    ctx = {"judge": FakeJudge([_labels_resp(labels)])}
    out = br.compute([_unit([_para(5), _para(5), _para(5)])], ctx)
    agg = out["aggregate"]
    assert agg["interiority_self_transition"] == 0.5
    assert agg["interiority_exit_to_action"] == 0.5


def test_transition_matrix():
    labels = [("DIALOGUE", None), ("DIALOGUE", None), ("ACTION", None)]
    ctx = {"judge": FakeJudge([_labels_resp(labels)])}
    out = br.compute([_unit([_para(5), _para(5), _para(5)])], ctx)
    row = out["transition_matrix"]["DIALOGUE"]
    assert row["_n"] == 2
    assert row["DIALOGUE"] == 0.5
    assert row["ACTION"] == 0.5


def test_runs_do_not_cross_unit_boundaries():
    labels = [("ACTION", None), ("ACTION", None)]
    ctx = {"judge": FakeJudge([_labels_resp(labels[:1]), _labels_resp(labels[:1])])}
    out = br.compute([_unit([_para(5)], 0), _unit([_para(5)], 1)], ctx)
    assert out["aggregate"]["n_segments"] == 2  # one per unit, not one merged run
    assert out["aggregate"]["switch_rate"] is None  # no within-unit transitions


def test_hole_splits_run_sequences():
    seqs = br._annotated([[(5, "ACTION", None), (5, None, None), (5, "ACTION", None)]])
    assert len(seqs) == 2
    assert br.words_per_segment(seqs)["n_segments"] == 2


def test_structural_gauge_groups_the_four_signals():
    gauge = _four_para_out()["structural_gauge"]
    assert set(gauge) == {"words_per_mode_segment", "interiority_self_transition",
                          "secondary_shading_rate", "setting_touch_rate",
                          "master_bands"}
    assert "interiority_self_transition" in gauge["master_bands"]
