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


def test_wrong_count_rejected():
    short = _labels_resp([("ACTION", None)])  # 1 item for a 2-paragraph batch
    good = _labels_resp([("ACTION", None), ("ACTION", None)])
    ctx = {"judge": FakeJudge([short, good])}
    out = br.compute([_unit([_para(5), _para(5)])], ctx)
    assert out["aggregate"]["n_unlabeled"] == 0


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
