"""Tests for opening_formula: within-text chapter-opening similarity census."""
from __future__ import annotations

from conftest import load_metric

of = load_metric("opening_formula")


def test_first_sentence_extraction():
    assert of.first_sentence("She ran. More text follows.") == "She ran."
    # documented approximation: terminal punctuation inside a closing quote
    # ends the sentence, so quoted exclamations cut short
    assert of.first_sentence('  "Stop!" he cried. Then quiet.') == '"Stop!"'


def test_first_sentence_fallback_without_terminator():
    text = "x" * 400
    assert of.first_sentence(text) == "x" * of.MAX_OPENING_CHARS


def test_formulaic_openings_detected():
    units = [
        "The morning came cold and grey over the harbor. Ships moved slowly.",
        "The morning came cold and grey over the hills. Nothing stirred at all.",
        "Rain hammered the windows without pause. She waited by the door.",
    ]
    out = of.compute(units)
    agg = out["aggregate"]
    assert agg["n_units"] == 3
    assert agg["n_pairs"] == 3
    assert agg["max_pair"] == [0, 1]
    assert agg["max"] >= of.HIGH_SIMILARITY
    assert [p["pair"] for p in out["per_pair_high"]] == [[0, 1]]
    assert out["repeated_openers"] == [{"opener": "the morning came", "count": 2}]


def test_varied_openings_clean():
    units = [
        "Dawn broke over the ridge at last. Birds began.",
        "Nobody spoke at breakfast that day. The tea cooled.",
        "A letter arrived with the noon post. It was thin.",
    ]
    out = of.compute(units)
    assert out["per_pair_high"] == []
    assert out["repeated_openers"] == []
    assert out["aggregate"]["high_pair_rate"] == 0.0


def test_identical_openings_similarity_one():
    units = ["Same opening line here. Different middle one.",
             "Same opening line here. Different middle two."]
    out = of.compute(units)
    assert out["aggregate"]["max"] == 1.0
    assert out["aggregate"]["high_pair_rate"] == 1.0


def test_single_unit_no_pairs():
    out = of.compute(["Only one chapter. Nothing to compare."])
    assert out["aggregate"]["n_pairs"] == 0
    assert out["aggregate"]["mean"] is None


def test_openings_recorded_and_schema():
    out = of.compute(["Alpha beta gamma. Rest.", "Delta epsilon zeta. Rest."])
    assert out["schema"] == "opening_formula/1"
    assert out["openings"] == ["Alpha beta gamma.", "Delta epsilon zeta."]
