"""Tests for the corpus-level ngram_diversity metric (distinct-n + Self-BLEU)."""
from __future__ import annotations

from conftest import load_metric

ng = load_metric("ngram_diversity")

_SENT = "the quick brown fox jumps over the lazy dog"  # 9 tokens, 8 distinct types


def test_single_run_distinct_1():
    # distinct unigrams = 8 ("the" repeats), total = 9 -> ratio 8/9.
    out = ng.compute([_SENT])
    d1 = out["distinct"]["distinct_1"]
    assert d1["distinct"] == 8
    assert d1["total"] == 9
    assert d1["ratio"] == round(8 / 9, 4)


def test_identical_runs_maximal_self_bleu():
    # Two identical runs are maximally similar -> Self-BLEU == 1.0 (least diverse).
    out = ng.compute([_SENT, _SENT])
    sb = out["self_bleu"]
    assert sb["mean"] == 1.0
    assert sb["per_run"] == [1.0, 1.0]
    # Pooled distinct-1: 8 distinct over 18 total.
    assert out["distinct"]["distinct_1"]["distinct"] == 8
    assert out["distinct"]["distinct_1"]["total"] == 18


def test_disjoint_runs_low_self_bleu():
    out = ng.compute(["alpha beta gamma delta", "one two three four"])
    # No shared n-grams -> Self-BLEU is at the smoothing floor, far below identical.
    assert out["self_bleu"]["mean"] < 0.5


def test_identical_more_similar_than_disjoint():
    identical = ng.compute([_SENT, _SENT])["self_bleu"]["mean"]
    disjoint = ng.compute(["alpha beta gamma delta", "one two three four"])["self_bleu"]["mean"]
    assert identical > disjoint


def test_single_run_self_bleu_undefined():
    sb = ng.compute([_SENT])["self_bleu"]
    assert sb["mean"] is None
    assert sb["per_run"] == []


def test_distinct_ratio_bounds():
    out = ng.compute([_SENT, "completely different words appear in this second run"])
    for n in (1, 2, 3):
        d = out["distinct"][f"distinct_{n}"]
        assert d["distinct"] <= d["total"]
        assert 0.0 <= d["ratio"] <= 1.0


def test_length_spread_zero_for_equal_lengths():
    out = ng.compute([_SENT, _SENT])
    assert out["tokens"]["length_spread"] == 0.0
    assert out["tokens"]["min"] == out["tokens"]["max"] == 9


def test_degenerate_runs_do_not_crash():
    # An empty run and a 1-token run (no higher-order n-grams) must be handled
    # gracefully alongside a normal run rather than raising.
    out = ng.compute(["", "solo", _SENT])
    sb = out["self_bleu"]
    assert len(sb["per_run"]) == 3
    # Empty candidate yields BLEU 0.0; all scores stay within [0, 1].
    assert sb["per_run"][0] == 0.0
    for b in sb["per_run"]:
        assert 0.0 <= b <= 1.0


def test_structure():
    out = ng.compute([_SENT, _SENT, "another run with several distinct tokens here"])
    assert out["schema"] == "ngram_diversity/1"
    assert out["runs"] == 3
    sb = out["self_bleu"]
    assert len(sb["per_run"]) == 3
    assert sb["min"] <= sb["mean"] <= sb["max"]
    for b in sb["per_run"]:
        assert 0.0 <= b <= 1.0
