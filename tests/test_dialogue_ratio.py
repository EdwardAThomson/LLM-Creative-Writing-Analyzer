"""Tests for the dialogue_ratio metric (words inside paired double quotes)."""
from __future__ import annotations

from conftest import load_metric

dr = load_metric("dialogue_ratio")


def test_known_quoted_span_ratio():
    # 10 total words; the quoted span "Hello there friend" is 3 words -> 0.3.
    text = '"Hello there friend," she said to the old man today.'
    r = dr.compute([text])["per_run"][0]
    assert r["total_words"] == 10
    assert r["dialogue_words"] == 3
    assert r["dialogue_ratio"] == 0.3
    assert r["quoted_passages"] == 1


def test_no_quotes_zero_ratio():
    r = dr.compute(["The cat sat quietly on the warm windowsill."])["per_run"][0]
    assert r["dialogue_words"] == 0
    assert r["dialogue_ratio"] == 0.0
    assert r["quoted_passages"] == 0


def test_all_dialogue_ratio_one():
    # Entire text is inside one quoted span -> ratio 1.0.
    r = dr.compute(['"every single word here is spoken"'])["per_run"][0]
    assert r["dialogue_ratio"] == 1.0
    assert r["dialogue_words"] == r["total_words"] == 6


def test_curly_quotes_counted():
    text = "“Yes indeed” she replied."
    r = dr.compute([text])["per_run"][0]
    # "Yes indeed" = 2 words inside curly quotes; total = 4 words.
    assert r["dialogue_words"] == 2
    assert r["total_words"] == 4
    assert r["quoted_passages"] == 1


def test_single_quotes_ignored():
    # Apostrophes / single quotes must NOT be treated as dialogue delimiters.
    r = dr.compute(["don't count Kepler's apostrophes as 'dialogue' please"])["per_run"][0]
    assert r["dialogue_words"] == 0


def test_multiple_passages_counted():
    text = '"first" then narration then "second passage here"'
    r = dr.compute([text])["per_run"][0]
    assert r["quoted_passages"] == 2
    assert r["dialogue_words"] == 1 + 3


def test_empty_text_none_ratio():
    r = dr.compute([""])["per_run"][0]
    assert r["total_words"] == 0
    assert r["dialogue_ratio"] is None


def test_ratio_bounds_and_structure():
    texts = [
        '"a b c" narration here',
        "no quotes at all in this one",
        '"all spoken words here now"',
    ]
    out = dr.compute(texts)
    assert out["schema"] == "dialogue_ratio/1"
    assert out["runs"] == 3
    assert len(out["per_run"]) == 3
    for r in out["per_run"]:
        if r["dialogue_ratio"] is not None:
            assert 0.0 <= r["dialogue_ratio"] <= 1.0
    agg = out["aggregate"]
    ratios = [r["dialogue_ratio"] for r in out["per_run"] if r["dialogue_ratio"] is not None]
    assert agg["n"] == len(ratios)
    assert agg["min"] <= agg["mean"] <= agg["max"]
    assert agg["min"] == min(ratios)
    assert agg["max"] == max(ratios)


def test_aggregate_empty_when_no_words():
    agg = dr.compute(["", ""])["aggregate"]
    assert agg["n"] == 0
    assert agg["mean"] is None
