"""Tests for the text_structure metric (regex structural profile, stdlib)."""
from __future__ import annotations

from conftest import load_metric

ts = load_metric("text_structure")


def test_basic_profile_counts():
    text = "One two three. Four five!\n\nSix seven eight nine?"
    r = ts.compute([text])["per_run"][0]
    assert r["words"] == 9
    assert r["sentences"] == 3
    assert r["paragraphs"] == 2
    assert r["words_per_sentence"] == 3.0
    assert r["sentences_per_paragraph"] == 1.5
    assert r["words_per_paragraph"] == 4.5


def test_paragraphs_per_1k_words():
    text = ("w " * 100).strip() + "\n\n" + ("v " * 100).strip()
    r = ts.compute([text])["per_run"][0]
    assert r["paragraphs_per_1k_words"] == 10.0


def test_closing_quotes_honored_in_sentence_split():
    r = ts.compute(['He said, "Stop!" Then he left.'])["per_run"][0]
    assert r["sentences"] == 2


def test_sentence_floor_of_one_when_words_present():
    r = ts.compute(["no terminal punctuation here"])["per_run"][0]
    assert r["sentences"] == 1


def test_empty_text():
    r = ts.compute([""])["per_run"][0]
    assert r["words"] == 0
    assert r["sentences"] == 0
    assert r["paragraphs"] == 0
    assert r["words_per_sentence"] is None


def test_aggregate_across_units():
    out = ts.compute(["Alpha beta. Gamma delta.", "Epsilon zeta."])
    assert out["schema"] == "text_structure/1"
    assert out["runs"] == 2
    agg = out["aggregate"]
    assert agg["words"]["n"] == 2
    assert agg["words"]["mean"] == 3.0
    assert agg["sentences"]["min"] == 1
    assert agg["sentences"]["max"] == 2
