"""Tests for the intra_text_repetition metric (within-story word/phrase overuse)."""
from __future__ import annotations

from conftest import load_metric

it = load_metric("intra_text_repetition")


def test_repeated_content_word_high_unigram_rate():
    # 10 identical content words: distinct 1, total 10 -> rate (10-1)/10 = 0.9.
    r = it.compute(["river " * 10])["per_run"][0]
    assert r["content_tokens"] == 10
    assert r["unigram"]["rep_rate"] == 0.9
    assert r["unigram"]["top"] == [("river", 10)]


def test_all_unique_content_words_zero_unigram_rate():
    text = "river mountain forest desert ocean valley canyon glacier meadow tundra"
    r = it.compute([text])["per_run"][0]
    assert r["unigram"]["rep_rate"] == 0.0
    assert r["bigram"]["rep_rate"] == 0.0
    assert r["unigram"]["top"] == []


def test_bigram_repetition_detected():
    # "she said" appears twice; 10 tokens -> 9 bigrams, 8 distinct -> 1/9 = 0.111.
    text = "she said hello and she said goodbye to everyone there"
    r = it.compute([text])["per_run"][0]
    assert r["bigram"]["rep_rate"] == 0.111
    assert ("she said", 2) in r["bigram"]["top"]


def test_stopwords_excluded_from_unigram():
    # Stopwords ("the") and short tokens are filtered from the content-word stream,
    # so their repetition does not inflate the unigram rate.
    text = "the the the the the the the the the the"
    r = it.compute([text])["per_run"][0]
    assert r["content_tokens"] == 0
    assert r["unigram"]["rep_rate"] is None


def test_short_text_below_n_returns_none():
    r = it.compute(["solo"])["per_run"][0]
    # One token: no bigram/trigram possible.
    assert r["bigram"]["rep_rate"] is None
    assert r["trigram"]["rep_rate"] is None


def test_reliability_flag():
    short = it.compute(["a few words only here"])["per_run"][0]
    assert short["reliable"] is False
    long_text = " ".join(f"token{i}" for i in range(40))
    assert it.compute([long_text])["per_run"][0]["reliable"] is True


def test_rates_in_unit_interval_and_structure():
    texts = [
        "echo echo echo echo echo echo echo echo echo echo echo echo",
        "each word here is entirely distinct from every neighbour around",
    ]
    out = it.compute(texts)
    assert out["schema"] == "intra_text_repetition/1"
    assert out["runs"] == 2
    assert len(out["per_run"]) == 2
    for r in out["per_run"]:
        for gram in ("unigram", "bigram", "trigram"):
            rate = r[gram]["rep_rate"]
            if rate is not None:
                assert 0.0 <= rate < 1.0


def test_aggregate_is_mean_of_per_run():
    texts = ["alpha alpha alpha alpha alpha alpha", "beta gamma delta epsilon zeta eta"]
    out = it.compute(texts)
    uni_vals = [r["unigram"]["rep_rate"] for r in out["per_run"] if r["unigram"]["rep_rate"] is not None]
    agg = out["aggregate"]["unigram"]
    assert agg["n"] == len(uni_vals)
    assert agg["mean"] == round(sum(uni_vals) / len(uni_vals), 3)
    assert agg["min"] <= agg["mean"] <= agg["max"]
