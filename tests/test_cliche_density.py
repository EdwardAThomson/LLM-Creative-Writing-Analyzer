"""Tests for the cliche_density metric (stock-phrase / slop-word / em-dash hits)."""
from __future__ import annotations

from conftest import load_metric

cd = load_metric("cliche_density")


def test_clean_text_zero_hits():
    text = "The cat sat quietly on the warm windowsill watching birds."
    r = cd.compute([text])["per_run"][0]
    assert r["words"] == 10
    assert r["cliche"]["hits"] == 0
    assert r["cliche"]["per_1k"] == 0.0
    assert r["slop_words"]["hits"] == 0
    assert r["em_dash"]["count"] == 0


def test_known_cliche_phrases_detected():
    text = "Once upon a time, in a world where heroes rose, a single tear fell."
    r = cd.compute([text])["per_run"][0]
    assert r["cliche"]["hits"] == 3
    assert set(r["cliche"]["phrase_hits"]) == {
        "in_a_world_where",
        "once_upon_a_time",
        "single_tear",
    }
    assert r["cliche"]["distinct"] == 3
    # by_category buckets the hits: openers (A) get 2, emotion (C) gets 1.
    assert r["cliche"]["by_category"]["A_opener"] == 2
    assert r["cliche"]["by_category"]["C_emotion"] == 1


def test_slop_words_separate_from_cliche():
    text = "She loved to delve into the rich tapestry of the ethereal realm."
    r = cd.compute([text])["per_run"][0]
    assert r["slop_words"]["hits"] == 4
    assert set(r["slop_words"]["word_hits"]) == {"delve", "tapestry", "ethereal", "realm"}
    # Slop words must NOT leak into the cliche headline count.
    assert r["cliche"]["hits"] == 0


def test_em_dash_both_forms_counted():
    r = cd.compute(["Wait— stop -- now."])["per_run"][0]
    assert r["em_dash"]["count"] == 2


def test_per_1k_normalization():
    # 5 cliche-free filler words + 1 cliche occurrence is awkward; instead use a
    # text of known length with one cliche, and check the per-1000-word scaling.
    # "a new chapter" is one E_closer cliche (3 words); pad to exactly 10 words.
    text = "Today marked a new chapter for everyone in town"  # 9 words... adjust
    r = cd.compute([text])["per_run"][0]
    hits = r["cliche"]["hits"]
    words = r["words"]
    assert hits == 1
    assert r["cliche"]["per_1k"] == round(hits / words * 1000, 2)


def test_empty_text_none_per_1k():
    r = cd.compute([""])["per_run"][0]
    assert r["words"] == 0
    assert r["cliche"]["per_1k"] is None
    assert r["slop_words"]["per_1k"] is None
    assert r["em_dash"]["per_1k"] is None


def test_structure_and_aggregate():
    texts = [
        "Once upon a time the story began here in earnest.",
        "A perfectly ordinary sentence with no tropes whatsoever today.",
    ]
    out = cd.compute(texts)
    assert out["schema"] == "cliche_density/1"
    assert out["lexicon_version"] == cd.LEXICON_VERSION
    assert out["runs"] == 2
    assert len(out["per_run"]) == 2
    agg = out["aggregate"]
    # aggregate is the mean of per-run per_1k values (ignoring Nones).
    cliche_vals = [r["cliche"]["per_1k"] for r in out["per_run"] if r["cliche"]["per_1k"] is not None]
    assert agg["cliche_per_1k"] == round(sum(cliche_vals) / len(cliche_vals), 2)
    assert min(cliche_vals) <= agg["cliche_per_1k"] <= max(cliche_vals)


def test_case_insensitive_matching():
    r = cd.compute(["ONCE UPON A TIME there was a dragon."])["per_run"][0]
    assert "once_upon_a_time" in r["cliche"]["phrase_hits"]
