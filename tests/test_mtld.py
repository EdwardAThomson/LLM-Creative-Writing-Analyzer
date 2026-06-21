"""Tests for the MTLD length-robust lexical diversity metric."""
from __future__ import annotations

from conftest import load_metric

mt = load_metric("mtld")


def test_all_same_token_low_mtld():
    # 20 identical tokens: TTR drops to <=0.72 every 2 tokens -> 10 factors ->
    # mtld = 20 / 10 = 2.0 (minimal diversity).
    r = mt.compute(["word " * 20])["per_run"][0]
    assert r["tokens"] == 20
    assert r["types"] == 1
    assert r["mtld"] == 2.0


def test_all_unique_tokens_high_mtld():
    # 10 all-unique tokens never cross the threshold -> factors==0 -> mtld == len.
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    r = mt.compute([text])["per_run"][0]
    assert r["tokens"] == 10
    assert r["types"] == 10
    assert r["ttr"] == 1.0
    assert r["mtld"] == 10.0


def test_diverse_higher_than_repetitive():
    diverse = mt.compute(["one two three four five six seven eight nine ten"])["per_run"][0]
    repetitive = mt.compute(["echo " * 10])["per_run"][0]
    assert diverse["mtld"] > repetitive["mtld"]


def test_reliability_flag():
    short = mt.compute(["just five short little words"])["per_run"][0]
    assert short["reliable"] is False
    long_text = " ".join(f"w{i}" for i in range(60))
    assert mt.compute([long_text])["per_run"][0]["reliable"] is True


def test_empty_text_none():
    r = mt.compute([""])["per_run"][0]
    assert r["tokens"] == 0
    assert r["mtld"] is None
    assert r["ttr"] is None


def test_tokenization_lowercases_and_keeps_apostrophes():
    r = mt.compute(["Don't Don't don't"])["per_run"][0]
    # All three normalize to the same token "don't" -> 3 tokens, 1 type.
    assert r["tokens"] == 3
    assert r["types"] == 1


def test_structure_and_aggregate():
    texts = [
        " ".join(f"u{i}" for i in range(60)),       # diverse, reliable
        "same " * 60,                                # repetitive, reliable
    ]
    out = mt.compute(texts)
    assert out["schema"] == "mtld/1"
    assert out["runs"] == 2
    assert len(out["per_run"]) == 2
    scored = [r["mtld"] for r in out["per_run"] if r["mtld"] is not None]
    agg = out["aggregate"]
    assert agg["n"] == len(scored)
    assert agg["mean"] == round(sum(scored) / len(scored), 2)
    assert agg["min"] <= agg["mean"] <= agg["max"]
    # both runs are reliable here, so the reliable-only aggregate matches.
    assert out["aggregate_reliable_only"]["n"] == 2
    assert out["unreliable_runs"] == 0
