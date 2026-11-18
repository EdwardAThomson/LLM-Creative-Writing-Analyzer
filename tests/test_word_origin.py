"""Tests for the word_origin metric (French/Latin vs Old English share)."""
from __future__ import annotations

from conftest import load_metric

wo = load_metric("word_origin")


def test_table_groups():
    assert wo.origin("hand") == "G"            # hand label (the Wordnet lacks it)
    assert wo.origin("sword") == "G"
    assert wo.origin("manner") == "R"
    assert wo.origin("the") == "G"             # grammar words are in the table
    assert wo.origin("melancholy") == "Gk"


def test_endings_and_suffix_rule():
    assert wo.origin("swords") == "G"
    assert wo.origin("walked") == "G"
    assert wo.origin("reconfiguration") == "R"
    assert wo.origin("zzxq") is None


def test_plain_vs_latinate_runs():
    plain = "The old man took his sword in hand and went down to the dark water."
    ornate = "The gentleman expressed considerable satisfaction at the occasion of her departure."
    r = wo.compute([plain, ornate])
    a, b = r["per_run"]
    assert a["latinate_pct"] < 10 < b["latinate_pct"]
    assert b["latinate_content_pct"] > b["latinate_pct"]     # grammar words dilute it
    assert a["origin_known_pct"] > 90
    assert r["aggregate"]["latinate_pct"]["n"] == 2
    assert "satisfaction" in r["top_latinate_words"]


def test_names_left_out():
    # "Montmorency" is only ever capitalised: a name, not an unknown word.
    r = wo.compute(["Montmorency took the sword. Montmorency went home."])["per_run"][0]
    assert r["origin_known_pct"] == 100.0
    assert r["words"] == 5                       # took the sword went home


def test_empty_run():
    r = wo.compute([""])
    assert r["per_run"][0]["latinate_pct"] is None
    assert r["aggregate"]["latinate_pct"]["n"] == 0


def test_not_in_frozen_manifests():
    # Library module only, until v3/st2 are frozen: the frozen series must not
    # pick it up (Hard invariant 4).
    mf = load_metric("_manifests")
    for version in ("v2", "st1"):
        assert "word_origin" not in mf.resolve(version)["metrics"]
