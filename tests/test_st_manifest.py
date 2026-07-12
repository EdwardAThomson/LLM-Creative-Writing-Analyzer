"""Tests for the st1 single-text manifest and the frozen-series invariants."""
from __future__ import annotations

from conftest import load_metric

mf = load_metric("_manifests")

ST1_METRICS = [
    "text_structure", "mtld", "burstiness", "dialogue_ratio",
    "intra_text_repetition", "cliche_density", "ngram_diversity",
    "phonetic_names", "self_similarity", "opening_formula", "entity_census",
]

V1_LEGACY = ["text_similarity", "vocabulary_diversity", "structure",
             "semantic_similarity", "entity_analysis"]


def test_st1_resolves_to_the_selection():
    res = mf.resolve("st1")
    assert res["metrics"] == ST1_METRICS
    assert res["chain"] == ["st1"]       # extends: null, own series
    assert res["legacy"] == set()        # nothing run by the v1 pipeline


def test_st1_does_not_pull_v1_legacy_names():
    assert not set(mf.resolve("st1")["metrics"]) & set(V1_LEGACY)


def test_frozen_vn_series_unchanged_by_st_modules():
    # Adding library modules must not alter what the frozen manifests resolve to.
    v2 = mf.resolve("v2")
    assert v2["metrics"] == V1_LEGACY + [
        "phonetic_names", "mtld", "burstiness", "dialogue_ratio",
        "intra_text_repetition", "cliche_density", "ngram_diversity",
        "opening_lines",
    ]
    # None of the new st module names collide with a vN name (a collision would
    # silently promote a module into the frozen v2 library run).
    new_modules = {"text_structure", "self_similarity", "opening_formula",
                   "entity_census"}
    assert not new_modules & set(v2["metrics"])
