"""Tests for self_similarity: the within-text adjacent-unit duplication detector.

The load-bearing case is the planted duplicate: a long verbatim block shared by
two adjacent units must produce a flagged similarity spike over a near-zero
median (the StoryDaemon defect class: 0.668 adjacent similarity vs ~0.02
baseline, ~9,200 verbatim chars).
"""
from __future__ import annotations

from conftest import load_metric

ss = load_metric("self_similarity")


def _chapter(tag, n=150):
    # fully distinct token stems per chapter -> healthy baseline similarity ~0
    return " ".join(f"{tag}tok{i}" for i in range(n))


def _book_with_planted_duplicate():
    chapters = [_chapter(t) for t in ("aa", "bb", "cc", "dd", "ee")]
    block = " ".join(f"dupword{i}" for i in range(110))  # ~700 verbatim chars
    chapters[2] += " " + block
    chapters[3] = block + " " + chapters[3]
    return chapters, block


def test_healthy_book_no_flags():
    out = ss.compute([_chapter(t) for t in ("aa", "bb", "cc", "dd")])
    assert out["aggregate"]["n_pairs"] == 3
    assert out["aggregate"]["max"] < 0.05
    assert out["flagged"] == []
    assert out["duplication_suspected"] is False


def test_planted_duplicate_is_flagged():
    chapters, block = _book_with_planted_duplicate()
    out = ss.compute(chapters)
    assert out["duplication_suspected"] is True
    assert [p["pair"] for p in out["flagged"]] == [[2, 3]]
    hit = out["flagged"][0]
    assert hit["similarity"] >= ss.FLAG_SIMILARITY
    assert hit["longest_match_words"] >= 110
    assert hit["longest_match_chars"] >= len(block) - 10
    # the spike stands over a near-zero baseline
    assert out["aggregate"]["median"] < 0.05
    assert out["aggregate"]["max_pair"] == [2, 3]
    assert out["aggregate"]["max_verbatim_chars"] == hit["longest_match_chars"]


def test_verbatim_block_flags_even_below_similarity_threshold():
    # A 500+ char verbatim run inside two much larger units: ratio stays small
    # but the verbatim gauge must still flag it.
    block = " ".join(f"dupword{i}" for i in range(90))  # ~630 chars
    a = _chapter("aa", 800) + " " + block
    b = block + " " + _chapter("bb", 800)
    out = ss.compute([a, b])
    p = out["per_pair"][0]
    assert p["similarity"] < ss.FLAG_SIMILARITY
    assert p["longest_match_chars"] >= ss.FLAG_MATCH_CHARS
    assert out["duplication_suspected"] is True


def test_identical_adjacent_units():
    ch = _chapter("aa")
    out = ss.compute([ch, ch])
    p = out["per_pair"][0]
    assert p["similarity"] == 1.0
    assert p["longest_match_words"] == 150
    assert out["flagged"] == [p]


def test_series_shape_and_pairs():
    out = ss.compute([_chapter(t) for t in ("aa", "bb", "cc")])
    assert [p["pair"] for p in out["per_pair"]] == [[0, 1], [1, 2]]
    assert out["runs"] == 3
    assert out["schema"] == "self_similarity/1"
    assert out["thresholds"] == {"similarity": ss.FLAG_SIMILARITY,
                                 "verbatim_chars": ss.FLAG_MATCH_CHARS}


def test_single_unit_no_pairs_no_crash():
    out = ss.compute([_chapter("aa")])
    assert out["aggregate"]["n_pairs"] == 0
    assert out["aggregate"]["max"] is None
    assert out["flagged"] == []
    assert out["semantic"]["available"] is False


def test_empty_units_similarity_zero():
    out = ss.compute(["", _chapter("aa")])
    assert out["per_pair"][0]["similarity"] == 0.0
    assert out["per_pair"][0]["longest_match_chars"] == 0


def test_semantic_gracefully_unavailable_without_embedding_dep():
    # In this test env the embedding dependency is absent (and the module is
    # loaded standalone), so the semantic block must degrade, not raise.
    out = ss.compute([_chapter("aa"), _chapter("bb")])
    assert out["semantic"]["available"] is False
    assert "reason" in out["semantic"]
