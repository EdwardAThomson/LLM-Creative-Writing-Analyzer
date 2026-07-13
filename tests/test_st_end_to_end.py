"""End-to-end single-text (st1) scoring on a synthetic fixture.

Drives the same path the CLI uses (``python -m utils.metrics --text book.txt
--segment chapters --benchmark st1``) minus the ``utils`` package import, which
this minimal env cannot perform (heavy eager deps; see conftest): manifest
resolution, segmentation via the reused narrative_dynamics layer
(``_textmode.segment_units``), and every stdlib st1 metric over the resulting
units, including the planted-duplicate detection through the full pipeline.
Zero LLM calls anywhere; heavy local-NLP metrics (burstiness, phonetic_names,
entity_census extraction) are exercised in the full environment instead.
"""
from __future__ import annotations

from conftest import load_metric

mf = load_metric("_manifests")
tm = load_metric("_textmode")

# the st1 metrics whose modules are stdlib-only (loadable + runnable here)
PURE_ST1 = ["text_structure", "mtld", "dialogue_ratio", "intra_text_repetition",
            "cliche_density", "ngram_diversity", "self_similarity",
            "opening_formula"]


def _chapter_body(tag, n=160):
    words = " ".join(f"{tag}tok{i}" for i in range(n))
    return f"The chapter opened on {tag} matters. {words}."


def _fixture_book():
    """Gutenberg-wrapped synthetic book; chapter III re-uses a ~700-char block
    of chapter II verbatim (the adjacent-duplication defect)."""
    block = " ".join(f"dupword{i}" for i in range(110))
    chapters = {
        "I": _chapter_body("aa"),
        "II": _chapter_body("bb") + " " + block,
        "III": block + " " + _chapter_body("cc"),
        "IV": _chapter_body("dd"),
    }
    body = "\n\n".join(f"CHAPTER {n}\n\n{text}" for n, text in chapters.items())
    return ("Frontmatter junk\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
            + body +
            "\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"), block


def test_st1_manifest_metrics_all_have_modules():
    import pathlib
    metrics_dir = pathlib.Path(__file__).parent.parent / "utils" / "metrics"
    for name in mf.resolve("st1")["metrics"]:
        assert (metrics_dir / f"{name}.py").is_file(), name


def test_segment_units_reuses_nd_layer_and_trims_gutenberg():
    text, _ = _fixture_book()
    info, labels, units = tm.segment_units(text, "chapters")
    assert info["strategy_used"] == "chapters"
    assert info["n_units"] == 4
    assert labels == ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV"]
    assert all("Gutenberg" not in u for u in units)


def test_segment_units_windows_strategy():
    text, _ = _fixture_book()
    info, labels, units = tm.segment_units(text, "windows", window_words=200)
    assert info["strategy_used"] == "windows"
    assert len(units) >= 2
    assert labels[0] == "w000"


def _fixture_book_with_front():
    """The fixture book with 220 words of front matter (title page / TOC
    stand-in) between the Gutenberg START marker and CHAPTER I."""
    text, block = _fixture_book()
    front = " ".join(f"frontword{i}" for i in range(220))
    return text.replace("***\n\nCHAPTER I", f"***\n\n{front}\n\nCHAPTER I"), block


def test_segment_units_excludes_front_by_default_and_records_it():
    # shakedown: Dracula's (front) unit (MTLD 24.3 vs 70-101 for chapters,
    # TOC names in the census) polluted the per-unit stats; the scoring layer
    # now drops it by default and the sidecar info says so.
    text, _ = _fixture_book_with_front()
    info, labels, units = tm.segment_units(text, "chapters")
    assert labels == ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV"]
    assert all("frontword" not in u for u in units)
    fm = info["front_matter"]
    assert fm["excluded"] is True
    assert fm["front_units"] == [{"index": 0, "label": "(front)", "words": 220}]
    assert fm["n_units_segmented"] == 5
    assert fm["n_units_scored"] == 4
    assert info["n_units"] == 5  # segmentation truth is untouched


def test_segment_units_include_front_opts_back_in():
    text, _ = _fixture_book_with_front()
    info, labels, units = tm.segment_units(text, "chapters", include_front=True)
    assert labels[0] == "(front)"
    assert len(units) == 5
    assert info["front_matter"]["excluded"] is False
    assert info["front_matter"]["n_units_scored"] == 5


def test_segment_units_records_absent_front_and_tail_trim():
    # nothing silent, nothing invented: without front matter or an end-marker
    # tail, both records are present and null
    text, _ = _fixture_book()
    info, _, _ = tm.segment_units(text, "chapters")
    assert info["front_matter"] is None
    assert info["tail_trim"] is None


def _fixture_book_with_preface():
    """The fixture book with a PREFACE heading before CHAPTER I: author
    apparatus, not story (generalizes the (front) exclusion above)."""
    text, block = _fixture_book()
    preface = " ".join(f"prefaceword{i}" for i in range(80))
    return text.replace(
        "***\n\nCHAPTER I", f"***\n\nPREFACE\n\n{preface}\n\nCHAPTER I"), block


def test_segment_units_excludes_apparatus_by_default_and_records_it():
    text, _ = _fixture_book_with_preface()
    info, labels, units = tm.segment_units(text, "chapters")
    assert labels == ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV"]
    assert all("prefaceword" not in u for u in units)
    ap = info["apparatus"]
    assert ap["excluded"] is True
    assert ap["apparatus_units"] == [{"index": 0, "label": "PREFACE", "words": 80}]
    assert ap["n_units_scored"] == 4
    assert info["n_units"] == 5  # segmentation truth is untouched


def test_segment_units_include_apparatus_opts_back_in():
    text, _ = _fixture_book_with_preface()
    info, labels, units = tm.segment_units(text, "chapters", include_apparatus=True)
    assert labels[0] == "PREFACE"
    assert len(units) == 5
    assert info["apparatus"]["excluded"] is False
    assert info["apparatus"]["n_units_scored"] == 5


def test_full_st1_pure_pass_over_segmented_fixture():
    text, block = _fixture_book()
    _, _, units = tm.segment_units(text, "chapters")
    ctx: dict = {}
    results = {name: load_metric(name).compute(units, ctx) for name in PURE_ST1}

    # every metric produced its self-describing shape over the 4 units
    for name, res in results.items():
        assert res["schema"].startswith(name + "/"), name
        assert res["runs"] == 4, name

    # the planted duplicate is exposed through the full segmentation pipeline
    ss = results["self_similarity"]
    assert ss["duplication_suspected"] is True
    assert [p["pair"] for p in ss["flagged"]] == [[1, 2]]
    assert ss["flagged"][0]["longest_match_chars"] >= len(block) - 10
    assert ss["aggregate"]["median"] < 0.05

    # the fixture's formulaic chapter openings are caught too (chapter III
    # opens with the duplicated block, so 3 of 4 share the formula)
    opening = results["opening_formula"]
    assert opening["repeated_openers"][0]["opener"] == "the chapter opened"
    assert opening["repeated_openers"][0]["count"] == 3

    # sanity on a couple of per-unit metrics
    assert results["text_structure"]["aggregate"]["paragraphs"]["mean"] >= 1
    assert all(r["mtld"] is not None for r in results["mtld"]["per_run"])
