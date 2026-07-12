"""Tests for the extract command (proposer -> canonical Markdown) and the md
ingestion strategy (the permanent analysis path). Pure stdlib, no LLM."""
from __future__ import annotations

import json

from benchmarks.narrative_dynamics import segmentation as seg
from benchmarks.narrative_dynamics.__main__ import _effective_strategy
from benchmarks.narrative_dynamics.__main__ import main as nd_main
from benchmarks.narrative_dynamics.extract import main as extract_main

from conftest import load_metric


def _words(tag, n=80):
    return " ".join(f"{tag}{i}" for i in range(n))


def _synthetic_book() -> str:
    # front matter + three chapters; chapter two carries a body line that
    # would read as a Markdown heading, to exercise the escape round trip
    return ("The Project Gutenberg eBook of Mini Novel\n\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK MINI NOVEL ***\n\n"
            + _words("front", 220) + "\n\n"
            "CHAPTER I\n\n" + _words("one", 80) + "\n\n"
            "CHAPTER II\n\n" + _words("two", 60)
            + "\n\n# not a heading, just a hash line\n\n" + _words("more", 30)
            + "\n\nCHAPTER III\n\n" + _words("three", 80) + "\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK MINI NOVEL ***\n")


def _extract(tmp_path, argv_extra=()):
    src = tmp_path / "mini.txt"
    src.write_text(_synthetic_book(), encoding="utf-8")
    out = tmp_path / "mini.md"
    rc = extract_main([str(src), "--out", str(out)] + list(argv_extra))
    return rc, src, out


# --- the md splitter ---------------------------------------------------------------

MD_DOC = ("<!--\nschema: nd-extract/1\nsource: book.txt\nsha256: abc\n-->\n\n"
          "# (front)\n\ntitle page words\n\n"
          "# CHAPTER I\n\nBody one.\n\n"
          "# CHAPTER II\n\nBody two.\n\\# escaped heading line\nmore\n")


def test_md_splitter_units_and_labels():
    units, prov = seg.segment_markdown(MD_DOC)
    assert [u["label"] for u in units] == ["(front)", "CHAPTER I", "CHAPTER II"]
    assert units[0]["text"] == "title page words"
    assert units[1]["text"] == "Body one."
    assert [u["index"] for u in units] == [0, 1, 2]
    assert all(u["words"] == seg.word_count(u["text"]) for u in units)


def test_md_splitter_reads_provenance():
    units, prov = seg.segment_markdown(MD_DOC)
    assert prov == {"schema": "nd-extract/1", "source": "book.txt",
                    "sha256": "abc"}


def test_md_splitter_without_provenance():
    units, prov = seg.segment_markdown("# A\n\nbody a\n\n# B\n\nbody b\n")
    assert prov is None
    assert [u["label"] for u in units] == ["A", "B"]


def test_md_splitter_unescapes_body_heading_lines():
    units, _ = seg.segment_markdown(MD_DOC)
    assert "\n# escaped heading line\n" in "\n" + units[2]["text"] + "\n"
    assert "\\#" not in units[2]["text"]


def test_md_splitter_only_top_level_headings_split():
    text = "# A\n\nbody\n## subheading stays\n  # indented stays\n#nospace stays\n"
    units, _ = seg.segment_markdown(text)
    assert len(units) == 1
    assert "## subheading stays" in units[0]["text"]
    assert "# indented stays" in units[0]["text"]
    assert "#nospace stays" in units[0]["text"]


def test_md_splitter_content_before_first_heading_becomes_front():
    units, _ = seg.segment_markdown("stray line\n\n# A\n\nbody\n")
    assert [u["label"] for u in units] == [seg.FRONT_LABEL, "A"]
    assert units[0]["text"] == "stray line"


def test_md_splitter_no_headings_single_document_unit():
    units, prov = seg.segment_markdown("<!--\nsource: x\n-->\n\njust prose here\n")
    assert [u["label"] for u in units] == ["(document)"]
    assert units[0]["text"] == "just prose here"
    assert prov == {"source": "x"}


def test_md_splitter_handles_crlf_and_empty_bodies():
    units, _ = seg.segment_markdown("# A\r\n\r\nbody\r\n\r\n# B\r\n")
    assert [u["label"] for u in units] == ["A", "B"]
    assert units[0]["text"] == "body"
    assert units[1]["text"] == ""
    assert units[1]["words"] == 0


def test_segment_md_strategy_result_shape():
    out = seg.segment(MD_DOC, strategy="md")
    assert out["strategy_requested"] == "md"
    assert out["strategy_used"] == "md"
    assert out["n_units"] == 3
    assert out["tail_trim"] is None
    assert out["provenance"]["schema"] == "nd-extract/1"
    # no heuristics: the fake TOC screen / min-chapters fallback never runs
    two_units = seg.segment("# A\n\nbody a\n\n# B\n\nbody b\n", strategy="md")
    assert two_units["strategy_used"] == "md"
    assert two_units["n_units"] == 2


# --- the extract command --------------------------------------------------------------

def test_extract_writes_markdown_sidecar_and_report(tmp_path, capsys):
    rc, src, out = _extract(tmp_path)
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert md.startswith("<!--\nschema: nd-extract/1\n")
    for key in ("source: mini.txt", "sha256: ", "extracted: ", "tool: ",
                "commit: ", "units: 4"):
        assert key in md.split("-->")[0]
    assert "\n# (front)\n" in md
    assert "\n# CHAPTER I\n" in md
    assert "\n\\# not a heading, just a hash line\n" in md  # escaped
    sidecar = json.loads((tmp_path / "mini.extract.json").read_text())
    assert sidecar["schema"] == "nd-extract/1"
    assert sidecar["labels"] == ["(front)", "CHAPTER I", "CHAPTER II",
                                 "CHAPTER III"]
    assert sidecar["unit_words"] == [220, 80, 98, 80]
    assert sidecar["strategy_used"] == "chapters"
    assert sidecar["warnings"]["front_unit"]["words"] == 220
    assert sidecar["warnings"]["escaped_body_lines"] == 1
    report = capsys.readouterr().out
    assert "EXTRACTION REPORT" in report
    assert "Units: 4" in report
    assert "front unit kept: 220 words" in report


def test_extract_expected_units_gate(tmp_path, capsys):
    rc, _, _ = _extract(tmp_path, ["--expected-units", "4"])
    assert rc == 0
    assert "Expected units: 4, found 4: OK" in capsys.readouterr().out
    rc, _, _ = _extract(tmp_path, ["--expected-units", "9"])
    assert rc == 1
    assert "found 4: MISMATCH" in capsys.readouterr().out
    sidecar = json.loads((tmp_path / "mini.extract.json").read_text())
    assert sidecar["expected_units"] == 9
    assert sidecar["expected_match"] is False


def test_extract_refuses_overwriting_input(tmp_path):
    src = tmp_path / "book.md"
    src.write_text("# A\n\nbody\n", encoding="utf-8")
    assert extract_main([str(src), "--out", str(src)]) == 2
    assert extract_main([str(tmp_path / "missing.txt")]) == 2


def test_extract_records_proposer_warnings(tmp_path, capsys):
    # a runt unit and a screened TOC run must surface in sidecar + report
    toc = "\n\n".join(f"CHAPTER {r}" for r in ["I", "II", "III"])
    text = (toc + "\n\n" + _words("ded", 60) + "\n\n"
            "CHAPTER I\n\n" + _words("one", 80) + "\n\n"
            "CHAPTER II\n\ntiny\n\n"
            "CHAPTER III\n\n" + _words("three", 80) + "\n\n"
            "CHAPTER IV\n\n" + _words("four", 80) + "\n")
    src = tmp_path / "warny.txt"
    src.write_text(text, encoding="utf-8")
    assert extract_main([str(src), "--out", str(tmp_path / "warny.md")]) == 0
    sidecar = json.loads((tmp_path / "warny.extract.json").read_text())
    assert sidecar["warnings"]["dropped_runts"] == [
        {"label": "CHAPTER II", "words": 1}]
    screened = [s["text"] for s in sidecar["warnings"]["screened_candidates"]]
    assert screened == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]
    report = capsys.readouterr().out
    assert "dropped runt units" in report
    assert "screened TOC/junction candidates" in report


def test_extract_records_page_anchor_strip(tmp_path, capsys):
    # Monte Cristo shape: standalone plate-anchor lines ("0185m") are stripped
    # during trimming and the count must be sidecar- and report-visible
    text = ("CHAPTER I\n\n" + _words("one", 80) + "\n\n0185m\n\n" + _words("oneb", 20)
            + "\n\nCHAPTER II\n\n" + _words("two", 80) + "\n\n0261m\n\n"
            + _words("twob", 20)
            + "\n\nCHAPTER III\n\n" + _words("three", 80) + "\n")
    src = tmp_path / "anchors.txt"
    src.write_text(text, encoding="utf-8")
    assert extract_main([str(src), "--out", str(tmp_path / "anchors.md")]) == 0
    sidecar = json.loads((tmp_path / "anchors.extract.json").read_text())
    assert sidecar["warnings"]["page_anchor_lines_removed"] == 2
    assert sidecar["unit_words"] == [100, 100, 80]  # anchors are not words
    md = (tmp_path / "anchors.md").read_text(encoding="utf-8")
    assert "0185m" not in md and "0261m" not in md
    assert "page_anchor_lines_removed: 2" in capsys.readouterr().out


# --- unmatched structural suspects in the extract report --------------------------------

def test_extract_reports_unmatched_suspects(tmp_path, capsys):
    # Collins shape: a narrator heading matching no pattern sits mid-body; it
    # must surface in the sidecar and the stdout count (diagnostic only)
    text = ("CHAPTER I\n\n" + _words("one", 80) + "\n\n"
            "THE STORY CONTINUED BY VINCENT GILMORE\n\n" + _words("g", 80)
            + "\n\nCHAPTER II\n\n" + _words("two", 80)
            + "\n\nCHAPTER III\n\n" + _words("three", 80) + "\n")
    src = tmp_path / "sus.txt"
    src.write_text(text, encoding="utf-8")
    assert extract_main([str(src), "--out", str(tmp_path / "sus.md")]) == 0
    sidecar = json.loads((tmp_path / "sus.extract.json").read_text())
    assert sidecar["unmatched_suspects"] == [
        {"unit_index": 0, "line": "THE STORY CONTINUED BY VINCENT GILMORE",
         "position_words_into_unit": 80}]
    assert "Unmatched structural suspects: 1" in capsys.readouterr().out


def test_extract_suspects_exclude_epistolary_headers(tmp_path, capsys):
    # Dracula shape: italic entry headers must not flood the suspect list
    text = ("CHAPTER I\n\n_Jonathan Harker’s Journal._\n\n" + _words("one", 80)
            + "\n\nCHAPTER II\n\n_Dr. Seward’s Diary._\n\n" + _words("two", 80)
            + "\n\nCHAPTER III\n\n" + _words("three", 80) + "\n")
    src = tmp_path / "drac.txt"
    src.write_text(text, encoding="utf-8")
    assert extract_main([str(src), "--out", str(tmp_path / "drac.md")]) == 0
    sidecar = json.loads((tmp_path / "drac.extract.json").read_text())
    assert sidecar["unmatched_suspects"] == []
    assert "Unmatched structural suspects: 0" in capsys.readouterr().out


# --- boundary overrides ------------------------------------------------------------------

COLLINS_SHAPED = ("I\n\n" + _words("w1", 80) + "\n\n"
                  "II\n\n" + _words("w2", 80) + "\n\n\n"
                  "THE STORY CONTINUED BY VINCENT GILMORE\n\n"
                  + _words("g", 80) + "\n\n"
                  "I\n\n" + _words("g1", 80) + "\n")


def _write_boundaries(tmp_path, headings):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"headings": headings}), encoding="utf-8")
    return p


def test_boundaries_override_splits_collins_shape(tmp_path, capsys):
    src = tmp_path / "collins.txt"
    src.write_text(COLLINS_SHAPED, encoding="utf-8")
    b = _write_boundaries(tmp_path, [
        {"match": "THE STORY CONTINUED BY VINCENT GILMORE",
         "label": "The Story Continued by Vincent Gilmore"}])
    out = tmp_path / "collins.md"
    rc = extract_main([str(src), "--out", str(out), "--boundaries", str(b)])
    assert rc == 0
    sidecar = json.loads((tmp_path / "collins.extract.json").read_text())
    assert sidecar["labels"] == [
        "I", "II", "The Story Continued by Vincent Gilmore", "I (2)"]
    assert [u.get("from_override", False) for u in sidecar["units"]] == [
        False, False, True, False]
    assert sidecar["boundary_overrides"]["entries"] == [
        {"match": "THE STORY CONTINUED BY VINCENT GILMORE",
         "label": "The Story Continued by Vincent Gilmore",
         "lines_matched": 1, "boundaries": 1}]
    md = out.read_text(encoding="utf-8")
    assert "\n# The Story Continued by Vincent Gilmore\n" in md
    # the heading line is excluded from bodies: unit II ends with its prose
    units, _ = seg.segment_markdown(md)
    assert units[1]["text"].endswith("w279")
    assert units[2]["text"] == _words("g", 80)
    assert "Boundary overrides: 1 entries, 1 matched lines, 1 boundaries" \
        in capsys.readouterr().out


def test_boundaries_override_runt_and_front_rules_apply(tmp_path):
    # an override whose body is thin drops as a runt, like any heading
    text = ("I\n\n" + _words("w1", 80) + "\n\n"
            "II\n\n" + _words("w2", 80) + "\n\n\n"
            "THE STORY CONCLUDED BY WALTER HARTRIGHT\n\n\n"
            "I\n\n" + _words("g1", 80) + "\n")
    src = tmp_path / "runt.txt"
    src.write_text(text, encoding="utf-8")
    b = _write_boundaries(tmp_path, [
        {"match": "THE STORY CONCLUDED BY WALTER HARTRIGHT"}])
    assert extract_main([str(src), "--out", str(tmp_path / "runt.md"),
                         "--boundaries", str(b)]) == 0
    sidecar = json.loads((tmp_path / "runt.extract.json").read_text())
    assert sidecar["labels"] == ["I", "II", "I (2)"]
    assert sidecar["warnings"]["dropped_runts"] == [
        {"label": "THE STORY CONCLUDED BY WALTER HARTRIGHT", "words": 0}]
    # label defaulted to the matched text in the runt record; the entry still
    # reports its boundary
    assert sidecar["boundary_overrides"]["entries"][0]["boundaries"] == 1


def test_boundaries_zero_match_fails_loudly(tmp_path, capsys):
    src = tmp_path / "collins.txt"
    src.write_text(COLLINS_SHAPED, encoding="utf-8")
    b = _write_boundaries(tmp_path, [
        {"match": "THE STORY CONTINUED BY VINCENT GILMORE"},
        {"match": "THE STORY CONTINUED BY NOBODY AT ALL"}])
    out = tmp_path / "collins.md"
    rc = extract_main([str(src), "--out", str(out), "--boundaries", str(b)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "matched zero lines" in err
    assert "THE STORY CONTINUED BY NOBODY AT ALL" in err
    assert not out.exists()  # nothing written on stale overrides
    assert not (tmp_path / "collins.extract.json").exists()


def test_boundaries_malformed_file_fails_loudly(tmp_path, capsys):
    src = tmp_path / "collins.txt"
    src.write_text(COLLINS_SHAPED, encoding="utf-8")
    for payload in ["not json at all", '{"headings": "x"}',
                    '{"headings": [{"label": "no match key"}]}',
                    '{"headings": []}']:
        b = tmp_path / "b.json"
        b.write_text(payload, encoding="utf-8")
        rc = extract_main([str(src), "--out", str(tmp_path / "c.md"),
                           "--boundaries", str(b)])
        assert rc == 2, payload
        assert "bad boundaries file" in capsys.readouterr().err


# --- duplicate-label disambiguation -------------------------------------------------------

def test_extract_disambiguates_duplicate_labels(tmp_path):
    # Wells shape: roman numerals restart in Book Two; md headings must stay
    # unique while the sidecar keeps the raw heading text per unit
    text = ("I.\n\n" + _words("a", 80) + "\n\nII.\n\n" + _words("b", 80)
            + "\n\nI.\n\n" + _words("c", 80) + "\n\nII.\n\n" + _words("d", 80)
            + "\n\nI.\n\n" + _words("e", 80) + "\n")
    src = tmp_path / "dup.txt"
    src.write_text(text, encoding="utf-8")
    out = tmp_path / "dup.md"
    assert extract_main([str(src), "--out", str(out)]) == 0
    sidecar = json.loads((tmp_path / "dup.extract.json").read_text())
    assert sidecar["labels"] == ["I.", "II.", "I. (2)", "II. (2)", "I. (3)"]
    assert [u["label_raw"] for u in sidecar["units"]] == [
        "I.", "II.", "I.", "II.", "I."]
    md = out.read_text(encoding="utf-8")
    assert "\n# I. (2)\n" in md and "\n# I. (3)\n" in md
    # round-trip parity on the disambiguated headings
    units, _ = seg.segment_markdown(md)
    assert [u["label"] for u in units] == sidecar["labels"]
    assert units[2]["text"] == _words("c", 80)


# --- round trip: extract -> md -> re-ingest ---------------------------------------------

def test_round_trip_unit_parity(tmp_path):
    rc, src, out = _extract(tmp_path)
    assert rc == 0
    direct = seg.segment(src.read_text(encoding="utf-8"), strategy="chapters")
    reingested = seg.segment(out.read_text(encoding="utf-8"), strategy="md")
    assert [u["label"] for u in reingested["units"]] == \
           [u["label"] for u in direct["units"]]
    assert [u["text"] for u in reingested["units"]] == \
           [u["text"] for u in direct["units"]]
    assert [u["words"] for u in reingested["units"]] == \
           [u["words"] for u in direct["units"]]


def test_round_trip_metric_equality(tmp_path, metric):
    # a pure library metric scores the md-ingested units identically to the
    # directly segmented ones (the extraction step must be metric-invisible)
    rc, src, out = _extract(tmp_path)
    assert rc == 0
    direct = seg.segment(src.read_text(encoding="utf-8"), strategy="chapters")
    reingested = seg.segment(out.read_text(encoding="utf-8"), strategy="md")
    of = metric("opening_formula")
    a = of.compute([u["text"] for u in direct["units"]])
    b = of.compute([u["text"] for u in reingested["units"]])
    assert a == b


# --- nd CLI integration -----------------------------------------------------------------

def test_effective_strategy_resolution():
    assert _effective_strategy("book.txt", None) == "chapters"
    assert _effective_strategy("book.md", None) == "md"
    assert _effective_strategy("book.MD", None) == "md"
    # heading heuristics never run on canonical Markdown
    assert _effective_strategy("book.md", "chapters") == "md"
    assert _effective_strategy("book.txt", "chapters") == "chapters"
    # an explicit windows request is honoured for any input
    assert _effective_strategy("book.md", "windows") == "windows"


def test_nd_cli_scores_extracted_md(tmp_path):
    rc, src, out = _extract(tmp_path)
    assert rc == 0
    assert nd_main([str(out), "--dry-run", "--metrics", "tension_trajectory"]) == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    seg_info = result["segmentation"]
    assert seg_info["strategy_used"] == "md"
    assert seg_info["n_units"] == 4
    assert seg_info["provenance"]["source"] == "mini.txt"
    # the (front) unit written by extract is still excluded at scoring time
    assert seg_info["front_matter"]["excluded"] is True
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert [u["label"] for u in per_unit] == ["CHAPTER I", "CHAPTER II",
                                              "CHAPTER III"]


def test_st1_textmode_segments_md(tmp_path):
    # the st1 path reuses the same layer through _textmode.segment_units
    rc, src, out = _extract(tmp_path)
    assert rc == 0
    tm = load_metric("_textmode")
    info, labels, texts = tm.segment_units(out.read_text(encoding="utf-8"), "md")
    assert info["strategy_used"] == "md"
    assert info["provenance"]["schema"] == "nd-extract/1"
    assert labels == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]  # front excluded
    assert info["front_matter"]["excluded"] is True
    assert texts[0].startswith("one0")
