"""Tests for the narrative-dynamics segmentation layer (pure, no LLM)."""
from __future__ import annotations

from benchmarks.narrative_dynamics import segmentation as seg

GUT = (
    "The Project Gutenberg eBook of Something\n\nLicense preamble here.\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
    "Actual body text.\n\nMore body.\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\nLicense tail.\n"
)


def test_strip_gutenberg_markers():
    body = seg.strip_gutenberg(GUT)
    assert body.startswith("Actual body text.")
    assert body.endswith("More body.")
    assert "License" not in body
    assert "Gutenberg" not in body


def test_strip_gutenberg_handles_crlf_and_this_variant():
    text = ("head\r\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\r\n"
            "body\r\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\r\ntail")
    assert seg.strip_gutenberg(text) == "body"


def test_strip_gutenberg_without_markers_is_passthrough():
    assert seg.strip_gutenberg("just\n\nprose") == "just\n\nprose"


def test_strip_gutenberg_removes_illustrations_and_collapses_blanks():
    text = "a\n\n\n\n[Illustration: A fine plate.]\n\nb"
    out = seg.strip_gutenberg(text)
    assert "[Illustration" not in out
    assert "\n\n\n" not in out


def test_split_paragraphs_joins_hard_wrapped_lines():
    text = "First line\nof one paragraph.\n\nSecond   paragraph\there."
    assert seg.split_paragraphs(text) == [
        "First line of one paragraph.",
        "Second paragraph here.",
    ]


def test_is_chapter_heading_accepts_common_forms():
    for h in ["CHAPTER IV", "Chapter 12.", "Chapter I.", "XII", "IV.", "3.",
              "PROLOGUE", "EPILOGUE", "FIRST NARRATIVE.", "BOOK II",
              "Chapter One"]:
        assert seg.is_chapter_heading(h), h


def test_is_chapter_heading_rejects_prose_and_long_lines():
    for h in ["He said chapter and verse.", "",
              "It was the best of times, it was the worst of times, it was the age "
              "of wisdom"]:
        assert not seg.is_chapter_heading(h), h
    # A bare "I" is ambiguous (roman-numeral chapter 1 vs the pronoun). It is
    # accepted, deliberately: dropping a real chapter-I heading would fold that
    # chapter into the front matter, and a lone blank-surrounded "I" line is
    # vanishingly rare in prose.
    assert seg.is_chapter_heading("I")
    assert seg.is_chapter_heading("II")


def test_detect_chapter_lines_requires_blank_surround():
    text = "CHAPTER I\n\nbody\nCHAPTER II in running prose\n\nCHAPTER II\n\nmore"
    lines = text.split("\n")
    hits = seg.detect_chapter_lines(text)
    assert [lines[i] for i in hits] == ["CHAPTER I", "CHAPTER II"]


def test_detect_chapter_lines_accepts_heading_with_title_line():
    # The Thirty-Nine Steps format: "Chapter I." immediately over a short title
    # line, no blank between them (the shakedown bug: these were all rejected).
    text = ("Chapter I.\nThe Man Who Died\n\n" + _words("a") + "\n\n"
            "Chapter II.\nThe Milkman Sets Out on his Travels\n\n" + _words("b") + "\n\n"
            "Chapter III.\nThe Adventure of the Literary Innkeeper\n\n" + _words("c"))
    lines = text.split("\n")
    hits = seg.detect_chapter_lines(text)
    assert [lines[i] for i in hits] == ["Chapter I.", "Chapter II.", "Chapter III."]


def test_segment_chapters_keeps_title_line_in_body():
    text = ("Chapter I.\nThe Man Who Died\n\n" + _words("a") + "\n\n"
            "Chapter II.\nThe Milkman\n\n" + _words("b") + "\n\n"
            "Chapter III.\nThe Innkeeper\n\n" + _words("c"))
    units = seg.segment_chapters(text)
    assert [u["label"] for u in units] == ["Chapter I.", "Chapter II.", "Chapter III."]
    assert units[0]["text"].startswith("The Man Who Died")
    assert units[1]["text"].startswith("The Milkman")


def test_detect_chapter_lines_rejects_heading_followed_by_prose():
    # A short first prose line is NOT a title: hard-wrapped prose continues on
    # the following line, so the candidate is rejected (guard purpose intact).
    text = ("intro para.\n\n"
            "IV.\nAnd the prose just runs on here\n"
            "continuing on a second wrapped line before any blank.\n\n"
            "more text.")
    assert seg.detect_chapter_lines(text) == []


def test_detect_chapter_lines_rejects_long_line_after_heading():
    long_line = "x" * 80  # too long to be a title line, and no blank after CHAPTER I
    text = f"CHAPTER I\n{long_line}\n\nbody."
    assert seg.detect_chapter_lines(text) == []


def test_toc_block_produces_no_headings():
    # A Contents block whose entries look exactly like body headings (heading +
    # title-line form) must be screened out; the body headings must survive.
    toc = "\n\n".join(f"CHAPTER {r}\nTitle {r}" for r in ["I", "II", "III"])
    dedication = _words("ded", 60)
    body = "\n\n".join(f"CHAPTER {r}\nTitle {r}\n\n" + _words(r.lower())
                       for r in ["I", "II", "III"])
    text = f"Contents\n\n{toc}\n\n{dedication}\n\n{body}"
    lines = text.split("\n")
    hits = seg.detect_chapter_lines(text)
    assert len(hits) == 3
    assert all(i > lines.index(dedication.split("\n")[0]) for i in hits)
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert [u["label"] for u in out["units"]] == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]
    # the dedication is not mislabeled as the tail of a TOC "chapter"
    assert all("ded0" not in u["text"] for u in out["units"])


def test_toc_blank_separated_entries_also_screened():
    # TOC entries each surrounded by blank lines passed even the OLD guard;
    # the density screen drops the run while keeping the real chapters.
    toc = "\n\n".join(f"CHAPTER {r}" for r in ["I", "II", "III", "IV"])
    body = "\n\n".join(f"CHAPTER {r}\n\n" + _words(r.lower())
                       for r in ["I", "II", "III", "IV"])
    text = f"{toc}\n\n{_words('front', 60)}\n\n{body}"
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert [u["label"] for u in out["units"]] == [
        "CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV"]
    assert len(seg.detect_chapter_lines(text)) == 4


def test_steps_style_toc_lines_are_not_headings():
    # The actual Steps TOC sets title on the SAME line with no punctuation
    # separator; such lines never match the heading patterns at all.
    for line in ["Chapter I    The Man Who Died",
                 "Chapter X    Various Parties Converging on the Sea"]:
        assert not seg.is_chapter_heading(line), line


def test_segment_mixed_heading_formats():
    # Synthetic multi-format fixture: old style (blank after), title-line
    # style, and a bare roman numeral; boundaries and labels all correct.
    a, b, c = _words("a"), _words("b"), _words("c")
    text = (f"CHAPTER 1.\n\n{a}\n\n"
            f"Chapter II.\nThe Title Line\n\n{b}\n\n"
            f"III.\n\n{c}")
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    units = out["units"]
    assert [u["label"] for u in units] == ["CHAPTER 1.", "Chapter II.", "III."]
    assert units[0]["text"] == a
    assert units[1]["text"] == "The Title Line\n\n" + b
    assert units[2]["text"] == c
    assert [u["words"] for u in units] == [80, 83, 80]


def test_title_line_style_still_falls_back_below_min_chapters():
    text = ("Chapter I.\nThe Man Who Died\n\n" + _words("a", 200) + "\n\n"
            "Chapter II.\nThe Milkman\n\n" + _words("b", 200))
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_requested"] == "chapters"
    assert out["strategy_used"].startswith("windows (fallback")


def _words(tag, n=80):
    return " ".join(f"{tag}{i}" for i in range(n))


def _chaptered_text(n_chapters=4, words_per=80):
    parts = []
    for c in range(1, n_chapters + 1):
        parts.append(f"CHAPTER {c}.")
        body = " ".join(f"w{c}x{i}" for i in range(words_per))
        parts.append(body)
    return "\n\n".join(parts)


def test_segment_chapters_splits_and_labels():
    units = seg.segment_chapters(_chaptered_text(4))
    assert len(units) == 4
    assert [u["label"] for u in units] == [f"CHAPTER {c}." for c in range(1, 5)]
    assert [u["index"] for u in units] == [0, 1, 2, 3]
    assert all(u["words"] == 80 for u in units)


def test_segment_chapters_returns_empty_below_min():
    # only 2 detectable chapters -> [] so segment() can fall back
    assert seg.segment_chapters(_chaptered_text(2)) == []


def test_segment_chapters_drops_heading_only_fragments():
    text = _chaptered_text(3) + "\n\nCHAPTER 99.\n\ntiny"
    units = seg.segment_chapters(text)
    assert [u["label"] for u in units] == ["CHAPTER 1.", "CHAPTER 2.", "CHAPTER 3."]


def test_segment_windows_snaps_to_paragraph_boundaries():
    paras = [" ".join(f"p{k}w{i}" for i in range(600)) for k in range(7)]
    units = seg.segment_windows("\n\n".join(paras), window_words=1500)
    # 600+600 < 1500, +600 = 1800 >= 1500 -> 3-paragraph windows; 7 paras ->
    # windows of 3, 3, and a trailing 1-paragraph unit (600 words, no runt merge).
    assert [u["words"] for u in units] == [1800, 1800, 600]
    assert units[0]["text"].count("\n\n") == 2  # paragraphs kept intact
    assert [u["label"] for u in units] == ["w000", "w001", "w002"]


def test_segment_windows_merges_trailing_runt():
    paras = [" ".join(f"a{i}" for i in range(1500)), "short tail runt"]
    units = seg.segment_windows("\n\n".join(paras), window_words=1500)
    assert len(units) == 1
    assert units[0]["words"] == 1503


def test_segment_chapters_strategy_with_fallback():
    out = seg.segment("no chapters here.\n\n" + " ".join(f"x{i}" for i in range(100)),
                      strategy="chapters")
    assert out["strategy_requested"] == "chapters"
    assert out["strategy_used"].startswith("windows (fallback")
    assert out["n_units"] == 1


def test_segment_reports_totals_and_uses_chapters_when_present():
    out = seg.segment(_chaptered_text(5), strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert out["n_units"] == 5
    assert out["total_words"] == 5 * 80


def test_segment_trims_gutenberg_by_default():
    text = GUT.replace("Actual body text.\n\nMore body.", _chaptered_text(3))
    out = seg.segment(text, strategy="chapters")
    assert out["n_units"] == 3
    assert "Gutenberg" not in out["units"][0]["text"]


def test_segment_rejects_unknown_strategy():
    try:
        seg.segment("x", strategy="scenes")
    except ValueError as e:
        assert "scenes" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_truncate_middle_short_text_untouched():
    text = " ".join(f"w{i}" for i in range(100))
    assert seg.truncate_middle(text, 50, 50) == text  # within the +200 grace


def test_truncate_middle_long_text_keeps_head_and_tail():
    words = [f"w{i}" for i in range(1000)]
    out = seg.truncate_middle(" ".join(words), 100, 100)
    assert out.startswith("w0 ")
    assert out.endswith(" w999")
    assert "[... middle omitted: about 800 words ...]" in out
