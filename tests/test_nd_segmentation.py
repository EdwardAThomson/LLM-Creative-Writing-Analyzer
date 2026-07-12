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


# --- trailing back-matter trim -------------------------------------------------------

# A Grosset & Dunlap style catalog, modeled on the one printed after THE END
# but inside the Gutenberg markers of the shakedown Dracula copy.
CATALOG_TAIL = (
    "                                THE END\n\n"
    "       *       *       *       *       *\n\n"
    "     There are more books of the sort you like; more than 500 titles\n"
    "     all told, in the list which you will find on the wrapper of this\n"
    "     book. Ask for the publishers' complete catalog.\n\n"
    "DETECTIVE STORIES BY J. S. FLETCHER\n\n"
    "THE SECRET OF THE BARBICAN\n\nGREEN INK\n\nTHE SAFETY PIN\n\n"
    "GROSSET & DUNLAP, Publishers, NEW YORK\n"
)

_LONG_PROSE = ("It was a long and uneventful journey through the low hills "
               "and the weather held fair for the whole of it. ") * 200


def test_tail_trim_removes_catalog_after_end_marker():
    text = _LONG_PROSE + "\n\n" + CATALOG_TAIL
    out, note = seg.trim_trailing_backmatter(text)
    assert note is not None
    assert out.endswith("THE END")           # marker line kept
    assert "GROSSET" not in out
    assert note["marker"] == "THE END"
    assert note["trimmed_words"] == seg.word_count(CATALOG_TAIL.split("THE END", 1)[1])
    assert note["non_narrative_line_ratio"] >= seg.TAIL_NONPROSE_RATIO
    assert "catalog" in note["vocabulary_hits"]
    assert len(note["vocabulary_hits"]) >= seg.TAIL_MIN_VOCAB_HITS


def test_tail_trim_no_marker_keeps_everything():
    # the Steps case: no end-marker line at all; the closing prose survives
    text = _LONG_PROSE + "\n\nBut I had done my best service, I think, before I put on khaki."
    out, note = seg.trim_trailing_backmatter(text)
    assert note is None
    assert out == text


def test_tail_trim_mid_text_marker_is_never_trusted():
    # "THE END" of Book One at ~50% of the text: in doubt, keep everything
    text = _LONG_PROSE + "\n\nTHE END\n\n" + _LONG_PROSE
    out, note = seg.trim_trailing_backmatter(text)
    assert note is None
    assert out == text


def test_tail_trim_keeps_prose_after_marker():
    # an epilogue in real prose after the marker: in doubt, keep everything
    epilogue = ("It remains only to add that the family prospered for many "
                "years afterward and the old house stood open to travellers "
                "until the road itself was moved. Nothing more was ever heard "
                "of the stranger, though the innkeeper claimed otherwise.")
    text = _LONG_PROSE + "\n\nTHE END\n\n" + epilogue
    out, note = seg.trim_trailing_backmatter(text)
    assert note is None
    assert out == text


def test_tail_trim_requires_catalog_vocabulary():
    # non-narrative-looking lines without publisher vocabulary: keep everything
    text = _LONG_PROSE + "\n\nTHE END\n\nALPHA\n\nBETA\n\nGAMMA\n\nDELTA\n"
    out, note = seg.trim_trailing_backmatter(text)
    assert note is None
    assert out == text


def test_tail_trim_marker_with_nothing_after_is_a_no_op():
    text = _LONG_PROSE + "\n\nTHE END"
    out, note = seg.trim_trailing_backmatter(text)
    assert note is None
    assert out == text


def test_segment_records_tail_trim_and_final_chapter_is_clean():
    # chapters long enough that the marker sits inside the last-5% zone
    body = "\n\n".join(f"CHAPTER {c}\n\n" + _words(f"c{c}", 800) for c in (1, 2, 3))
    out = seg.segment(body + "\n\n" + CATALOG_TAIL, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert out["tail_trim"] is not None
    assert out["tail_trim"]["marker"] == "THE END"
    assert all("GROSSET" not in u["text"] for u in out["units"])
    # material before the marker (the Harker-NOTE position) is untouched
    assert out["units"][-1]["text"].rstrip().endswith("THE END")


def test_segment_without_trim_reports_tail_trim_none():
    out = seg.segment(_chaptered_text(3), strategy="chapters")
    assert out["tail_trim"] is None


# --- front-matter scoring policy -------------------------------------------------------

def _units_with_front():
    text = _words("front", 220) + "\n\n" + _chaptered_text(3)
    units = seg.segment_chapters(text)
    assert units[0]["label"] == seg.FRONT_LABEL  # fixture sanity
    return units


def test_exclude_front_matter_default_excludes_and_records():
    units = _units_with_front()
    kept, record = seg.exclude_front_matter(units)
    assert [u["label"] for u in kept] == ["CHAPTER 1.", "CHAPTER 2.", "CHAPTER 3."]
    assert [u["index"] for u in kept] == [1, 2, 3]  # original indices kept
    assert record["excluded"] is True
    assert record["front_units"] == [{"index": 0, "label": "(front)", "words": 220}]
    assert record["n_units_segmented"] == 4
    assert record["n_units_scored"] == 3
    assert "--include-front" in record["policy"]


def test_exclude_front_matter_opt_in_keeps_front():
    units = _units_with_front()
    kept, record = seg.exclude_front_matter(units, include_front=True)
    assert kept == units
    assert record["excluded"] is False
    assert record["n_units_scored"] == 4


def test_exclude_front_matter_without_front_unit_is_silent():
    units = seg.segment_chapters(_chaptered_text(3))
    kept, record = seg.exclude_front_matter(units)
    assert kept == units
    assert record is None


# --- TOC density screen ------------------------------------------------------------

def test_toc_density_screen_is_the_only_standing_guard():
    # Regression for the density screen itself. On the real shakedown books it
    # never fired (Dracula's TOC entries died at the heading regex and
    # blank-context guards), so this fixture is built to pass every candidate
    # guard: each blank-line-separated TOC entry matches the heading regex,
    # has a blank line (or the text edge) above, and a blank line below.
    entries = ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV", "CHAPTER V"]
    for e in entries:
        assert seg.is_chapter_heading(e)
    toc = "\n\n".join(entries)
    preface = _words("pref", 60)  # >= MIN_UNIT_WORDS: breaks the run before the body
    body = "\n\n".join(f"{e}\n\n" + _words(e.split()[-1].lower(), 120) for e in entries)
    text = f"{toc}\n\n{preface}\n\n{body}"

    # Control: the same shape with a sub-threshold run (TOC_MIN_RUN - 1
    # entries) leaks straight through the candidate guards, proving the
    # density screen is the only defense standing against this fixture.
    control = "\n\n".join(entries[:seg.TOC_MIN_RUN - 1]) + f"\n\n{preface}\n\n{body}"
    assert len(seg.detect_chapter_lines(control)) == (seg.TOC_MIN_RUN - 1) + len(entries)

    # The screen drops the whole TOC run; every real chapter survives.
    lines = text.split("\n")
    hits = seg.detect_chapter_lines(text)
    assert [lines[i] for i in hits] == entries
    assert min(hits) > lines.index(preface)  # the body headings, not the TOC
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert [u["label"] for u in out["units"]] == entries
    assert all("pref0" not in u["text"] for u in out["units"])


def test_truncate_middle_short_text_untouched():
    text = " ".join(f"w{i}" for i in range(100))
    assert seg.truncate_middle(text, 50, 50) == text  # within the +200 grace


def test_truncate_middle_long_text_keeps_head_and_tail():
    words = [f"w{i}" for i in range(1000)]
    out = seg.truncate_middle(" ".join(words), 100, 100)
    assert out.startswith("w0 ")
    assert out.endswith(" w999")
    assert "[... middle omitted: about 800 words ...]" in out
