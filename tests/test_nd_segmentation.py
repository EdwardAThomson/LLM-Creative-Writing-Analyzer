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


# --- apparatus scoring policy --------------------------------------------------------

def test_is_apparatus_label_matches_the_closed_vocabulary():
    for label in ("PREFACE", "preface", "  Preface  ", "NOTE TO THE THIRD EDITION",
                  "note to the third edition", "FOOTNOTES", "DEDICATION",
                  "APPENDIX B", "APPENDIX", "FOREWORD", "AFTERWORD", "ERRATA",
                  "GLOSSARY", "EPIGRAPH", "CONTENTS", "NOTE",
                  "TRANSLATOR'S NOTE", "AUTHOR'S NOTE", "PUBLISHER'S NOTE",
                  "INTRODUCTORY NOTE", "PREFACE TO THE READER", "PREFACE."):
        assert seg.is_apparatus_label(label), label


def test_is_apparatus_label_does_not_match_story_sections():
    # The critical negative gate: labels containing apparatus-ish vocabulary
    # words that ARE the story (The Woman in White's epistolary narrators)
    # must never match, nor must ordinary chapter/structural headings, nor
    # the conservatively-deferred PROLOGUE/EPILOGUE.
    for label in ("The Story Begun by Walter Hartright",
                  "The Story Continued by Marian Halcombe",
                  "1. The Narrative of Hester Pinhorn",
                  "The Story Continued by Isidor, Ottavio, Baldassare Fosco",
                  "CHAPTER I.", "CHAPTER I", "Chapter 12.", "IV", "12.",
                  "PROLOGUE", "EPILOGUE", "INTRODUCTION", "INTERLUDE", "ENVOI"):
        assert not seg.is_apparatus_label(label), label


def _units_with_apparatus():
    text = ("CHAPTER I\n\n" + _words("one", 120)
            + "\n\nPREFACE\n\n" + _words("pref", 120)
            + "\n\nCHAPTER II\n\n" + _words("two", 120)
            + "\n\nCHAPTER III\n\n" + _words("three", 120))
    units = seg.segment_chapters(text)
    assert [u["label"] for u in units] == ["CHAPTER I", "PREFACE", "CHAPTER II", "CHAPTER III"]
    return units


def test_exclude_apparatus_default_excludes_and_records():
    units = _units_with_apparatus()
    kept, record = seg.exclude_apparatus(units)
    assert [u["label"] for u in kept] == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]
    assert [u["index"] for u in kept] == [0, 2, 3]  # original indices kept
    assert record["excluded"] is True
    assert record["apparatus_units"] == [{"index": 1, "label": "PREFACE", "words": 120}]
    assert record["n_units_segmented"] == 4
    assert record["n_units_scored"] == 3
    assert "--include-apparatus" in record["policy"]


def test_exclude_apparatus_opt_in_keeps_apparatus():
    units = _units_with_apparatus()
    kept, record = seg.exclude_apparatus(units, include_apparatus=True)
    assert kept == units
    assert record["excluded"] is False
    assert record["n_units_scored"] == 4


def test_exclude_apparatus_without_apparatus_unit_is_silent():
    units = seg.segment_chapters(_chaptered_text(3))
    kept, record = seg.exclude_apparatus(units)
    assert kept == units
    assert record is None


def test_exclude_non_story_chains_front_and_apparatus():
    text = (_words("front", 220) + "\n\nPREFACE\n\n" + _words("pref", 120)
            + "\n\n" + _chaptered_text(3))
    units = seg.segment_chapters(text)
    assert [u["label"] for u in units] == [
        seg.FRONT_LABEL, "PREFACE", "CHAPTER 1.", "CHAPTER 2.", "CHAPTER 3."]
    kept, records = seg.exclude_non_story(units)
    assert [u["label"] for u in kept] == ["CHAPTER 1.", "CHAPTER 2.", "CHAPTER 3."]
    assert records["front_matter"]["excluded"] is True
    assert records["apparatus"]["excluded"] is True
    assert records["apparatus"]["apparatus_units"] == [
        {"index": 1, "label": "PREFACE", "words": 120}]
    # both opt back in independently
    kept_all, records_all = seg.exclude_non_story(
        units, include_front=True, include_apparatus=True)
    assert kept_all == units
    assert records_all["front_matter"]["excluded"] is False
    assert records_all["apparatus"]["excluded"] is False


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


# --- illustration blocks containing structure (Pride and Prejudice shakedown) --------

# The George Allen illustrated edition sets the only "Chapter I." heading of
# the text INSIDE an illustration block, and closes the book with THE / END
# inside another; both shapes verbatim from the shakedown copy.
PP_CHAPTER_BLOCK = "[Illustration: ·PRIDE AND PREJUDICE·\n\n\n\n\nChapter I.]"
PP_END_BLOCK = ("[Illustration:\n\n"
                "                                  THE\n"
                "                                  END\n"
                "                                   ]")


def test_illustration_block_preserves_chapter_heading():
    text = "front words here.\n\n" + PP_CHAPTER_BLOCK + "\n\nIt is a truth."
    out = seg.strip_gutenberg(text)
    assert "[Illustration" not in out
    assert "PRIDE AND PREJUDICE" not in out    # the caption itself still goes
    lines = out.split("\n")
    i = lines.index("Chapter I.")              # emitted standalone
    assert not lines[i - 1].strip() and not lines[i + 1].strip()


def test_illustration_block_preserves_end_marker():
    text = "closing prose.\n\n" + PP_END_BLOCK + "\n\nPRINTER COLOPHON"
    out = seg.strip_gutenberg(text)
    assert "[Illustration" not in out
    assert "END" in out.split("\n")            # the end-marker line survives
    assert "THE" not in out.split("\n")        # bare THE is not a marker line


def test_illustration_caption_only_blocks_still_removed_entirely():
    text = "a\n\n[Illustration: “He came down to see the place”]\n\nb"
    out = seg.strip_gutenberg(text)
    assert "came down" not in out
    assert out == "a\n\nb"


def test_pride_shape_chapter_one_recovered_end_to_end():
    text = (_words("front", 220) + "\n\n" + PP_CHAPTER_BLOCK + "\n\n"
            + _words("one", 80) + "\n\nCHAPTER II.\n\n" + _words("two", 80)
            + "\n\nCHAPTER III.\n\n" + _words("three", 80) + "\n\n"
            + PP_END_BLOCK + "\n")
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == [
        seg.FRONT_LABEL, "Chapter I.", "CHAPTER II.", "CHAPTER III."]
    assert out["units"][1]["text"].startswith("one0")
    assert out["units"][-1]["text"].rstrip().endswith("END")


# --- PRELUDE / FINALE structural units (Middlemarch shakedown) ------------------------

def test_prelude_and_finale_are_headings():
    # Middlemarch shape: without these patterns the Prelude fell into front
    # matter and the Finale fused into Chapter 86.
    assert seg.is_chapter_heading("PRELUDE.")
    assert seg.is_chapter_heading("FINALE.")


def test_prelude_and_finale_segment_as_units():
    text = ("PRELUDE.\n\n" + _words("pre", 80)
            + "\n\nCHAPTER I.\n\n" + _words("c1", 80)
            + "\n\nCHAPTER II.\n\n" + _words("c2", 80)
            + "\n\nFINALE.\n\n" + _words("fin", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == [
        "PRELUDE.", "CHAPTER I.", "CHAPTER II.", "FINALE."]
    assert out["units"][0]["text"] == _words("pre", 80)
    assert out["units"][-1]["text"] == _words("fin", 80)


# --- TOC screen body-exemption (War and Peace / Moonstone shakedown) -------------------

def test_toc_run_tail_with_substantial_body_survives():
    # War and Peace shape: the real Book One CHAPTER I sat at the END of the
    # TOC run (only "BOOK ONE: 1805", 3 words, between the last TOC entry and
    # it) with 2,015 words following, and was deleted with the run.
    toc = "\n\n".join(f"CHAPTER {r}" for r in ["I", "II", "III", "IV", "V"])
    text = ("Contents\n\n" + toc + "\n\nBOOK ONE: 1805\n\nCHAPTER I\n\n"
            + _words("b1", seg.TOC_BODY_EXEMPT_WORDS + 50)
            + "\n\nCHAPTER II\n\n" + _words("b2", 120)
            + "\n\nCHAPTER III\n\n" + _words("b3", 120))
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert [u["label"] for u in out["units"]] == ["CHAPTER I", "CHAPTER II",
                                                  "CHAPTER III"]
    assert out["units"][0]["text"].startswith("b10 ")  # the real chapter body
    # the TOC entries were screened and recorded
    screened = out["chapter_detection"]["screened_candidates"]
    assert [s["text"] for s in screened] == [
        "CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV", "CHAPTER V"]


def test_junction_run_keeps_the_substantial_chapter():
    # Moonstone shape: the junction run SECOND PERIOD. -> FIRST NARRATIVE. ->
    # CHAPTER I lost all three, fusing two narrators. The envelope headings
    # have thin bodies and may still drop; the CHAPTER I with a substantial
    # body must survive.
    text = ("CHAPTER XXII\n\n" + _words("a", 150)
            + "\n\nCHAPTER XXIII\n\n" + _words("b", 150)
            + "\n\nTHE END OF THE FIRST PERIOD.\n\n"
            "SECOND PERIOD.\n\n"
            "THE DISCOVERY OF THE TRUTH. (1848-1849.)\n\n"
            "The Events related in several Narratives.\n\n"
            "FIRST NARRATIVE.\n\n"
            "Contributed by Miss Clack; niece of the late Sir John Verinder.\n\n"
            "CHAPTER I\n\n" + _words("clack", seg.TOC_BODY_EXEMPT_WORDS + 100)
            + "\n\nCHAPTER II\n\n" + _words("clack2", 150))
    out = seg.segment(text, strategy="chapters")
    labels = [u["label"] for u in out["units"]]
    assert "CHAPTER I" in labels[2:]  # the narrator-opening chapter survives
    assert labels == ["CHAPTER XXII", "CHAPTER XXIII", "CHAPTER I", "CHAPTER II"]
    assert out["units"][2]["text"].startswith("clack0")
    screened = [s["text"] for s in out["chapter_detection"]["screened_candidates"]]
    assert screened == ["SECOND PERIOD.", "FIRST NARRATIVE."]


def test_toc_run_tail_below_exemption_bar_stays_screened():
    # The original protection stands: a TOC whose last entry is followed by a
    # sub-exemption gap (a dedication, 60 words) must screen completely; the
    # dedication is never absorbed as a fake chapter.
    entries = ["CHAPTER I", "CHAPTER II", "CHAPTER III", "CHAPTER IV"]
    toc = "\n\n".join(entries)
    dedication = _words("ded", 60)
    body = "\n\n".join(f"{e}\n\n" + _words(e.split()[-1].lower(), 120)
                       for e in entries)
    out = seg.segment(f"{toc}\n\n{dedication}\n\n{body}", strategy="chapters")
    assert [u["label"] for u in out["units"]] == entries
    assert all("ded0" not in u["text"] for u in out["units"])


def test_dense_latent_lines_screen_guard_passing_neighbors():
    # Middlemarch TOC shape: the packed " CHAPTER n." entries fail the
    # blank-context guards line by line, but the first and last entries of the
    # block (" PRELUDE.", " FINALE.") pass them. Density measured over ALL
    # heading-like lines screens those two; the body PRELUDE./FINALE. (with
    # real bodies) survive.
    toc = (" PRELUDE.\n\n BOOK I. MISS BROOKE.\n" +
           "\n".join(f" CHAPTER {r}." for r in ["I", "II", "III", "IV", "V"]) +
           "\n\n FINALE.")
    text = ("Title Page\n\nContents\n\n" + toc + "\n\nPRELUDE.\n\n"
            + _words("pre", 250) + "\n\nCHAPTER I.\n\n" + _words("c1", 120)
            + "\n\nCHAPTER II.\n\n" + _words("c2", 120)
            + "\n\nFINALE.\n\n" + _words("fin", 250))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == [
        "PRELUDE.", "CHAPTER I.", "CHAPTER II.", "FINALE."]
    assert out["units"][0]["text"].startswith("pre0")
    assert out["units"][-1]["text"].startswith("fin0")


# --- orphan heading-like tail lines (War and Peace shakedown) --------------------------

def test_orphan_heading_line_stripped_from_unit_tail():
    # "BOOK EIGHT: 1811 - 12" matches no heading pattern and landed verbatim
    # at the tail of the preceding chapter; it is stripped and recorded.
    text = ("CHAPTER I\n\n" + _words("a", 80)
            + "\n\nBOOK EIGHT: 1811 - 12\n\nCHAPTER II\n\n" + _words("b", 80)
            + "\n\nCHAPTER III\n\n" + _words("c", 80))
    out = seg.segment(text, strategy="chapters")
    u = out["units"][0]
    assert u["text"] == _words("a", 80)
    assert u["stripped_tail"] == ["BOOK EIGHT: 1811 - 12"]
    assert u["words"] == 80
    assert out["chapter_detection"]["stripped_tails"] == [
        {"unit_label": "CHAPTER I", "lines": ["BOOK EIGHT: 1811 - 12"]}]


def test_orphan_tail_strip_leaves_prose_and_final_unit_alone():
    # A prose tail (terminal punctuation) is never stripped, and the final
    # unit is never processed (THE END lives there).
    text = ("CHAPTER I\n\n" + _words("a", 80) + "\n\nHe went to Moscow.\n\n"
            "CHAPTER II\n\n" + _words("b", 80)
            + "\n\nCHAPTER III\n\n" + _words("c", 80) + "\n\nTHE END")
    out = seg.segment(text, strategy="chapters")
    assert out["units"][0]["text"].endswith("He went to Moscow.")
    assert "stripped_tail" not in out["units"][0]
    assert out["units"][-1]["text"].rstrip().endswith("THE END")


def test_orphan_tail_requires_standalone_line():
    # A short title-case line that is the wrapped tail of a paragraph (no
    # blank line above it) stays: only standalone orphans are stripped.
    body = _words("a", 60) + "\nAnd So It Ends"
    text = ("CHAPTER I\n\n" + body + "\n\nCHAPTER II\n\n" + _words("b", 80)
            + "\n\nCHAPTER III\n\n" + _words("c", 80))
    out = seg.segment(text, strategy="chapters")
    assert out["units"][0]["text"].endswith("And So It Ends")


# --- spelled-number BOOK/PART/VOLUME headings (War of the Worlds extraction) -----------

def test_spelled_number_book_headings():
    # Wells body headings verbatim: "BOOK ONE" / "BOOK TWO" matched nothing
    # and leaked into the adjacent chapter units.
    for h in ["BOOK ONE", "BOOK TWO", "PART THREE", "VOLUME TWENTY",
              "BOOK ONE.", "BOOK SEVENTEEN"]:
        assert seg.is_chapter_heading(h), h
    for h in ["BOOK TWENTYONE", "Book One", "BOOK ONE.—THE COMING OF THE MARTIANS"]:
        assert not seg.is_chapter_heading(h), h


def test_wells_book_headings_become_runt_dropped_boundaries():
    # The Wells junction shape verbatim: a book heading over its short title
    # line, directly over the next chapter. The book headings become
    # boundaries whose title-only bodies drop as runts, so no furniture leaks
    # into the chapter tails.
    text = ("BOOK ONE\nTHE COMING OF THE MARTIANS\n\n\n"
            "I.\nTHE EVE OF THE WAR.\n\n" + _words("one", 80) + "\n\n"
            "II.\nTHE FALLING STAR.\n\n" + _words("two", 80) + "\n\n\n"
            "BOOK TWO\nTHE EARTH UNDER THE MARTIANS.\n\n\n"
            "I.\nUNDER FOOT.\n\n" + _words("uf", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == ["I.", "II.", "I."]
    assert "BOOK TWO" not in out["units"][1]["text"]
    assert "MARTIANS" not in out["units"][1]["text"].split("THE FALLING STAR.")[1]
    runts = [r["label"] for r in out["chapter_detection"]["dropped_runts"]]
    assert runts == ["BOOK ONE", "BOOK TWO"]


# --- PREFACE / PREAMBLE named units (Bleak House / Woman in White evidence) ------------

def test_preface_and_preamble_are_headings():
    # Bleak House sets a bare PREFACE over the author's preface, which merged
    # into the front unit; a PREAMBLE is the same shape in other editions.
    for h in ["PREFACE", "PREAMBLE", "PREFACE."]:
        assert seg.is_chapter_heading(h), h
    # the TOC's title-case "Preface" line stays unmatched (conservative)
    assert not seg.is_chapter_heading("Preface")


def test_preface_unit_recovered():
    text = ("PREFACE\n\n" + _words("pref", 80) + "\n\n"
            "CHAPTER I\nIn Chancery\n\n" + _words("c1", 80) + "\n\n"
            "CHAPTER II\nIn Fashion\n\n" + _words("c2", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == ["PREFACE", "CHAPTER I",
                                                  "CHAPTER II"]
    assert out["units"][0]["text"] == _words("pref", 80)


# --- EPOCH ordinal units with THE prefix (Woman in White evidence) ----------------------

def test_epoch_headings_with_optional_the_prefix():
    # Collins body headings verbatim: "THE SECOND EPOCH" / "THE THIRD EPOCH".
    for h in ["THE SECOND EPOCH", "THE THIRD EPOCH", "FIRST EPOCH",
              "THE FIRST PERIOD."]:
        assert seg.is_chapter_heading(h), h
    for h in ["First Epoch", "THE EPOCH", "THE END OF THE FIRST PERIOD."]:
        assert not seg.is_chapter_heading(h), h


def test_epoch_junction_keeps_closing_unit_clean():
    # The Woman in White epoch junction verbatim: the epoch heading stands
    # over the narrator line, over the diary's "I". The epoch boundary's
    # narrator-only body drops as a runt; nothing leaks into chapter bodies.
    text = ("I\n\n" + _words("a", 80) + "\n\n"
            "II\n\n" + _words("b", 80) + "\n\n"
            "[The First Epoch of the Story closes here.]\n\n\n\n"
            "THE SECOND EPOCH\n\n"
            "THE STORY CONTINUED BY MARIAN HALCOMBE.\n\n\n\n"
            "I\n\nBLACKWATER PARK, HAMPSHIRE.\n\n" + _words("d", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == ["I", "II", "I"]
    assert out["units"][1]["text"].endswith(
        "[The First Epoch of the Story closes here.]")
    assert out["units"][2]["text"].startswith("BLACKWATER PARK, HAMPSHIRE.")


# --- multi-line orphan tail blocks (Wells / Collins extraction evidence) -----------------

def test_orphan_block_wells_shape_stripped():
    # War of the Worlds unit 17 tail verbatim: "BOOK TWO" over "THE EARTH
    # UNDER THE MARTIANS." The old single-line loop broke because the line
    # above the title line is non-blank, and on the trailing period.
    body = _words("a", 80) + "\n\n\nBOOK TWO\nTHE EARTH UNDER THE MARTIANS."
    out, stripped = seg._strip_orphan_tail(body)
    assert stripped == ["BOOK TWO", "THE EARTH UNDER THE MARTIANS."]
    assert out == _words("a", 80)


def test_orphan_block_collins_shape_stripped():
    # Woman in White junction scaffolding verbatim: the narrator heading over
    # the parenthetical subtitle; a trailing ")" is not prose-terminal here.
    body = (_words("a", 80) + "\n\nThe End of Hartright’s Narrative.\n\n\n"
            "THE STORY CONTINUED BY VINCENT GILMORE\n\n"
            "(of Chancery Lane, Solicitor)")
    out, stripped = seg._strip_orphan_tail(body)
    assert stripped == ["THE STORY CONTINUED BY VINCENT GILMORE",
                        "(of Chancery Lane, Solicitor)"]
    assert out.endswith("The End of Hartright’s Narrative.")


def test_orphan_block_sentence_line_needs_heading_like_top():
    # a lone all-caps sentence-punctuated line reads as prose and stays,
    # with or without a clean prose line above it
    for body in [_words("a", 80) + "\n\nHE SAID IT WAS OVER.",
                 _words("a", 80) + "\n\nTHE EARTH UNDER THE MARTIANS."]:
        out, stripped = seg._strip_orphan_tail(body)
        assert stripped == []
        assert out == body


def test_orphan_block_single_parenthetical_stays():
    # the ")" nuance is scoped to blocks: a lone parenthetical is not stripped
    body = _words("a", 80) + "\n\n(of Chancery Lane, Solicitor)"
    out, stripped = seg._strip_orphan_tail(body)
    assert stripped == []
    assert out == body


def test_orphan_block_end_marker_protected():
    # THE END must survive for the tail trimmer, alone or under a heading
    body = _words("a", 80) + "\n\nBOOK TWO\n\nTHE END"
    out, stripped = seg._strip_orphan_tail(body)
    assert stripped == []
    assert out == body


def test_orphan_block_requires_standalone_top():
    # four stacked structural lines exceed the block cap, so the top of the
    # collected block has a non-blank line above it: nothing is stripped
    body = (_words("a", 80) + "\n\nFIRST LINE HERE\nSECOND LINE HERE\n"
            "THIRD LINE HERE\nFOURTH LINE HERE")
    out, stripped = seg._strip_orphan_tail(body)
    assert stripped == []
    assert out == body


def test_collins_junction_scaffolding_stripped_end_to_end():
    # segment-level: the narrator heading + subtitle land at the tail of the
    # preceding chapter (the next "I" is the detected boundary) and are
    # stripped and recorded
    text = ("I\n\n" + _words("a", 80) + "\n\n"
            "II\n\n" + _words("b", 80) + "\n\nThe End of Hartright’s Narrative.\n\n\n"
            "THE STORY CONTINUED BY VINCENT GILMORE\n\n"
            "(of Chancery Lane, Solicitor)\n\n\n"
            "I\n\n" + _words("g", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == ["I", "II", "I"]
    assert out["units"][1]["text"].endswith("The End of Hartright’s Narrative.")
    assert out["chapter_detection"]["stripped_tails"] == [
        {"unit_label": "II",
         "lines": ["THE STORY CONTINUED BY VINCENT GILMORE",
                   "(of Chancery Lane, Solicitor)"]}]


# --- unmatched structural suspects (diagnostic; Collins invisibility evidence) ----------

def test_unmatched_suspects_flags_collins_narrator_headers():
    # Without overrides, the Woman in White narrator headings matched no
    # pattern and fused invisibly; the scan must surface them per unit.
    units = [{"index": 0, "text": _words("x", 40)},
             {"index": 1, "text": (_words("a", 40) + "\n\n"
                                   "THE STORY CONTINUED BY VINCENT GILMORE\n\n"
                                   + _words("b", 40) + "\n\n"
                                   "THE STORY CONCLUDED BY WALTER HARTRIGHT\n\n"
                                   + _words("c", 40))}]
    sus = seg.find_unmatched_suspects(units)
    assert [s["line"] for s in sus] == [
        "THE STORY CONTINUED BY VINCENT GILMORE",
        "THE STORY CONCLUDED BY WALTER HARTRIGHT"]
    assert [s["unit_index"] for s in sus] == [1, 1]
    # positions count every word above the line, including earlier suspects
    assert [s["position_words_into_unit"] for s in sus] == [40, 86]


def test_unmatched_suspects_ignore_epistolary_and_prose():
    # Dracula-style italic entry headers (leading underscore, trailing
    # period) must not flood the report; nor do prose lines, pattern-matched
    # headings, end markers, or non-standalone lines.
    body = ("_Dr. Seward’s Diary._\n\n" + _words("a", 30) + "\n\n"
            "_Mina Murray’s Journal._\n\n" + _words("b", 30) + "\n\n"
            "He crossed the room.\n\n"
            "CHAPTER II\n\n"
            "A Fine Title Case Prose Line Ending Here.\n\n"
            "WRAPPED CAPS LINE\n" + _words("c", 10) + "\n\nTHE END")
    sus = seg.find_unmatched_suspects([{"index": 0, "text": body}])
    assert sus == []


def test_unmatched_suspects_allow_trailing_parenthesis():
    # the ")" nuance from the orphan stripper applies here too
    body = _words("a", 20) + "\n\n(Housekeeper at Blackwater Park)\n\n" + _words("b", 20)
    sus = seg.find_unmatched_suspects([{"index": 3, "text": body}])
    assert sus == [{"unit_index": 3, "line": "(Housekeeper at Blackwater Park)",
                    "position_words_into_unit": 20}]


# --- windows-mode front-matter note (documented, no behavior change) --------------------

def test_windows_result_carries_front_matter_note():
    paras = "\n\n".join(" ".join(f"p{k}w{i}" for i in range(300)) for k in range(3))
    out = seg.segment(paras, strategy="windows", window_words=400)
    assert "front-matter exclusion" in out["note"]
    fallback = seg.segment("no chapters.\n\n" + _words("x", 100), strategy="chapters")
    assert fallback["strategy_used"].startswith("windows (fallback")
    assert "front-matter exclusion" in fallback["note"]


# --- wrapped/long headings (Monte Cristo Chapter 61 extraction evidence) ---------------

# The wild Dumas lead line verbatim: 66 chars, over the old magic 60 cap, its
# title wrapped onto a second line ("Peaches") in the Gutenberg copy.
DUMAS_WRAP_LEAD = "Chapter 61. How a Gardener May Get Rid of the Dormice that Eat His"


def test_long_heading_cap_admits_dumas_chapter_61():
    assert len(DUMAS_WRAP_LEAD) == 66          # over the old inline 60 cap
    assert len(DUMAS_WRAP_LEAD) <= seg.MAX_HEADING_CHARS
    assert seg.is_chapter_heading(DUMAS_WRAP_LEAD)
    # a prose line starting "Chapter 61 " (no dot after the number) never matches
    assert not seg.is_chapter_heading(
        "Chapter 61 was long past when the gardener came to the gate")


def test_wrapped_heading_dumas_shape_recovered():
    # The exact Dumas shape: the wrapped heading (leading space and all)
    # between real chapters. One heading, label joined across both lines, the
    # continuation line never becomes body text.
    text = ("Chapter 60. The Telegraph\n\n" + _words("sixty", 80) + "\n\n\n"
            " " + DUMAS_WRAP_LEAD + "\n Peaches\n\n" + _words("gard", 80)
            + "\n\n\nChapter 62. Ghosts\n\n" + _words("gh", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == [
        "Chapter 60. The Telegraph",
        DUMAS_WRAP_LEAD + " Peaches",
        "Chapter 62. Ghosts"]
    assert out["units"][0]["text"] == _words("sixty", 80)   # no fusion
    assert out["units"][1]["text"] == _words("gard", 80)    # no "Peaches" body line


def test_wrap_continuation_is_conservative():
    # the title-line convention is untouched: a bare heading over a title line
    # keeps the title in the body (no title text after the numbered prefix)
    assert not seg._wrap_continuation(["Chapter I.", "The Man Who Died", ""], 0)
    # a prose continuation (comma, lowercase words) is never joined
    assert not seg._wrap_continuation(
        [DUMAS_WRAP_LEAD, "peaches, though he tried,", ""], 0)
    assert not seg._wrap_continuation([DUMAS_WRAP_LEAD, "peaches and cream", ""], 0)
    # the wild continuation is; an optional terminal period is allowed
    assert seg._wrap_continuation([DUMAS_WRAP_LEAD, "Peaches", ""], 0)
    assert seg._wrap_continuation([DUMAS_WRAP_LEAD, "Peaches.", ""], 0)
    # no blank line after the continuation: hard-wrapped prose, not a title
    assert not seg._wrap_continuation([DUMAS_WRAP_LEAD, "Peaches", "and more prose"], 0)


def test_suspects_surface_wrapped_heading_lead_without_blank_below():
    # Fix facet (c): a non-standalone line that PREFIX-matches a numbered
    # heading but fails full heading detection (here: over the raised cap)
    # must surface as a suspect even though the wrap breaks blank-below.
    lead = "Chapter 71. " + " ".join(["Word"] * 15)
    assert len(lead) > seg.MAX_HEADING_CHARS
    assert not seg.is_chapter_heading(lead)
    body = _words("a", 40) + "\n\n" + lead + "\nContinuation Title\n\n" + _words("b", 40)
    sus = seg.find_unmatched_suspects([{"index": 2, "text": body}])
    assert sus == [{"unit_index": 2, "line": lead,
                    "position_words_into_unit": 40}]
    # hard-wrapped prose paragraphs starting "Chapter 61. ..." stay out: an
    # in-cap lead is excluded as a full pattern match (visible elsewhere),
    # an over-cap one dies on the connective-aware cap ratio
    for prose in [
        "Chapter 61. That was the year when the gardener first came down\n"
        "to the house and stayed.",
        "Chapter 61. That was the year when the gardener first came down to the "
        "house from town\nand stayed on.",
    ]:
        sus = seg.find_unmatched_suspects(
            [{"index": 0, "text": _words("a", 40) + "\n\n" + prose}])
        assert sus == [], prose


# --- BOOK the-ordinal headings (Tale of Two Cities extraction evidence) -----------------

TOTC_BOOKS = ["Book the First--Recalled to Life",
              "Book the Second--the Golden Thread",
              "Book the Third--the Track of a Storm"]


def test_totc_book_heading_forms():
    # the three wild headings verbatim, plus case/title variants
    for h in TOTC_BOOKS + ["BOOK THE FIRST", "BOOK THE FIRST--RECALLED TO LIFE",
                           "Book the First", "Book the Twentieth.",
                           "PART THE SECOND"]:
        assert seg.is_chapter_heading(h), h
    # negatives: prose after "the", and the pinned shakedown non-matches (the
    # trailing-title group is scoped to the the-ordinal branch so these stay out)
    for h in ["Book the passage for Tuesday", "Book One",
              "BOOK ONE.—THE COMING OF THE MARTIANS", "BOOK EIGHT: 1811 - 12",
              "The end of the first book."]:
        assert not seg.is_chapter_heading(h), h


def test_totc_shape_book_boundaries_end_to_end():
    # Compact Tale of Two Cities shape: an indented TOC repeating the Book
    # headings (leading spaces per the wild file) over packed chapter entries,
    # then the body. The TOC copies screen under the existing TOC handling;
    # the body Book headings become boundaries whose empty bodies drop as
    # runts (the sidecar-visible record); no Book furniture leaks into
    # chapter bodies. The paratext "The end of the first book." is prose-shaped
    # and stays in the preceding chapter tail (documented out of scope).
    def toc_section(book, chapters):
        entries = "\n".join(f"     CHAPTER {r}      T{r}" for r in chapters)
        return f"     {book}\n\n{entries}"
    toc = "\n\n".join(toc_section(b, ["I", "II"]) for b in TOTC_BOOKS)
    body = (TOTC_BOOKS[0] + "\n\n\n"
            "CHAPTER I.\nThe Period\n\n" + _words("c1", 80) + "\n\n"
            "CHAPTER II.\nThe Mail\n\n" + _words("c2", 80) + "\n\n"
            "The end of the first book.\n\n\n"
            + TOTC_BOOKS[1] + "\n\n\n"
            "CHAPTER I.\nFive Years Later\n\n" + _words("c3", 80) + "\n\n"
            + TOTC_BOOKS[2] + "\n\n\n"
            "CHAPTER I.\nIn Secret\n\n" + _words("c4", 80))
    text = ("A TALE OF TWO CITIES\n\nCONTENTS\n\n" + toc + "\n\n"
            + _words("scrap", 60) + "\n\n" + body)
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    assert [u["label"] for u in out["units"]] == [
        "CHAPTER I.", "CHAPTER II.", "CHAPTER I.", "CHAPTER I."]
    det = out["chapter_detection"]
    assert [s["text"] for s in det["screened_candidates"]] == TOTC_BOOKS
    assert det["dropped_runts"] == [
        {"label": b, "words": 0} for b in TOTC_BOOKS]
    assert all("Book the" not in u["text"] for u in out["units"])
    assert out["units"][1]["text"].endswith("The end of the first book.")


def test_totc_shape_long_inline_toc_no_longer_leaks_as_chapter_units():
    # Wild-file-scale regression for the actual Tale of Two Cities bug: Book
    # the Second/Third's TOC chapter lists are long enough (15 entries, ~75
    # words each) to clear MIN_UNIT_WORDS as a whole block, so BEFORE the fix
    # they survived as two spurious "chapter" units whose entire "body" was
    # nothing but their own TOC entries -- is_chapter_heading never matches
    # the inline "CHAPTER <n>      <title>" TOC format (a whitespace-gap, not
    # punctuation, separates the number from the title), so those entries
    # were invisible to the TOC density screen and never registered as a
    # dense run. Book the First's TOC block is kept short (6 entries, ~24
    # words) so it still drops as a plain runt, exactly like the compact
    # fixture above; this test isolates the long-block leak that the compact
    # one is too small to reproduce.
    def toc_section(book, n):
        entries = "\n".join(f"     CHAPTER {r}      Title Words {r}"
                            for r in range(1, n + 1))
        return f"     {book}\n\n{entries}"
    toc = "\n\n".join([toc_section(TOTC_BOOKS[0], 6),
                       toc_section(TOTC_BOOKS[1], 15),
                       toc_section(TOTC_BOOKS[2], 15)])
    # bodies exceed TOC_BODY_EXEMPT_WORDS so the real chapters keep their
    # existing body-exemption protection (War and Peace / Moonstone shape)
    body = (TOTC_BOOKS[0] + "\n\n\n"
            "CHAPTER 1.\nThe Period\n\n"
            + _words("c1", seg.TOC_BODY_EXEMPT_WORDS + 50) + "\n\n"
            + TOTC_BOOKS[1] + "\n\n\n"
            "CHAPTER 1.\nFive Years Later\n\n"
            + _words("c2", seg.TOC_BODY_EXEMPT_WORDS + 50) + "\n\n"
            + TOTC_BOOKS[2] + "\n\n\n"
            "CHAPTER 1.\nIn Secret\n\n"
            + _words("c3", seg.TOC_BODY_EXEMPT_WORDS + 50))
    text = "A TALE OF TWO CITIES\n\nCONTENTS\n\n" + toc + "\n\n" + body
    out = seg.segment(text, strategy="chapters")
    assert out["strategy_used"] == "chapters"
    # exactly the 3 real chapters: no spurious Book-heading units, and no
    # "(front)" unit either (the TOC dropped as a scrap like before the fix)
    assert [u["label"] for u in out["units"]] == [
        "CHAPTER 1.", "CHAPTER 1.", "CHAPTER 1."]
    assert "c10 " in out["units"][0]["text"]
    assert "c20 " in out["units"][1]["text"]
    assert "c30 " in out["units"][2]["text"]
    assert all("Title Words" not in u["text"] and "Book the" not in u["text"]
              for u in out["units"])


# --- suspects: connective exemption after a structural keyword (ToTC evidence) ----------

def test_suspect_cap_ratio_exempts_connectives_after_keyword():
    # Pre-fix, the plain 0.7 ratio missed 2 of the 3 ToTC Book headings
    # ("...First--Recalled to Life" 3/5, "...the Track of a Storm" 4/7);
    # with connectives out of the denominator all three read structural.
    for line in TOTC_BOOKS:
        assert seg._suspect_cap_ratio_ok(line), line


def test_suspects_surface_keyword_heading_with_connectives():
    # end-to-end on an unmatched analogue ("Last" is not a spelled ordinal,
    # so no pattern matches and the suspect scan is the only visibility)
    line = "Part the Last--the Reckoning of a Storm"
    assert not seg.is_chapter_heading(line)
    assert seg._is_suspect_line(line)
    body = _words("a", 30) + "\n\n" + line + "\n\n" + _words("b", 30)
    sus = seg.find_unmatched_suspects([{"index": 5, "text": body}])
    assert sus == [{"unit_index": 5, "line": line,
                    "position_words_into_unit": 30}]


def test_suspects_do_not_flood_on_prose_book_or_the_lines():
    # ordinary prose starting with a keyword or "The" stays out; the
    # prose-shaped paratext "The end of the first book." is documented out of
    # scope (terminal period, lowercase words: catching it would flood)
    for line in ["Book the passage for Tuesday and send word ahead",
                 "The end of the first book.",
                 "The mail came late over the hill that night"]:
        assert not seg._is_suspect_line(line), line


# --- NOTE structural units (Jane Eyre extraction evidence) ------------------------------

def test_note_heading_forms():
    for h in ["NOTE TO THE THIRD EDITION", "NOTE", "NOTE."]:
        assert seg.is_chapter_heading(h), h
    # the colon form is an aside, not a heading; mixed case and other NOTE
    # phrasings stay out (uppercase-led, EDITION-suffixed form only)
    for h in ["NOTE: this is an aside", "Note to the third edition",
              "NOTE TO THE READER"]:
        assert not seg.is_chapter_heading(h), h


def test_jane_eyre_note_edition_unit_recovered():
    # the Jane Eyre shape: PREFACE, then the standalone third-edition note,
    # then the chapters; the note becomes its own unit
    text = ("PREFACE\n\n" + _words("pref", 80) + "\n\n"
            "NOTE TO THE THIRD EDITION\n\n" + _words("note", 80) + "\n\n"
            "CHAPTER I\n\n" + _words("c1", 80) + "\n\n"
            "CHAPTER II\n\n" + _words("c2", 80))
    out = seg.segment(text, strategy="chapters")
    assert [u["label"] for u in out["units"]] == [
        "PREFACE", "NOTE TO THE THIRD EDITION", "CHAPTER I", "CHAPTER II"]
    assert out["units"][1]["text"] == _words("note", 80)


# --- page-anchor noise strip (Monte Cristo extraction evidence) --------------------------

def test_strip_page_anchors_wild_shape():
    # the wild shape: prose, blank, anchor, blank, prose; anchors go, prose
    # stays, leftover blank runs collapse
    text = (_words("a", 30) + "\n\n0185m\n\n" + _words("b", 30)
            + "\n\n0023m\n\n" + _words("c", 30))
    out, removed = seg.strip_page_anchors(text)
    assert removed == 2
    assert "0185m" not in out and "0023m" not in out
    assert "\n\n\n" not in out
    assert out.startswith("a0 ") and out.endswith(" c29")


def test_strip_page_anchors_conservative_guards():
    # a prose line ending "...100m", sub-3-digit standalone lines (the poem /
    # measurement case behind the 3+ digit floor), and a non-standalone
    # anchor are all untouched
    for text in [_words("a", 10) + " it measured 100m",
                 _words("a", 10) + "\n\n5m\n\n" + _words("b", 10),
                 _words("a", 10) + "\n\n12m\n\n" + _words("b", 10),
                 _words("a", 10) + "\n0185m\n" + _words("b", 10)]:
        out, removed = seg.strip_page_anchors(text)
        assert removed == 0
        assert out == text


def test_segment_records_page_anchor_strip():
    text = ("CHAPTER 1.\n\n" + _words("a", 80) + "\n\n0185m\n\n" + _words("a2", 20)
            + "\n\nCHAPTER 2.\n\n" + _words("b", 80)
            + "\n\nCHAPTER 3.\n\n" + _words("c", 80))
    out = seg.segment(text, strategy="chapters")
    assert out["page_anchor_lines_removed"] == 1
    assert all("0185m" not in u["text"] for u in out["units"])
    assert out["units"][0]["words"] == 100  # the anchor is not a word
    # trimming disabled: nothing stripped, count stays 0
    out2 = seg.segment(text, strategy="chapters", trim_gutenberg=False)
    assert out2["page_anchor_lines_removed"] == 0
    assert any("0185m" in u["text"] for u in out2["units"])


def test_truncate_middle_short_text_untouched():
    text = " ".join(f"w{i}" for i in range(100))
    assert seg.truncate_middle(text, 50, 50) == text  # within the +200 grace


def test_truncate_middle_long_text_keeps_head_and_tail():
    words = [f"w{i}" for i in range(1000)]
    out = seg.truncate_middle(" ".join(words), 100, 100)
    assert out.startswith("w0 ")
    assert out.endswith(" w999")
    assert "[... middle omitted: about 800 words ...]" in out
