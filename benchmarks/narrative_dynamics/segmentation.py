"""Segmentation layer: arbitrary-length text to ordered analysis units. Pure, no LLM.

Every narrative-dynamics metric consumes the same unit list produced here, so the
segmentation policy is decided once per document and recorded in the output. Three
selectable strategies:

* ``chapters``: detect Gutenberg-style chapter headings (``CHAPTER IV``,
  ``Chapter 12.``, bare roman numerals, ``PROLOGUE``/``EPILOGUE``/``PRELUDE``/
  ``FINALE``...). Adapted from the StoryDaemon masters-study extraction scripts.
  Falls back to fixed windows when a text has too few detectable headings, and
  says so in the result. Architecture note (2026-07): the heading heuristics are
  a PROPOSER for the one-time extraction step
  (``python -m benchmarks.narrative_dynamics.extract``), not a forever-parser;
  ongoing analysis is meant to consume the canonical Markdown that extraction
  emits, via the ``md`` strategy below.
* ``windows``: fixed ~1500-word windows snapped to paragraph boundaries (1500
  words is this repo's existing benchmark framing for one story-sized unit).
  Windows-mode caveat: it performs NO front-matter exclusion; any title page /
  TOC / preface is scored inside the first window(s). The segment() result
  carries a ``note`` saying so. (Extraction-first makes this path secondary.)
* ``md``: canonical-Markdown ingestion, auto-selected for ``.md`` inputs. Splits
  on top-level ``# `` headings and reads the extract provenance header; NO
  heuristics. This is the permanent analysis path for extracted books.

Plus a Project Gutenberg frontmatter/license trimmer, since the masters corpus
files carry the header and license tail as downloaded, a conservative
trailing back-matter trimmer for publisher catalogs printed INSIDE the
Gutenberg markers (see ``trim_trailing_backmatter``), and a page-anchor noise
strip for standalone HTML-conversion plate anchors like ``0185m`` (see
``strip_page_anchors``).

A unit is a dict: ``{"index": int, "label": str, "text": str, "words": int}``.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_WORDS = 1500  # the repo's established story-unit framing
MIN_UNIT_WORDS = 50          # drop degenerate units (heading-only fragments)
MIN_CHAPTERS = 3             # fewer detected chapters than this -> fall back
MAX_TITLE_LINE_CHARS = 60    # a chapter-title line under a heading stays short
# Longest line is_chapter_heading will consider. Was a magic 60 inline, which
# made long headings doubly invisible: Monte Cristo's "Chapter 61. How a
# Gardener May Get Rid of the Dormice that Eat His" (66 chars, its title
# wrapped onto a second line) matched nothing, fused into its neighbor, AND
# stayed off the suspects report. 80 clears the wild case with margin while
# still rejecting typical hard-wrapped prose (Gutenberg wraps near 72).
MAX_HEADING_CHARS = 80
TOC_MIN_RUN = 3              # 3+ near-adjacent heading candidates = a contents list
FRONT_LABEL = "(front)"      # label of the kept pre-first-heading unit

# A screened-run member survives when this many words follow it before the next
# heading-like line: a real chapter body, not a TOC gap. Deliberately larger
# than MIN_UNIT_WORDS: the last entry of a contents list is followed by real
# front matter (a dedication easily clears 50 words) and must still screen, or
# it absorbs that front matter as a fake chapter. Calibration from the
# shakedown corpus: the largest TOC-to-body front-matter gap is ~60 words (the
# dedication shape in the regression fixtures); the smallest real chapter body
# at a run tail is 171 words (the Moonstone Prologue's "I"). Wild findings the
# exemption recovers: War and Peace's Book One CHAPTER I sat at the end of the
# TOC run with 2,015 words following and was deleted; the Moonstone junction
# run SECOND PERIOD -> FIRST NARRATIVE -> CHAPTER I lost all three, fusing two
# narrators (the thin-bodied envelope headings may still drop; the CHAPTER I
# with a substantial body must survive).
TOC_BODY_EXEMPT_WORDS = MIN_UNIT_WORDS * 2

# NON_STORY: two parallel categories of labeled unit that extraction keeps
# (nothing is silently dropped from the canonical Markdown) but that the
# SCORING layer excludes by default because they are not narrative: the
# pre-first-heading ``(front)`` unit (see ``FRONT_LABEL``/``exclude_front_matter``)
# and author-apparatus units such as prefaces or footnotes (see
# ``is_apparatus_label``/``exclude_apparatus``). Both are opt-in-able back into
# scoring (``--include-front`` / ``--include-apparatus``) and both record their
# exclusion in the sidecar so it is never silent.

# --- Gutenberg trimming -----------------------------------------------------------

_GUT_START = re.compile(r"^\s*\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_GUT_END = re.compile(r"^\s*\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_ILLUSTRATION = re.compile(r"\[Illustration:?[^\]]*\]", re.S)


def _replace_illustration(m: "re.Match[str]") -> str:
    """Replacement for one ``[Illustration: ...]`` block: keep structural lines.

    Shakedown evidence (Pride and Prejudice, George Allen illustrated edition):
    the block ``[Illustration: ·PRIDE AND PREJUDICE· ... Chapter I.]`` carries
    the only "Chapter I." heading in the text, and ``[Illustration: THE END]``
    carries the end marker; deleting the blocks whole lost Chapter I into the
    front matter and erased the end marker. So before dropping a block, its
    inner text is scanned for lines matching the chapter-heading patterns or
    the end-marker pattern; those lines are emitted (blank-separated, so the
    heading context guards still see them) in place of the block, and only the
    rest of the block is deleted.
    """
    inner = m.group(0)
    inner = inner[len("[Illustration"):].lstrip(":")
    if inner.endswith("]"):
        inner = inner[:-1]
    kept = [line.strip() for line in inner.split("\n")
            if is_chapter_heading(line) or _END_MARKER.match(line)]
    if not kept:
        return ""
    return "\n\n" + "\n\n".join(kept) + "\n\n"


def strip_gutenberg(text: str) -> str:
    """Trim Project Gutenberg frontmatter and license tail; normalize newlines.

    Keeps only the body between the ``*** START OF ... ***`` and
    ``*** END OF ... ***`` markers (either marker may be absent). Also removes
    ``[Illustration: ...]`` blocks (preserving any structural lines inside
    them, see ``_replace_illustration``) and collapses 3+ blank lines.
    """
    text = text.replace("\r\n", "\n")
    m = _GUT_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUT_END.search(text)
    if m:
        text = text[:m.start()]
    text = _ILLUSTRATION.sub(_replace_illustration, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- trailing back-matter ----------------------------------------------------------

# An explicit end-marker on a line of its own (decorative whitespace, asterisks
# or italic underscores allowed around it). Explicit case forms, not re.I: a
# lowercase "the end" mid-line is prose, never a marker.
_END_MARKER = re.compile(
    r"^[ \t*_]*(?:THE\s+END|The\s+End|FINIS|Finis|END)[.!]?[ \t*_]*$", re.M)

TAIL_MARKER_ZONE = 0.95      # marker must sit in the last ~5% of the text
TAIL_NONPROSE_RATIO = 0.5    # at least half the post-marker lines look non-narrative
TAIL_MIN_VOCAB_HITS = 2      # and catalog vocabulary must corroborate
_NONPROSE_MAX_CHARS = 40     # a line this short is a title/entry, not prose
_TITLECASE_RATIO = 0.7       # share of capitalized words that reads as a title line

# Publisher/catalog vocabulary (lowercase substring match). Small documented
# list drawn from the shakedown evidence: the Grosset & Dunlap catalog printed
# after THE END inside Dracula's Gutenberg markers.
_CATALOG_VOCAB = (
    "publisher", "catalog", "books are sold", "ask for", "wrapper",
    "complete list", "titles", "grosset", "dunlap",
)

# A printer's colophon (Pride and Prejudice/Chiswick Press evidence: "END"
# then "CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO." / "TOOKS COURT,
# CHANCERY LANE, LONDON.") is a second, publisher-vocabulary-free back-matter
# shape: unlike a catalog blurb-plus-title-list (long, needs the vocabulary
# corroboration above), a colophon is just a couple of short non-narrative
# lines naming a press/printer and an address, standing alone as the very
# last thing in the document. Key on that structural texture only (block
# size, non-narrative-ness, multi-word lines), never on a specific press
# name, so it generalizes to any printed edition's imprint.
MAX_COLOPHON_LINES = 3            # an imprint block stays this short
_COLOPHON_MIN_WORDS_PER_LINE = 2  # an address/imprint reads as a phrase, not
                                   # a bare word-list entry (catalog titles,
                                   # single capitalized items)


def _looks_like_colophon(lines: list[str]) -> bool:
    """General texture of a printer's colophon trailing the end-marker: a
    short block of standalone non-narrative lines, each carrying more than
    one word. Deliberately requires no vocabulary match (unlike the catalog
    path) so it generalizes across printers/publishers; the line-count cap
    and per-line word-count floor are what keep it from also swallowing a
    longer catalog blurb or a bare list of single-word titles."""
    return (bool(lines) and len(lines) <= MAX_COLOPHON_LINES
            and all(_looks_non_narrative(l)
                    and len(l.split()) >= _COLOPHON_MIN_WORDS_PER_LINE
                    for l in lines))


def _looks_non_narrative(line: str) -> bool:
    """Simple per-line signal: short, all-caps, or title-case lines are the
    texture of a publisher catalog / title list, not of prose."""
    s = line.strip()
    if len(s) < _NONPROSE_MAX_CHARS:
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    words = s.split()
    return sum(1 for w in words if w[:1].isupper()) / len(words) >= _TITLECASE_RATIO


def trim_trailing_backmatter(text: str) -> tuple[str, Optional[dict]]:
    """Trim publisher back-matter after an explicit end-marker line.

    Shakedown evidence: Dracula's Gutenberg body carries a Grosset & Dunlap
    catalog AFTER "THE END" but INSIDE the ``*** END`` marker, so
    ``strip_gutenberg`` keeps it and 238 words of ads contaminated the final
    chapter. Conservative by design; a trim happens only when every signal
    agrees:

    * the LAST end-marker line (``THE END`` / ``FINIS`` / standalone ``END``)
      sits in the final ~5% of the text (a mid-text marker is never trusted),
    * at least ``TAIL_NONPROSE_RATIO`` of the post-marker lines read as
      non-narrative (short / title-case / all-caps), and
    * EITHER at least ``TAIL_MIN_VOCAB_HITS`` distinct catalog-vocabulary
      terms corroborate (a publisher's catalog blurb + title list), OR the
      post-marker lines are a short printer's-colophon-shaped block (see
      ``_looks_like_colophon``) -- a press/printer imprint carries no
      catalog vocabulary at all, so it needs its own, purely structural,
      corroboration instead.

    In any doubt the text is returned unchanged: a false trim (losing real
    prose, e.g. an epilogue or author's note after the marker) is worse than
    keeping ads. Material BEFORE the marker (e.g. Dracula's closing NOTE by
    Harker) is untouched. A catalog trim keeps the marker line itself (it
    reads as the story's own closing note -- c.f. Emma's bare "FINIS" with
    nothing after it at all, left untouched by the early-return above -- with
    only publisher ephemera following); a colophon trim removes the marker
    too, since there it is paired with a press/printer imprint and both are
    the printer's apparatus rather than the author's, so the last real prose
    line becomes the true close. Returns ``(text, note)`` where ``note`` is a
    sidecar-ready record of what was trimmed and why, or None when nothing
    was trimmed.
    """
    matches = list(_END_MARKER.finditer(text))
    if not matches:
        return text, None
    m = matches[-1]
    if m.start() < TAIL_MARKER_ZONE * len(text):
        return text, None  # marker not in the tail: in doubt, keep everything
    after = text[m.end():]
    lines = [l for l in after.split("\n") if l.strip()]
    if not lines:
        return text, None  # nothing after the marker
    nonprose_ratio = sum(1 for l in lines if _looks_non_narrative(l)) / len(lines)
    vocab_hits = [t for t in _CATALOG_VOCAB if t in after.lower()]
    is_colophon = _looks_like_colophon(lines)
    if nonprose_ratio < TAIL_NONPROSE_RATIO:
        return text, None  # could be prose (epilogue, note): in doubt, keep
    catalog_trim = len(vocab_hits) >= TAIL_MIN_VOCAB_HITS
    if not catalog_trim and not is_colophon:
        return text, None  # no catalog vocab and not colophon-shaped: keep
    cut = m.end() if catalog_trim else m.start()
    note = {
        "marker": text[m.start():m.end()].strip(),
        "trimmed_words": word_count(after),
        "trimmed_lines": len(lines),
        "non_narrative_line_ratio": round(nonprose_ratio, 3),
        "vocabulary_hits": vocab_hits,
        "colophon": is_colophon,
        "marker_kept": catalog_trim,
        "reason": (
            "end-marker line in the last 5% of the text followed by "
            "non-narrative catalog material; trimmed at the marker "
            "(marker line kept)" if catalog_trim else
            "printer's colophon block (short, non-narrative press/printer "
            "imprint, no catalog vocabulary) after the end-marker line; "
            "marker and colophon both trimmed as printer's apparatus"),
    }
    logger.info("trailing back-matter trimmed: %s", note)
    return text[:cut].rstrip(), note


# --- page-anchor noise -------------------------------------------------------------

# Standalone lines like "0185m": HTML-conversion plate anchors, not prose.
# Monte Cristo extraction evidence: the shakedown copy carries 443 of them
# (zero-padded, 4 to 6 digits in the wild), each alone between blank lines,
# and they contaminated unit bodies and word counts. Judgment call on the
# digit floor: 3+ digits required, so a legitimate short line like "5m" or
# "12m" (a measurement, a poem line) is never clipped; every wild anchor has
# at least 4 digits, so 3 keeps a margin without loosening toward prose.
_PAGE_ANCHOR = re.compile(r"^\d{3,}m$")


def strip_page_anchors(text: str) -> tuple[str, int]:
    """Remove standalone page-anchor lines; return ``(text, lines_removed)``.

    Conservative by design: the line must match ``_PAGE_ANCHOR`` exactly (no
    surrounding text, no stripping of indentation) and be standalone (blank
    line or text edge above AND below), so a prose line ending "...100m" or
    an anchor-like token inside a paragraph is never touched. Runs during
    trimming, alongside the Gutenberg strip; callers record the count in the
    sidecar/warnings so the removal is never silent.
    """
    lines = text.split("\n")
    kept: list[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if (_PAGE_ANCHOR.match(line)
                and (i == 0 or not lines[i - 1].strip())
                and (i == len(lines) - 1 or not lines[i + 1].strip())):
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return text, 0
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)), removed


# --- paragraphs --------------------------------------------------------------------

def split_paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs, hard-wrapped lines joined, whitespace squashed."""
    paras = []
    for block in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s+", " ", block).strip()
        if p:
            paras.append(p)
    return paras


def word_count(text: str) -> int:
    return len(text.split())


# --- chapter detection -------------------------------------------------------------

_ROMAN = r"[IVXLCDM]+"
# Spelled-out number words accepted in BOOK/PART/VOLUME headings (uppercase
# only, conservative). Wells extraction evidence: the body headings
# "BOOK ONE" / "BOOK TWO" matched nothing and leaked into chapter units.
_SPELLED_NUMBER = ("ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
                   "ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|"
                   "SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY")
# Title-Case ordinal words accepted after a "the"/"THE" in BOOK/PART/VOLUME
# headings (uppercase ordinals are already covered by the THE [A-Z]+ branch).
# Tale of Two Cities extraction evidence: the body headings "Book the
# First--Recalled to Life" (and Second/Third) matched nothing and leaked
# silently. Range mirrors the uppercase spelled numbers above (up to twenty).
_SPELLED_ORDINAL_TITLE = ("First|Second|Third|Fourth|Fifth|Sixth|Seventh|"
                          "Eighth|Ninth|Tenth|Eleventh|Twelfth|Thirteenth|"
                          "Fourteenth|Fifteenth|Sixteenth|Seventeenth|"
                          "Eighteenth|Nineteenth|Twentieth")
_CHAPTER_PATTERNS = [
    # CHAPTER IV / Chapter 12. / CHAPTER THE FIRST, optionally followed by a
    # title after a dot, colon, hyphen, or em dash (the em dash in the class
    # below is a literal matched in Gutenberg headings, e.g. Eddison's)
    re.compile(rf"^(?:CHAPTER|Chapter)\s+(?:{_ROMAN}|\d+|[A-Z][A-Za-z-]+)\.?(?:\s*[.:—-]\s*\S.*)?$"),
    # BOOK II / PART THE SECOND / VOLUME I / BOOK ONE (treated as boundaries
    # too), plus the Tale of Two Cities forms: a mixed-case "the" with a
    # spelled ordinal and an optional trailing title mirroring the CHAPTER
    # suffix ("Book the First--Recalled to Life"; the "--" is covered because
    # "-" matches the separator class and the second "-" starts the title).
    # The trailing-title group is deliberately scoped to the the-ordinal
    # branch only: the roman/digit/spelled-cardinal branches stay title-free
    # because the Wells TOC form "BOOK ONE.—THE COMING OF THE MARTIANS" and
    # the War and Peace orphan "BOOK EIGHT: 1811 - 12" are pinned NON-matches
    # (see the shakedown fixtures; both must keep failing every pattern).
    re.compile(rf"^(?:BOOK|PART|VOLUME|Book|Part|Volume)\s+"
               rf"(?:{_ROMAN}|\d+|{_SPELLED_NUMBER}"
               rf"|(?:THE|the)\s+(?:[A-Z]+|{_SPELLED_ORDINAL_TITLE})"
               rf"(?:\s*[.:—-]\s*\S.*)?"
               rf")\.?$"),
    # bare roman numeral heading lines: "IV." / "XII"
    re.compile(rf"^{_ROMAN}\.?$"),
    # numbered headings: "12." on a line of its own
    re.compile(r"^\d{1,3}\.?$"),
    # named structural units. PRELUDE/FINALE per the Middlemarch shakedown:
    # without them the Prelude fell into front matter and the Finale fused
    # into Chapter 86. PREFACE per Bleak House (the preface merged into the
    # front unit); PREAMBLE is the same shape in other editions. EPOCH (with
    # an optional THE prefix on the ordinal form) per The Woman in White:
    # the body headings are "THE SECOND EPOCH" / "THE THIRD EPOCH".
    # NOTE per Jane Eyre: the standalone "NOTE TO THE THIRD EDITION" heading
    # matched nothing. Uppercase-led like the rest of this alternation, and
    # only the bare form or the "TO THE <words> EDITION" form; a colon form
    # ("NOTE: ...") is an aside, not a heading, and must not match.
    re.compile(r"^(?:PROLOGUE|EPILOGUE|PRELUDE|FINALE|INTRODUCTION|CONCLUSION|"
               r"PREFACE|PREAMBLE|NOTE(?:\s+TO\s+THE\s+[A-Z]+(?:\s+[A-Z]+)*\s+EDITION)?|"
               r"(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)"
               r"\s+(?:NARRATIVE|PERIOD|BOOK|PART|EPOCH))\.?$"),
]


def is_chapter_heading(line: str) -> bool:
    """True when a stripped line looks like a chapter-style structural heading."""
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return False
    return any(p.match(s) for p in _CHAPTER_PATTERNS)


# A numbered CHAPTER heading prefix: "Chapter 61." / "CHAPTER IV." followed by
# a space or the line end. Used for the two wrapped-heading facets (Monte
# Cristo evidence, Chapter 61): joining a wrapped title continuation line into
# the heading, and surfacing over-long or non-standalone leads as suspects.
_NUMBERED_HEADING_PREFIX = re.compile(
    rf"^(?:CHAPTER|Chapter)\s+(?:\d+|{_ROMAN})\.(?:\s|$)")


def _wrap_continuation(lines: list[str], i: int) -> bool:
    """True when ``lines[i + 1]`` is the wrapped continuation of a numbered
    heading-with-title at ``lines[i]``.

    Monte Cristo evidence: "Chapter 61. How a Gardener May Get Rid of the
    Dormice that Eat His" wraps its title onto a second line ("Peaches");
    the two lines are ONE heading and the continuation must not become body
    text. Deliberately conservative, so the title-line convention (a bare
    "Chapter I." over "The Man Who Died", where the title line stays in the
    body) is untouched: the lead line must be a full pattern match that
    ALREADY carries title text after the numbered prefix, and the next line
    must be short, title-cased (lowercase connectives allowed), free of prose
    punctuation except an optional terminal period, and followed by a blank
    line or the text edge.
    """
    s = lines[i].strip()
    m = _NUMBERED_HEADING_PREFIX.match(s)
    if not m or not s[m.end():].strip() or not is_chapter_heading(s):
        return False
    if i + 1 >= len(lines):
        return False
    c = lines[i + 1].strip()
    if not c or len(c) > MAX_TITLE_LINE_CHARS or is_chapter_heading(c):
        return False
    if i + 2 < len(lines) and lines[i + 2].strip():
        return False  # continuation must be followed by a blank line / edge
    core = c[:-1] if c.endswith(".") else c
    if not core or core[-1] in _TERMINAL_PUNCT:
        return False  # prose-punctuated: only a terminal period is allowed
    alpha_words = [w for w in core.split() if w[:1].isalpha()]
    if not alpha_words:
        return False
    return all(w[:1].isupper() or w.lower() in _HEADING_CONNECTIVES
               for w in alpha_words)


def _is_title_line(lines: list[str], i: int) -> bool:
    """True when ``lines[i]`` reads as a standalone chapter-title line.

    Used for headings with no blank line under them: many Gutenberg editions set
    ``Chapter I.`` immediately over the chapter title (``The Man Who Died``). A
    title line is short, not itself a heading, and is followed by a blank line
    or the text edge; a hard-wrapped prose paragraph continues on the next
    line, so prose starting right under a candidate never qualifies.
    """
    s = lines[i].strip()
    if not s or len(s) > MAX_TITLE_LINE_CHARS or is_chapter_heading(s):
        return False
    return i + 1 >= len(lines) or not lines[i + 1].strip()


def _match_override_lines(lines: list[str], overrides) -> dict[int, dict]:
    """Line index -> boundary-override entry, by exact stripped-line equality.

    ``overrides`` is the reviewed data list from an extract ``--boundaries``
    file: dicts with a required ``match`` (exact stripped line text) and an
    optional ``label`` (defaults to the matched text).
    """
    if not overrides:
        return {}
    by_text: dict[str, dict] = {}
    for entry in overrides:
        by_text.setdefault(entry["match"], entry)
    return {i: by_text[line.strip()] for i, line in enumerate(lines)
            if line.strip() in by_text}


# TOC-style inline chapter-entry line: keyword + number + a MULTI-SPACE
# columnar gap + a title, all on one line ("CHAPTER I      Five Years
# Later"). ``is_chapter_heading`` deliberately does not match this shape (its
# own inline-title form requires a punctuation separator: ". "/": "/"-"/"--",
# see the CHAPTER pattern above), so a TOC block built from these lines is
# invisible to the density measure in ``_screen_toc_runs``: the entries read
# as ordinary prose words between the surrounding Book headings, the run
# never registers as dense, and the Book heading survives with the whole
# entry list as its fake body. Tale of Two Cities evidence: Book the
# Second/Third's 24/15-line inline TOC entries (2-4 space gaps in the wild
# file) each contribute a few "prose" words, comfortably clearing
# MIN_UNIT_WORDS across the block, while Book the First's shorter 6-entry
# block stays under it and already dropped as a runt. The 2-space floor
# matches the tightest gap actually observed ("CHAPTER XVIII  Nine Days");
# a single space is the ordinary "keyword number title" prose/heading shape
# and must not match. This predicate feeds ONLY the density measurement
# (below); it never adds a line to ``hits``, so a real chapter heading using
# this columnar layout is unaffected unless it is also short enough to fail
# the body-exemption, the same safety net every other TOC case relies on.
_TOC_INLINE_ENTRY = re.compile(
    rf"^(?:CHAPTER|Chapter|BOOK|Book|PART|Part|VOLUME|Volume)\s+"
    rf"(?:{_ROMAN}|\d+|{_SPELLED_NUMBER})\.?\s{{2,}}\S.*$")


def _is_toc_entry_line(line: str) -> bool:
    """True for a TOC-style inline chapter/book entry (see ``_TOC_INLINE_ENTRY``)."""
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return False
    return bool(_TOC_INLINE_ENTRY.match(s))


def _screen_toc_runs(lines: list[str], hits: list[int],
                     report: Optional[dict] = None,
                     extra_latent: Optional[set[int]] = None) -> list[int]:
    """Drop dense runs of heading-like lines (a table of contents, not structure).

    Density is measured over ALL lines matching the heading patterns, PLUS
    TOC-style inline entry lines (not only the context-guard-passing
    candidates): a Contents block whose entries sit on adjacent lines fails
    the blank-line guards entry by entry, yet its first and last entries can
    pass them and read as boundaries (the Middlemarch TOC: " PRELUDE." and
    " FINALE." pass, the 94 packed " CHAPTER n." lines between them do not).
    A run is ``TOC_MIN_RUN`` or more consecutive heading-like lines with
    fewer than ``MIN_UNIT_WORDS`` words between each adjacent pair.

    Body exemption (War and Peace / Moonstone shakedown): a candidate inside a
    dense run is NOT dropped when at least ``TOC_BODY_EXEMPT_WORDS`` of text
    follow it before the next heading-like line (or the end): that is a real
    chapter body, not a TOC gap. W&P's Book One CHAPTER I ended the TOC run
    with 2,015 words following; Moonstone's junction run SECOND PERIOD ->
    FIRST NARRATIVE -> CHAPTER I lost the narrator-opening CHAPTER I (the two
    thin-bodied envelope headings may still drop, which is acceptable). The
    *last* entry of a plain TOC stays screened: the front matter following it
    (dedication, preface scraps) stays under the exemption bar, so it cannot
    absorb that front matter as a fake chapter.

    ``extra_latent`` adds line indices that count as heading-like for the
    density measurement even though no pattern matches them: boundary-override
    lines, whose TOC copies would otherwise read as boundaries.
    """
    if not hits:
        return hits
    latent = [i for i, line in enumerate(lines)
              if is_chapter_heading(line) or _is_toc_entry_line(line)
              or (extra_latent and i in extra_latent)]
    if len(latent) < TOC_MIN_RUN:
        return hits
    dense: set[int] = set()
    run = [0]

    def flush(run: list[int]) -> None:
        if len(run) >= TOC_MIN_RUN:
            dense.update(latent[k] for k in run)

    for k in range(1, len(latent)):
        between = " ".join(lines[latent[k - 1] + 1 : latent[k]])
        if word_count(between) < MIN_UNIT_WORDS:
            run.append(k)
        else:
            flush(run)
            run = [k]
    flush(run)

    def following_body_words(i: int) -> int:
        nxt = next((j for j in latent if j > i), len(lines))
        return word_count(" ".join(lines[i + 1 : nxt]))

    kept, screened = [], []
    for i in hits:
        if i in dense and following_body_words(i) < TOC_BODY_EXEMPT_WORDS:
            screened.append(i)
        else:
            kept.append(i)
    if report is not None and screened:
        report["screened_candidates"] = [
            {"line": i, "text": lines[i].strip()} for i in screened]
    return kept


def _raw_heading_hits(lines: list[str], override_lines: dict[int, dict]) -> list[int]:
    """Context-guarded heading-candidate indices, BEFORE TOC-run screening.

    A heading counts when preceded by a blank line (or the text edge) and
    followed by either a blank line / the text edge, or a short standalone
    title line (see ``_is_title_line``): Gutenberg editions commonly set the
    chapter title directly under the heading with no intervening blank. The
    blank-before guard still screens roman numerals and short phrases inside
    running prose. This is the raw candidate list ``detect_chapter_lines``
    screens with ``_screen_toc_runs``; ``segment_chapters`` also uses it
    directly (unscreened) to anchor the front-matter boundary, see there.
    """
    n = len(lines)
    hits = []
    for i, line in enumerate(lines):
        if not (is_chapter_heading(line) or i in override_lines):
            continue
        if i > 0 and lines[i - 1].strip():
            continue  # no blank line (or edge) above
        next_blank = i == n - 1 or not lines[i + 1].strip()
        if next_blank or _is_title_line(lines, i + 1):
            hits.append(i)
    return hits


def detect_chapter_lines(text: str, report: Optional[dict] = None,
                         overrides: Optional[list[dict]] = None) -> list[int]:
    """Indices of heading lines in ``text.split('\\n')``.

    A heading counts when preceded by a blank line (or the text edge) and
    followed by either a blank line / the text edge, or a short standalone
    title line (see ``_is_title_line``): Gutenberg editions commonly set the
    chapter title directly under the heading with no intervening blank. The
    blank-before guard still screens roman numerals and short phrases inside
    running prose, and ``_screen_toc_runs`` screens contents listings. When a
    ``report`` dict is passed, screened candidates are recorded in it (used by
    the extract command's verification report).

    ``overrides`` (see ``_match_override_lines``) adds reviewed exact-text
    boundary lines. They pass through the same context guards and TOC screen
    as pattern-matched headings; when a ``report`` dict is passed, per-entry
    match/boundary counts (``override_entries``) and the surviving override
    boundaries (``override_boundaries``) are recorded in it.
    """
    lines = text.split("\n")
    override_lines = _match_override_lines(lines, overrides)
    hits = _raw_heading_hits(lines, override_lines)
    hits = _screen_toc_runs(lines, hits, report,
                            extra_latent=set(override_lines) or None)
    if report is not None and overrides:
        report["override_entries"] = [
            {"match": e["match"],
             "label": e.get("label") or e["match"],
             "lines_matched": sum(1 for j in override_lines
                                  if lines[j].strip() == e["match"]),
             "boundaries": sum(1 for j in hits if j in override_lines
                               and lines[j].strip() == e["match"])}
            for e in overrides]
        boundaries = [{"line": j, "text": lines[j].strip(),
                       "label": (override_lines[j].get("label")
                                 or lines[j].strip())}
                      for j in hits if j in override_lines]
        if boundaries:
            report["override_boundaries"] = boundaries
    return hits


# Characters that end a prose line: a paragraph's final line virtually always
# closes with one of these, an orphan heading line does not. A trailing ")"
# is deliberately NOT in the set: a short parenthetical scaffolding line such
# as "(of Chancery Lane, Solicitor)" under a narrator heading is part of the
# orphan block (The Woman in White junction shape), not prose.
_TERMINAL_PUNCT = ".!?,;\"'”’]*_…"
# Sentence-final punctuation. A line ending in one of these may still be
# stripped, but only inside a multi-line orphan block whose top line is
# heading-like (the Wells shape: "BOOK TWO" over "THE EARTH UNDER THE
# MARTIANS."); it is never stripped on its own or from the top of a block.
_SENTENCE_PUNCT = ".!?\"”"
MAX_ORPHAN_BLOCK_LINES = 3   # orphan blocks stay small: heading plus title-ish lines


def _looks_structural(s: str) -> bool:
    """Short, title-case/all-caps, alpha-bearing: the texture of a heading or
    junction-scaffolding line rather than prose."""
    if not s or len(s) > MAX_TITLE_LINE_CHARS:
        return False
    alpha_words = [w for w in s.split() if w[:1].isalpha()]
    return bool(alpha_words) and all(w[:1].isupper() for w in alpha_words)


def _strip_orphan_tail(body: str) -> tuple[str, list[str]]:
    """Strip trailing orphan heading-like lines or blocks from a unit body.

    Shakedown evidence (War and Peace): a structural heading line matching NO
    pattern ("BOOK EIGHT: 1811 - 12") lands verbatim at the tail of the
    preceding unit, because the following CHAPTER I is the detected boundary.
    Extraction evidence (Wells, Collins): the orphan can be a small BLOCK, a
    heading-like line over a short title line ("BOOK TWO" + "THE EARTH UNDER
    THE MARTIANS.") or a narrator heading over a parenthetical subtitle
    ("THE STORY CONTINUED BY VINCENT GILMORE" + "(of Chancery Lane,
    Solicitor)"), where the single-line loop broke on the non-blank line
    above or on the trailing punctuation.

    Conservative rules: at most ``MAX_ORPHAN_BLOCK_LINES`` structural-looking
    lines (blank lines between them allowed) are collected from the tail; the
    block must be standalone (blank line or edge above its top line); an
    end-marker line (THE END must survive for the tail trimmer) or a
    prose-looking line ends collection. A line ending in sentence punctuation
    (. ! ? ") is stripped only inside a multi-line block whose top line is
    heading-like (a clean structural line or a pattern-matching heading),
    never on its own. A single-line orphan keeps the old rules: no terminal
    punctuation at all (a trailing ")" only counts as non-terminal inside a
    block) and not a pattern-matching heading (those are boundaries or
    screened candidates, not orphans). Stripped lines are returned so the
    caller can record them in the unit metadata; only units followed by a
    detected boundary are processed (see segment_chapters).
    """
    lines = body.split("\n")
    block: list[int] = []  # candidate line indices, collected bottom-up
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    while i >= 0 and len(block) < MAX_ORPHAN_BLOCK_LINES:
        s = lines[i].strip()
        if not s:
            i -= 1
            continue  # blank lines inside the block are allowed
        if not _looks_structural(s) or _END_MARKER.match(s):
            break
        if s[-1] in _TERMINAL_PUNCT and s[-1] not in _SENTENCE_PUNCT:
            break  # ends like broken prose (comma, semicolon, ...): never strip
        block.append(i)
        i -= 1
    # a sentence-punctuated line may not top the block: without a heading-like
    # line above it, it reads as prose and stays
    while block and lines[block[-1]].strip()[-1] in _SENTENCE_PUNCT:
        block.pop()
    if block:
        top = block[-1]
        top_s = lines[top].strip()
        if top > 0 and lines[top - 1].strip():
            block = []  # not standalone: a wrapped prose tail stays
        elif len(block) == 1 and (top_s[-1] in _TERMINAL_PUNCT + ")"
                                  or is_chapter_heading(top_s)):
            block = []  # single lines keep the old conservative rules
        elif len(block) > 1 and (top_s[-1] in _TERMINAL_PUNCT + ")"
                                 and not is_chapter_heading(top_s)):
            block = []  # block top must be heading-like
    if not block:
        return body, []
    stripped = [lines[j].strip() for j in reversed(block)]
    return "\n".join(lines[: block[-1]]).rstrip(), stripped


def segment_chapters(text: str, min_unit_words: int = MIN_UNIT_WORDS,
                     report: Optional[dict] = None,
                     overrides: Optional[list[dict]] = None) -> list[dict]:
    """Split at detected chapter headings. Returns [] if fewer than MIN_CHAPTERS.

    When a ``report`` dict is passed, it collects what the split left out
    (screened TOC candidates, dropped runt units, dropped front-matter scraps,
    stripped orphan tail lines) for the extract command's verification report.
    ``overrides`` adds reviewed exact-text boundary lines (see
    ``detect_chapter_lines``); a unit whose heading came from an override
    carries ``source: "override"`` and uses the entry's label (defaulting to
    the matched line text). Runt and front-matter rules apply to override
    units exactly as to pattern-matched ones.
    """
    lines = text.split("\n")
    idxs = detect_chapter_lines(text, report, overrides)
    if len(idxs) < MIN_CHAPTERS:
        return []
    override_lines = _match_override_lines(lines, overrides)
    units: list[dict] = []
    followed_by_boundary: list[bool] = []
    # Text before the first heading is kept only if it is substantial (a preface
    # or unlabeled opening); tiny scraps of front matter are dropped. ``pre``
    # spans everything up to the first SURVIVING heading (idxs[0]) exactly as
    # before -- nothing before it is ever silently discarded from accounting,
    # it is either kept whole as the "(front)" unit or dropped whole as a
    # reported scrap (Monte Cristo evidence: its 586-word front unit, kept,
    # legitimately contains its own screened 117-chapter TOC plus a "VOLUME
    # ONE" divider; that content must stay put, not vanish into a gap between
    # a separate "front boundary" and idxs[0]).
    #
    # The KEEP/DROP decision, however, excludes ``_is_toc_entry_line`` matches
    # from the word count: a columnar TOC entry list inflates the raw count
    # with structural filler, not prose. Tale of Two Cities evidence: its
    # screened TOC (all 3 Books' inline chapter lists, ~215 words of entry
    # lines) sits before the first surviving heading, so raw ``pre`` is 245
    # words -- over the keep threshold -- while the real, non-entry-line
    # content in that span (title, "CONTENTS", the 3 Book headings, ~30
    # words) is not. Judging the threshold on the raw count would keep a
    # "(front)" unit whose "content" is almost entirely the TOC entries the
    # fix just screened out of the chapter list. Deliberately narrower than
    # excluding every ``is_chapter_heading`` match too: Monte Cristo's own
    # 117-entry TOC uses the ordinary punctuation-separated heading form
    # ("Chapter 103. Maximilian"), which the *existing* density screen
    # already handles on its own terms (see ``_screen_toc_runs``), and its
    # 586-word front unit is deliberately KEPT (frozen corpus) -- excluding
    # ``is_chapter_heading`` lines too would gut that count and drop it.
    pre_lines = lines[: idxs[0]]
    pre = "\n".join(pre_lines).strip()
    pre_prose_words = word_count(" ".join(
        l for l in pre_lines if not _is_toc_entry_line(l)))
    if pre_prose_words >= min_unit_words * 4:
        units.append({"label": FRONT_LABEL, "text": pre})
        followed_by_boundary.append(True)
    elif pre and report is not None:
        report["dropped_front_words"] = word_count(pre)
    for k, i in enumerate(idxs):
        j = idxs[k + 1] if k + 1 < len(idxs) else len(lines)
        body_start = i + 1
        raw_label = lines[i].strip()
        if not override_lines.get(i) and _wrap_continuation(lines, i):
            # wrapped long heading (Monte Cristo Chapter 61): the title
            # continues on the next line; the label joins both lines and the
            # continuation line never becomes body text
            raw_label = raw_label + " " + lines[i + 1].strip()
            body_start = i + 2
        body = "\n".join(lines[body_start : j]).strip()
        entry = override_lines.get(i)
        label = (entry.get("label") or entry["match"]) if entry \
            else raw_label
        if word_count(body) < min_unit_words:
            if report is not None:
                report.setdefault("dropped_runts", []).append(
                    {"label": label, "words": word_count(body)})
            continue
        unit = {"label": label, "text": body}
        if entry:
            unit["source"] = "override"
        units.append(unit)
        followed_by_boundary.append(k + 1 < len(idxs))
    for u, followed in zip(units, followed_by_boundary):
        if not followed:
            continue  # never touch the final unit's tail (THE END lives there)
        body, stripped = _strip_orphan_tail(u["text"])
        if stripped:
            u["text"] = body
            u["stripped_tail"] = stripped
            if report is not None:
                report.setdefault("stripped_tails", []).append(
                    {"unit_label": u["label"], "lines": stripped})
    for n, u in enumerate(units):
        u["index"] = n
        u["words"] = word_count(u["text"])
    return units


# --- unmatched structural suspects (diagnostic only) ---------------------------------

SUSPECT_MAX_CHARS = 70       # a structural line stays short
_SUSPECT_LEAD_CHARS = "_\"'“‘"  # italic/quoted epistolary headers, not structure
_SUSPECT_CAP_RATIO = 0.7     # share of capitalized words that reads as a title

# Lowercase connective words that structural headings legitimately carry
# ("Book the Third--the Track of a Storm"). When a line STARTS with a
# structural keyword, these are exempted from the cap-ratio denominator.
_HEADING_CONNECTIVES = frozenset(("the", "of", "a", "an", "to", "and", "in"))
_STRUCTURAL_KEYWORD_LEAD = re.compile(
    r"^(?:BOOK|Book|PART|Part|VOLUME|Volume|CHAPTER|Chapter)\s")


def _suspect_cap_ratio_ok(s: str) -> bool:
    """Cap-ratio test for suspect lines, connective-aware for keyword leads.

    Tale of Two Cities evidence: "Book the First--Recalled to Life" (3 of 5
    capitalized) and "Book the Third--the Track of a Storm" (4 of 7) fell
    under the plain 0.7 ratio and never surfaced, while "Book the Second--the
    Golden Thread" (4 of 5) did: 2 of 3 headings missed. When the line starts
    with a structural keyword (Book/Part/Volume/Chapter, either case), the
    connective words are dropped from the denominator; ordinary prose lines
    starting with those keywords still fail because their remaining words are
    lowercase.
    """
    words = [w.lstrip("(\"'“‘[") for w in s.split()]
    alpha_words = [w for w in words if w[:1].isalpha()]
    if not alpha_words:
        return False
    if _STRUCTURAL_KEYWORD_LEAD.match(s):
        counted = [w for w in alpha_words
                   if w.rstrip(".,:;").lower() not in _HEADING_CONNECTIVES]
        alpha_words = counted or alpha_words
    caps = sum(1 for w in alpha_words if w[:1].isupper())
    return caps / len(alpha_words) >= _SUSPECT_CAP_RATIO


def _is_suspect_line(s: str) -> bool:
    """One stripped line: looks structural yet matches no heading pattern.

    Calibrated for precision over recall. Epistolary italic entry headers
    (Dracula's "_Dr. Seward's Diary._") are excluded by the leading
    underscore/quote guard and the sentence-punctuation guard; a trailing
    ")" does NOT count as prose-terminal (the fix-2 nuance), so parenthetical
    junction scaffolding still surfaces. The words must read as mostly
    uppercase / Title Case (at least ``_SUSPECT_CAP_RATIO`` of the
    alpha-leading words capitalized, after stripping opening brackets and
    quotes, and exempting lowercase connectives when the line starts with a
    structural keyword; see ``_suspect_cap_ratio_ok``), which drops ordinary
    prose lines.

    Out of scope, deliberately: prose-shaped paratext such as Tale of Two
    Cities' "The end of the first book." is NOT chased here. It reads exactly
    like a sentence (lowercase words, terminal period), so any cap-ratio or
    punctuation loosening wide enough to catch it would flood the report with
    real prose; it stays in the preceding unit's tail.
    """
    if not s or len(s) > SUSPECT_MAX_CHARS:
        return False
    if s[0] in _SUSPECT_LEAD_CHARS:
        return False
    if s[-1] in _TERMINAL_PUNCT:
        return False
    if is_chapter_heading(s) or _END_MARKER.match(s):
        return False  # matched a pattern / end marker: visible elsewhere
    return _suspect_cap_ratio_ok(s)


def find_unmatched_suspects(units: list[dict]) -> list[dict]:
    """Scan unit bodies for standalone lines that look structural but matched
    no heading pattern (the heading-invisibility fix).

    Extraction evidence (The Woman in White): the narrator headings
    ("THE STORY CONTINUED BY VINCENT GILMORE", ...) matched no pattern and
    fused silently into neighboring units; nothing in the report showed they
    existed. This scan is purely diagnostic (no behavior change to
    segmentation): a line counts when it is standalone (blank lines or the
    body edge above AND below) and passes ``_is_suspect_line``.

    Wrapped-heading exception (Monte Cristo evidence, Chapter 61): a wrapped
    numbered heading is non-standalone (its title continuation sits directly
    below, so blank-below fails) and can exceed ``SUSPECT_MAX_CHARS``, which
    made the miss doubly invisible. A line that PREFIX-matches a numbered
    heading ("Chapter 61. ..." per ``_NUMBERED_HEADING_PREFIX``) but is not a
    full pattern match therefore surfaces with only blank-above required, no
    length cap (over-length is the very failure mode being caught), no prose
    terminal punctuation, and the connective-aware cap ratio (which keeps
    hard-wrapped prose paragraphs starting "Chapter 61. That was..." out).

    Returns ``[{"unit_index", "line", "position_words_into_unit"}, ...]`` for
    the extract sidecar; the stdout report summarizes the count.
    """
    suspects: list[dict] = []
    for u in units:
        lines = u["text"].split("\n")
        words_seen = 0
        for i, line in enumerate(lines):
            s = line.strip()
            blank_above = i == 0 or not lines[i - 1].strip()
            blank_below = i == len(lines) - 1 or not lines[i + 1].strip()
            hit = False
            if s and blank_above:
                if blank_below and _is_suspect_line(s):
                    hit = True
                elif (_NUMBERED_HEADING_PREFIX.match(s)
                        and not is_chapter_heading(s)
                        and s[-1] not in _TERMINAL_PUNCT
                        and _suspect_cap_ratio_ok(s)):
                    hit = True  # wrapped/over-long numbered heading lead
            if hit:
                suspects.append({"unit_index": u["index"], "line": s,
                                 "position_words_into_unit": words_seen})
            words_seen += word_count(line)
    return suspects


# --- fixed windows -----------------------------------------------------------------

def segment_windows(text: str, window_words: int = DEFAULT_WINDOW_WORDS,
                    min_unit_words: int = MIN_UNIT_WORDS) -> list[dict]:
    """~window_words-sized units snapped to paragraph boundaries.

    Paragraphs accumulate until the window reaches the target, then the window
    closes at that paragraph boundary (so a window can overshoot by at most one
    paragraph, and never splits a paragraph). A trailing runt below
    ``min_unit_words`` is merged into the previous window.
    """
    paras = split_paragraphs(text)
    units: list[dict] = []
    cur: list[str] = []
    cur_words = 0
    for p in paras:
        cur.append(p)
        cur_words += word_count(p)
        if cur_words >= window_words:
            units.append({"text": "\n\n".join(cur)})
            cur, cur_words = [], 0
    if cur:
        tail = "\n\n".join(cur)
        if units and word_count(tail) < min_unit_words:
            units[-1]["text"] += "\n\n" + tail
        else:
            units.append({"text": tail})
    for n, u in enumerate(units):
        u["index"] = n
        u["label"] = f"w{n:03d}"
        u["words"] = word_count(u["text"])
    return units


# --- front-matter scoring policy ----------------------------------------------------

def exclude_front_matter(units: list[dict],
                         include_front: bool = False) -> tuple[list[dict], Optional[dict]]:
    """Scoring-layer policy for the pre-first-heading ``(front)`` unit.

    Segmentation keeps substantial front matter as a labeled unit so nothing
    is silently dropped, but front matter is not a chapter: the two-book
    shakedown showed Dracula's (front) unit (title page, TOC, epigraph) scoring
    MTLD 24.3 against 70-101 for real chapters and feeding TOC names into the
    entity census. Scoring therefore excludes it by default;
    ``include_front=True`` (the ``--include-front`` CLI flag) opts back in.

    Returns ``(units_to_score, record)``. The record lists the front unit(s),
    whether they were excluded, and the segmented/scored counts; callers embed
    it in the sidecar so the exclusion is never silent. It is None when the
    segmentation produced no front unit. Kept units retain their original
    ``index`` values so per-unit metric records stay traceable to the
    segmentation.
    """
    front = [u for u in units if u["label"] == FRONT_LABEL]
    if not front:
        return units, None
    kept = units if include_front else [u for u in units if u["label"] != FRONT_LABEL]
    record = {
        "front_units": [{"index": u["index"], "label": u["label"], "words": u["words"]}
                        for u in front],
        "excluded": not include_front,
        "policy": ("excluded from scoring by default; pass --include-front to score it"
                   if not include_front else "scored: --include-front"),
        "n_units_segmented": len(units),
        "n_units_scored": len(kept),
    }
    return kept, record


# --- apparatus scoring policy --------------------------------------------------------

# Closed vocabulary of author-apparatus labels (prefaces, editorial notes,
# footnotes, ...): not story, generalizing the ``(front)`` policy above to
# labeled headings anywhere in the document. Matched case-insensitively
# against the FULL trimmed label -- anchored, never a substring search -- so
# real story sections that happen to contain apparatus-ish vocabulary words
# never match: The Woman in White's narrator headings ("The Story Begun by
# Walter Hartright", "1. The Narrative of Hester Pinhorn") ARE the epistolary
# novel and must stay scored. PROLOGUE/EPILOGUE/INTRODUCTION/INTERLUDE/ENVOI
# are deliberately NOT in this vocabulary: they are frequently narrative, none
# appear in the current corpus, and this is a conservative default, not a
# proven boundary.
#
# Bare "NOTE" is deliberately EXCLUDED from this set (Dracula evidence): the
# novel's final unit is a standalone "NOTE" that is Jonathan Harker's in-world
# epilogue ("Seven years ago..."), i.e. story, not editorial apparatus. Only
# QUALIFIED note forms are apparatus: the "NOTE TO THE ... EDITION" prefix
# (Jane Eyre) and the possessive "TRANSLATOR'S / AUTHOR'S / PUBLISHER'S NOTE".
_APPARATUS_EXACT = frozenset({
    "PREFACE", "FOREWORD", "AFTERWORD", "DEDICATION", "FOOTNOTES", "APPENDIX",
    "ERRATA", "GLOSSARY", "EPIGRAPH", "CONTENTS",
    "TRANSLATOR'S NOTE", "AUTHOR'S NOTE", "PUBLISHER'S NOTE",
    "INTRODUCTORY NOTE",
})
# Prefix forms: "NOTE TO THE THIRD EDITION" (Jane Eyre), "APPENDIX A", and the
# "PREFACE TO THE ..." shape some editions use for a revised-edition preface.
_APPARATUS_PREFIXES = ("NOTE TO THE ", "PREFACE TO THE ", "APPENDIX ")


def is_apparatus_label(label: str) -> bool:
    """True when ``label`` is author apparatus (front/back matter), not story.

    Whole-label match only, case-insensitive, with curly apostrophes
    normalized and an optional trailing period stripped (editions vary on
    ``PREFACE`` vs ``PREFACE.``). See the module-level vocabulary comment for
    what is and is not included, and why the match is anchored rather than a
    substring search.
    """
    s = label.strip().upper().replace("’", "'").replace("‘", "'")
    if s.endswith("."):
        s = s[:-1]
    if s in _APPARATUS_EXACT:
        return True
    return s.startswith(_APPARATUS_PREFIXES)


def exclude_apparatus(units: list[dict],
                      include_apparatus: bool = False) -> tuple[list[dict], Optional[dict]]:
    """Scoring-layer policy for author-apparatus units (see ``is_apparatus_label``).

    Same shape and rationale as ``exclude_front_matter``, generalized from the
    single pre-first-heading unit to any labeled apparatus heading anywhere in
    the document: extraction keeps these units (nothing silently dropped from
    the canonical Markdown), but scoring an editorial preface, a translator's
    note, or a footnotes appendix alongside real chapters distorts per-unit
    stats the same way the shakedown showed for the ``(front)`` unit. Excluded
    from scoring by default; ``include_apparatus=True`` (the
    ``--include-apparatus`` CLI flag) opts back in.

    Returns ``(units_to_score, record)``. The record lists the apparatus
    unit(s), whether they were excluded, and the units-in/units-out counts for
    THIS call (i.e. of the units passed in -- typically already past
    ``exclude_front_matter``, so these counts compose rather than duplicate
    that record's). None when none of the input units is apparatus. Kept units
    retain their original ``index`` values.
    """
    apparatus = [u for u in units if is_apparatus_label(u["label"])]
    if not apparatus:
        return units, None
    kept = units if include_apparatus \
        else [u for u in units if not is_apparatus_label(u["label"])]
    record = {
        "apparatus_units": [{"index": u["index"], "label": u["label"], "words": u["words"]}
                            for u in apparatus],
        "excluded": not include_apparatus,
        "policy": ("excluded from scoring by default; pass --include-apparatus to score it"
                   if not include_apparatus else "scored: --include-apparatus"),
        "n_units_segmented": len(units),
        "n_units_scored": len(kept),
    }
    return kept, record


def exclude_non_story(units: list[dict], include_front: bool = False,
                      include_apparatus: bool = False) -> tuple[list[dict], dict]:
    """Apply both scoring-layer exclusions in sequence: front matter, then
    apparatus. Convenience wrapper so callers (the ``nd`` CLI, ``utils.metrics``
    single-text mode) don't duplicate the two-call chain.

    Returns ``(units_to_score, records)`` where ``records`` is
    ``{"front_matter": ..., "apparatus": ...}`` -- each either the record shape
    documented on the matching function, or None when that category was
    absent. Callers embed both under those two keys in the sidecar; existing
    ``front_matter`` readers are unaffected, ``apparatus`` is additive.
    """
    units, front_record = exclude_front_matter(units, include_front=include_front)
    units, apparatus_record = exclude_apparatus(units, include_apparatus=include_apparatus)
    return units, {"front_matter": front_record, "apparatus": apparatus_record}


# --- canonical Markdown ingestion ----------------------------------------------------
# The permanent analysis path: the extract command emits one canonical .md per
# book (provenance header + one "# <label>" heading per unit); this splitter
# re-ingests it with NO heuristics. Kept deliberately trivial.

_MD_PROVENANCE = re.compile(r"\A\s*<!--(.*?)-->", re.S)
_MD_HEADING = re.compile(r"^# (.+)$")
_MD_ESCAPED = re.compile(r"^\\(# .*)$")

# Note printed for windows-mode results: unlike chapters/md, windows performs
# no front-matter exclusion (there is no "(front)" unit to exclude), so any
# title page / TOC / preface is scored inside the first window(s).
WINDOWS_FRONT_NOTE = ("windows segmentation performs no front-matter exclusion; "
                      "leading front matter is scored inside the first window(s)")


def parse_provenance(text: str) -> tuple[Optional[dict], str]:
    """Read the extract provenance header (a leading HTML comment of
    ``key: value`` lines). Returns ``(provenance_or_None, remaining_text)``."""
    m = _MD_PROVENANCE.match(text)
    if not m:
        return None, text
    prov: dict = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        prov[key.strip()] = value.strip()
    return (prov or None), text[m.end():]


def segment_markdown(text: str) -> tuple[list[dict], Optional[dict]]:
    """Split canonical Markdown on top-level ``# `` headings. No heuristics.

    Returns ``(units, provenance)``. Every heading becomes a unit, verbatim: no
    word-count floor, no TOC screen, no Gutenberg trim (extraction already did
    all of that, once). ``\\# `` body lines are unescaped (the extract command
    escapes them). Non-blank content before the first heading is kept as a
    ``(front)`` unit; a file with no headings at all becomes one
    ``(document)`` unit.
    """
    provenance, text = parse_provenance(text)
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    heads = [i for i, line in enumerate(lines) if _MD_HEADING.match(line)]

    def body(lo: int, hi: int) -> str:
        chunk = [_MD_ESCAPED.sub(r"\1", line) for line in lines[lo:hi]]
        return "\n".join(chunk).strip()

    units: list[dict] = []
    if not heads:
        whole = body(0, len(lines))
        if whole:
            units.append({"label": "(document)", "text": whole})
    else:
        pre = body(0, heads[0])
        if pre:
            units.append({"label": FRONT_LABEL, "text": pre})
        for k, i in enumerate(heads):
            j = heads[k + 1] if k + 1 < len(heads) else len(lines)
            units.append({"label": _MD_HEADING.match(lines[i]).group(1).strip(),
                          "text": body(i + 1, j)})
    for n, u in enumerate(units):
        u["index"] = n
        u["words"] = word_count(u["text"])
    return units, provenance


# --- top-level entry ---------------------------------------------------------------

def segment(text: str, strategy: str = "chapters",
            window_words: int = DEFAULT_WINDOW_WORDS,
            trim_gutenberg: bool = True,
            overrides: Optional[list[dict]] = None) -> dict:
    """Segment a document into ordered units.

    Returns ``{"strategy_requested", "strategy_used", "units", "n_units",
    "total_words", "tail_trim"}``. ``chapters`` falls back to ``windows`` when
    too few headings are detectable; ``strategy_used`` records what actually
    happened. ``tail_trim`` records a trailing back-matter trim (see
    ``trim_trailing_backmatter``; applied with the Gutenberg trim), or None.
    ``chapters``/``windows`` results carry ``page_anchor_lines_removed``, the
    count of standalone page-anchor lines stripped during trimming (see
    ``strip_page_anchors``; 0 when trimming is disabled).
    ``chapters`` results also carry ``chapter_detection`` (what the proposer
    screened/dropped; see ``segment_chapters``); ``md`` results carry the
    ``provenance`` header of the extracted file; ``windows`` results carry a
    ``note`` recording that no front-matter exclusion applies. ``overrides``
    (chapters strategy only) adds reviewed exact-text boundary lines; see
    ``segment_chapters``.
    """
    if strategy not in ("chapters", "windows", "md"):
        raise ValueError(f"unknown segmentation strategy: {strategy!r}")

    if strategy == "md":
        # canonical Markdown: extraction already trimmed and screened, once
        units, provenance = segment_markdown(text)
        return {
            "strategy_requested": strategy,
            "strategy_used": "md",
            "units": units,
            "n_units": len(units),
            "total_words": sum(u["words"] for u in units),
            "tail_trim": None,
            "provenance": provenance,
        }

    tail_note = None
    page_anchors_removed = 0
    if trim_gutenberg:
        text = strip_gutenberg(text)
        text, page_anchors_removed = strip_page_anchors(text)
        text, tail_note = trim_trailing_backmatter(text)
    else:
        text = text.replace("\r\n", "\n").strip()

    used = strategy
    detection: dict = {}
    units = (segment_chapters(text, report=detection, overrides=overrides)
             if strategy == "chapters" else [])
    if not units:
        units = segment_windows(text, window_words=window_words)
        if strategy == "chapters":
            used = "windows (fallback: fewer than "f"{MIN_CHAPTERS} chapter headings detected)"
    result = {
        "strategy_requested": strategy,
        "strategy_used": used,
        "units": units,
        "n_units": len(units),
        "total_words": sum(u["words"] for u in units),
        "tail_trim": tail_note,
        "page_anchor_lines_removed": page_anchors_removed,
    }
    if used == "chapters":
        result["chapter_detection"] = detection or None
    if used.startswith("windows"):
        result["note"] = WINDOWS_FRONT_NOTE
    return result


# --- long-unit truncation (shared by the LLM metrics) --------------------------------

def truncate_middle(text: str, head_words: int, tail_words: int) -> str:
    """Keep the first ``head_words`` and last ``tail_words`` of a long unit.

    Matches the masters-study long-chapter policy: the omission is announced
    inline so the judge knows material was skipped. Texts within ~200 words of
    the budget are returned whole.
    """
    words = text.split()
    if len(words) <= head_words + tail_words + 200:
        return text
    omitted = len(words) - head_words - tail_words
    return (" ".join(words[:head_words])
            + f"\n\n[... middle omitted: about {omitted} words ...]\n\n"
            + " ".join(words[-tail_words:]))
