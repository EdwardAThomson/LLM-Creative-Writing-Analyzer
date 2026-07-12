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
files carry the header and license tail as downloaded, and a conservative
trailing back-matter trimmer for publisher catalogs printed INSIDE the
Gutenberg markers (see ``trim_trailing_backmatter``).

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
    * at least ``TAIL_MIN_VOCAB_HITS`` distinct catalog-vocabulary terms
      corroborate.

    In any doubt the text is returned unchanged: a false trim (losing real
    prose, e.g. an epilogue or author's note after the marker) is worse than
    keeping ads. The marker line itself is kept; material BEFORE the marker
    (e.g. Dracula's closing NOTE by Harker) is untouched. Returns
    ``(text, note)`` where ``note`` is a sidecar-ready record of what was
    trimmed and why, or None when nothing was trimmed.
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
    if nonprose_ratio < TAIL_NONPROSE_RATIO or len(vocab_hits) < TAIL_MIN_VOCAB_HITS:
        return text, None  # could be prose (epilogue, note): in doubt, keep
    note = {
        "marker": text[m.start():m.end()].strip(),
        "trimmed_words": word_count(after),
        "trimmed_lines": len(lines),
        "non_narrative_line_ratio": round(nonprose_ratio, 3),
        "vocabulary_hits": vocab_hits,
        "reason": ("end-marker line in the last 5% of the text followed by "
                   "non-narrative catalog material; trimmed at the marker "
                   "(marker line kept)"),
    }
    logger.info("trailing back-matter trimmed: %s", note)
    return text[:m.end()].rstrip(), note


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
_CHAPTER_PATTERNS = [
    # CHAPTER IV / Chapter 12. / CHAPTER THE FIRST, optionally followed by a
    # title after a dot, colon, hyphen, or em dash (the em dash in the class
    # below is a literal matched in Gutenberg headings, e.g. Eddison's)
    re.compile(rf"^(?:CHAPTER|Chapter)\s+(?:{_ROMAN}|\d+|[A-Z][A-Za-z-]+)\.?(?:\s*[.:—-]\s*\S.*)?$"),
    # BOOK II / PART THE SECOND / VOLUME I (treated as boundaries too)
    re.compile(rf"^(?:BOOK|PART|VOLUME|Book|Part|Volume)\s+(?:{_ROMAN}|\d+|THE\s+[A-Z]+)\.?$"),
    # bare roman numeral heading lines: "IV." / "XII"
    re.compile(rf"^{_ROMAN}\.?$"),
    # numbered headings: "12." on a line of its own
    re.compile(r"^\d{1,3}\.?$"),
    # named structural units. PRELUDE/FINALE per the Middlemarch shakedown:
    # without them the Prelude fell into front matter and the Finale fused
    # into Chapter 86.
    re.compile(r"^(?:PROLOGUE|EPILOGUE|PRELUDE|FINALE|INTRODUCTION|CONCLUSION|"
               r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)"
               r"\s+(?:NARRATIVE|PERIOD|BOOK|PART))\.?$"),
]


def is_chapter_heading(line: str) -> bool:
    """True when a stripped line looks like a chapter-style structural heading."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    return any(p.match(s) for p in _CHAPTER_PATTERNS)


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


def _screen_toc_runs(lines: list[str], hits: list[int],
                     report: Optional[dict] = None) -> list[int]:
    """Drop dense runs of heading-like lines (a table of contents, not structure).

    Density is measured over ALL lines matching the heading patterns (not only
    the context-guard-passing candidates): a Contents block whose entries sit
    on adjacent lines fails the blank-line guards entry by entry, yet its first
    and last entries can pass them and read as boundaries (the Middlemarch TOC:
    " PRELUDE." and " FINALE." pass, the 94 packed " CHAPTER n." lines between
    them do not). A run is ``TOC_MIN_RUN`` or more consecutive heading-like
    lines with fewer than ``MIN_UNIT_WORDS`` words between each adjacent pair.

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
    """
    if not hits:
        return hits
    latent = [i for i, line in enumerate(lines) if is_chapter_heading(line)]
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


def detect_chapter_lines(text: str, report: Optional[dict] = None) -> list[int]:
    """Indices of heading lines in ``text.split('\\n')``.

    A heading counts when preceded by a blank line (or the text edge) and
    followed by either a blank line / the text edge, or a short standalone
    title line (see ``_is_title_line``): Gutenberg editions commonly set the
    chapter title directly under the heading with no intervening blank. The
    blank-before guard still screens roman numerals and short phrases inside
    running prose, and ``_screen_toc_runs`` screens contents listings. When a
    ``report`` dict is passed, screened candidates are recorded in it (used by
    the extract command's verification report).
    """
    lines = text.split("\n")
    n = len(lines)
    hits = []
    for i, line in enumerate(lines):
        if not is_chapter_heading(line):
            continue
        if i > 0 and lines[i - 1].strip():
            continue  # no blank line (or edge) above
        next_blank = i == n - 1 or not lines[i + 1].strip()
        if next_blank or _is_title_line(lines, i + 1):
            hits.append(i)
    return _screen_toc_runs(lines, hits, report)


# Characters that end a prose line: a paragraph's final line virtually always
# closes with one of these, an orphan heading line does not.
_TERMINAL_PUNCT = ".!?,;\"'”’)]*_…"


def _strip_orphan_tail(body: str) -> tuple[str, list[str]]:
    """Strip trailing orphan heading-like lines from a unit body.

    Shakedown evidence (War and Peace): a structural heading line matching NO
    pattern ("BOOK EIGHT: 1811 - 12") lands verbatim at the tail of the
    preceding unit, because the following CHAPTER I is the detected boundary.
    Cheap conservative mitigation: strip a trailing line only when it is
    standalone (blank line above), short (heading-length), has no terminal
    punctuation, is title-case or all-caps, and is neither an end-marker line
    (THE END must survive for the tail trimmer) nor a pattern-matching heading
    (those are boundaries or screened candidates, not orphans). Stripped lines
    are returned so the caller can record them in the unit metadata; only
    units followed by a detected boundary are processed (see segment_chapters).
    """
    lines = body.split("\n")
    stripped: list[str] = []
    while lines:
        if not lines[-1].strip():
            lines.pop()
            continue
        s = lines[-1].strip()
        if len(s) > MAX_TITLE_LINE_CHARS or len(lines) < 2 or lines[-2].strip():
            break  # long, or not standalone: a wrapped prose tail stays
        if s[-1] in _TERMINAL_PUNCT:
            break  # ends like prose
        if _END_MARKER.match(s) or is_chapter_heading(s):
            break
        alpha_words = [w for w in s.split() if w[:1].isalpha()]
        if not alpha_words or not all(w[:1].isupper() for w in alpha_words):
            break  # not title-case/all-caps
        stripped.insert(0, s)
        lines.pop()
    if not stripped:
        return body, []
    return "\n".join(lines).rstrip(), stripped


def segment_chapters(text: str, min_unit_words: int = MIN_UNIT_WORDS,
                     report: Optional[dict] = None) -> list[dict]:
    """Split at detected chapter headings. Returns [] if fewer than MIN_CHAPTERS.

    When a ``report`` dict is passed, it collects what the split left out
    (screened TOC candidates, dropped runt units, dropped front-matter scraps,
    stripped orphan tail lines) for the extract command's verification report.
    """
    lines = text.split("\n")
    idxs = detect_chapter_lines(text, report)
    if len(idxs) < MIN_CHAPTERS:
        return []
    units: list[dict] = []
    followed_by_boundary: list[bool] = []
    # Text before the first heading is kept only if it is substantial (a preface
    # or unlabeled opening); tiny scraps of front matter are dropped.
    pre = "\n".join(lines[: idxs[0]]).strip()
    if word_count(pre) >= min_unit_words * 4:
        units.append({"label": FRONT_LABEL, "text": pre})
        followed_by_boundary.append(True)
    elif pre and report is not None:
        report["dropped_front_words"] = word_count(pre)
    for k, i in enumerate(idxs):
        j = idxs[k + 1] if k + 1 < len(idxs) else len(lines)
        body = "\n".join(lines[i + 1 : j]).strip()
        if word_count(body) < min_unit_words:
            if report is not None:
                report.setdefault("dropped_runts", []).append(
                    {"label": lines[i].strip(), "words": word_count(body)})
            continue
        units.append({"label": lines[i].strip(), "text": body})
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
            trim_gutenberg: bool = True) -> dict:
    """Segment a document into ordered units.

    Returns ``{"strategy_requested", "strategy_used", "units", "n_units",
    "total_words", "tail_trim"}``. ``chapters`` falls back to ``windows`` when
    too few headings are detectable; ``strategy_used`` records what actually
    happened. ``tail_trim`` records a trailing back-matter trim (see
    ``trim_trailing_backmatter``; applied with the Gutenberg trim), or None.
    ``chapters`` results also carry ``chapter_detection`` (what the proposer
    screened/dropped; see ``segment_chapters``); ``md`` results carry the
    ``provenance`` header of the extracted file; ``windows`` results carry a
    ``note`` recording that no front-matter exclusion applies.
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
    if trim_gutenberg:
        text = strip_gutenberg(text)
        text, tail_note = trim_trailing_backmatter(text)
    else:
        text = text.replace("\r\n", "\n").strip()

    used = strategy
    detection: dict = {}
    units = segment_chapters(text, report=detection) if strategy == "chapters" else []
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
