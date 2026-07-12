"""Segmentation layer: arbitrary-length text to ordered analysis units. Pure, no LLM.

Every narrative-dynamics metric consumes the same unit list produced here, so the
segmentation policy is decided once per document and recorded in the output. Two
selectable strategies:

* ``chapters``: detect Gutenberg-style chapter headings (``CHAPTER IV``,
  ``Chapter 12.``, bare roman numerals, ``PROLOGUE``/``EPILOGUE``...). Adapted from
  the StoryDaemon masters-study extraction scripts. Falls back to fixed windows
  when a text has too few detectable headings, and says so in the result.
* ``windows``: fixed ~1500-word windows snapped to paragraph boundaries (1500
  words is this repo's existing benchmark framing for one story-sized unit).

Plus a Project Gutenberg frontmatter/license trimmer, since the masters corpus
files carry the header and license tail as downloaded.

A unit is a dict: ``{"index": int, "label": str, "text": str, "words": int}``.
"""
from __future__ import annotations

import re
from typing import Optional

DEFAULT_WINDOW_WORDS = 1500  # the repo's established story-unit framing
MIN_UNIT_WORDS = 50          # drop degenerate units (heading-only fragments)
MIN_CHAPTERS = 3             # fewer detected chapters than this -> fall back

# --- Gutenberg trimming -----------------------------------------------------------

_GUT_START = re.compile(r"^\s*\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_GUT_END = re.compile(r"^\s*\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_ILLUSTRATION = re.compile(r"\[Illustration:?[^\]]*\]", re.S)


def strip_gutenberg(text: str) -> str:
    """Trim Project Gutenberg frontmatter and license tail; normalize newlines.

    Keeps only the body between the ``*** START OF ... ***`` and
    ``*** END OF ... ***`` markers (either marker may be absent). Also removes
    ``[Illustration: ...]`` blocks and collapses 3+ blank lines.
    """
    text = text.replace("\r\n", "\n")
    m = _GUT_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUT_END.search(text)
    if m:
        text = text[:m.start()]
    text = _ILLUSTRATION.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    # named structural units
    re.compile(r"^(?:PROLOGUE|EPILOGUE|INTRODUCTION|CONCLUSION|"
               r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)"
               r"\s+(?:NARRATIVE|PERIOD|BOOK|PART))\.?$"),
]


def is_chapter_heading(line: str) -> bool:
    """True when a stripped line looks like a chapter-style structural heading."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    return any(p.match(s) for p in _CHAPTER_PATTERNS)


def detect_chapter_lines(text: str) -> list[int]:
    """Indices of heading lines in ``text.split('\\n')``.

    A heading only counts when surrounded by blank lines (or the text edge), which
    screens out roman numerals and short phrases inside running prose.
    """
    lines = text.split("\n")
    hits = []
    for i, line in enumerate(lines):
        if not is_chapter_heading(line):
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i == len(lines) - 1 or not lines[i + 1].strip()
        if prev_blank and next_blank:
            hits.append(i)
    return hits


def segment_chapters(text: str, min_unit_words: int = MIN_UNIT_WORDS) -> list[dict]:
    """Split at detected chapter headings. Returns [] if fewer than MIN_CHAPTERS."""
    lines = text.split("\n")
    idxs = detect_chapter_lines(text)
    if len(idxs) < MIN_CHAPTERS:
        return []
    units: list[dict] = []
    # Text before the first heading is kept only if it is substantial (a preface
    # or unlabeled opening); tiny scraps of front matter are dropped.
    pre = "\n".join(lines[: idxs[0]]).strip()
    if word_count(pre) >= min_unit_words * 4:
        units.append({"label": "(front)", "text": pre})
    for k, i in enumerate(idxs):
        j = idxs[k + 1] if k + 1 < len(idxs) else len(lines)
        body = "\n".join(lines[i + 1 : j]).strip()
        if word_count(body) < min_unit_words:
            continue
        units.append({"label": lines[i].strip(), "text": body})
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


# --- top-level entry ---------------------------------------------------------------

def segment(text: str, strategy: str = "chapters",
            window_words: int = DEFAULT_WINDOW_WORDS,
            trim_gutenberg: bool = True) -> dict:
    """Segment a document into ordered units.

    Returns ``{"strategy_requested", "strategy_used", "units", "n_units",
    "total_words"}``. ``chapters`` falls back to ``windows`` when too few
    headings are detectable; ``strategy_used`` records what actually happened.
    """
    if strategy not in ("chapters", "windows"):
        raise ValueError(f"unknown segmentation strategy: {strategy!r}")
    if trim_gutenberg:
        text = strip_gutenberg(text)
    else:
        text = text.replace("\r\n", "\n").strip()

    used = strategy
    units = segment_chapters(text) if strategy == "chapters" else []
    if not units:
        units = segment_windows(text, window_words=window_words)
        if strategy == "chapters":
            used = "windows (fallback: fewer than "f"{MIN_CHAPTERS} chapter headings detected)"
    return {
        "strategy_requested": strategy,
        "strategy_used": used,
        "units": units,
        "n_units": len(units),
        "total_words": sum(u["words"] for u in units),
    }


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
