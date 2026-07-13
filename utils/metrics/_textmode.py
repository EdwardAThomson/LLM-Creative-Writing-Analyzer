"""Raw-text input support for the retroactive scorer (scoring-only mode).

Lets ``python -m utils.metrics --text`` score arbitrary user-supplied text with
the v2+ metric library, with no generation step and no results_*.json: one file
becomes a single-response set; a directory becomes one set whose "runs" are its
text files in sorted order (so the cross-run metrics compare the files, and the
per-text metrics score each file).

Pure stdlib helper (underscore-prefixed: not a metric, excluded from
``available()``). Kept free of relative imports so tests can load it by file
path like the other pure modules (see tests/conftest.py).
"""
from __future__ import annotations

import glob
import os

TEXT_EXTENSIONS = (".txt", ".md")


def collect_texts(path: str) -> tuple[str, list[str], list[str]]:
    """(group_name, source_names, texts) for a text file or a directory of them.

    The group name plays the role a model key plays in results-JSON mode: it
    keys the sidecar's ``models`` mapping. For a file it is the file stem; for a
    directory, the directory name.
    """
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return os.path.splitext(os.path.basename(path))[0], [os.path.basename(path)], [f.read()]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no such file or directory: {path}")
    files: list[str] = []
    for ext in TEXT_EXTENSIONS:
        files.extend(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
    files = sorted(f for f in files if not f.endswith(".nd.txt"))
    if not files:
        raise FileNotFoundError(
            f"no {'/'.join(TEXT_EXTENSIONS)} files under {path}")
    texts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            texts.append(fh.read())
    group = os.path.basename(os.path.normpath(path))
    return group, [os.path.relpath(f, path) for f in files], texts


def sidecar_path(path: str, group: str) -> str:
    """Where the text-mode sidecar goes: next to the input, never over it."""
    if os.path.isdir(path):
        return os.path.join(path, f"{group}.metrics.json")
    return os.path.splitext(path)[0] + ".metrics.json"


def segment_units(text: str, strategy: str, window_words: int = 1500,
                  include_front: bool = False, include_apparatus: bool = False):
    """Segment one document into units via the narrative_dynamics layer (reused,
    not duplicated). Returns ``(segmentation_info, unit_labels, unit_texts)``.

    This is what turns ``--text`` into single-text (st1) mode: the units become
    the "runs" the library metrics score. The import is absolute and local so
    this module stays file-path loadable; it resolves when the repo root is on
    ``sys.path`` (true for ``python -m utils.metrics`` run from the repo root,
    and for the test harness).

    Strategies: ``chapters`` (heading heuristics; a one-time proposer, see
    ``benchmarks/narrative_dynamics/extract.py``), ``windows`` (fixed ~N-word
    windows; note that windows-mode performs NO front-matter/apparatus
    exclusion -- it has no headings to exclude by -- so any title page / TOC /
    preface is scored inside the first window(s), which the segmentation info
    records in its ``note`` field), and ``md`` (canonical extracted Markdown
    split on ``# `` headings, no heuristics; auto-selected by the CLI for
    ``.md`` inputs).

    The pre-first-heading ``(front)`` unit is excluded from scoring unless
    ``include_front`` (the ``--include-front`` flag): the shakedown showed it
    polluting per-unit stats (Dracula's TOC/title page). Author-apparatus
    units (prefaces, editorial notes, footnotes, ...) are likewise excluded
    unless ``include_apparatus`` (``--include-apparatus``): they are not
    story, generalizing the same exclusion. Both exclusions are recorded in
    ``segmentation_info["front_matter"]`` / ``["apparatus"]`` so the sidecar
    states what was left out; any trailing back-matter trim is likewise
    recorded in ``segmentation_info["tail_trim"]``.
    """
    from benchmarks.narrative_dynamics import segmentation as seg  # local, stdlib

    res = seg.segment(text, strategy=strategy, window_words=window_words)
    units, non_story = seg.exclude_non_story(
        res["units"], include_front=include_front,
        include_apparatus=include_apparatus)
    info = {k: v for k, v in res.items() if k != "units"}
    info["front_matter"] = non_story["front_matter"]
    info["apparatus"] = non_story["apparatus"]
    labels = [u["label"] for u in units]
    return info, labels, [u["text"] for u in units]
