"""One-time extraction: raw book text to canonical Markdown. Scoring-free.

    python -m benchmarks.narrative_dynamics.extract <file.txt>
        [--out <file.md>] [--expected-units N] [--no-gutenberg-trim]

The architecture decision this implements (2026-07): the chapter-heading
heuristics in ``segmentation.py`` are a PROPOSER, run once per book here, with
a human-checkable verification report; ongoing analysis consumes the canonical
Markdown this command emits (via the ``md`` segmentation strategy), never the
heuristics. Extraction runs the Gutenberg/back-matter trimming plus the
chapter proposer and writes:

* ``<out>.md`` — a provenance header (HTML comment: source filename, sha256,
  extraction date, tool version/commit) then one ``# <label>`` heading per
  unit with its body. Body lines that would read as headings (``# ...``) are
  escaped with a backslash; the md splitter unescapes them.
* a ``.extract.json`` sidecar next to the output — units found, labels in
  order, per-unit word counts, and warnings (front-unit summary, tail trim,
  dropped runts, screened candidate runs, stripped tails, dropped front
  scraps).
* a VERIFICATION REPORT on stdout. With ``--expected-units N`` the exit code
  is nonzero on a count mismatch, so fleet runs can gate on it.

No LLM, no metrics: pure stdlib, like the segmentation layer it drives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from . import segmentation

EXTRACT_SCHEMA = "nd-extract/1"
TOOL_NAME = "llm_creative_writing-analyser benchmarks.narrative_dynamics.extract"


def _tool_commit() -> str:
    """Short git commit of the tool, for provenance; "unknown" outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _escape_body(text: str) -> tuple[str, int]:
    """Backslash-escape body lines that would parse as top-level headings."""
    lines = text.split("\n")
    escaped = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i] = "\\" + line
            escaped += 1
    return "\n".join(lines), escaped


def render_markdown(units: list[dict], provenance: dict) -> tuple[str, int]:
    """Canonical Markdown: provenance comment, then ``# <label>`` + body per unit.

    Returns ``(markdown, n_escaped_lines)``.
    """
    head = "\n".join(f"{k}: {v}" for k, v in provenance.items())
    parts = [f"<!--\n{head}\n-->"]
    escaped_total = 0
    for u in units:
        body, escaped = _escape_body(u["text"])
        escaped_total += escaped
        parts.append(f"# {u['label']}\n\n{body}")
    return "\n\n".join(parts) + "\n", escaped_total


def _collect_warnings(seg_result: dict, escaped_lines: int) -> dict:
    """Sidecar/report warnings: everything extraction changed or left out."""
    units = seg_result["units"]
    detection = seg_result.get("chapter_detection") or {}
    front = [u for u in units if u["label"] == segmentation.FRONT_LABEL]
    warnings: dict = {}
    if front:
        u = front[0]
        warnings["front_unit"] = {
            "words": u["words"],
            "begins": " ".join(u["text"].split()[:15]),
        }
    if seg_result.get("tail_trim"):
        warnings["tail_trim"] = seg_result["tail_trim"]
    for key in ("dropped_runts", "screened_candidates", "stripped_tails",
                "dropped_front_words"):
        if detection.get(key):
            warnings[key] = detection[key]
    if seg_result["strategy_used"] != "chapters":
        warnings["strategy_fallback"] = seg_result["strategy_used"]
    if seg_result.get("note"):
        warnings["note"] = seg_result["note"]
    if escaped_lines:
        warnings["escaped_body_lines"] = escaped_lines
    return warnings


def _print_report(source: str, sidecar: dict) -> None:
    print("EXTRACTION REPORT")
    print(f"Source: {source} (sha256 {sidecar['sha256'][:12]}...)")
    print(f"Strategy: {sidecar['strategy_used']}")
    print(f"Units: {sidecar['n_units']} ({sidecar['total_words']} words)")
    labels = sidecar["labels"]
    shown = labels if len(labels) <= 12 else labels[:6] + ["..."] + labels[-5:]
    print("Labels: " + ", ".join(shown))
    warnings = sidecar["warnings"]
    if warnings:
        print("Warnings:")
        for key, value in warnings.items():
            if key == "front_unit":
                print(f"  - front unit kept: {value['words']} words, "
                      f"begins \"{value['begins']}\"")
            elif key == "tail_trim":
                print(f"  - tail trimmed at \"{value['marker']}\": "
                      f"{value['trimmed_words']} words of back-matter removed")
            elif key == "dropped_runts":
                print(f"  - dropped runt units (thin body): "
                      + ", ".join(f"{r['label']} ({r['words']}w)" for r in value))
            elif key == "screened_candidates":
                print(f"  - screened TOC/junction candidates: "
                      + ", ".join(s["text"] for s in value))
            elif key == "stripped_tails":
                print(f"  - stripped orphan tail lines: "
                      + "; ".join(f"{t['unit_label']}: {' / '.join(t['lines'])}"
                                  for t in value))
            elif key == "dropped_front_words":
                print(f"  - dropped pre-heading front scrap: {value} words")
            else:
                print(f"  - {key}: {value}")
    else:
        print("Warnings: none")
    if sidecar.get("expected_units") is not None:
        status = "OK" if sidecar["expected_match"] else "MISMATCH"
        print(f"Expected units: {sidecar['expected_units']}, "
              f"found {sidecar['n_units']}: {status}")


def extract_file(path: str, out_path: str, expected_units: int | None = None,
                 trim_gutenberg: bool = True) -> tuple[dict, str]:
    """Run trimming + the proposer on one file; return (sidecar_dict, markdown)."""
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    seg_result = segmentation.segment(text, strategy="chapters",
                                      trim_gutenberg=trim_gutenberg)
    units = seg_result["units"]
    provenance = {
        "schema": EXTRACT_SCHEMA,
        "source": os.path.basename(path),
        "sha256": sha,
        "extracted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": TOOL_NAME,
        "commit": _tool_commit(),
        "units": len(units),
    }
    markdown, escaped_lines = render_markdown(units, provenance)
    sidecar = {
        "schema": EXTRACT_SCHEMA,
        "source": os.path.basename(path),
        "sha256": sha,
        "extracted": provenance["extracted"],
        "tool": {"name": TOOL_NAME, "commit": provenance["commit"]},
        "out": os.path.basename(out_path),
        "strategy_used": seg_result["strategy_used"],
        "n_units": len(units),
        "total_words": seg_result["total_words"],
        "labels": [u["label"] for u in units],
        "unit_words": [u["words"] for u in units],
        "warnings": _collect_warnings(seg_result, escaped_lines),
    }
    if expected_units is not None:
        sidecar["expected_units"] = expected_units
        sidecar["expected_match"] = (len(units) == expected_units)
    return sidecar, markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.narrative_dynamics.extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Raw book text file (e.g. a Gutenberg .txt)")
    parser.add_argument("--out", help="Canonical Markdown output path "
                                      "(default: <input stem>.md next to the input)")
    parser.add_argument("--expected-units", type=int, default=None,
                        help="Ground-truth unit count; exit nonzero on mismatch")
    parser.add_argument("--no-gutenberg-trim", action="store_true",
                        help="Do not strip Project Gutenberg frontmatter/license")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2
    out_path = args.out or os.path.splitext(args.path)[0] + ".md"
    if os.path.abspath(out_path) == os.path.abspath(args.path):
        print("error: output path equals the input path", file=sys.stderr)
        return 2

    sidecar, markdown = extract_file(
        args.path, out_path, expected_units=args.expected_units,
        trim_gutenberg=not args.no_gutenberg_trim)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    sidecar_path = os.path.splitext(out_path)[0] + ".extract.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    _print_report(os.path.basename(args.path), sidecar)
    print(f"wrote {out_path} / {sidecar_path}")
    if sidecar.get("expected_units") is not None and not sidecar["expected_match"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
