"""One-time extraction: raw book text to canonical Markdown. Scoring-free.

    python -m benchmarks.narrative_dynamics.extract <file.txt>
        [--out <file.md>] [--expected-units N] [--no-gutenberg-trim]
        [--boundaries <file.json>]

The architecture decision this implements (2026-07): the chapter-heading
heuristics in ``segmentation.py`` are a PROPOSER, run once per book here, with
a human-checkable verification report; ongoing analysis consumes the canonical
Markdown this command emits (via the ``md`` segmentation strategy), never the
heuristics. Extraction runs the Gutenberg/back-matter trimming plus the
chapter proposer and writes:

* ``<out>.md`` — a provenance header (HTML comment: source filename, sha256,
  extraction date, tool version/commit) then one ``# <label>`` heading per
  unit with its body. Body lines that would read as headings (``# ...``) are
  escaped with a backslash; the md splitter unescapes them. When the same
  label occurs more than once (e.g. roman numerals restarting per book), the
  later occurrences are suffixed deterministically ("I.", "I. (2)", ...) so
  md headings stay unique; the sidecar keeps the raw text per unit
  (``label_raw``).
* a ``.extract.json`` sidecar next to the output — units found, labels in
  order, per-unit word counts (plus per-unit records with ``label_raw`` and
  override provenance), warnings (front-unit summary, tail trim, dropped
  runts, screened candidate runs, stripped tails, dropped front scraps), and
  ``unmatched_suspects``: standalone body lines that look structural but
  matched no heading pattern (purely diagnostic; review candidates for a
  ``--boundaries`` file).
* a VERIFICATION REPORT on stdout. With ``--expected-units N`` the exit code
  is nonzero on a count mismatch, so fleet runs can gate on it.
  ``--expected-units`` counts the ``(front)`` unit when it is kept.

``--boundaries <file.json>`` loads a reviewed data file of additional heading
lines: ``{"headings": [{"match": "<exact line text>", "label": "<optional>"},
...]}``. Each ``match`` is compared by exact stripped-line equality; matching
lines become boundaries exactly like pattern-matched headings (unit splitting,
exclusion from bodies, runt/front rules as usual). The sidecar records which
boundaries came from overrides, and an entry that matches zero lines is a
hard error (exit nonzero) so stale overrides are caught.

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


def load_boundaries(path: str) -> list[dict]:
    """Load and validate a ``--boundaries`` overrides file.

    Expected shape: ``{"headings": [{"match": str, "label": str?}, ...]}``.
    Raises ValueError with a human-readable message on any malformation.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("headings"), list):
        raise ValueError('boundaries file must be {"headings": [...]}')
    entries = []
    for k, e in enumerate(data["headings"]):
        if (not isinstance(e, dict) or not isinstance(e.get("match"), str)
                or not e["match"].strip()):
            raise ValueError(f'headings[{k}] needs a non-empty string "match"')
        if "label" in e and not isinstance(e["label"], str):
            raise ValueError(f'headings[{k}] "label" must be a string')
        entries.append({"match": e["match"].strip(),
                        "label": e.get("label")})
    if not entries:
        raise ValueError("boundaries file has no headings entries")
    return entries


def disambiguate_labels(units: list[dict]) -> list[dict]:
    """Suffix repeated labels deterministically: "I.", "I. (2)", "I. (3)"...

    Keeps md headings unique without inventing structure. The raw heading
    text is preserved on every unit as ``label_raw``.
    """
    seen: dict[str, int] = {}
    for u in units:
        raw = u["label"]
        n = seen[raw] = seen.get(raw, 0) + 1
        u["label_raw"] = raw
        if n > 1:
            u["label"] = f"{raw} ({n})"
    return units


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
    bo = sidecar.get("boundary_overrides")
    if bo:
        matched = sum(e["lines_matched"] for e in bo["entries"])
        print(f"Boundary overrides: {len(bo['entries'])} entries, "
              f"{matched} matched lines, {len(bo['boundaries'])} boundaries")
    suspects = sidecar.get("unmatched_suspects", [])
    print(f"Unmatched structural suspects: {len(suspects)}"
          + (" (diagnostic; lines listed in the sidecar)" if suspects else ""))
    if sidecar.get("expected_units") is not None:
        status = "OK" if sidecar["expected_match"] else "MISMATCH"
        print(f"Expected units: {sidecar['expected_units']}, "
              f"found {sidecar['n_units']}: {status}")


def extract_file(path: str, out_path: str, expected_units: int | None = None,
                 trim_gutenberg: bool = True,
                 overrides: list[dict] | None = None) -> tuple[dict, str]:
    """Run trimming + the proposer on one file; return (sidecar_dict, markdown)."""
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    seg_result = segmentation.segment(text, strategy="chapters",
                                      trim_gutenberg=trim_gutenberg,
                                      overrides=overrides)
    units = disambiguate_labels(seg_result["units"])
    suspects = segmentation.find_unmatched_suspects(units)
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
    unit_records = []
    for u in units:
        rec = {"label": u["label"], "label_raw": u.get("label_raw", u["label"]),
               "words": u["words"]}
        if u.get("source") == "override":
            rec["from_override"] = True
        unit_records.append(rec)
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
        "units": unit_records,
        "unmatched_suspects": suspects,
        "warnings": _collect_warnings(seg_result, escaped_lines),
    }
    if overrides is not None:
        detection = seg_result.get("chapter_detection") or {}
        sidecar["boundary_overrides"] = {
            "entries": detection.get("override_entries", []),
            "boundaries": detection.get("override_boundaries", []),
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
                        help="Ground-truth unit count; exit nonzero on mismatch "
                             "(counts the (front) unit when it is kept)")
    parser.add_argument("--no-gutenberg-trim", action="store_true",
                        help="Do not strip Project Gutenberg frontmatter/license")
    parser.add_argument("--boundaries", default=None,
                        help="Reviewed JSON data file of additional heading "
                             'lines: {"headings": [{"match": "<exact line '
                             'text>", "label": "<optional>"}]}. Matched by '
                             "exact stripped-line equality; matching lines "
                             "become boundaries exactly like pattern-matched "
                             "headings. An entry matching zero lines is a "
                             "hard error")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2
    out_path = args.out or os.path.splitext(args.path)[0] + ".md"
    if os.path.abspath(out_path) == os.path.abspath(args.path):
        print("error: output path equals the input path", file=sys.stderr)
        return 2
    overrides = None
    if args.boundaries:
        try:
            overrides = load_boundaries(args.boundaries)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"error: bad boundaries file {args.boundaries}: {e}",
                  file=sys.stderr)
            return 2

    sidecar, markdown = extract_file(
        args.path, out_path, expected_units=args.expected_units,
        trim_gutenberg=not args.no_gutenberg_trim, overrides=overrides)

    if overrides is not None:
        if sidecar["strategy_used"] != "chapters":
            print("error: --boundaries supplied but chapter segmentation "
                  f"fell back ({sidecar['strategy_used']}); nothing written",
                  file=sys.stderr)
            return 2
        stale = [e for e in sidecar["boundary_overrides"]["entries"]
                 if e["lines_matched"] == 0]
        if stale:
            for e in stale:
                print(f"error: boundary override matched zero lines: "
                      f"{e['match']!r}", file=sys.stderr)
            print("error: stale boundary overrides; nothing written",
                  file=sys.stderr)
            return 2

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
