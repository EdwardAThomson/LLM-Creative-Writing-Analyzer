"""Scoring-only CLI for the Narrative Dynamics benchmark: no generation step.

    python -m benchmarks.narrative_dynamics <text-file | directory>
        [--benchmark nd1 | --metrics name,name]
        [--segmentation chapters|windows|md] [--window-words 1500]
        [--no-gutenberg-trim] [--include-front]
        [--judge-model MODEL | --dry-run]
        [--aliases aliases.json] [--reference ref.json]
        [--make-reference OUT.json]
        [--out-dir DIR] [--list]

Analyzes user-supplied text (one file, or every ``*.txt``/``*.md`` in a
directory) and writes, per document, a self-describing JSON sidecar
(``<stem>.nd.json``) and a text report (``<stem>.nd.txt``), following the
``utils/metrics`` retroactive-scorer pattern: input files are never touched.
``.md`` inputs are treated as canonical extracted Markdown (see
``extract.py``) and split on their ``# `` headings with no heuristics.

``--dry-run`` exercises the full pipeline with a zero-spend placeholder judge
(the output is stamped so its numbers cannot be mistaken for measurements).
``--make-reference`` additionally aggregates this run's documents into a
reference-distribution JSON for later ``--reference`` comparisons (this is how
the masters reference data gets built when a real run happens).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from . import (DEFAULT_BENCHMARK, SCHEMA, available, compute_document,
               resolve_benchmark, segmentation)
from .judge import DEFAULT_JUDGE_MODEL, AiHelperJudge, DryRunJudge, describe_judge
from .reference import compare, load_reference, make_reference
from .report import render_text

TEXT_EXTENSIONS = (".txt", ".md")


def _find_text_files(directory: str) -> list[str]:
    hits: list[str] = []
    for ext in TEXT_EXTENSIONS:
        hits.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
    # never re-ingest our own outputs
    return sorted(h for h in hits if not h.endswith(".nd.txt"))


def _title_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _effective_strategy(path: str, requested: str | None) -> str:
    """Resolve the segmentation strategy for one input file.

    ``.md`` inputs are canonical extracted Markdown (see ``extract.py``) and
    take the heuristic-free ``md`` splitter; heading heuristics never run on
    them (requesting ``chapters`` for an .md resolves to ``md`` too). An
    explicit ``windows`` request is honoured for any input. Default for raw
    text stays ``chapters``.
    """
    is_md = path.lower().endswith(".md")
    if requested in (None, "chapters"):
        return "md" if is_md else "chapters"
    return requested


def score_file(path: str, names, ctx: dict, args, benchmark_version) -> dict:
    """Segment, score, and write the sidecar + text report for one document."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    seg = segmentation.segment(
        text, strategy=_effective_strategy(path, args.segmentation),
        window_words=args.window_words,
        trim_gutenberg=not args.no_gutenberg_trim)
    # scoring-layer policy: the "(front)" unit is excluded unless --include-front;
    # the record lands in the sidecar so the exclusion is never silent
    units, front_record = segmentation.exclude_front_matter(
        seg["units"], include_front=args.include_front)
    seg_info = {k: v for k, v in seg.items() if k != "units"}
    seg_info["front_matter"] = front_record
    doc_ctx = dict(ctx)  # fresh per document: unit_tensions must not leak across
    doc_ctx["title"] = _title_from_path(path)
    metrics = compute_document(units, names, doc_ctx)
    result = {
        "schema": SCHEMA,
        "source": os.path.basename(path),
        "benchmark": benchmark_version,
        "metrics_run": list(metrics),
        "judge": describe_judge(ctx["judge"]),
        "segmentation": seg_info,
        "metrics": metrics,
    }
    comparison = None
    if args.reference:
        comparison = compare(metrics, args.reference)
        result["comparison"] = comparison

    out_dir = args.out_dir or os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, _title_from_path(path))
    with open(stem + ".nd.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(stem + ".nd.txt", "w", encoding="utf-8") as f:
        f.write(render_text(os.path.basename(path), result, comparison))
    print(f"wrote {stem}.nd.json / .nd.txt")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.narrative_dynamics", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?",
                        help="A text file, or a directory of *.txt/*.md files")
    parser.add_argument("--metrics", help="Comma-separated metric names (default: benchmark set)")
    parser.add_argument("--benchmark", default=None,
                        help=f"ndN manifest to run (default: {DEFAULT_BENCHMARK}); "
                             "mutually exclusive with --metrics")
    parser.add_argument("--segmentation", choices=["chapters", "windows", "md"],
                        default=None,
                        help="Unit strategy: chapter-heading detection (falls back to "
                             "windows), fixed ~N-word windows, or canonical-Markdown "
                             "splitting (default: chapters for raw text; md is "
                             "auto-selected for .md inputs)")
    parser.add_argument("--window-words", type=int,
                        default=segmentation.DEFAULT_WINDOW_WORDS,
                        help="Target words per window unit (default: %(default)s)")
    parser.add_argument("--no-gutenberg-trim", action="store_true",
                        help="Do not strip Project Gutenberg frontmatter/license")
    parser.add_argument("--include-front", action="store_true",
                        help="Score the pre-first-heading \"(front)\" unit too "
                             "(excluded by default: front matter distorts "
                             "per-unit stats; the sidecar records the exclusion)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="ai_helper model key for the judge (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Placeholder judge, zero LLM spend (pipeline check only)")
    parser.add_argument("--aliases",
                        help="JSON file mapping name variants to canonical names "
                             "(thread clustering)")
    parser.add_argument("--reference",
                        help="nd_reference JSON to compare aggregates against")
    parser.add_argument("--make-reference", metavar="OUT_JSON",
                        help="Also aggregate this run's documents into a reference file")
    parser.add_argument("--out-dir", help="Output directory (default: next to each input)")
    parser.add_argument("--list", action="store_true",
                        help="List available metrics and exit")
    args = parser.parse_args(argv)

    if args.list or not args.path:
        print("Available metrics:", ", ".join(available()) or "(none)")
        return 0 if args.list else 2

    if args.benchmark and args.metrics:
        print("error: pass only one of --benchmark / --metrics", file=sys.stderr)
        return 2

    benchmark_version = None
    if args.metrics:
        names = [s.strip() for s in args.metrics.split(",")]
        unknown = [n for n in names if n not in available()]
        if unknown:
            print(f"error: unknown metrics: {', '.join(unknown)}", file=sys.stderr)
            return 2
    else:
        benchmark_version = args.benchmark or DEFAULT_BENCHMARK
        try:
            names = resolve_benchmark(benchmark_version)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    ctx: dict = {}
    ctx["judge"] = DryRunJudge() if args.dry_run else AiHelperJudge(args.judge_model)
    if args.aliases:
        with open(args.aliases) as f:
            ctx["aliases"] = json.load(f)
    if args.reference:
        try:
            args.reference = load_reference(args.reference)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if os.path.isdir(args.path):
        files = _find_text_files(args.path)
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        print(f"error: no such file or directory: {args.path}", file=sys.stderr)
        return 1
    if not files:
        print(f"no {'/'.join(TEXT_EXTENSIONS)} files under {args.path}", file=sys.stderr)
        return 1

    scored: dict[str, dict] = {}
    failures = 0
    for path in files:
        print(f"[{os.path.basename(path)}]", file=sys.stderr)
        try:
            scored[_title_from_path(path)] = score_file(
                path, names, ctx, args, benchmark_version)
        except Exception as e:  # keep the batch alive; report at the end
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    if args.make_reference and scored:
        ref = make_reference(
            scored,
            description=(f"generated by python -m benchmarks.narrative_dynamics; "
                         f"judge {describe_judge(ctx['judge'])}"),
            benchmark=benchmark_version)
        with open(args.make_reference, "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=2)
        print(f"wrote reference: {args.make_reference}")

    print(f"scored {len(scored)}/{len(files)} documents")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
