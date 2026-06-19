"""Retroactively score saved runs with v2+ metrics — no regeneration needed.

    python -m utils.metrics <results.json | results_dir/>
                            [--benchmark vN | --metrics name,name]
                            [--model NAME] [--out FILE] [--list]

Reads the v1 results JSON (``results[model]['responses'][i]['response_text']``),
runs the requested metrics per model, and writes a **sidecar** JSON. It never
touches the input file, so the frozen v1 artifacts stay byte-identical — which is
what lets the 2025/2026 corpora be re-scored with new metrics safely.

Pass a **directory** instead of a file to (re-)score every ``results_*.json`` under
it into its own sidecar — the batch shares one spaCy/embedding model load. Use this
to refresh the whole corpus whenever a metric or lexicon version changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from . import SCHEMA, available, compute


def _extract(results: dict, only_model: str | None) -> dict[str, list[str]]:
    """model -> [response_text, ...] from a v1 results dict."""
    texts: dict[str, list[str]] = {}
    for model, payload in results.items():
        if only_model and model != only_model:
            continue
        responses = (payload or {}).get("responses", [])
        texts[model] = [r.get("response_text", "") for r in responses]
    return texts


def _find_result_files(directory: str) -> list[str]:
    """Every ``results_*.json`` under ``directory`` (recursive), excluding sidecars."""
    hits = glob.glob(os.path.join(directory, "**", "results_*.json"), recursive=True)
    return sorted(f for f in hits if not f.endswith(".metrics.json"))


def _score_file(path: str, names, only_model, benchmark_version, ctx: dict, out_path=None) -> int:
    """Score one results file into its sidecar. Returns 0 on success, 1 if no model."""
    with open(path) as f:
        results = json.load(f)
    by_model = _extract(results, only_model)
    if not by_model:
        print(f"  no matching model in {path}", file=sys.stderr)
        return 1
    out = {
        "schema": SCHEMA,
        "source": os.path.basename(path),
        "benchmark": benchmark_version,  # null for ad-hoc --metrics / default runs
        "metrics_run": names if names is not None else available(),
        "models": {},
    }
    for model, texts in by_model.items():
        print(f"  scoring {model} ({len(texts)} runs)...", file=sys.stderr)
        out["models"][model] = compute(texts, names, ctx)
    out_path = out_path or (os.path.splitext(path)[0] + ".metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m utils.metrics", description=__doc__)
    parser.add_argument("results_json", nargs="?", help="A v1 results_*.json file, or a directory to score all of them")
    parser.add_argument("--metrics", help="Comma-separated metric names (default: all)")
    parser.add_argument(
        "--benchmark",
        help="Benchmark version (e.g. v2): run its cumulative library metrics. "
        "Mutually exclusive with --metrics.",
    )
    parser.add_argument("--model", help="Only score this model key")
    parser.add_argument("--out", help="Sidecar output path (default: <input>.metrics.json)")
    parser.add_argument("--list", action="store_true", help="List available metrics and exit")
    args = parser.parse_args(argv)

    if args.list or not args.results_json:
        print("Available metrics:", ", ".join(available()) or "(none)")
        return 0 if args.list else 2

    if args.benchmark and args.metrics:
        print("error: pass only one of --benchmark / --metrics", file=sys.stderr)
        return 2

    benchmark_version = None
    if args.benchmark:
        from ._manifests import library_metrics

        benchmark_version = args.benchmark
        try:
            names = library_metrics(args.benchmark)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not names:
            print(f"benchmark {args.benchmark} resolves to no library metrics", file=sys.stderr)
            return 1
    elif args.metrics:
        names = [s.strip() for s in args.metrics.split(",")]
    else:
        names = None

    ctx: dict = {}  # shared scratch: caches spaCy + embedding model across the batch

    if os.path.isdir(args.results_json):
        if args.out:
            print("error: --out is not valid in directory mode (each file gets its own sidecar)", file=sys.stderr)
            return 2
        files = _find_result_files(args.results_json)
        if not files:
            print(f"no results_*.json under {args.results_json}", file=sys.stderr)
            return 1
        failures = 0
        for path in files:
            print(f"[{os.path.basename(path)}]", file=sys.stderr)
            failures += _score_file(path, names, args.model, benchmark_version, ctx)
        print(f"scored {len(files) - failures}/{len(files)} files")
        return 1 if failures else 0

    return _score_file(args.results_json, names, args.model, benchmark_version, ctx, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
