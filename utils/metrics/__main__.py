"""Retroactively score saved runs with v2+ metrics — no regeneration needed.

    python -m utils.metrics <results.json> [--benchmark vN | --metrics name,name]
                            [--model NAME] [--out FILE] [--list]

Reads the v1 results JSON (``results[model]['responses'][i]['response_text']``),
runs the requested metrics per model, and writes a **sidecar** JSON. It never
touches the input file, so the frozen v1 artifacts stay byte-identical — which is
what lets the 2025/2026 corpora be re-scored with new metrics safely.
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m utils.metrics", description=__doc__)
    parser.add_argument("results_json", nargs="?", help="Path to a v1 results_*.json file")
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

    with open(args.results_json) as f:
        results = json.load(f)

    by_model = _extract(results, args.model)
    if not by_model:
        print(f"No matching model in {args.results_json}", file=sys.stderr)
        return 1

    ctx: dict = {}  # shared scratch (caches the spaCy model across models)
    out = {
        "schema": SCHEMA,
        "source": os.path.basename(args.results_json),
        "benchmark": benchmark_version,  # null for ad-hoc --metrics / default runs
        "metrics_run": names if names is not None else available(),
        "models": {},
    }
    for model, texts in by_model.items():
        print(f"scoring {model} ({len(texts)} runs)...", file=sys.stderr)
        out["models"][model] = compute(texts, names, ctx)

    out_path = args.out or (os.path.splitext(args.results_json)[0] + ".metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
