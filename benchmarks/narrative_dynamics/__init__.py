"""Narrative Dynamics: the long-range story-structure benchmark (scoring-only).

The third benchmark in this repo, alongside the frozen v1 pipeline and the v2+
metrics library. Its object of study is different: not "how similar are N runs
of one prompt" but "what long-range structure does ONE arbitrary-length text
have": tension trajectory, block/mode rhythm, and thread architecture. It ports
StoryDaemon's validated gauges as versioned rubric artifacts (see ``rubrics/``,
each with a provenance header and a re-verification caveat).

Design mirrors ``utils/metrics``:
  * one module per metric, filename == metric name, exposing
    ``compute(units, ctx) -> dict`` (units come from ``segmentation.segment``);
  * benchmark versions are frozen manifests (``benchmarks/nd1.yaml``, its own
    series, same extends/add scheme as vN);
  * heavy/spendy work is behind one seam: ``ctx["judge"]`` (see ``judge.py``),
    so every metric runs with a real model, a test fake, or ``--dry-run``;
  * a metric that raises is captured per metric, never killing the batch.

Scoring-only CLI: ``python -m benchmarks.narrative_dynamics <file|dir>``.
This package imports only the stdlib until a real LLM call happens.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Iterable, Optional

SCHEMA = "narrative_dynamics/1"
DEFAULT_BENCHMARK = "nd1"

# Canonical execution order: tension first so thread_architecture can read
# ctx["unit_tensions"] for its switch-delta analysis.
METRIC_ORDER = ["tension_trajectory", "block_rhythm", "thread_architecture"]

_NON_METRIC_MODULES = {"segmentation", "judge", "clustering", "reference",
                       "report", "rubrics", "__main__"}

_BENCHMARKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available() -> list[str]:
    """Names of metric modules in this package (discovery imports nothing)."""
    return sorted(
        m.name for m in pkgutil.iter_modules(__path__)
        if not m.name.startswith("_") and m.name not in _NON_METRIC_MODULES
    )


def _ordered(names: Iterable[str]) -> list[str]:
    names = list(names)
    return ([n for n in METRIC_ORDER if n in names]
            + [n for n in names if n not in METRIC_ORDER])


def _load(name: str):
    mod = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(mod, "compute"):
        raise AttributeError(f"metric '{name}' has no compute(units, ctx)")
    return mod


def compute_document(units: list[dict], names: Optional[Iterable[str]] = None,
                     ctx: Optional[dict] = None) -> dict:
    """Run the named metrics (default: all) over one document's units.

    Returns ``{metric_name: result_dict}``. A metric that raises is captured as
    ``{"error": ...}`` so one failing metric never kills the run. Metrics run in
    canonical order (tension before threads) regardless of request order.
    """
    requested = _ordered(names if names is not None else available())
    ctx = ctx if ctx is not None else {}
    results: dict = {}
    for name in requested:
        try:
            results[name] = _load(name).compute(units, ctx)
        except Exception as e:  # resilient: isolate per-metric failures
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    return results


# --- manifest resolution (the ndN series) --------------------------------------------
# Self-contained twin of utils/metrics/_manifests.py: importing that module would
# trigger the eager utils/__init__ (heavy deps), which this package must not do.

def _load_manifest(version: str) -> dict:
    path = os.path.join(_BENCHMARKS_DIR, f"{version}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no benchmark manifest: {path}")
    data: dict = {}
    current_list_key: Optional[str] = None
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith("- "):
                if current_list_key is None:
                    raise ValueError(f"{path}: list item with no key: {raw!r}")
                data[current_list_key].append(stripped[2:].strip().strip("'\""))
                continue
            if ":" not in stripped:
                raise ValueError(f"{path}: cannot parse line: {raw!r}")
            key, _, rest = stripped.partition(":")
            key, rest = key.strip(), rest.strip()
            if rest == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = None if rest in ("null", "~") else rest.strip("'\"")
                current_list_key = None
    return data


def resolve_benchmark(version: str, _seen: Optional[set] = None) -> list[str]:
    """Cumulative metric names for an ndN manifest (walks the extends chain)."""
    _seen = _seen if _seen is not None else set()
    if version in _seen:
        raise ValueError(f"cyclic 'extends' through {version}")
    _seen.add(version)
    manifest = _load_manifest(version)
    parent = manifest.get("extends")
    metrics = resolve_benchmark(parent, _seen) if parent else []
    for name in manifest.get("add", []):
        if name not in metrics:
            metrics.append(name)
    return metrics
