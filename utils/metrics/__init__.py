"""Modular, opt-in creative-writing metrics (v2+).

v1 lives in ``utils/text_analysis.py`` and is **frozen** — it carries the
longitudinal meaning of the 2025/2026 study and must not change. This package is
the growing library of *additional* metrics. Each metric is a module named after
the metric (filename == metric name) exposing the contract in ``_base.py``::

    NAME = "phonetic_names"
    def compute(responses: list[str], ctx: dict) -> dict: ...

Design notes (see METRICS_ROADMAP.md → Architecture):
  * Modules are keyed by stable name, organized by metric — never by version.
    Benchmark "versions" are cumulative manifests elsewhere, not subpackages here.
  * Heavy dependencies are imported lazily *inside* ``compute()`` so an unused
    metric never loads its dependency and the core/CLI path stays stdlib-only.
  * Results from here must be written to a separate sidecar, never merged into the
    v1 ``analysis`` dict (``write_json_results`` dumps the whole dict).
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable, Optional

SCHEMA = "metrics/1"


def available() -> list[str]:
    """Names of all metric modules in this package (filename == metric name).

    Discovery does not import the modules, so listing is free of their deps.
    """
    return sorted(
        m.name
        for m in pkgutil.iter_modules(__path__)
        if not m.name.startswith("_") and m.name != "__main__"
    )


def _load(name: str):
    mod = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(mod, "compute"):
        raise AttributeError(f"metric '{name}' has no compute(responses, ctx)")
    return mod


def compute(
    responses: list[str],
    names: Optional[Iterable[str]] = None,
    ctx: Optional[dict] = None,
) -> dict:
    """Run the named metrics (default: all available) over ``responses``.

    Returns ``{metric_name: result_dict}``. A metric that raises is captured as
    ``{"error": ...}`` so one failing metric never kills the batch. ``ctx`` is a
    shared scratch dict metrics may use to cache/communicate work (e.g. a loaded
    spaCy model under ``ctx['_nlp']``).
    """
    requested = list(names) if names is not None else available()
    ctx = ctx if ctx is not None else {}
    results: dict = {}
    for name in requested:
        try:
            results[name] = _load(name).compute(responses, ctx)
        except Exception as e:  # resilient: isolate per-metric failures
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    return results
