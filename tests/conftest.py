"""Test harness bootstrap.

Production package import is intentionally avoided: ``utils/__init__.py`` eagerly
imports ``utils.text_analysis``, which pulls in ``sentence_transformers`` (a heavy
dep that is not installed in this environment). So ``from utils.metrics import X``
fails at import time. Instead we load each pure metric module directly by file
path, bypassing the package ``__init__``.

Only the stdlib-only metric modules can be loaded this way (they import nothing
beyond ``re``/``collections``/``math``). The heavy modules (``_base``,
``burstiness``, ``opening_lines``, ``phonetic_names``) are out of scope.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "utils" / "metrics"

# The benchmarks package (benchmarks/narrative_dynamics) imports only the stdlib
# at import time, so unlike `utils` it can be imported normally; it just needs
# the repo root on sys.path (pytest only adds tests/ for us).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_metric(name: str):
    """Import a metric module by file path (no package import side effects)."""
    spec = importlib.util.spec_from_file_location("m_" + name, _METRICS_DIR / (name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def metric():
    """Fixture returning the ``load_metric`` loader (session-cached modules)."""
    cache: dict = {}

    def _get(name: str):
        if name not in cache:
            cache[name] = load_metric(name)
        return cache[name]

    return _get
