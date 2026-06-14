"""Resolve frozen, cumulative benchmark manifests (``benchmarks/vN.yaml``).

A benchmark *version* is a frozen named SELECTION over the shared metric library,
expressed as ``extends: <prior>`` + ``add: [names]`` (METRICS_ROADMAP — Architecture).
Resolving ``vN`` walks the extends chain and unions the ``add`` lists into the full
cumulative metric set.

This module parses the manifests with a **tiny stdlib-only loader** for the trivial
subset they use (scalar ``key: value`` lines, ``null``, and one ``- item`` list).
The manifests are authored here and frozen, so a full YAML dependency would buy
nothing and would violate the stdlib-only core path (CLAUDE.md). If manifests ever
need real YAML, swap ``_load_manifest`` for a lazy ``import yaml``.
"""
from __future__ import annotations

import os
from typing import Optional

# repo_root/benchmarks  (this file is repo_root/utils/metrics/manifests.py)
_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
)


def _strip_comment(line: str) -> str:
    """Drop a ``#`` comment. Manifest values are unquoted, so this is safe."""
    return line.split("#", 1)[0]


def _coerce(value: str):
    v = value.strip()
    if v in ("null", "~", ""):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    return v.strip("'\"")


def _load_manifest(version: str) -> dict:
    """Parse one ``benchmarks/<version>.yaml`` into a dict (scalars + ``add`` list)."""
    path = os.path.join(_BENCHMARKS_DIR, f"{version}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no benchmark manifest: {path}")

    data: dict = {}
    current_list_key: Optional[str] = None
    with open(path) as f:
        for raw in f:
            line = _strip_comment(raw).rstrip()
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith("- "):  # item of the active list key
                if current_list_key is None:
                    raise ValueError(f"{path}: list item with no key: {raw!r}")
                data[current_list_key].append(_coerce(stripped[2:]))
                continue
            if ":" not in stripped:
                raise ValueError(f"{path}: cannot parse line: {raw!r}")
            key, _, rest = stripped.partition(":")
            key = key.strip()
            if rest.strip() == "":  # opens a list (e.g. "add:")
                data[key] = []
                current_list_key = key
            else:
                data[key] = _coerce(rest)
                current_list_key = None
    return data


def resolve(version: str, _seen: Optional[set] = None) -> dict:
    """Resolve a benchmark version to its cumulative metric set.

    Returns ``{"version", "chain": [oldest..newest], "metrics": [cumulative names],
    "legacy": {names declared by legacy (v1-pipeline) manifests}}``. ``metrics``
    preserves chain order (base first) and de-duplicates.
    """
    _seen = _seen if _seen is not None else set()
    if version in _seen:
        raise ValueError(f"cyclic 'extends' through {version}")
    _seen.add(version)

    manifest = _load_manifest(version)
    parent = manifest.get("extends")
    add = list(manifest.get("add", []))
    is_legacy = bool(manifest.get("legacy"))

    if parent:
        base = resolve(parent, _seen)
        chain = base["chain"] + [version]
        metrics = list(base["metrics"])
        legacy = set(base["legacy"])
    else:
        chain, metrics, legacy = [version], [], set()

    for name in add:
        if name not in metrics:
            metrics.append(name)
    if is_legacy:
        legacy.update(add)

    return {"version": version, "chain": chain, "metrics": metrics, "legacy": legacy}


def library_metrics(version: str) -> list[str]:
    """Cumulative metric names for ``version`` that are runnable utils/metrics modules.

    Filters out legacy v1-pipeline names (which have no module here) by intersecting
    with the installed library, preserving manifest order.
    """
    from . import available  # local import avoids a cycle at package import time

    have = set(available())
    return [name for name in resolve(version)["metrics"] if name in have]
