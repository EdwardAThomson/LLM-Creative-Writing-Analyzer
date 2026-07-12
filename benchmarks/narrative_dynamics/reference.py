"""Masters-comparison hook: reference distributions for narrative-dynamics aggregates.

A *reference file* is a JSON snapshot of the aggregate values a chosen reference
corpus produced (the intended first use: the 26-work masters corpus, generated
later, when a real judged run happens). This module ships the format, a loader
with validation, a builder (``make_reference``), and the comparison
(``compare``) that the report renders. No reference data ships with the code;
the format is the contract.

Format (``schema: nd_reference/1``)::

    {
      "schema": "nd_reference/1",
      "description": "<what corpus, which judge, when>",
      "benchmark": "nd1",
      "documents": ["dracula", ...],            # provenance of the values
      "metrics": {
        "tension_trajectory": {
          "mean_register": {"values": [...], "mean": .., "std": ..,
                             "min": .., "max": ..},
          ...
        },
        ...
      }
    }

Comparison keys per metric are the scalar aggregate fields in ``REFERENCE_KEYS``.
``compare`` reports, per key: the document's value, the reference mean and range,
and whether the value falls inside the observed reference range. Being outside
the range is a flag for attention, not a verdict (the reference corpora are
calibration, not pass/fail thresholds).
"""
from __future__ import annotations

import json
from typing import Optional

REFERENCE_SCHEMA = "nd_reference/1"

# Scalar aggregate fields worth comparing, per metric (dict path: result
# ["metrics"][metric]["aggregate"][key]).
REFERENCE_KEYS = {
    "tension_trajectory": [
        "mean_register", "std", "volatility", "calm_share", "high_share",
        "peak_height", "peak_position", "tail_mean", "final_tension",
    ],
    "block_rhythm": [
        "words_per_mode_segment", "max_segment_words", "switch_rate",
        "secondary_shading_rate", "setting_touch_rate",
        "interiority_share", "interiority_self_transition",
    ],
    "thread_architecture": [
        "n_threads_2plus", "switch_rate", "mean_run", "max_run",
        "first_convergence",
    ],
}


def _summary(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"values": values, "mean": round(mean, 3), "std": round(var ** 0.5, 3),
            "min": min(values), "max": max(values)}


def make_reference(documents: dict[str, dict], description: str = "",
                   benchmark: Optional[str] = None) -> dict:
    """Build a reference dict from scored documents.

    ``documents`` maps a document name to its narrative-dynamics result dict
    (the ``{"metrics": {...}}`` shape the CLI writes). Metrics or keys missing
    from a document are simply skipped.
    """
    metrics: dict = {}
    for metric, keys in REFERENCE_KEYS.items():
        per_key: dict = {}
        for key in keys:
            values = []
            for doc in documents.values():
                agg = (doc.get("metrics", {}).get(metric, {}) or {}).get("aggregate", {})
                v = agg.get(key)
                if isinstance(v, (int, float)):
                    values.append(v)
            if values:
                per_key[key] = _summary(values)
        if per_key:
            metrics[metric] = per_key
    return {
        "schema": REFERENCE_SCHEMA,
        "description": description,
        "benchmark": benchmark,
        "documents": sorted(documents),
        "metrics": metrics,
    }


def load_reference(path: str) -> dict:
    with open(path) as f:
        ref = json.load(f)
    if ref.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {REFERENCE_SCHEMA!r}, got {ref.get('schema')!r}")
    if not isinstance(ref.get("metrics"), dict):
        raise ValueError(f"{path}: missing 'metrics' mapping")
    return ref


def compare(result_metrics: dict, reference: dict) -> dict:
    """Compare one document's metric results against a reference.

    Returns ``{metric: {key: {"value", "ref_mean", "ref_min", "ref_max",
    "within_range"}}}`` for every reference key present on both sides.
    """
    out: dict = {}
    for metric, per_key in reference.get("metrics", {}).items():
        agg = (result_metrics.get(metric, {}) or {}).get("aggregate", {})
        rows: dict = {}
        for key, ref in per_key.items():
            value = agg.get(key)
            if not isinstance(value, (int, float)):
                continue
            rows[key] = {
                "value": value,
                "ref_mean": ref.get("mean"),
                "ref_min": ref.get("min"),
                "ref_max": ref.get("max"),
                "within_range": (ref.get("min") is not None
                                 and ref.get("max") is not None
                                 and ref["min"] <= value <= ref["max"]),
            }
        if rows:
            out[metric] = rows
    return out
