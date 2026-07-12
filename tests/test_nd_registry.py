"""Tests for the narrative_dynamics registry, ordering, and manifest resolution."""
from __future__ import annotations

import pytest

import benchmarks.narrative_dynamics as nd
from benchmarks.narrative_dynamics.judge import FakeJudge


def test_available_lists_exactly_the_metric_modules():
    assert nd.available() == ["block_rhythm", "tension_trajectory",
                              "thread_architecture"]


def test_resolve_nd1_manifest():
    assert nd.resolve_benchmark("nd1") == ["tension_trajectory", "block_rhythm",
                                           "thread_architecture"]


def test_resolve_unknown_manifest_raises():
    with pytest.raises(FileNotFoundError):
        nd.resolve_benchmark("nd99")


def test_metric_order_canonicalized():
    # tension must run before thread_architecture regardless of request order
    assert nd._ordered(["thread_architecture", "tension_trajectory"]) == [
        "tension_trajectory", "thread_architecture"]
    # unknown-to-order names go last, preserving request order
    assert nd._ordered(["zeta", "block_rhythm"]) == ["block_rhythm", "zeta"]


def test_compute_document_isolates_metric_failures():
    units = [{"index": 0, "label": "u0", "text": "hello world", "words": 2}]
    # no judge in ctx: every LLM metric fails, each captured per metric
    out = nd.compute_document(units, ["tension_trajectory"], ctx={})
    assert "error" in out["tension_trajectory"]
    assert "JudgeError" in out["tension_trajectory"]["error"]


def test_compute_document_runs_requested_metrics_only():
    units = [{"index": 0, "label": "u0", "text": "hello world there", "words": 3}]
    ctx = {"judge": FakeJudge(['{"tension_level": 4, "rationale": "r"}'])}
    out = nd.compute_document(units, ["tension_trajectory"], ctx)
    assert list(out) == ["tension_trajectory"]
    assert out["tension_trajectory"]["aggregate"]["mean_register"] == 4
