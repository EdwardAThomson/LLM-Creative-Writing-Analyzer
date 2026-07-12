"""Tests for utils/metrics/_textmode.py (raw-text scoring-only input helper).

Loaded by file path like the other pure metric-library modules: importing the
``utils`` package would pull heavy deps (see conftest).
"""
from __future__ import annotations

import pytest

from conftest import load_metric

tm = load_metric("_textmode")


def test_single_file(tmp_path):
    p = tmp_path / "story.txt"
    p.write_text("Once there was a fox.")
    group, sources, texts = tm.collect_texts(str(p))
    assert group == "story"
    assert sources == ["story.txt"]
    assert texts == ["Once there was a fox."]


def test_directory_collects_txt_and_md_sorted(tmp_path):
    d = tmp_path / "corpus"
    (d / "sub").mkdir(parents=True)
    (d / "b.txt").write_text("B")
    (d / "a.md").write_text("A")
    (d / "sub" / "c.txt").write_text("C")
    (d / "ignored.json").write_text("{}")
    group, sources, texts = tm.collect_texts(str(d))
    assert group == "corpus"
    assert sources == ["a.md", "b.txt", "sub/c.txt"]
    assert texts == ["A", "B", "C"]


def test_directory_skips_nd_report_outputs(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("A")
    (d / "a.nd.txt").write_text("a narrative-dynamics report, not an input")
    _, sources, _ = tm.collect_texts(str(d))
    assert sources == ["a.txt"]


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tm.collect_texts(str(tmp_path / "nope"))


def test_empty_directory_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        tm.collect_texts(str(d))


def test_sidecar_path_for_file_and_directory(tmp_path):
    f = tmp_path / "story.txt"
    f.write_text("x")
    assert tm.sidecar_path(str(f), "story") == str(tmp_path / "story.metrics.json")
    d = tmp_path / "corpus"
    d.mkdir()
    assert tm.sidecar_path(str(d), "corpus") == str(d / "corpus.metrics.json")
