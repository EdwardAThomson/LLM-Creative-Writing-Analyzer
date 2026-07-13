"""End-to-end tests: a synthetic multi-chapter text through scoring-only mode.

Covers both entry styles: the CLI (``python -m benchmarks.narrative_dynamics``,
here via ``main(argv)``) with the zero-spend dry-run judge, and the library path
(``segment`` + ``compute_document``) with scripted fake judge responses.
"""
from __future__ import annotations

import json
import re

import benchmarks.narrative_dynamics as nd
from benchmarks.narrative_dynamics import segmentation
from benchmarks.narrative_dynamics.__main__ import main
from benchmarks.narrative_dynamics.judge import FakeJudge
from benchmarks.narrative_dynamics.reference import load_reference

SYNTHETIC_NOVEL = """The Project Gutenberg eBook of Mini Novel

*** START OF THE PROJECT GUTENBERG EBOOK MINI NOVEL ***

CHAPTER I

Alice walked into the cold morning air and considered the road ahead. The village lay quiet under a thin frost, and the church bell counted seven. She pulled her coat tighter and set out along the lane toward the mill, where Tom would already be waiting with the cart.

"You are late," Tom said, not unkindly, when she arrived. "The river rose in the night. We shall have to take the long way round by the bridge."

They loaded the cart in silence. Alice thought about the letter in her pocket and what it might mean for both of them, and decided to say nothing until the crossing was behind them.

CHAPTER II

The bridge was out. Half its planking hung splintered over the brown flood, and a knot of villagers stood at the near end arguing about ropes. Tom swore softly and turned the horse.

"There is the ford at Hazel Bend," Alice said. "If we are quick."

The ford nearly took them. Midway across, the current caught the cart and swung it wide, and for a long moment Alice was certain the horse would go down. Tom hauled on the reins, shouting, and somehow they came up streaming on the far bank.

CHAPTER III

That evening, dry and shaken, they read the letter together by the fire. The mill was sold. The new owner would keep them both on, at better wages, and the long dread of the winter lifted all at once.

"Well," said Tom, and laughed for the first time in a week. "All that water for good news."

Alice folded the letter and watched the flames settle. Tomorrow there would be planking to haul for the bridge, and the whole village would turn out.

*** END OF THE PROJECT GUTENBERG EBOOK MINI NOVEL ***
"""


def _write_novel(tmp_path, name="mini.txt"):
    p = tmp_path / name
    p.write_text(SYNTHETIC_NOVEL, encoding="utf-8")
    return p


# --- CLI, dry-run judge ---------------------------------------------------------------

def test_cli_dry_run_single_file(tmp_path, capsys):
    p = _write_novel(tmp_path)
    rc = main([str(p), "--dry-run"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    assert result["schema"] == "narrative_dynamics/1"
    assert result["benchmark"] == "nd1"
    assert result["judge"] == "dry-run"
    assert result["segmentation"]["strategy_used"] == "chapters"
    assert result["segmentation"]["n_units"] == 3
    assert set(result["metrics"]) == {"tension_trajectory", "block_rhythm",
                                      "thread_architecture"}
    for m in result["metrics"].values():
        assert "error" not in m
    report = (tmp_path / "mini.nd.txt").read_text()
    assert "NARRATIVE DYNAMICS ANALYSIS" in report
    assert "dry-run judge; all LLM-derived numbers are placeholders" in report
    assert "TENSION TRAJECTORY" in report
    assert "BLOCK RHYTHM" in report
    assert "THREAD ARCHITECTURE" in report


def test_cli_dry_run_directory_and_make_reference(tmp_path):
    _write_novel(tmp_path, "one.txt")
    _write_novel(tmp_path, "two.txt")
    ref_path = tmp_path / "masters_ref.json"
    rc = main([str(tmp_path), "--dry-run", "--make-reference", str(ref_path)])
    assert rc == 0
    assert (tmp_path / "one.nd.json").exists()
    assert (tmp_path / "two.nd.json").exists()
    ref = load_reference(str(ref_path))
    assert ref["documents"] == ["one", "two"]
    assert "tension_trajectory" in ref["metrics"]


def test_cli_reference_comparison(tmp_path):
    p = _write_novel(tmp_path)
    ref_path = tmp_path / "ref.json"
    assert main([str(p), "--dry-run", "--make-reference", str(ref_path)]) == 0
    # second pass compares against the reference built from the same dry run:
    # every value must fall inside its (degenerate) reference range
    assert main([str(p), "--dry-run", "--reference", str(ref_path)]) == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    rows = [r for metric in result["comparison"].values() for r in metric.values()]
    assert rows and all(r["within_range"] for r in rows)
    assert "MASTERS COMPARISON" in (tmp_path / "mini.nd.txt").read_text()


def test_cli_metric_subset(tmp_path):
    p = _write_novel(tmp_path)
    rc = main([str(p), "--dry-run", "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    assert list(result["metrics"]) == ["tension_trajectory"]
    assert result["benchmark"] is None  # ad hoc selection, not a manifest


def test_cli_rejects_unknown_metric(tmp_path):
    p = _write_novel(tmp_path)
    assert main([str(p), "--dry-run", "--metrics", "nope"]) == 2


def test_cli_rejects_benchmark_plus_metrics(tmp_path):
    p = _write_novel(tmp_path)
    assert main([str(p), "--benchmark", "nd1", "--metrics", "tension_trajectory"]) == 2


def test_cli_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "tension_trajectory" in out


def _novel_with_front_and_catalog():
    """The synthetic novel plus 220 words of front matter and a publisher
    catalog after THE END (both inside the Gutenberg markers), padded so the
    end-marker sits in the last-5% zone; the shakedown Dracula shape."""
    front = " ".join(f"frontword{i}" for i in range(220))
    padding = ("The road ran on through the low hills and the weather held "
               "fair for the whole of that long uneventful day. ") * 150
    catalog = ("                                THE END\n\n"
               "     More stories of the sort you like; more than 500 titles in\n"
               "     the list on the wrapper. Ask for the publishers' catalog.\n\n"
               "STORIES BY J. S. FLETCHER\n\nGREEN INK\n\nTHE SAFETY PIN\n\n"
               "GROSSET & DUNLAP, Publishers, NEW YORK\n")
    text = SYNTHETIC_NOVEL.replace("***\n\nCHAPTER I", f"***\n\n{front}\n\nCHAPTER I")
    return text.replace(
        "\n\n*** END OF",
        f"\n\n{padding}\n\n{catalog}\n\n*** END OF")


def test_cli_front_excluded_by_default_and_recorded(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(_novel_with_front_and_catalog(), encoding="utf-8")
    rc = main([str(p), "--dry-run", "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    seg_info = result["segmentation"]
    assert seg_info["n_units"] == 4  # segmentation truth: (front) + 3 chapters
    fm = seg_info["front_matter"]
    assert fm["excluded"] is True
    assert fm["front_units"] == [{"index": 0, "label": "(front)", "words": 220}]
    assert fm["n_units_scored"] == 3
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert [u["label"] for u in per_unit] == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]
    assert [u["index"] for u in per_unit] == [1, 2, 3]  # traceable to segmentation
    report = (tmp_path / "mini.nd.txt").read_text()
    assert "excluded from scoring" in report
    assert "--include-front" in report


def test_cli_include_front_scores_the_front_unit(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(_novel_with_front_and_catalog(), encoding="utf-8")
    rc = main([str(p), "--dry-run", "--include-front",
               "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    fm = result["segmentation"]["front_matter"]
    assert fm["excluded"] is False
    assert fm["n_units_scored"] == 4
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert per_unit[0]["label"] == "(front)"


def _novel_with_preface():
    """The synthetic novel plus a PREFACE heading before CHAPTER I: author
    apparatus, not story."""
    preface = ("This edition has been prepared with care. " * 20)
    return SYNTHETIC_NOVEL.replace(
        "***\n\nCHAPTER I", f"***\n\nPREFACE\n\n{preface}\n\nCHAPTER I")


def test_cli_apparatus_excluded_by_default_and_recorded(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(_novel_with_preface(), encoding="utf-8")
    rc = main([str(p), "--dry-run", "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    seg_info = result["segmentation"]
    assert seg_info["n_units"] == 4  # segmentation truth: PREFACE + 3 chapters
    ap = seg_info["apparatus"]
    assert ap["excluded"] is True
    assert ap["apparatus_units"] == [{"index": 0, "label": "PREFACE", "words": 140}]
    assert ap["n_units_scored"] == 3
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert [u["label"] for u in per_unit] == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]
    report = (tmp_path / "mini.nd.txt").read_text()
    assert "excluded from scoring" in report
    assert "--include-apparatus" in report


def test_cli_include_apparatus_scores_the_preface(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(_novel_with_preface(), encoding="utf-8")
    rc = main([str(p), "--dry-run", "--include-apparatus",
               "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    ap = result["segmentation"]["apparatus"]
    assert ap["excluded"] is False
    assert ap["n_units_scored"] == 4
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert per_unit[0]["label"] == "PREFACE"


def test_cli_tail_trim_recorded_in_sidecar_and_report(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(_novel_with_front_and_catalog(), encoding="utf-8")
    assert main([str(p), "--dry-run", "--metrics", "tension_trajectory"]) == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    tt = result["segmentation"]["tail_trim"]
    assert tt is not None
    assert tt["marker"] == "THE END"
    assert "catalog" in tt["vocabulary_hits"]
    per_unit = result["metrics"]["tension_trajectory"]["per_unit"]
    assert all("GROSSET" not in u.get("label", "") for u in per_unit)
    report = (tmp_path / "mini.nd.txt").read_text()
    assert "Tail trim:" in report
    # the untouched synthetic novel records no trim and no front unit
    q = tmp_path / "plain.txt"
    q.write_text(SYNTHETIC_NOVEL, encoding="utf-8")
    assert main([str(q), "--dry-run", "--metrics", "tension_trajectory"]) == 0
    plain = json.loads((tmp_path / "plain.nd.json").read_text())
    assert plain["segmentation"]["tail_trim"] is None
    assert plain["segmentation"]["front_matter"] is None


def test_cli_windows_strategy(tmp_path):
    p = _write_novel(tmp_path)
    rc = main([str(p), "--dry-run", "--segmentation", "windows",
               "--window-words", "120", "--metrics", "tension_trajectory"])
    assert rc == 0
    result = json.loads((tmp_path / "mini.nd.json").read_text())
    assert result["segmentation"]["strategy_used"] == "windows"
    assert result["segmentation"]["n_units"] >= 2


# --- library path, scripted fake judge --------------------------------------------------

def _scripted_judge():
    """Prompt-aware fake: rising-then-falling tension, one shared-cast thread,
    a fixed block pattern per chapter (D/A/I with one SETTING shade)."""
    def fake(prompt):
        if '"tension_level"' in prompt:
            level = {"CHAPTER I": 3, "CHAPTER II": 7, "CHAPTER III": 2}[
                next(k for k in ("CHAPTER III", "CHAPTER II", "CHAPTER I") if k in prompt)]
            return json.dumps({"tension_level": level, "rationale": "scripted"})
        if '"principal_cast"' in prompt:
            return json.dumps({"pov": "Alice", "principal_cast": ["Alice", "Tom"],
                               "strand": "Alice and Tom cross the river"})
        if "block-type labels" in prompt:
            n = len(re.findall(r"^\[\d+\] ", prompt, re.M))
            labels = [("ACTION", "SETTING"), ("DIALOGUE", None), ("INTERIORITY", None)]
            return json.dumps([
                {"n": i + 1, "primary": labels[i % 3][0], "secondary": labels[i % 3][1]}
                for i in range(n)])
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")
    return FakeJudge(fake)


def test_full_pass_with_scripted_fakes():
    seg = segmentation.segment(SYNTHETIC_NOVEL, strategy="chapters")
    assert seg["n_units"] == 3
    ctx = {"judge": _scripted_judge()}
    out = nd.compute_document(seg["units"], None, ctx)

    tens = out["tension_trajectory"]["aggregate"]
    assert [r["tension"] for r in out["tension_trajectory"]["per_unit"]] == [3, 7, 2]
    assert tens["mean_register"] == 4.0
    assert tens["volatility"] == 4.5
    assert tens["peak_height"] == 7
    assert tens["ending_mode"] == "wind_down"

    blocks = out["block_rhythm"]["aggregate"]
    assert blocks["n_paragraphs"] == 9
    assert blocks["distribution"]["ACTION"] == round(3 / 9, 3)
    assert blocks["secondary_shading_rate"] == round(3 / 9, 3)
    assert blocks["setting_touch_rate"] == round(3 / 9, 3)
    assert blocks["switch_rate"] == 1.0  # the scripted pattern never repeats a mode

    threads = out["thread_architecture"]["aggregate"]
    assert threads["n_threads"] == 1
    assert threads["switch_rate"] == 0.0
    # tension coupling flowed through ctx from the tension metric
    assert out["thread_architecture"]["threads"][0]["mean_tension"] == 4.0
