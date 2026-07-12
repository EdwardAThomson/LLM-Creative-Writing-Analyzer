"""Tests for the narrative-dynamics judge seam (fakes, dry-run, JSON parsing)."""
from __future__ import annotations

import pytest

from benchmarks.narrative_dynamics import judge as jd


# --- FakeJudge -----------------------------------------------------------------------

def test_fake_judge_list_consumed_in_order():
    j = jd.FakeJudge(["one", "two"])
    assert j("p1") == "one"
    assert j("p2") == "two"
    assert j.calls == 2
    assert j.prompts == ["p1", "p2"]


def test_fake_judge_exhausted_raises():
    j = jd.FakeJudge([])
    with pytest.raises(jd.JudgeError):
        j("p")


def test_fake_judge_callable_form():
    j = jd.FakeJudge(lambda prompt: prompt.upper())
    assert j("abc") == "ABC"
    assert j.describe() == "fake"


# --- DryRunJudge ---------------------------------------------------------------------

def test_dry_run_judge_tension_shape():
    j = jd.DryRunJudge()
    out = j('rate it. {"tension_level": <integer 0-10>}')
    assert jd.extract_json_object(out)["tension_level"] == 5


def test_dry_run_judge_cast_shape():
    j = jd.DryRunJudge()
    obj = jd.extract_json_object(j('respond {"principal_cast": [...]}'))
    assert obj["pov"]
    assert isinstance(obj["principal_cast"], list)


def test_dry_run_judge_block_shape_counts_paragraphs():
    prompt = ("You are annotating prose paragraphs with block-type labels.\n"
              "[1] First para.\n\n[2] Second para.\n\n[3] Third para.\n")
    arr = jd.extract_json_array(jd.DryRunJudge()(prompt))
    assert [x["n"] for x in arr] == [1, 2, 3]
    assert all(x["primary"] == "ACTION" for x in arr)


def test_dry_run_judge_unknown_prompt_raises():
    with pytest.raises(jd.JudgeError):
        jd.DryRunJudge()("tell me a story")


# --- JSON extraction -----------------------------------------------------------------

def test_extract_json_object_ignores_fences_and_prose():
    raw = 'Sure!\n```json\n{"tension_level": 7, "rationale": "x"}\n```\nDone.'
    assert jd.extract_json_object(raw)["tension_level"] == 7


def test_extract_json_object_missing_raises():
    with pytest.raises(ValueError):
        jd.extract_json_object("no json here")


def test_extract_json_array_repairs_unterminated_label():
    raw = '[{"n": 1, "primary": "ACTION", "secondary": "SETTING}]'
    arr = jd.extract_json_array(raw)
    assert arr[0]["secondary"] == "SETTING"


def test_extract_json_array_missing_raises():
    with pytest.raises(ValueError):
        jd.extract_json_array("{}")


# --- ask_json retry protocol ----------------------------------------------------------

def test_ask_json_retries_once_then_succeeds():
    j = jd.FakeJudge(["garbage", '{"a": 1}'])
    assert jd.ask_json(j, "p", jd.extract_json_object) == {"a": 1}
    assert j.calls == 2


def test_ask_json_raises_after_two_failures():
    j = jd.FakeJudge(["bad", "worse"])
    with pytest.raises(jd.JudgeError):
        jd.ask_json(j, "p", jd.extract_json_object)


def test_require_judge():
    with pytest.raises(jd.JudgeError):
        jd.require_judge({})
    sentinel = jd.FakeJudge([])
    assert jd.require_judge({"judge": sentinel}) is sentinel


def test_describe_judge_fallbacks():
    assert jd.describe_judge(jd.DryRunJudge()) == "dry-run"
    def plain(prompt):
        return ""
    assert jd.describe_judge(plain) == "plain"
