"""Tests for the durable judge cache + call-budget governor (``cache.py``).

Two layers are covered:
  * unit level, directly against ``ask_json`` with a counting stub judge (no
    real API/CLI calls) -- cache hits, resume across separate ``JudgeCache``
    instances (simulating separate process runs), auto-invalidation on prompt
    change, failures never cached, the budget cap, and identity separation
    across different ``describe()`` strings;
  * CLI level, via ``main(argv)`` with ``--dry-run`` (still zero spend) --
    the cache file appears and is reused across two runs, ``--max-calls``
    stops cleanly with no incomplete sidecar and resumes on re-run, and
    ``--no-cache`` disables persistence.
"""
from __future__ import annotations

import json

import pytest

from benchmarks.narrative_dynamics import cache as ndcache
from benchmarks.narrative_dynamics import judge as jd
from benchmarks.narrative_dynamics.__main__ import main

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


class CountingJudge:
    """Fake judge: implements the judge interface (``__call__`` + ``describe``),
    counts calls, and records every prompt it was asked (to prove none repeat)."""

    def __init__(self, respond, judge_id="stub"):
        self._respond = respond  # callable(prompt) -> str
        self.calls = 0
        self.prompts: list[str] = []
        self._id = judge_id

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self._respond(prompt)

    def describe(self) -> str:
        return self._id


def _judge_for(mapping: dict, judge_id="stub") -> CountingJudge:
    """A CountingJudge that answers deterministically by prompt text."""
    return CountingJudge(lambda prompt: mapping[prompt], judge_id)


# --- unit level: ask_json + JudgeCache + CallBudget -----------------------------------

def test_cache_hit_makes_zero_new_calls(tmp_path):
    cache = ndcache.JudgeCache(str(tmp_path / "c.jsonl"))
    judge = _judge_for({"prompt-A": '{"v": 1}'})
    ctx = {"cache": cache}
    r1 = jd.ask_json(judge, "prompt-A", jd.extract_json_object, ctx=ctx)
    r2 = jd.ask_json(judge, "prompt-A", jd.extract_json_object, ctx=ctx)
    assert r1 == r2 == {"v": 1}
    assert judge.calls == 1  # second call was a cache hit
    cache.close()


def test_resume_reopens_cache_and_skips_completed(tmp_path):
    cache_path = str(tmp_path / "c.jsonl")
    prompts = ["p1", "p2", "p3"]
    responses = {"p1": '{"v": 1}', "p2": '{"v": 2}', "p3": '{"v": 3}'}

    # "run 1": budget allows only 2 real calls
    judge1 = _judge_for(responses)
    cache1 = ndcache.JudgeCache(cache_path)
    ctx1 = {"cache": cache1, "budget": ndcache.CallBudget(2)}
    results, stopped = [], False
    for p in prompts:
        try:
            results.append(jd.ask_json(judge1, p, jd.extract_json_object, ctx=ctx1))
        except ndcache.BudgetExhausted:
            stopped = True
            break
    cache1.close()
    assert stopped is True
    assert judge1.calls == 2
    assert len(results) == 2
    assert len(cache1) == 2  # both completed calls durable on disk

    # "run 2": fresh process would reopen the same cache path
    judge2 = _judge_for(responses)
    cache2 = ndcache.JudgeCache(cache_path)  # reloads the 2 prior entries
    ctx2 = {"cache": cache2, "budget": ndcache.CallBudget(None)}
    final = [jd.ask_json(judge2, p, jd.extract_json_object, ctx=ctx2) for p in prompts]
    cache2.close()

    assert final == [{"v": 1}, {"v": 2}, {"v": 3}]
    assert judge2.calls == 1  # only p3 was a real call this run
    all_prompts_sent = judge1.prompts + judge2.prompts
    assert all_prompts_sent == ["p1", "p2", "p3"]  # no prompt sent twice


def test_prompt_change_invalidates_only_that_entry(tmp_path):
    cache = ndcache.JudgeCache(str(tmp_path / "c.jsonl"))
    judge = _judge_for({
        "prompt-A": '{"v": 1}',
        "prompt-B": '{"v": 2}',
        "prompt-A-changed": '{"v": 99}',
    })
    ctx = {"cache": cache}
    r_a = jd.ask_json(judge, "prompt-A", jd.extract_json_object, ctx=ctx)
    r_b = jd.ask_json(judge, "prompt-B", jd.extract_json_object, ctx=ctx)
    assert judge.calls == 2

    # unchanged prompt: hit, no new call
    assert jd.ask_json(judge, "prompt-A", jd.extract_json_object, ctx=ctx) == r_a
    assert judge.calls == 2

    # changed prompt text: miss, re-calls
    r_changed = jd.ask_json(judge, "prompt-A-changed", jd.extract_json_object, ctx=ctx)
    assert r_changed == {"v": 99}
    assert judge.calls == 3

    # the untouched prompt-B still hits
    assert jd.ask_json(judge, "prompt-B", jd.extract_json_object, ctx=ctx) == r_b
    assert judge.calls == 3
    cache.close()


def test_only_successful_raw_is_cached_not_failed_attempt(tmp_path):
    cache = ndcache.JudgeCache(str(tmp_path / "c.jsonl"))
    attempts = iter(["not json at all", '{"v": 42}'])
    judge = CountingJudge(lambda prompt: next(attempts))
    ctx = {"cache": cache}

    result = jd.ask_json(judge, "p", jd.extract_json_object, ctx=ctx)
    assert result == {"v": 42}
    assert judge.calls == 2  # one failure, one re-ask that succeeded

    key = cache.make_key("p", judge.describe())
    assert cache.get(key) == '{"v": 42}'  # only the successful raw was cached
    cache.close()


def test_total_failure_not_cached_then_resume_succeeds(tmp_path):
    cache_path = str(tmp_path / "c.jsonl")

    judge1 = CountingJudge(lambda prompt: "still not json")
    cache1 = ndcache.JudgeCache(cache_path)
    with pytest.raises(jd.JudgeError):
        jd.ask_json(judge1, "p", jd.extract_json_object, ctx={"cache": cache1})
    assert judge1.calls == 2  # both tries exhausted
    assert len(cache1) == 0  # nothing durable: never cache a failure
    cache1.close()

    # later run: same (still-empty) cache, a judge that now succeeds
    judge2 = CountingJudge(lambda prompt: '{"v": 7}')
    cache2 = ndcache.JudgeCache(cache_path)
    result = jd.ask_json(judge2, "p", jd.extract_json_object, ctx={"cache": cache2})
    assert result == {"v": 7}
    assert judge2.calls == 1
    assert len(cache2) == 1
    cache2.close()


def test_budget_stops_after_exactly_k_real_calls(tmp_path):
    cache = ndcache.JudgeCache(str(tmp_path / "c.jsonl"))
    responses = {f"p{i}": json.dumps({"v": i}) for i in range(5)}
    judge = _judge_for(responses)
    budget = ndcache.CallBudget(3)
    ctx = {"cache": cache, "budget": budget}

    completed = 0
    stopped_exc = None
    for i in range(5):
        try:
            jd.ask_json(judge, f"p{i}", jd.extract_json_object, ctx=ctx)
            completed += 1
        except ndcache.BudgetExhausted as e:
            stopped_exc = e
            break

    assert completed == 3
    assert judge.calls == 3
    assert len(cache) == 3  # cache holds exactly the completed work
    assert stopped_exc is not None
    assert stopped_exc.calls_this_run == 3
    assert stopped_exc.max_calls == 3
    assert stopped_exc.cache_path == cache.path
    cache.close()


def test_different_judge_identities_do_not_collide(tmp_path):
    """dry-run-style vs model-style descriptors, same prompt: no collision."""
    cache = ndcache.JudgeCache(str(tmp_path / "c.jsonl"))
    judge_dry = _judge_for({"same-prompt": '{"v": "dry"}'}, judge_id="dry-run")
    judge_model = _judge_for({"same-prompt": '{"v": "real"}'},
                              judge_id="ai_helper:claude-cli-haiku")
    ctx = {"cache": cache}

    r1 = jd.ask_json(judge_dry, "same-prompt", jd.extract_json_object, ctx=ctx)
    r2 = jd.ask_json(judge_model, "same-prompt", jd.extract_json_object, ctx=ctx)
    assert r1 == {"v": "dry"}
    assert r2 == {"v": "real"}
    assert judge_dry.calls == 1
    assert judge_model.calls == 1
    assert len(cache) == 2  # two distinct entries, not one shared/colliding one

    # each judge's own subsequent ask on the same prompt is a hit
    assert jd.ask_json(judge_dry, "same-prompt", jd.extract_json_object, ctx=ctx) == r1
    assert jd.ask_json(judge_model, "same-prompt", jd.extract_json_object, ctx=ctx) == r2
    assert judge_dry.calls == 1
    assert judge_model.calls == 1
    cache.close()


def test_ask_json_uncached_when_no_ctx_matches_original_behaviour():
    """No ctx (or no cache/budget in it) reproduces the pre-cache behaviour exactly."""
    judge = _judge_for({"p": '{"v": 1}'})
    assert jd.ask_json(judge, "p", jd.extract_json_object) == {"v": 1}
    assert jd.ask_json(judge, "p", jd.extract_json_object) == {"v": 1}
    assert judge.calls == 2  # no cache in play: every call is real


# --- CLI level: --dry-run (zero spend) ------------------------------------------------

def test_cli_dry_run_cache_hit_on_rerun(tmp_path):
    p = _write_novel(tmp_path)
    assert main([str(p), "--dry-run", "--metrics", "tension_trajectory"]) == 0
    cache_path = tmp_path / "mini.nd.cache.jsonl"
    assert cache_path.exists()
    lines_first = cache_path.read_text().splitlines()
    assert len(lines_first) == 3  # one real call per chapter unit

    # re-run the identical command: every call should be a cache hit
    assert main([str(p), "--dry-run", "--metrics", "tension_trajectory"]) == 0
    lines_second = cache_path.read_text().splitlines()
    assert lines_second == lines_first  # no new lines appended


def test_cli_max_calls_stops_cleanly_then_resumes(tmp_path):
    p = _write_novel(tmp_path)
    rc = main([str(p), "--dry-run", "--metrics", "tension_trajectory",
               "--max-calls", "2"])
    assert rc == 3
    assert not (tmp_path / "mini.nd.json").exists()
    assert not (tmp_path / "mini.nd.txt").exists()
    cache_path = tmp_path / "mini.nd.cache.jsonl"
    assert cache_path.exists()
    lines = cache_path.read_text().splitlines()
    assert len(lines) == 2

    # re-run the same command, no cap: finishes using 2 cache hits + 1 new call
    rc2 = main([str(p), "--dry-run", "--metrics", "tension_trajectory"])
    assert rc2 == 0
    assert (tmp_path / "mini.nd.json").exists()
    lines = cache_path.read_text().splitlines()
    assert len(lines) == 3
    keys = [json.loads(line)["key"] for line in lines]
    assert len(keys) == len(set(keys))  # no prompt was ever answered twice


def test_cli_no_cache_flag_disables_persistence(tmp_path):
    p = _write_novel(tmp_path)
    rc = main([str(p), "--dry-run", "--metrics", "tension_trajectory", "--no-cache"])
    assert rc == 0
    assert (tmp_path / "mini.nd.json").exists()
    assert not (tmp_path / "mini.nd.cache.jsonl").exists()
