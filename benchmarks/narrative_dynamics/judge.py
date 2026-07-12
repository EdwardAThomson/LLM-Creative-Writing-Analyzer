"""The LLM-judge seam for narrative-dynamics metrics.

All LLM traffic goes through one callable, ``judge(prompt) -> str``, carried in
the shared ``ctx`` dict under ``ctx["judge"]``. Three implementations:

* ``AiHelperJudge(model)``: routes through this repo's ``ai_helper.send_prompt``
  (lazy import, so nothing here needs the API SDKs until a real call happens).
* ``FakeJudge(responses)``: deterministic canned responses for tests; accepts a
  list (consumed in order) or a callable ``prompt -> str``.
* ``DryRunJudge()``: prompt-aware placeholder answers, so every metric is
  runnable end to end with zero spend (``--dry-run`` on the CLI). Its numbers
  are meaningless by design; the output is stamped accordingly.

JSON extraction and the one-retry protocol (``ask_json``) live here too, adapted
from the StoryDaemon study drivers, so every metric parses judge output the same
way and degrades gracefully instead of dying mid-batch.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional, Union

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"  # the study's annotator family


class JudgeError(RuntimeError):
    """A judge call or its parse failed after retries."""


class AiHelperJudge:
    """Route judge prompts through ai_helper.send_prompt (the repo's one dispatch seam)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        self.model = model
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        from ai_helper import send_prompt  # lazy: pulls the API SDKs

        self.calls += 1
        out = send_prompt(prompt, model=self.model)
        if out is None:
            raise JudgeError(f"model {self.model} returned no content")
        return out

    def describe(self) -> str:
        return f"ai_helper:{self.model}"


class FakeJudge:
    """Canned responses for tests: a list consumed in order, or a prompt->str callable."""

    def __init__(self, responses: Union[list, Callable[[str], str]]):
        self._fn = responses if callable(responses) else None
        self._queue = list(responses) if not callable(responses) else None
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        if not self._queue:
            raise JudgeError("FakeJudge exhausted: no canned response left")
        return self._queue.pop(0)

    def describe(self) -> str:
        return "fake"


class DryRunJudge:
    """Zero-spend placeholder judge. Recognizes each metric's prompt shape and
    returns syntactically valid, semantically meaningless answers."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if '"tension_level"' in prompt:
            return '{"tension_level": 5, "rationale": "dry-run placeholder"}'
        if '"principal_cast"' in prompt:
            return ('{"pov": "Placeholder Protagonist", '
                    '"principal_cast": ["Placeholder Protagonist", "Placeholder Ally"], '
                    '"strand": "dry-run placeholder strand"}')
        if "block-type labels" in prompt:
            n = len(re.findall(r"^\[\d+\] ", prompt, re.M))
            items = ", ".join(
                f'{{"n": {i}, "primary": "ACTION", "secondary": null}}'
                for i in range(1, n + 1)
            )
            return f"[{items}]"
        raise JudgeError("DryRunJudge: unrecognized prompt shape")

    def describe(self) -> str:
        return "dry-run"


def describe_judge(judge) -> str:
    fn = getattr(judge, "describe", None)
    return fn() if callable(fn) else getattr(judge, "__name__", "injected")


def require_judge(ctx: Optional[dict]):
    judge = (ctx or {}).get("judge")
    if judge is None:
        raise JudgeError(
            "no judge in ctx: set ctx['judge'] (AiHelperJudge / FakeJudge / DryRunJudge)"
        )
    return judge


# --- JSON extraction (adapted from the source study drivers) ------------------------

_FENCE = re.compile(r"```(json)?")
# repair a rare model typo: unterminated label string, e.g. `"secondary": "ACTION}`
_UNTERMINATED = re.compile(r':\s*"([A-Z_]+)\s*([,}\]])')


def extract_json_object(text: str) -> dict:
    text = _FENCE.sub("", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0), strict=False)


def extract_json_array(text: str) -> list:
    text = _FENCE.sub("", text)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in response")
    return json.loads(_UNTERMINATED.sub(r': "\1"\2', m.group(0)), strict=False)


def ask_json(judge, prompt: str, parse: Callable[[str], object], tries: int = 2):
    """Call the judge and parse; one re-ask on a malformed response (study protocol).

    Raises ``JudgeError`` after ``tries`` failures. Callers that prefer holes over
    hard failure catch it and record the error per unit/batch.
    """
    last = None
    for _ in range(tries):
        raw = judge(prompt)
        try:
            return parse(raw)
        except Exception as e:  # malformed response: re-ask once
            last = e
    raise JudgeError(f"judge response unparseable after {tries} tries: {last}")
