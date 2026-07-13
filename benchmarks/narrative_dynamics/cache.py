"""Durable judge-call cache + call-budget governor for narrative-dynamics runs.

Long ``nd1`` runs make hundreds of LLM calls (three metrics x many units); any
one of them can time out or the whole process can be killed partway through.
Two pieces make that survivable:

* ``JudgeCache``: an append-only JSONL of *successful* judge calls, keyed by a
  hash of ``(prompt, judge identity)``. ``ask_json`` (see ``judge.py``) checks
  it before calling the judge and appends to it after a call parses; a call
  that never parses is never written, so a re-run retries exactly the calls
  that didn't complete last time -- re-running the same command *is* resume.
  Each ``put`` is flushed + fsynced immediately, so a crash loses at most the
  one in-flight call.
* ``CallBudget`` / ``BudgetExhausted``: an optional cap on real (cache-miss)
  judge calls per run, so a long run can be deliberately paused (e.g. to check
  cost) and continued later by re-running the same command.

Both are backend-agnostic: they sit above the ``judge(prompt) -> str``
callable and never inspect which implementation is in use beyond its
``describe()`` string.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional


class BudgetExhausted(RuntimeError):
    """Raised by ``CallBudget.consume`` when the per-run call cap is reached.

    Deliberately NOT a ``JudgeError`` subclass: metrics catch ``JudgeError``
    around individual calls to turn a parse failure into a "hole", but a
    budget stop must propagate past that (and past ``compute_document``'s
    per-metric isolation) all the way to the CLI, which prints a resumable
    message and exits without writing the incomplete sidecar.
    """

    def __init__(self, calls_this_run: int, max_calls: int,
                 cache_path: Optional[str] = None):
        self.calls_this_run = calls_this_run
        self.max_calls = max_calls
        self.cache_path = cache_path
        msg = f"stopped at budget: {max_calls} calls this run"
        if cache_path:
            msg += f", cache at {cache_path}"
        else:
            msg += " (--no-cache: no progress was saved)"
        msg += ", re-run the same command to continue"
        super().__init__(msg)


class CallBudget:
    """Counts real (cache-miss) judge calls across a run; caps at ``max_calls``."""

    def __init__(self, max_calls: Optional[int] = None):
        self.max_calls = max_calls
        self.calls_made = 0

    def consume(self, cache_path: Optional[str] = None) -> None:
        """Register one real call about to be made; raise if the cap is hit first."""
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise BudgetExhausted(self.calls_made, self.max_calls, cache_path)
        self.calls_made += 1


class JudgeCache:
    """Append-only JSONL cache of successful judge calls.

    One line per successful call: ``{"key": ..., "judge": ..., "response": ...}``
    where ``response`` is the raw model text (re-parsed on a hit, so retry/repair
    logic never has to be duplicated). Existing entries are loaded into an
    in-memory index once at construction; new entries are appended and
    flushed+fsynced immediately.
    """

    def __init__(self, path: str):
        self.path = path
        self._index: dict[str, str] = {}
        self._load()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # tolerate a truncated last line from a mid-write crash
                self._index[rec["key"]] = rec["response"]

    @staticmethod
    def make_key(prompt: str, judge_id: str) -> str:
        """Hash of (prompt, judge identity): dry-run/model/version never collide."""
        h = hashlib.sha256()
        h.update(judge_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._index.get(key)

    def put(self, key: str, judge_id: str, response: str) -> None:
        rec = {"key": key, "judge": judge_id, "response": response}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._index[key] = response

    def close(self) -> None:
        self._fh.close()

    def __len__(self) -> int:
        return len(self._index)

    def __enter__(self) -> "JudgeCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
