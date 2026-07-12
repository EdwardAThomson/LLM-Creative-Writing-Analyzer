"""Deterministic cast-based thread clustering. Pure code, no LLM.

Port of the clustering rules from StoryDaemon's masters threads/tension study
(docs/MASTERS_THREADS_TENSION_STUDY.md, source commit
abb21b7be9ae5c42c710b406d58e906e8d8d1e50), fixed there before interpretation:

* A unit's signature is its canonical principal cast plus a ``pov:`` token
  (POV is the strongest thread-identity signal in epistolary/single-narrator
  books, so it participates in assignment similarity).
* Units are processed in narrative order. A thread's profile is the set of
  signature elements present in at least 50 percent of its units (majority cast).
* A unit joins the thread with the highest Jaccard overlap between its signature
  and the thread profile if that overlap is >= 0.3 (ties go to the most recently
  active thread); otherwise it opens a new thread.
* Convergence detection is cast-only (POV tokens excluded): a unit whose cast
  covers at least half of the cast profile of 2+ established (2+ unit) threads
  records a merge event at that unit.

Known edge (documented in the study): the rule fragments single-POV books whose
supporting cast rotates completely (The Thirty-Nine Steps reads as 5 threads at
0.3 but 1 at 0.2), so threshold sensitivity is always reported alongside.

Name canonicalization (honorific stripping plus an optional alias map) is also
ported; the alias lookup is two-level, raw form first, so 'mrs bennet' and
'mr bennet' stay distinct even though title-stripping would collapse them.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

DEFAULT_THETA = 0.3
SENSITIVITY_THETAS = (0.2, 0.3, 0.4)
CONVERGENCE_COVERAGE = 0.5   # unit cast must cover >= this fraction of a thread's cast profile
ESTABLISHED_MIN_UNITS = 2    # threads this large participate in convergence detection

TITLES_RE = re.compile(
    r"^(mr|mrs|miss|ms|dr|doctor|sir|lady|lord|count|countess|professor|prof|"
    r"sergeant|captain|colonel|rev|reverend|madame|mademoiselle|the|hon)\.?\s+", re.I)


def _base(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s'-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(name: str) -> str:
    """Lowercase, strip punctuation, iteratively strip leading honorifics."""
    s = _base(name)
    prev = None
    while prev != s:
        prev = s
        s = TITLES_RE.sub("", s)
    return s


def canon(name: str, aliases: Optional[dict] = None) -> str:
    """Canonical token for a name. Alias lookup is two-level: the raw lowercased
    form (titles kept) first, then the title-stripped form."""
    amap = aliases or {}
    raw = _base(name)
    if raw in amap:
        return amap[raw]
    s = norm_name(name)
    return amap.get(s, s)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def majority_profile(members: list[dict], field: str) -> set:
    """Elements present in >= 50 percent of the member units' ``field`` sets."""
    n = len(members)
    cnt = Counter(x for m in members for x in m[field])
    return {x for x, k in cnt.items() if k >= n / 2}


def build_signatures(cast_records: list[dict], aliases: Optional[dict] = None) -> list[dict]:
    """Canonicalize raw extraction records into clustering inputs.

    Input records: ``{"pov": str, "principal_cast": [str, ...]}`` per unit, in
    narrative order. Output per unit: ``{"cast": sorted canonical names,
    "sig": cast plus pov: token, "pov": canonical pov}``.
    """
    out = []
    for rec in cast_records:
        cast = sorted({canon(n, aliases) for n in rec.get("principal_cast", []) if norm_name(n)})
        pov = canon(rec.get("pov", ""), aliases) if rec.get("pov") else ""
        sig = sorted(set(cast) | ({f"pov:{pov}"} if pov else set()))
        out.append({"cast": cast, "sig": sig, "pov": pov})
    return out


def cluster(signatures: list[dict], theta: float = DEFAULT_THETA):
    """Cluster unit signatures into threads.

    Returns ``(threads, assign, merges)`` where ``threads`` is a list of
    ``{"id": int, "members": [unit_index, ...]}``, ``assign`` is the thread id per
    unit, and ``merges`` is ``[(unit_index, [thread_ids]), ...]`` convergence
    events (cast-only coverage of 2+ established threads).
    """
    threads: list[dict] = []       # {"id", "members": [idx], "_units": [sig dicts]}
    assign: list[int] = []
    merges: list[tuple] = []
    for i, u in enumerate(signatures):
        sims = [(jaccard(u["sig"], majority_profile(t["_units"], "sig")), t) for t in threads]
        best = max(sims, key=lambda x: (x[0], x[1]["members"][-1]))[1] if sims else None
        best_sim = max((s for s, _ in sims), default=0.0)
        covered = []
        for _, t in sims:
            if len(t["members"]) < ESTABLISHED_MIN_UNITS:
                continue
            p = majority_profile(t["_units"], "cast")
            if p and len(set(u["cast"]) & p) / len(p) >= CONVERGENCE_COVERAGE:
                covered.append(t["id"])
        if len(covered) >= 2:
            merges.append((i, sorted(covered)))
        if best is not None and best_sim >= theta:
            best["members"].append(i)
            best["_units"].append(u)
            assign.append(best["id"])
        else:
            t = {"id": len(threads), "members": [i], "_units": [u]}
            threads.append(t)
            assign.append(t["id"])
    return ([{"id": t["id"], "members": t["members"]} for t in threads], assign, merges)


def run_lengths(assign: list[int]) -> list[int]:
    """Lengths of consecutive same-thread runs, in order."""
    if not assign:
        return []
    runs = []
    cur = 1
    for a, b in zip(assign, assign[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def threshold_sensitivity(signatures: list[dict],
                          thetas=SENSITIVITY_THETAS) -> dict:
    """Thread count at each threshold (the study's honesty check on theta)."""
    return {str(th): len(cluster(signatures, th)[0]) for th in thetas}
