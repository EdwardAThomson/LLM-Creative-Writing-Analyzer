"""Block rhythm: per-paragraph block-type annotation plus mode-dynamics aggregations.

Each unit's paragraphs are annotated with the 7-type block rubric
(``rubrics/block_types.py``, ported from StoryDaemon with provenance) by an LLM
judge, in batches of 20, one re-ask per malformed batch; a batch that fails twice
becomes a hole (unlabeled paragraphs), never a hard failure.

The aggregations are the source study's *validated* findings, in their robust
forms:
  * type distribution and secondary-shading rate,
  * words-per-mode-segment (the word-normalized commitment gauge; the
    paragraph-count form was shown to be a paragraph-length artifact),
  * interiority self-transition rate (the exit rule: the study's single most
    actionable number),
  * setting-touch rate (primary or secondary),
plus the transition matrix and switch rate for context. The four validated
signals are grouped as ``structural_gauge`` with the master reference bands.

Runs and transitions never cross a unit boundary (study convention).
"""
from __future__ import annotations

from typing import Optional

from . import segmentation
from .judge import JudgeError, ask_json, extract_json_array, require_judge
from .rubrics import block_types

NAME = "block_rhythm"
REQUIRES_LLM = True
SCHEMA = "block_rhythm/1"

BATCH_SIZE = 20  # source-study batching
LABELS = block_types.LABELS


def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return round(x, nd) if x is not None else None


def _parse_batch(raw: str, expected_n: int) -> list[dict]:
    arr = extract_json_array(raw)
    if len(arr) != expected_n:
        raise ValueError(f"expected {expected_n} items, got {len(arr)}")
    out = []
    for i, item in enumerate(arr):
        if item.get("n") != i + 1:
            raise ValueError(f"item {i}: n mismatch ({item.get('n')})")
        prim = item.get("primary")
        if prim not in LABELS:
            raise ValueError(f"item {i}: bad primary {prim!r}")
        sec = item.get("secondary")
        if sec is not None and sec not in LABELS:
            raise ValueError(f"item {i}: bad secondary {sec!r}")
        out.append({"primary": prim, "secondary": sec})
    return out


def annotate_unit(paragraphs: list[str], ctx: dict) -> list[Optional[dict]]:
    """Label a unit's paragraphs in batches; failed batches yield None labels."""
    judge = require_judge(ctx)
    labels: list[Optional[dict]] = []
    for start in range(0, len(paragraphs), BATCH_SIZE):
        batch = paragraphs[start:start + BATCH_SIZE]
        prompt = block_types.render_annotation_prompt(batch)
        try:
            labels.extend(ask_json(judge, prompt,
                                    lambda raw: _parse_batch(raw, len(batch)), ctx=ctx))
        except JudgeError:
            labels.extend([None] * len(batch))
    return labels


# --- pure aggregations ----------------------------------------------------------------

def _annotated(unit_seqs: list[list[tuple]]) -> list[list[tuple]]:
    """Drop unlabeled paragraphs; a hole also splits the sequence so runs and
    transitions never span an unannotated gap."""
    out = []
    for seq in unit_seqs:
        cur: list[tuple] = []
        for item in seq:
            if item[1] is None:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(item)
        if cur:
            out.append(cur)
    return out


def distribution(seqs: list[list[tuple]]) -> dict:
    counts = {lab: 0 for lab in LABELS}
    n = 0
    for seq in seqs:
        for _, prim, _ in seq:
            counts[prim] += 1
            n += 1
    return {lab: _round(c / n) if n else None for lab, c in counts.items()}


def secondary_shading_rate(seqs: list[list[tuple]]) -> Optional[float]:
    labs = [(p, s) for seq in seqs for (_, p, s) in seq]
    return _round(sum(1 for _, s in labs if s) / len(labs)) if labs else None


def setting_touch_rate(seqs: list[list[tuple]]) -> Optional[float]:
    labs = [(p, s) for seq in seqs for (_, p, s) in seq]
    if not labs:
        return None
    return _round(sum(1 for p, s in labs if p == "SETTING" or s == "SETTING") / len(labs))


def mode_segments(seqs: list[list[tuple]]) -> list[tuple[str, int, int]]:
    """(label, n_paragraphs, n_words) per consecutive same-primary run, within units."""
    runs = []
    for seq in seqs:
        cur_lab, cur_n, cur_w = None, 0, 0
        for words, prim, _ in seq:
            if prim == cur_lab:
                cur_n += 1
                cur_w += words
            else:
                if cur_lab is not None:
                    runs.append((cur_lab, cur_n, cur_w))
                cur_lab, cur_n, cur_w = prim, 1, words
        if cur_lab is not None:
            runs.append((cur_lab, cur_n, cur_w))
    return runs


def words_per_segment(seqs: list[list[tuple]]) -> dict:
    runs = mode_segments(seqs)
    if not runs:
        return {"mean": None, "max": None, "n_segments": 0}
    words = [w for _, _, w in runs]
    return {"mean": _round(sum(words) / len(words), 1), "max": max(words),
            "n_segments": len(runs)}


def transition_matrix(seqs: list[list[tuple]]) -> dict:
    """Row-normalized primary -> next-primary probabilities, plus row counts."""
    counts = {a: {b: 0 for b in LABELS} for a in LABELS}
    for seq in seqs:
        labs = [p for _, p, _ in seq]
        for a, b in zip(labs, labs[1:]):
            counts[a][b] += 1
    mat = {}
    for a in LABELS:
        tot = sum(counts[a].values())
        mat[a] = {b: _round(counts[a][b] / tot) if tot else 0.0 for b in LABELS}
        mat[a]["_n"] = tot
    return mat


def interiority_dynamics(seqs: list[list[tuple]]) -> dict:
    n_all = sum(len(seq) for seq in seqs)
    n_int = sum(1 for seq in seqs for (_, p, _) in seq if p == "INTERIORITY")
    trans = {"INTERIORITY": 0, "ACTION": 0, "DIALOGUE": 0, "_other": 0}
    tot = 0
    for seq in seqs:
        labs = [p for _, p, _ in seq]
        for a, b in zip(labs, labs[1:]):
            if a == "INTERIORITY":
                tot += 1
                trans[b if b in trans else "_other"] += 1
    return {
        "share": _round(n_int / n_all) if n_all else None,
        "self_transition": _round(trans["INTERIORITY"] / tot) if tot else None,
        "exit_to_action": _round(trans["ACTION"] / tot) if tot else None,
        "exit_to_dialogue": _round(trans["DIALOGUE"] / tot) if tot else None,
        "n_transitions": tot,
    }


def switch_rate(seqs: list[list[tuple]]) -> Optional[float]:
    sw = tot = 0
    for seq in seqs:
        labs = [p for _, p, _ in seq]
        for a, b in zip(labs, labs[1:]):
            tot += 1
            sw += a != b
    return _round(sw / tot) if tot else None


def compute(units: list[dict], ctx: Optional[dict] = None) -> dict:
    ctx = ctx if ctx is not None else {}
    per_unit = []
    unit_seqs: list[list[tuple]] = []
    n_paras = n_failed = 0
    for u in units:
        paras = segmentation.split_paragraphs(u["text"])
        labels = annotate_unit(paras, ctx)
        seq = [(segmentation.word_count(p),
                lab["primary"] if lab else None,
                lab["secondary"] if lab else None)
               for p, lab in zip(paras, labels)]
        unit_seqs.append(seq)
        n_paras += len(paras)
        n_failed += sum(1 for lab in labels if lab is None)
        per_unit.append({
            "index": u["index"], "label": u["label"],
            "n_paragraphs": len(paras),
            "n_unlabeled": sum(1 for lab in labels if lab is None),
            "labels": [[p, s] for _, p, s in seq],
        })

    seqs = _annotated(unit_seqs)
    wps = words_per_segment(seqs)
    interiority = interiority_dynamics(seqs)
    shading = secondary_shading_rate(seqs)
    setting = setting_touch_rate(seqs)
    aggregate = {
        "n_paragraphs": n_paras,
        "n_unlabeled": n_failed,
        "distribution": distribution(seqs),
        "secondary_shading_rate": shading,
        "setting_touch_rate": setting,
        "words_per_mode_segment": wps["mean"],
        "max_segment_words": wps["max"],
        "n_segments": wps["n_segments"],
        "switch_rate": switch_rate(seqs),
        "interiority_share": interiority["share"],
        "interiority_self_transition": interiority["self_transition"],
        "interiority_exit_to_action": interiority["exit_to_action"],
        "interiority_exit_to_dialogue": interiority["exit_to_dialogue"],
    }
    return {
        "schema": SCHEMA,
        "rubric": {"version": block_types.RUBRIC_VERSION, **block_types.PROVENANCE},
        "method": (f"per-paragraph LLM annotation (batches of {BATCH_SIZE}, one "
                   "re-ask); deterministic dynamics over the labels; runs never "
                   "cross unit boundaries"),
        "per_unit": per_unit,
        "aggregate": aggregate,
        "transition_matrix": transition_matrix(seqs),
        "structural_gauge": {
            "words_per_mode_segment": wps["mean"],
            "interiority_self_transition": interiority["self_transition"],
            "secondary_shading_rate": shading,
            "setting_touch_rate": setting,
            "master_bands": block_types.MASTER_BANDS,
        },
        "note": ("structural_gauge groups the source study's four validated "
                 "signals; master_bands are calibration context from the source "
                 "harness, not pass/fail thresholds (see rubric provenance and "
                 "the re-verification caveat)."),
    }
