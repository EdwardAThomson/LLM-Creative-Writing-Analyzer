"""The anchored 0-10 dramatic-tension rubric (versioned artifact, ported verbatim).

PROVENANCE
----------
Source repo:   StoryDaemon (https://github.com/EdwardAThomson/StoryDaemon, local
               checkout /home/edward/Projects/StoryDaemon)
Source file:   novel_agent/agent/tension_scale.py (TENSION_ANCHORS)
Source commit: abb21b7be9ae5c42c710b406d58e906e8d8d1e50 (2026-07-12)
Prompt frame:  docs/MASTERS_THREADS_TENSION_STUDY.md chapter-scoring protocol
               (rate stakes/threat/uncertainty/pressure, not dramatic words;
               rate the unit as a whole; long units truncated head+tail).

Reliability as measured THERE (not here):
  On a 20-chapter stratified masters re-pass (annotator anthropic/claude-haiku-4.5
  via OpenRouter, first pass temperature 0, re-pass temperature 0.8, shuffled):
  within-1 agreement 20/20 = 100 percent, exact 17/20 = 85 percent, MAD 0.15.
  Single-annotator self-consistency, no human gold labels.

Re-verification caveat: see rubrics/__init__.py REVERIFICATION_CAVEAT. Those
numbers do not transfer to this harness automatically; re-verify before trusting.

The anchor definitions below are copied verbatim from the source (the scorer-side
``definition`` strings; the writer-side ``directive`` strings are out of scope for
a scoring-only benchmark and are not ported).
"""
from __future__ import annotations

from typing import List, NamedTuple

from . import REVERIFICATION_CAVEAT

RUBRIC_VERSION = "tension_anchors/1"

PROVENANCE = {
    "source_repo": "StoryDaemon",
    "source_file": "novel_agent/agent/tension_scale.py",
    "source_commit": "abb21b7be9ae5c42c710b406d58e906e8d8d1e50",
    "ported": "2026-07-12",
    "reliability_as_measured_there": (
        "20-chapter masters re-pass (claude-haiku-4.5, temp 0 vs 0.8 shuffled): "
        "within-1 100% (20/20), exact 85% (17/20), MAD 0.15; single-annotator "
        "self-consistency, no human gold labels"
    ),
    "reverification_note": REVERIFICATION_CAVEAT,
}


class TensionBand(NamedTuple):
    lo: int
    hi: int
    name: str
    definition: str


TENSION_ANCHORS: List[TensionBand] = [
    TensionBand(0, 1, "none", "calm, safe, no stakes or conflict"),
    TensionBand(2, 3, "minimal", "routine, faint unease or anticipation"),
    TensionBand(4, 5, "rising", "complications or open questions, outcome uncertain"),
    TensionBand(6, 7, "high",
                "active conflict, real danger, or significant stakes pressing now"),
    TensionBand(8, 9, "very high",
                "imminent threat, violence, or a critical irreversible decision happening now"),
    TensionBand(10, 10, "peak climax",
                "a story-defining, life-or-death moment at its breaking point"),
]


def band_for(level: float) -> TensionBand:
    """Return the band a 0-10 tension level falls in (clamped to range)."""
    lvl = max(0, min(10, int(round(level))))
    for band in TENSION_ANCHORS:
        if band.lo <= lvl <= band.hi:
            return band
    return TENSION_ANCHORS[-1]


def _range_label(band: TensionBand) -> str:
    return str(band.lo) if band.lo == band.hi else f"{band.lo}-{band.hi}"


def scorer_anchor_block() -> str:
    """The 'Anchors:' lines for the tension-scorer prompt (verbatim source rendering)."""
    lines = ["Anchors:"]
    for b in TENSION_ANCHORS:
        lines.append(f"- {_range_label(b):<4} {b.name}: {b.definition}")
    return "\n".join(lines)


TENSION_PROMPT_TEMPLATE = """You are rating the DRAMATIC TENSION of one unit of a longer narrative, 0 to 10.

Rate stakes, threat, uncertainty, and pressure on the point-of-view character's goals, NOT the presence of dramatic words. A quiet conversation can be highly tense; a loud action scene can be low-stakes. Rate the unit as a whole (its dominant register, letting a genuine climax within the unit raise the score). Use the FULL range; do not default to the middle.

{anchors}

Unit ({label}) from {title}:
\"\"\"
{text}
\"\"\"

Respond with JSON only, no other text:
{{"tension_level": <integer 0-10>, "rationale": "<one short sentence>"}}"""


def render_tension_prompt(title: str, label: str, text: str) -> str:
    return TENSION_PROMPT_TEMPLATE.format(
        anchors=scorer_anchor_block(), title=title, label=label, text=text
    )
