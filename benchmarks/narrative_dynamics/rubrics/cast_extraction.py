"""The per-unit cast/strand extraction prompt (versioned artifact, ported verbatim).

PROVENANCE
----------
Source repo:   StoryDaemon (https://github.com/EdwardAThomson/StoryDaemon, local
               checkout /home/edward/Projects/StoryDaemon)
Source files:  docs/MASTERS_THREADS_TENSION_STUDY.md (protocol + reliability);
               cast prompt from that study's driver script.
Source commit: abb21b7be9ae5c42c710b406d58e906e8d8d1e50 (2026-07-12)

Reliability as measured THERE (not here), annotator anthropic/claude-haiku-4.5
via OpenRouter, 20-chapter stratified re-pass (temp 0 vs 0.8, shuffled):
  * Cast set overlap: mean Jaccard 0.95 (17/20 identical sets); the three
    disagreements were borderline-presence characters.
  * POV match: 20/20 = 100 percent.
  Single-annotator self-consistency, no human gold labels.

The deterministic clustering rules that consume these extractions (majority-cast
profile, Jaccard >= 0.3 assignment, cast-only convergence) are code, not prompt:
see ``clustering.py``, ported from the same study.

Re-verification caveat: see rubrics/__init__.py REVERIFICATION_CAVEAT. Those
numbers do not transfer to this harness automatically; re-verify before trusting.
"""
from __future__ import annotations

from . import REVERIFICATION_CAVEAT

RUBRIC_VERSION = "cast_extraction/1"

PROVENANCE = {
    "source_repo": "StoryDaemon",
    "source_file": "docs/MASTERS_THREADS_TENSION_STUDY.md",
    "source_commit": "abb21b7be9ae5c42c710b406d58e906e8d8d1e50",
    "ported": "2026-07-12",
    "reliability_as_measured_there": (
        "20-chapter re-pass (claude-haiku-4.5): cast mean Jaccard 0.95 "
        "(17/20 identical), POV match 100% (20/20); single-annotator "
        "self-consistency, no human gold labels"
    ),
    "reverification_note": REVERIFICATION_CAVEAT,
}

CAST_PROMPT_TEMPLATE = """You are a careful literary annotator. Below is one unit of a longer narrative ({title}). Identify the narrative strand this unit belongs to.

Respond with JSON only, no other text:
{{"pov": "<the point-of-view or focal character whose experience this unit follows; for diary/letter units, the writer of the dominant entry>",
 "principal_cast": ["<3 to 6 characters who are PRESENT in the unit and drive its events, including the POV character. Use each character's fullest canonical name (e.g. 'Mina Harker' not 'Mina', 'Elizabeth Bennet' not 'Lizzy'). Exclude characters who are only mentioned, remembered, or discussed in their absence>"],
 "strand": "<one line: which ongoing story strand is this unit advancing, stated as 'WHO is doing WHAT toward WHAT end'>"}}

UNIT ({label}):
\"\"\"
{text}
\"\"\""""


def render_cast_prompt(title: str, label: str, text: str) -> str:
    return CAST_PROMPT_TEMPLATE.format(title=title, label=label, text=text)
