"""Per-unit temporal-placement rubric (versioned artifact, ported from StoryScope).

PROVENANCE
----------
Source project: StoryScope (Russell, Rajendhran, Pham, Iyyer, Wieting),
                "StoryScope: Investigating idiosyncrasies in AI fiction",
                arXiv:2604.03136 v4 (2026). Code + taxonomy:
                https://github.com/jenna-russell/storyscope
Source artifact: data/taxonomy.json (the released 304-feature NarraBench-grounded
                taxonomy), temporal_structure + events dimensions. The placement
                categories and the projection targets below are ported from these
                feature definitions (question + values + detection_method strings,
                condensed, not the full closed-form scales):
                  TMP_ORD_010  Degree of Chronological Discontinuity (scale 1-5)
                  TMP_ORD_001  Global Chronological Structure (5 categories)
                  TMP_ORD_004  Depth of Flashback Nesting (none/single/multi)
                  TMP_ORD_007  Number of interwoven temporal strands (1/2/3+)
                  EVT_TYP_009  Balance of present-time vs flashback events (3 cats)
Ported: 2026-07-17.

WHY A PER-UNIT PORT (not a verbatim whole-story rating). StoryScope rates each
~5,000-word story ONCE, whole. This benchmark's unit of analysis is the
segmentation unit, so the port judges each unit's dominant temporal placement in
isolation (exactly as tension_trajectory scores tension per unit) and derives the
discontinuity/nesting/strand quantities DETERMINISTICALLY from the placement
sequence. That keeps the full temporal trajectory (strictly richer than a single
whole-story label), scales to arbitrary length, and needs one judge call per unit.
The whole-story scales are reconstructed in ``chronology_map``'s
``storyscope_projection`` block for validation against their gold labels; those
projection bands are HEURISTIC and provisional (to be calibrated on their dev
split), the measured per-unit quantities are the primary output.

Reliability as measured THERE (not here):
  StoryScope annotator = Gemini 3 Flash (minimal thinking). Feature-assignment
  repeatability over 5 independent runs: Krippendorff's alpha = 0.88. Human
  validation on a 240-feature subset: mean Cohen's kappa = 0.84. These are
  corpus-wide, across all 304 features, NOT per-feature and NOT for this
  per-unit adaptation.

Re-verification caveat: see rubrics/__init__.py REVERIFICATION_CAVEAT. Those
numbers were measured with a different annotator model, a whole-story prompt, and
a different corpus; they do not transfer to this per-unit harness. Re-verify
(double-pass agreement on a stratified sample) before trusting findings here.
"""
from __future__ import annotations

from typing import List, NamedTuple

from . import REVERIFICATION_CAVEAT

RUBRIC_VERSION = "chronology_scale/1"

PROVENANCE = {
    "source_project": "StoryScope",
    "source_artifact": "data/taxonomy.json (temporal_structure + events dimensions)",
    "source_paper": "arXiv:2604.03136v4",
    "source_features": ["TMP_ORD_010", "TMP_ORD_001", "TMP_ORD_004",
                        "TMP_ORD_007", "EVT_TYP_009"],
    "ported": "2026-07-17",
    "adaptation": ("per-unit temporal-placement annotation; the whole-story scales "
                   "are reconstructed deterministically from the placement sequence "
                   "(see chronology_map.storyscope_projection)"),
    "reliability_as_measured_there": (
        "StoryScope annotator Gemini 3 Flash: corpus-wide feature-assignment "
        "repeatability Krippendorff alpha 0.88 (5 runs), human agreement mean "
        "Cohen kappa 0.84 on a 240-feature subset; not per-feature, whole-story "
        "prompt, not this per-unit adaptation"
    ),
    "reverification_note": REVERIFICATION_CAVEAT,
}


class Placement(NamedTuple):
    key: str
    name: str
    definition: str


# The unit's DOMINANT temporal position relative to the story's narrative present
# (the main forward-moving spine). Ported from TMP_ORD_001 / EVT_TYP_009 language.
PLACEMENTS: List[Placement] = [
    Placement("present", "present progression",
              "the unit unfolds along the story's forward-moving present timeline "
              "(the main spine of narrated action)"),
    Placement("analepsis", "flashback / recollection",
              "the unit is dominantly set EARLIER than the present spine: a "
              "flashback, memory, recounted case history, or retrospective scene"),
    Placement("prolepsis", "flash-forward",
              "the unit is dominantly set LATER than the present spine: a "
              "flash-forward, anticipated or foretold future scene"),
    Placement("mixed", "braided",
              "the unit substantially cross-cuts present and past (or future) "
              "within itself, rather than sitting mostly in one time"),
]
PLACEMENT_KEYS = [p.key for p in PLACEMENTS]

# Flashback nesting depth OBSERVED in this unit (ported from TMP_ORD_004).
NESTING_LEVELS = ["none", "single", "multi"]


def _placement_block() -> str:
    lines = ["Temporal placements (choose the single best fit for the unit as a whole):"]
    for p in PLACEMENTS:
        lines.append(f'- "{p.key}" ({p.name}): {p.definition}')
    return "\n".join(lines)


CHRONOLOGY_PROMPT_TEMPLATE = """You are mapping the TEMPORAL PLACEMENT of one unit of a longer narrative.

Relative to the story's "narrative present" (its main forward-moving spine of action), where in time does THIS unit mostly sit? Judge the unit as a whole (its dominant register), and judge story-time, not the order of sentences: a calm memory told in the present tense is still a flashback if it depicts earlier events.

{placements}

Also report:
- "internal_jump": true if a sharp time jump happens WITHIN this unit (a scene-to-scene leap backward or forward, a section break across months/years), false if the unit stays in one continuous time.
- "nesting": "none" if no flashback; "single" if it contains a flashback from the present; "multi" if a flashback sits inside another flashback (a remembered scene in which a character recalls a still-earlier event).

Unit ({label}) from {title}:
\"\"\"
{text}
\"\"\"

Respond with JSON only, no other text:
{{"placement": "<present|analepsis|prolepsis|mixed>", "internal_jump": <true|false>, "nesting": "<none|single|multi>", "rationale": "<one short sentence>"}}"""


def render_chronology_prompt(title: str, label: str, text: str) -> str:
    return CHRONOLOGY_PROMPT_TEMPLATE.format(
        placements=_placement_block(), title=title, label=label, text=text
    )
