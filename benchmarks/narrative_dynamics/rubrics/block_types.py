"""The 7-type prose block rubric (versioned artifact, ported verbatim).

PROVENANCE
----------
Source repo:   StoryDaemon (https://github.com/EdwardAThomson/StoryDaemon, local
               checkout /home/edward/Projects/StoryDaemon)
Source files:  docs/BLOCK_DECOMPOSITION_STUDY.md (rubric + validated findings);
               annotation prompt from that study's driver script.
Source commit: abb21b7be9ae5c42c710b406d58e906e8d8d1e50 (2026-07-12)

Reliability as measured THERE (not here), annotator anthropic/claude-haiku-4.5
via OpenRouter, temperature 0, batches of 20 paragraphs:
  * Self-consistency: 96 percent (24/25) on contiguous in-context passages;
    80 percent on a hard, stratified, decontextualized 60-paragraph sample.
  * Cross-model agreement (vs google/gemini-3.5-flash): 80 percent (48/60),
    Cohen's kappa 0.75; in 10 of 12 disagreements one judge's secondary label
    equaled the other's primary.
  * Known-answer probes: 8/8 unambiguous probes recovered exactly, blind.
  * Coverage: an OTHER-allowed re-pass proposed OTHER for 2/60 (3.3 percent),
    both EXPOSITION (a low-frequency residual currently absorbed by LORE, the
    label with the worst cross-model agreement, 1/3). Treat LORE-based rates
    with wide error bars.

Validated findings the aggregations here port:
  * Words-per-mode-segment is the robust (word-normalized) form of the
    commitment finding; the paragraph-count form is a paragraph-length artifact.
    Masters hold a mode for ~90-188 words per segment (Conrad, the exception,
    58.6); generated prose measured 50-66.
  * The interiority EXIT rule is the load-bearing constraint, not the quantity:
    pooled master INTERIORITY self-transition 14.3 percent vs generated
    40.4 percent. An interiority share up to ~24 percent is master behavior;
    a self-transition rate above ~0.3 is not.
  * Secondary shading: masters shade ~26-54 percent of paragraphs with a
    secondary mode (generated ~24); masters touch SETTING (primary or
    secondary) in ~5-25 percent of paragraphs.

Re-verification caveat: see rubrics/__init__.py REVERIFICATION_CAVEAT. Those
numbers do not transfer to this harness automatically; re-verify before trusting.
"""
from __future__ import annotations

from . import REVERIFICATION_CAVEAT

RUBRIC_VERSION = "block_types/1"

PROVENANCE = {
    "source_repo": "StoryDaemon",
    "source_file": "docs/BLOCK_DECOMPOSITION_STUDY.md",
    "source_commit": "abb21b7be9ae5c42c710b406d58e906e8d8d1e50",
    "ported": "2026-07-12",
    "reliability_as_measured_there": (
        "claude-haiku-4.5 annotator: self-consistency 96% in-context / 80% "
        "decontextualized; cross-model (gemini-3.5-flash) 80%, kappa 0.75; "
        "known-answer probes 8/8; OTHER rate 3.3% (EXPOSITION residual in LORE, "
        "the noisiest label at 1/3 cross-model)"
    ),
    "reverification_note": REVERIFICATION_CAVEAT,
}

LABELS = ["SETTING", "CHARACTER_DESC", "LORE", "DIALOGUE", "ACTION",
          "INTERIORITY", "TRANSITION"]

# Carrier modes form runs and drive scenes; texture modes almost never run longer
# than one paragraph and mostly appear as secondary shading (study finding).
CARRIER_LABELS = ["DIALOGUE", "ACTION", "INTERIORITY"]
TEXTURE_LABELS = ["SETTING", "CHARACTER_DESC", "LORE"]

# Annotation prompt, verbatim from the source study's driver script.
ANNOTATION_RUBRIC = """You are annotating prose paragraphs from a novel with block-type labels.
Assign each paragraph exactly ONE primary label (its DOMINANT mode), and optionally
ONE secondary label if the paragraph is genuinely mixed. Labels:

- SETTING: description of place, atmosphere, weather, light, objects in the scene.
- CHARACTER_DESC: description of a character's appearance, dress, manner, bearing.
- LORE: history, backstory, flashback, world facts, how things came to be; includes
  one-sentence asides about the past or the wider world dropped into other material.
- DIALOGUE: paragraph dominated by quoted speech (a line of dialogue plus its tag
  counts as DIALOGUE even if short).
- ACTION: events happening now: movement, physical activity, things done or observed
  as they occur, procedural activity (searching, working, fighting, travelling).
- INTERIORITY: a character's thoughts, feelings, reasoning, deliberation, judgments.
- TRANSITION: connective tissue that moves time or place ("Three days later...",
  "They came at dusk to..."): brief, its job is the shift itself, not the new scene.

Rules:
- Pick the mode occupying the most words / carrying the paragraph's main job.
- Quoted speech with heavy narration around it: DIALOGUE only if speech dominates.
- First-person narration of what the narrator did = ACTION; of what they thought/felt
  = INTERIORITY; of what a place looked like = SETTING.
- Use TRANSITION sparingly: only when the shift is the paragraph's main content.

Respond with ONLY a JSON array, one object per paragraph, in the same order:
[{"n": 1, "primary": "SETTING", "secondary": null}, ...]
secondary must be a label string or null. No prose, no markdown fences."""

# Master reference bands for the four-signal structural gauge (study aggregates,
# six-master corpus; informative context shipped alongside the measurements, not
# pass/fail thresholds). Same provenance and caveat as the rubric.
MASTER_BANDS = {
    "words_per_mode_segment": {"master_range": [58.6, 187.8],
                               "note": "five of six masters 90-188; Conrad 58.6"},
    "interiority_self_transition": {"master_range": [0.0, 0.28],
                                    "note": "pooled masters 0.143; generated 0.404"},
    "secondary_shading_rate": {"master_range": [0.26, 0.54],
                               "note": "generated measured ~0.24"},
    "setting_touch_rate": {"master_range": [0.05, 0.25],
                           "note": "primary or secondary SETTING; generated ~0.09"},
}


def render_annotation_prompt(paragraphs: list[str]) -> str:
    lines = [ANNOTATION_RUBRIC, "", f"Paragraphs ({len(paragraphs)}):", ""]
    for i, p in enumerate(paragraphs, 1):
        lines.append(f"[{i}] {p}")
        lines.append("")
    return "\n".join(lines)
