"""Phonetic clustering of character surnames across runs.

Closes the gap in report §6.6: the "V-surname" signature recurs by *sound* even
when exact tokens never repeat (Claude Opus 4.8 produced a distinct V-initial
surname in 8/10 runs with zero token-level repeats). The v1 name metric counts
repeated exact tokens and is blind to this. Here we key each surname by its
leading sound and report how many runs share each sound.

Keying (``leading_sound/1``) — the leading sound of the surname:
  * digraphs that are a single sound are kept whole (``SH``, ``CH``, ``TH``;
    ``PH``/``GH`` → ``F``, ``WH`` → ``W``);
  * a leading vowel keys to that vowel (A/E/I/O/U), so vowel-initial names do not
    collapse into one mega-bucket;
  * otherwise the leading consonant letter — so **V stays distinct from F**. The
    V-surname family (Vance/Voss/Venn/Vale) is the whole point of this metric, and
    a phonetic encoder like Double Metaphone would wrongly fold V into F.

Surnames are hyphen-split upstream (see ``_base``), so ``Okafor-Voss`` contributes
its ``Voss`` sound — the exact case the v1 token-splitter misses.

Caveats:
  * The unit is each PERSON span's *final token* — a surname for multi-token
    names, but a given name when spaCy tags one alone. This is kept deliberately
    (loose): a V-sound given name is still evidence of the signature.
  * Like the v1 name metric, this inherits spaCy NER false positives — e.g. a
    place such as "Veyra Colony" tagged ``PERSON`` can inflate a sound's run
    count. Cross-check against the raw PERSON lists before quoting a number
    (report §4.4, §6.6).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ._base import person_surnames_by_run

NAME = "phonetic_names"
METHOD = "leading_sound/1"

# Two-letter onsets that represent a single sound (kept whole instead of keying on
# just the first letter). PH/GH map to the F sound; everything else keys to itself.
_DIGRAPHS = {"SH": "SH", "CH": "CH", "TH": "TH", "PH": "F", "GH": "F", "WH": "W"}


def _lead_sound(name: str) -> str:
    """Leading-sound key for a surname (see module docstring)."""
    s = "".join(c for c in name.upper() if c.isalpha())
    if not s:
        return "?"
    if s[:2] in _DIGRAPHS:
        return _DIGRAPHS[s[:2]]
    return s[0]  # leading vowel keys to itself; consonant stays distinct (V != F)


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    surnames_by_run = person_surnames_by_run(responses, ctx)
    n_runs = len(surnames_by_run)

    runs_with_sound: dict[str, set[int]] = defaultdict(set)
    examples: dict[str, set[str]] = defaultdict(set)
    total_surnames = 0
    for i, surnames in enumerate(surnames_by_run):
        for surname in surnames:
            total_surnames += 1
            key = _lead_sound(surname)
            runs_with_sound[key].add(i)
            examples[key].add(surname)

    # A "repeated sound" = a leading sound appearing in more than one run.
    repeated = [
        {
            "sound": key,
            "runs": len(run_ids),
            "of": n_runs,
            "distinct_names": sorted(examples[key]),
        }
        for key, run_ids in sorted(
            runs_with_sound.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
        if len(run_ids) > 1
    ]
    max_recurrence = max((r["runs"] for r in repeated), default=0)

    return {
        "schema": "phonetic_names/1",
        "method": METHOD,
        "runs": n_runs,
        "total_names": total_surnames,
        "distinct_sounds": len(runs_with_sound),
        "max_sound_recurrence": f"{max_recurrence}/{n_runs}" if n_runs else "0/0",
        "repeated_sounds": repeated,
        "note": (
            "Recurrence of a character name's leading SOUND across runs, not the "
            "exact token; complements the v1 name-component metric (report §6.6). "
            "Counts each PERSON span's final token and may include spaCy NER false "
            "positives (e.g. a place mislabeled PERSON) — verify before quoting."
        ),
    }
