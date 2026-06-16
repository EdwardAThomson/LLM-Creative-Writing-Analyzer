"""Cliché / stock-phrase density — reliance on worn, generic phrasing.

NOTE ON FRAMING: this is **not** an AI-detector. Most of the phrase lexicon (the
opener, emotional-shorthand, and time-dilation clichés) are overused *human*
clichés — pulp-fiction staples that long predate LLMs. Elevated density flags
*generic prose*, whoever wrote it. The few signals that lean genuinely
model-ward are isolated: em-dash density and the "it's not just X, it's Y"
cadence, plus the high-FP single-word register list (reported separately).

Three independent signals, all length-normalized to **per 1000 words**:
  * ``cliche`` — multi-word stock-phrase hits (the trustworthy headline; low FP),
    with a per-category breakdown so the report can say *which kind* dominates.
  * ``slop_words`` — single LLM-register words (delve/tapestry/…). High FP in
    fiction, so kept as a SEPARATE sub-score, never folded into ``cliche``.
  * ``em_dash`` — em-dash (— or --) density; a structural cadence tell.

The lexicon is the metric's meaning, so it is versioned (``lexicon_version``) and
frozen once shipped; additions bump the version. Pure stdlib (regex + counts).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

NAME = "cliche_density"
# Lexicon v1 — frozen after a tuning pass over the 13-model / 140-run corpus
# (2026-06): dead-but-canonical clichés kept for genre-generality; mix_of_and
# narrowed to emotion pairs to kill literal-mixture FPs ("sweat and hydraulic");
# couldnt_help_but broadened to catch "can't help but"; "in the end" dropped
# (too ordinary); "testament" removed from the single-word list (the testament_to
# phrase already covers it). Additions bump this version.
LEXICON_VERSION = "1"

# Emotion words for the "a mix of <emotion> and <emotion>" cliché — requiring both
# slots be emotions removes literal-mixture false positives while keeping the
# emotional-shorthand tell. Curated, not exhaustive (recall cost is accepted).
_EMOTIONS = sorted(
    """
    fear dread terror panic anxiety unease apprehension relief hope despair
    joy happiness sadness sorrow grief anger rage fury resentment guilt shame
    pride love longing desire jealousy envy disgust contempt surprise shock
    disbelief suspicion distrust curiosity wonder awe excitement anticipation
    determination resolve confusion doubt regret nostalgia melancholy warmth
    tenderness affection gratitude admiration embarrassment frustration
    irritation annoyance bitterness contentment calm serenity exhilaration elation
    """.split()
)
_EMO = "|".join(_EMOTIONS)

# (category, label, pattern). Case-insensitive. Patterns are bounded (no nested
# unbounded quantifiers) to avoid catastrophic backtracking.
_LEXICON: list[tuple[str, str, str]] = [
    # A — narrative-opener clichés (strong tell, low FP)
    ("A_opener", "in_a_world_where", r"\bin a world where\b"),
    ("A_opener", "little_did_know", r"\blittle did (?:he|she|they|i|we|you) know\b"),
    ("A_opener", "once_upon_a_time", r"\bonce upon a time\b"),
    ("A_opener", "unbeknownst", r"\bunbeknownst to (?:him|her|them|us|me)\b"),
    ("A_opener", "dark_and_stormy", r"\bit was a dark and stormy night\b"),
    # B — summary / epithet abstractions (tell-not-show, low FP)
    ("B_epithet", "testament_to", r"\btestament to\b"),
    ("B_epithet", "stark_reminder", r"\ba stark reminder\b"),
    ("B_epithet", "beacon_of", r"\ba beacon of (?:hope|light)\b"),
    ("B_epithet", "glimmer_of_hope", r"\ba glimmer of hope\b"),
    ("B_epithet", "embodiment_of", r"\bthe embodiment of\b"),
    ("B_epithet", "force_to_reckon", r"\ba force to be reckoned with\b"),
    # C — emotional shorthand (the big bucket, medium FP)
    ("C_emotion", "couldnt_help_but", r"\b(?:can['’]?t|cannot|could ?n['’]?t|could not) help but\b"),
    ("C_emotion", "mix_of_and", rf"\ba mix(?:ture)? of (?:{_EMO}) and (?:{_EMO})\b"),
    ("C_emotion", "wave_washed_over", r"\bwave of \w+ washed over\b"),
    ("C_emotion", "pang_of", r"\ba pang of\b"),
    ("C_emotion", "heart_pounded", r"\b(?:his|her|my|their) heart (?:pounded|raced|hammered|thundered)\b"),
    ("C_emotion", "breath_caught", r"\b(?:his|her|my|their) breath caught\b"),
    ("C_emotion", "shiver_spine", r"\bshivers? (?:ran|crawl(?:ed|ing)|went|going) down (?:his|her|my|their|the) spine\b"),
    ("C_emotion", "sent_shivers", r"\bsent (?:a )?shivers? (?:down|up) (?:his|her|my|their|the) spine\b"),
    ("C_emotion", "lump_in_throat", r"\ba lump in (?:his|her|my|their) throat\b"),
    ("C_emotion", "eyes_widened", r"\b(?:his|her|my|their) eyes widened\b"),
    ("C_emotion", "tears_welled", r"\btears welled\b"),
    ("C_emotion", "single_tear", r"\ba single tear\b"),
    ("C_emotion", "blood_ran_cold", r"\b(?:his|her|my|their) blood ran cold\b"),
    ("C_emotion", "knuckles_white", r"\bknuckles (?:white|turning white|whitening)\b"),
    # D — time-dilation clichés (medium FP)
    ("D_time", "in_that_moment", r"\bin that moment\b"),
    ("D_time", "time_stood_still", r"\btime (?:seemed to )?(?:slow(?:ed|ly)?|stood still|stopped|stand still)\b"),
    ("D_time", "world_faded", r"\bthe world around (?:him|her|me|them) (?:faded|melted away|disappeared)\b"),
    ("D_time", "felt_like_eternity", r"\bfor what felt like (?:an )?(?:eternity|forever)\b"),
    ("D_time", "as_if_time_itself", r"\bas if time itself\b"),
    # E — transformation / closer clichés (medium FP; in_the_end is risky)
    ("E_closer", "never_be_the_same", r"\bnever be the same\b"),
    ("E_closer", "forever_changed", r"\bforever changed\b"),
    ("E_closer", "new_chapter", r"\ba new chapter\b"),
    ("E_closer", "beginning_of_the_end", r"\bthe beginning of the end\b"),
    ("E_closer", "and_so_it_began", r"\band so it began\b"),
    # F — structural rhetorical tics (GPT cadence)
    ("F_struct", "not_only_but", r"\bnot only\b[\w\s,'-]{1,40}?\bbut\b"),
    ("F_struct", "not_just_its", r"\bit[’']?s not just\b[\w\s,'-]{1,40}?\bit[’']?s\b"),
    ("F_struct", "more_than_just", r"\bmore than just\b"),
]

# G — single LLM-register words (HIGH FP in fiction → separate sub-score)
_SLOP_WORDS = frozenset(
    """
    delve tapestry realm beacon intricate nuanced multifaceted palpable myriad
    kaleidoscope symphony ethereal unwavering bustling nestled gleaming cacophony
    labyrinthine shimmering whirlwind
    """.split()
)

_COMPILED = [(cat, label, re.compile(pat, re.IGNORECASE)) for cat, label, pat in _LEXICON]
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SLOP_RE = re.compile(r"\b(" + "|".join(sorted(_SLOP_WORDS)) + r")\b", re.IGNORECASE)
_EM_DASH = re.compile(r"—|--")

_CATEGORIES = sorted({cat for cat, _, _ in _LEXICON})


def _per_1k(count: int, words: int) -> Optional[float]:
    return round(count / words * 1000, 2) if words else None


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    per_run = []
    for text in responses:
        words = len(_WORD.findall(text))

        phrase_hits: Counter = Counter()
        by_category: Counter = Counter()
        for cat, label, rx in _COMPILED:
            n = len(rx.findall(text))
            if n:
                phrase_hits[label] = n
                by_category[cat] += n
        cliche_total = sum(phrase_hits.values())

        slop = Counter(m.group(1).lower() for m in _SLOP_RE.finditer(text))
        slop_total = sum(slop.values())
        em = len(_EM_DASH.findall(text))

        per_run.append(
            {
                "words": words,
                "cliche": {
                    "hits": cliche_total,
                    "per_1k": _per_1k(cliche_total, words),
                    "distinct": len(phrase_hits),
                    "by_category": {c: by_category.get(c, 0) for c in _CATEGORIES},
                    "phrase_hits": dict(phrase_hits),  # only non-zero
                },
                "slop_words": {
                    "hits": slop_total,
                    "per_1k": _per_1k(slop_total, words),
                    "distinct": len(slop),
                    "word_hits": dict(slop),
                },
                "em_dash": {"count": em, "per_1k": _per_1k(em, words)},
            }
        )

    def _mean_per_1k(path: str) -> Optional[float]:
        vals = [r[path]["per_1k"] for r in per_run if r[path]["per_1k"] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "schema": "cliche_density/1",
        "lexicon_version": LEXICON_VERSION,
        "lexicon_size": len(_LEXICON),
        "method": "stock-phrase / register-word / em-dash hits per 1000 words",
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": {
            "cliche_per_1k": _mean_per_1k("cliche"),
            "slop_words_per_1k": _mean_per_1k("slop_words"),
            "em_dash_per_1k": _mean_per_1k("em_dash"),
        },
        "note": (
            "Cliché / stock-phrase density (per 1000 words) — a GENERIC-PROSE signal, "
            "not an AI detector: most phrases are overused human clichés. 'cliche' is "
            "the trustworthy multi-word headline; 'slop_words' (single register words) "
            "is high-FP and kept separate; 'em_dash' is a structural cadence tell."
        ),
    }
