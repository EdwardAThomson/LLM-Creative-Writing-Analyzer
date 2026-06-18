"""Opening-line diversity — how alike the first sentences of the N runs are.

Identical or formulaic openings are a notorious model tell that whole-text
similarity dilutes (METRICS_ROADMAP Phase 1): if 6 of 10 runs open with "The year
was 2147…", the full-text similarity barely moves but the opening is a dead
giveaway. This metric looks only at the first sentence of each run.

Three lenses (high similarity / low distinct count = formulaic openings):
  * lexical — mean & max pairwise **Jaccard** (word-set overlap, order-insensitive)
    and **difflib sequence ratio** (phrasing-level, catches near-duplicates);
    plus the most common opening prefix (first 3 words) and its share, the most
    legible form of the tell, and the count of distinct openings.
  * semantic — mean/median/min/max pairwise cosine over sentence embeddings
    (``all-MiniLM-L6-v2``, the same model as the v1 semantic metric, so the numbers
    are comparable). Catches *paraphrased* openings lexical overlap misses.

The actual opening lines are echoed (truncated) in the output so a high score can
be eyeballed. Openings come from the shared spaCy segmenter (``sentences_by_run``).
Lexical lenses are stdlib (difflib + sets + Counter); the semantic lens lazily
loads the embedding model.

Title handling: models often emit a ``Title: "…"`` / markdown-heading line before
the prose, and spaCy frequently merges it into the first "sentence". Such leading
scaffolding is stripped before segmentation so the opening is the first PROSE line
(``title_blocks_stripped`` reports how many runs had one); ``suspected_title_openings``
then flags any short title-like opening that survived stripping.
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

from ._base import get_sentence_model, sentences_by_run

NAME = "opening_lines"
_PREFIX_WORDS = 3
_ECHO_CHARS = 140

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_WS = re.compile(r"\s+")
_TITLE_MAX_WORDS = 6

# A leading line is scaffolding (stripped before segmentation) if it is a markdown
# heading, a "Title:/Chapter:/Prologue" label, or a line that is ENTIRELY a quoted
# title (≤80 chars). Real opening dialogue ("Run!" she shouted.) has text after the
# quote, so it does not match the whole-line-quoted patterns.
_SCAFFOLD_LINE = re.compile(
    r"""^\s*(?:
        \#{1,6}\s+.* |                                   # markdown heading
        (?:title|chapter|prologue|epilogue|epigraph)\s*[:.\-—].* |  # labelled heading
        ["“][^"”]{1,80}["”] |                            # bare double-quoted title
        ['‘][^'’]{1,80}['’]                              # bare single-quoted title
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _strip_scaffolding(text: str) -> tuple[str, bool]:
    """Drop leading blank/title/heading lines so the opening is the first PROSE line.

    Returns (cleaned_text, stripped_anything). Falls back to the original text if
    stripping would remove everything.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or _SCAFFOLD_LINE.match(lines[i])):
        i += 1
    cleaned = "\n".join(lines[i:]).strip()
    return (cleaned, i > 0) if cleaned else (text, False)


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def _normalize(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip(" \t\n\"'“”‘’.!?—-")


def _looks_like_title(sentence: str) -> bool:
    """Short first segment with no sentence-terminal punctuation → probably a title."""
    words = sentence.split()
    return len(words) <= _TITLE_MAX_WORDS and not re.search(r"[.!?…]\s*$", sentence.strip())


def _pairwise(values: list, fn) -> dict:
    scores = [fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    if not scores:
        return {"mean": None, "max": None}
    return {"mean": _round(sum(scores) / len(scores)), "max": _round(max(scores))}


def _semantic(openings: list[str], ctx: Optional[dict]) -> dict:
    if len(openings) < 2:
        return {"note": "needs >=2 usable openings"}
    try:
        import numpy as np  # lazy

        model = get_sentence_model(ctx)
        emb = np.asarray(model.encode(openings), dtype="float32")
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sims = emb @ emb.T
        iu = np.triu_indices(len(openings), k=1)
        scores = sims[iu]
        return {
            "model": "all-MiniLM-L6-v2",
            "mean": _round(float(scores.mean())),
            "median": _round(float(np.median(scores))),
            "min": _round(float(scores.min())),
            "max": _round(float(scores.max())),
        }
    except ImportError:
        return {"error": "sentence-transformers / numpy not installed"}
    except Exception as e:  # never let the heavy path kill the metric
        return {"error": f"{type(e).__name__}: {e}"}


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    cleaned = [_strip_scaffolding(t) for t in responses]
    title_blocks_stripped = sum(1 for _, stripped in cleaned if stripped)
    sents_by_run = sentences_by_run([c for c, _ in cleaned], ctx)
    raw_openings = [s[0] if s else None for s in sents_by_run]
    openings = [o for o in raw_openings if o]  # usable (non-empty)
    n_usable = len(openings)

    norm = [_normalize(o) for o in openings]
    token_sets = [set(_WORD.findall(o.lower())) for o in openings]
    prefixes = [" ".join(_WORD.findall(o.lower())[:_PREFIX_WORDS]) for o in openings]
    prefix_counts = Counter(p for p in prefixes if p)
    top_prefix = prefix_counts.most_common(1)[0] if prefix_counts else None

    def _jaccard(a: set, b: set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    lexical = {
        "jaccard": _pairwise(token_sets, _jaccard),
        "sequence_ratio": _pairwise(norm, lambda a, b: SequenceMatcher(None, a, b).ratio()),
        "distinct_openings": len(set(norm)),
        "shared_prefix": {
            "top": [[g, c] for g, c in prefix_counts.most_common(3)],
            "max_share": f"{top_prefix[1]}/{n_usable}" if top_prefix else f"0/{n_usable}",
        },
    }

    return {
        "schema": "opening_lines/1",
        "method": "first-sentence similarity: Jaccard + difflib + semantic cosine (all-MiniLM-L6-v2)",
        "runs": len(responses),
        "usable_openings": n_usable,
        "title_blocks_stripped": title_blocks_stripped,
        "suspected_title_openings": sum(1 for o in openings if _looks_like_title(o)),
        "openings": [o[:_ECHO_CHARS] for o in openings],
        "lexical": lexical,
        "semantic": _semantic(openings, ctx),
        "note": (
            "Similarity of the FIRST SENTENCE across runs (a tell whole-text "
            "similarity dilutes). High jaccard/sequence_ratio/semantic or low "
            "distinct_openings = formulaic openings; shared_prefix.max_share shows "
            "how many runs share the same opening words. Leading title/heading "
            "scaffolding is stripped before segmentation (title_blocks_stripped "
            "counts it); suspected_title_openings flags any short title-like opening "
            "that survived stripping."
        ),
    }
