"""Word origin: share of words from French/Latin vs Old English (Germanic).

A register fingerprint the other metrics miss. Plain English prose is mostly
Old English words (hand, look, sword); formal, abstract or "elevated" prose
leans on French and Latin ones (occasion, manner, illumination). Published
reference points, counted over every word:

- Tolkien 89% Germanic / 8% Latinate; Sanderson 73% / 25% (a five-author
  table from The Legendarium channel, one short passage each);
- Morris/Dunsany/Eddison shelf about 10%, Austen and Ferrier about 19-20%
  (measured with this table, whole books, StoryDaemon 2026-11-14).

On the saved 13-model corpus (one sci-fi prompt) models range from about 15%
(2026 Claude) to about 32% (early-2025 Gemini Pro), with little overlap
between their per-run ranges, so it separates models.

Method: each word is looked up in a frozen table (``_data/word_origin.json``,
``TABLE_VERSION``) built in StoryDaemon from the Etymological Wordnet (from
Wiktionary, CC BY-SA 3.0) plus about 1,500 hand labels for its gaps and errors
(checked blind against 300 words the Wordnet resolves: 96% agree). A word the
table lacks is tried without its ending (stones, walked), then by a Latin
suffix rule (-tion, -ity, -ous); otherwise it is left out of the shares and
counted in ``origin_known_pct``. Names (words only ever capitalised in the
run) are left out.

Two shares per run: ``latinate_pct`` over every word, comparable to the
published table; ``latinate_content_pct`` without the small grammar words
(the, of, he, was), which are nearly all Old English in any text and so hide
differences. Sci-fi raises both for every model (technology words are Latin
and Greek), so compare models on the same prompt, not against novels.

Pure stdlib. A changed table is a new ``TABLE_VERSION``; re-score the corpus.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

NAME = "word_origin"
TABLE_VERSION = "1"
_TABLE_PATH = Path(__file__).resolve().parent / "_data" / "word_origin.json"

_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_LATIN_SUFFIX = re.compile(r"(tion|sion|ity|ment|ance|ence|ancy|ency|ous|ive|ible|able|"
                           r"ate|ure|ary|ory|ical|ial|ual|ient|iant|ism|ist|ize|ise|ify)s?$")
_ENDINGS = (("ies", "y"), ("ied", "y"), ("ier", "y"), ("iest", "y"), ("ily", "y"),
            ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
            ("ly", ""), ("er", ""), ("est", ""), ("ness", ""), ("ful", ""),
            ("less", ""), ("ment", ""), ("ion", "e"), ("ation", "e"))
_GRAMMAR = frozenset("""a an the and or but nor if so as than that this these those then
there here where when while what which who whom whose why how not no yes of to in on at
by for from with into onto upon over under about after before between through without
within against among up down out off again once i me my mine myself we us our ours you
your yours he him his himself she her hers herself it its itself they them their theirs
themselves is am are was were be been being have has had having do does did doing will
would shall should can could may might must ought all any both each few more most other
some such only own same too very just also even still yet ever never""".split())

_TABLE: Optional[dict] = None


def _table() -> dict:
    global _TABLE
    if _TABLE is None:
        groups = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))["groups"]
        _TABLE = {w: g for g, words in groups.items() for w in words.split()}
    return _TABLE


def _stems(w: str) -> list:
    out = [w[:-2]] if w.endswith("'s") else []
    for suffix, repl in _ENDINGS:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[: len(w) - len(suffix)] + repl
            out.append(base)
            if len(base) > 3 and base[-1] == base[-2] and not repl:   # stopped -> stop
                out.append(base[:-1])
    return out


def origin(word: str) -> Optional[str]:
    """'G' (Old English/Norse), 'R' (French/Latin), 'Gk', 'O' (other) or None."""
    table = _table()
    if word in table:
        return table[word]
    for stem in _stems(word):
        if stem in table:
            return table[stem]
    if len(word) > 6 and _LATIN_SUFFIX.search(word):
        return "R"
    return None


def _words(text: str) -> list:
    """Lowercased words, leaving out names (only ever capitalised) and contractions."""
    text = text.replace("’", "'").replace("‘", "'")
    found = [(w.lower(), w[0].isupper()) for w in _WORD.findall(text)]
    lower = {w for w, cap in found if not cap}
    return [w for w, cap in found if (not cap or w in lower) and "'" not in w]


def _pct(part: int, whole: int) -> Optional[float]:
    return round(100 * part / whole, 1) if whole else None


def _run(text: str, latinate_words: Counter) -> dict:
    words = _words(text)
    groups = [(w, origin(w)) for w in words]
    known = [(w, g) for w, g in groups if g]
    content = [(w, g) for w, g in known if w not in _GRAMMAR]
    count = Counter(g for _, g in known)
    latinate_words.update(w for w, g in content if g == "R")
    return {
        "words": len(words),
        "origin_known_pct": _pct(len(known), len(words)),
        "latinate_pct": _pct(count["R"], len(known)),
        "germanic_pct": _pct(count["G"], len(known)),
        "greek_pct": _pct(count["Gk"], len(known)),
        "latinate_content_pct": _pct(sum(1 for _, g in content if g == "R"), len(content)),
    }


def _summary(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None, "std": None}
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return {"n": len(vals), "mean": round(mean, 1), "min": min(vals),
            "max": max(vals), "std": round(std, 1)}


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    latinate_words: Counter = Counter()
    per_run = [_run(text, latinate_words) for text in responses]
    return {
        "schema": "word_origin/1",
        "table_version": TABLE_VERSION,
        "method": ("share of words by origin (French/Latin vs Old English/Norse) over "
                   "words the frozen table places; names left out"),
        "runs": len(responses),
        "per_run": per_run,
        "aggregate": {
            "latinate_pct": _summary([r["latinate_pct"] for r in per_run]),
            "latinate_content_pct": _summary([r["latinate_content_pct"] for r in per_run]),
            "origin_known_pct": _summary([r["origin_known_pct"] for r in per_run]),
        },
        "top_latinate_words": dict(latinate_words.most_common(25)),
        "note": ("latinate_pct counts every word (comparable to published tables: "
                 "Tolkien 8%, Sanderson 25%); latinate_content_pct leaves out grammar "
                 "words. Genre moves both (sci-fi vocabulary is Latin and Greek), so "
                 "compare models on the same prompt."),
    }
