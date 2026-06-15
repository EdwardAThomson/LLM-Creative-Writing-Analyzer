"""Shared contract and helpers for ``utils/metrics`` modules.

Contract — each metric module ``<name>.py`` defines::

    NAME = "<name>"                               # must match the filename
    def compute(responses: list[str], ctx: dict) -> dict: ...

``compute`` returns a JSON-serializable dict. **Import heavy deps inside
``compute``**, not at module top level, so unused metrics stay cheap. ``ctx`` is a
scratch dict the runner may pre-populate and that metrics may read/write to share
work (e.g. a cached spaCy model).
"""
from __future__ import annotations

import re
from typing import Optional

# Minimal spaCy-PERSON false-positive stop-list (report §4.4, §6.3, §6.6). These
# are non-names spaCy's en_core_web_sm mislabels as PERSON for this prompt; extend
# as new ones surface. Matched case-insensitively.
PERSON_STOPLIST = {
    "minds", "kepler", "metropolis", "metallic", "dirt", "frozen", "alert",
    "continuum", "meridian", "colony", "station", "concord", "senate",
    "parliament", "assembly", "demarchy", "synod", "chamber", "arbiter",
    "steward", "hierophant", "lattice", "reach", "tier", "earth", "moon",
}

_POSSESSIVE = re.compile(r"['’]s$")


def _surname_candidates(person_text: str) -> list[str]:
    """Surname parts of a PERSON span.

    Surname = last whitespace token (matches the v1 component convention), then
    split on hyphens so e.g. ``Dane Okafor-Voss`` yields ``Okafor`` and ``Voss``.
    Splitting hyphens is deliberate: the v1 token-splitter treats ``Okafor-Voss``
    as atomic and so misses the recurring ``Voss`` sound (report §6.6).
    """
    tokens = person_text.split()
    if not tokens:
        return []
    surname = _POSSESSIVE.sub("", tokens[-1])
    parts = []
    for seg in surname.split("-"):
        seg = re.sub(r"[^A-Za-z]", "", seg).strip()
        if len(seg) > 1 and not seg.isupper() and seg.lower() not in PERSON_STOPLIST:
            parts.append(seg)
    return parts


def _get_nlp(ctx: Optional[dict]):
    """Load (and cache in ctx) the spaCy model. Lazy import."""
    nlp = (ctx or {}).get("_nlp")
    if nlp is None:
        import spacy  # lazy

        nlp = spacy.load("en_core_web_sm")
        if ctx is not None:
            ctx["_nlp"] = nlp
    return nlp


def person_surnames_by_run(responses: list[str], ctx: Optional[dict] = None) -> list[list[str]]:
    """Per-run list of character surnames extracted from spaCy PERSON entities.

    False positives are filtered against ``PERSON_STOPLIST`` and all-caps acronyms
    are dropped. Returns one list of surname strings per response.
    """
    nlp = _get_nlp(ctx)
    out: list[list[str]] = []
    for text in responses:
        doc = nlp(text)
        names: list[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.extend(_surname_candidates(ent.text))
        out.append(names)
    return out


def sentences_by_run(responses: list[str], ctx: Optional[dict] = None) -> list[list[str]]:
    """Per-run list of sentence strings via the shared spaCy sentence segmenter.

    A generic seam (other metrics — opening-line diversity, readability — can reuse
    it). Empty/whitespace-only sentences are dropped. Uses the same cached
    ``ctx['_nlp']`` model as the entity helpers, so a v2 run segments once.
    """
    nlp = _get_nlp(ctx)
    out: list[list[str]] = []
    for text in responses:
        doc = nlp(text)
        out.append([s.text.strip() for s in doc.sents if s.text.strip()])
    return out
