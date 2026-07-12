"""Single-text entity census: cast size, entity density, name-component inventory.

The single-text (st1) reframing of the vN entity analysis: cross-run entity
OVERLAP is meaningless for one book (its characters are supposed to recur), so
this module instead takes a census. Per unit and overall: how many distinct
character name components exist (cast size), how densely entities occur, which
name components recur across units (the actual cast) versus appearing once
(walk-ons).

Extraction uses the same spaCy NER as v1 (lazy import inside ``compute``, so
the module imports cheaply), with the shared PERSON false-positive stop-list
from ``_base``. The census itself (``census``) is a pure function over
pre-extracted ``(label, text)`` pairs, so all counting logic is testable
without spaCy. Local NLP only: zero LLM calls.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

NAME = "entity_census"

TOP_COMPONENTS = 20
PLACE_LABELS = {"GPE", "LOC", "FAC"}
_POSSESSIVE = re.compile(r"['’]s$")


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return round(x, nd) if x is not None else None


def person_components(person_text: str, stoplist: Iterable[str] = ()) -> list[str]:
    """Name components of a PERSON span: whitespace-split, hyphen-split,
    possessives stripped, 1-char / all-caps / stop-listed parts dropped.
    Mirrors the v1 component convention (full names count part by part)."""
    stop = {s.lower() for s in stoplist}
    parts = []
    for token in person_text.split():
        token = _POSSESSIVE.sub("", token)
        for seg in token.split("-"):
            seg = re.sub(r"[^A-Za-z]", "", seg)
            if len(seg) > 1 and not seg.isupper() and seg.lower() not in stop:
                parts.append(seg)
    return parts


def census(entities_by_unit: list[list[tuple]], words_by_unit: list[int],
           stoplist: Iterable[str] = ()) -> dict:
    """Pure census over per-unit ``(label, text)`` entity lists."""
    total_words = sum(words_by_unit)
    label_counts: Counter = Counter()
    component_mentions: Counter = Counter()
    component_units: dict[str, set] = {}
    per_unit = []
    for i, (ents, words) in enumerate(zip(entities_by_unit, words_by_unit)):
        n_person = 0
        for label, text in ents:
            label_counts[label] += 1
            if label == "PERSON":
                n_person += 1
                for comp in person_components(text, stoplist):
                    key = comp.lower()
                    component_mentions[key] += 1
                    component_units.setdefault(key, set()).add(i)
        per_unit.append({
            "unit": i,
            "words": words,
            "entities": len(ents),
            "person_mentions": n_person,
            "distinct_components": len({
                c.lower() for label, text in ents if label == "PERSON"
                for c in person_components(text, stoplist)}),
        })

    recurring = {c for c, units in component_units.items() if len(units) >= 2}
    top = [{"component": c, "mentions": m, "units": len(component_units[c])}
           for c, m in component_mentions.most_common(TOP_COMPONENTS)]
    total_entities = sum(label_counts.values())
    person_mentions = label_counts.get("PERSON", 0)
    return {
        "per_unit": per_unit,
        "aggregate": {
            "total_words": total_words,
            "cast_size": len(component_mentions),
            "recurring_cast_size": len(recurring),
            "walk_on_count": len(component_mentions) - len(recurring),
            "person_mentions": person_mentions,
            "person_mentions_per_1k": _round(person_mentions / total_words * 1000)
                                      if total_words else None,
            "entity_mentions_per_1k": _round(total_entities / total_words * 1000)
                                      if total_words else None,
            "place_mentions_per_1k": _round(
                sum(label_counts[l] for l in PLACE_LABELS) / total_words * 1000)
                if total_words else None,
        },
        "label_counts": dict(label_counts),
        "top_components": top,
    }


def compute(responses: list[str], ctx: Optional[dict] = None) -> dict:
    from ._base import PERSON_STOPLIST, _get_nlp  # lazy; heavy (spaCy)

    nlp = _get_nlp(ctx)
    entities_by_unit = []
    words_by_unit = []
    for text in responses:
        doc = nlp(text)
        entities_by_unit.append([(ent.label_, ent.text) for ent in doc.ents])
        words_by_unit.append(len(text.split()))

    result = census(entities_by_unit, words_by_unit, PERSON_STOPLIST)
    result.update({
        "schema": "entity_census/1",
        "method": ("spaCy NER census over units; PERSON components split per the "
                   "v1 convention, filtered by the shared stop-list"),
        "runs": len(responses),
        "note": (
            "Single-text census (st1): cast_size counts distinct PERSON name "
            "components; recurring_cast_size counts those appearing in 2+ units "
            "(walk-ons appear once). NER false-positive caveats from the v1 "
            "analysis apply here too."
        ),
    })
    return result
