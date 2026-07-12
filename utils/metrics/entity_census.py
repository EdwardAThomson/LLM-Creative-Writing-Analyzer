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

Known NER false NEGATIVES (six-book shakedown, en_core_web_sm): principal
characters can be entirely mislabeled and vanish from the PERSON census.
Observed: War and Peace's Natasha (``Natásha``) labeled GPE in 41/41 mentions;
Napoleon labeled ORG in 79 percent of mentions; Pride and Prejudice's Lydia
labeled GPE/ORG. The ``capitalized_recurring`` field is the deterministic
fallback census that makes such invisible principals visible; it is reported
SEPARATELY so it never pollutes the NER-based numbers.

Unicode: accented names are NFKD-normalized with combining marks dropped
(``Kutúzov`` -> ``kutuzov``) instead of the old ``[^A-Za-z]`` deletion, which
mangled them (``Kutúzov`` -> ``kutzov``).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

NAME = "entity_census"

TOP_COMPONENTS = 20
PLACE_LABELS = {"GPE", "LOC", "FAC"}
_POSSESSIVE = re.compile(r"['’]s$")


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return round(x, nd) if x is not None else None


def fold_marks(text: str) -> str:
    """NFKD-decompose and drop combining marks: ``Kutúzov`` -> ``Kutuzov``.

    Letters that do not decompose (ø, æ, ß) pass through unchanged; they are
    letters and must not be deleted.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def person_components(person_text: str, stoplist: Iterable[str] = ()) -> list[str]:
    """Name components of a PERSON span: whitespace-split, hyphen-split,
    possessives stripped, 1-char / all-caps / stop-listed parts dropped.
    Mirrors the v1 component convention (full names count part by part).
    Accents are folded (NFKD, combining marks dropped), never deleted: the
    W&P shakedown showed ``[^A-Za-z]`` deletion turning Kutúzov into kutzov."""
    stop = {s.lower() for s in stoplist}
    parts = []
    for token in person_text.split():
        token = _POSSESSIVE.sub("", token)
        for seg in token.split("-"):
            seg = "".join(c for c in fold_marks(seg) if c.isalpha())
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


# --- deterministic capitalized-token census (NER-independent fallback) ---------------

# Words routinely capitalized mid-sentence without naming a specific person:
# honorifics/ranks, kinship terms, deity/interjections, calendar terms, and the
# nationality/language adjectives that saturate the shakedown corpus. Frozen
# small list, folded lowercase; extend as new noise surfaces.
CAPITALIZED_STOPWORDS = frozenset((
    # honorifics and forms of address
    "mr", "mrs", "ms", "miss", "dr", "st", "prof", "sir", "madam", "madame",
    "mademoiselle", "monsieur", "messieurs", "herr", "frau", "senor", "senora",
    "don", "dona", "lord", "lady", "dame", "master", "mister", "esquire",
    # kinship used as address
    "father", "mother", "mamma", "papa", "uncle", "aunt", "cousin", "brother",
    "sister", "grandmamma", "grandpapa",
    # rank and station
    "captain", "colonel", "general", "major", "sergeant", "lieutenant",
    "admiral", "count", "countess", "prince", "princess", "baron", "baroness",
    "duke", "duchess", "emperor", "empress", "king", "queen", "tsar", "czar",
    "doctor", "professor", "reverend", "bishop", "abbe",
    # deity and interjections
    "god", "heaven", "providence", "oh", "ah", "alas",
    # calendar
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "christmas",
    "easter", "michaelmas",
    # nationality / language adjectives
    "english", "french", "russian", "german", "italian", "austrian", "polish",
    "spanish", "latin", "greek", "turkish", "american", "british", "european",
    "christian",
))

# Unicode-aware word token: letters (any script), optional inner
# apostrophes/hyphens. [^\W\d_] is "letter" in re's unicode mode.
_CAP_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")


def capitalized_census(unit_texts: list[str],
                       stoplist: Iterable[str] = ()) -> dict:
    """Deterministic recurring-capitalized-token census over unit texts.

    The NER-independent fallback for the false-negative problem documented in
    the module docstring (Natásha GPE 41/41, Napoleon ORG 79 percent, Lydia
    GPE/ORG): a principal character recurs as a capitalized token whatever the
    NER thinks it is. Counted occurrences are filtered three ways:

    * sentence-start filter: an occurrence counts only when the preceding
      significant character (whitespace-skipping) is a letter, digit, or
      comma, i.e. clearly mid-sentence; grammar-capitalized sentence and
      dialogue openers never count;
    * stopword filter: ``CAPITALIZED_STOPWORDS`` plus the caller's stoplist;
    * form filter: all-caps tokens (acronyms, headings) and single letters are
      dropped; possessives are stripped; keys are NFKD-folded lowercase
      (``Natásha`` -> ``natasha``).

    Recurring means appearing in 2 or more units, mirroring the PERSON census
    convention. Reported separately from the NER numbers, never merged.
    """
    stop = {fold_marks(s).lower() for s in stoplist} | set(CAPITALIZED_STOPWORDS)
    mentions: Counter = Counter()
    token_units: dict[str, set] = {}
    for i, text in enumerate(unit_texts):
        for m in _CAP_WORD.finditer(text):
            word = m.group(0)
            if len(word) < 2 or not word[0].isupper() or word.isupper():
                continue
            j = m.start() - 1
            while j >= 0 and text[j] in " \t":
                j -= 1
            if j < 0:
                continue
            prev = text[j]
            if not (prev.isalpha() or prev.isdigit() or prev == ","):
                continue  # sentence/dialogue start or after punctuation: skip
            key = fold_marks(_POSSESSIVE.sub("", word)).lower()
            if len(key) < 2 or key in stop:
                continue
            mentions[key] += 1
            token_units.setdefault(key, set()).add(i)
    recurring = [k for k in mentions if len(token_units[k]) >= 2]
    top = [{"token": k, "mentions": c, "units": len(token_units[k])}
           for k, c in mentions.most_common()
           if len(token_units[k]) >= 2][:TOP_COMPONENTS]
    return {
        "recurring_count": len(recurring),
        "top": top,
        "note": ("Deterministic mid-sentence capitalized-token census; the "
                 "NER-independent fallback for PERSON false negatives (see "
                 "module docstring). Separate from the NER-based numbers."),
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
    result["capitalized_recurring"] = capitalized_census(responses, PERSON_STOPLIST)
    result.update({
        "schema": "entity_census/1",
        "method": ("spaCy NER census over units; PERSON components split per the "
                   "v1 convention, filtered by the shared stop-list; plus a "
                   "deterministic capitalized-token census (separate field)"),
        "runs": len(responses),
        "note": (
            "Single-text census (st1): cast_size counts distinct PERSON name "
            "components; recurring_cast_size counts those appearing in 2+ units "
            "(walk-ons appear once). NER false-positive caveats from the v1 "
            "analysis apply here too; NER false NEGATIVES (see module "
            "docstring) are covered by the separate capitalized_recurring "
            "census."
        ),
    })
    return result
