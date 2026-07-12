"""Tests for entity_census (pure census logic; spaCy extraction is out of scope
for this env, per the house convention for heavy modules)."""
from __future__ import annotations

from conftest import load_metric

ec = load_metric("entity_census")


# --- component splitting ----------------------------------------------------------------

def test_person_components_splits_full_names():
    assert ec.person_components("Mina Harker") == ["Mina", "Harker"]


def test_person_components_hyphen_and_possessive():
    assert ec.person_components("Dane Okafor-Voss's") == ["Dane", "Okafor", "Voss"]


def test_person_components_drops_short_caps_and_stoplisted():
    assert ec.person_components("J ALERT Kepler", stoplist={"kepler"}) == []


# --- pure census ------------------------------------------------------------------------

def _entities():
    return [
        [("PERSON", "Mina Harker"), ("PERSON", "Jonathan Harker"), ("GPE", "London")],
        [("PERSON", "Mina Harker"), ("PERSON", "Renfield"), ("LOC", "the moor")],
        [("PERSON", "Renfield"), ("ORG", "the Council")],
    ]


def test_census_cast_and_recurrence():
    out = ec.census(_entities(), [1000, 1000, 1000])
    agg = out["aggregate"]
    # components: mina, harker, jonathan, renfield
    assert agg["cast_size"] == 4
    # mina (units 0,1), harker (0,1), renfield (1,2) recur; jonathan is a walk-on
    assert agg["recurring_cast_size"] == 3
    assert agg["walk_on_count"] == 1
    assert agg["person_mentions"] == 5


def test_census_densities_per_1k():
    out = ec.census(_entities(), [1000, 1000, 1000])
    agg = out["aggregate"]
    assert agg["total_words"] == 3000
    assert agg["person_mentions_per_1k"] == round(5 / 3, 2)
    assert agg["entity_mentions_per_1k"] == round(8 / 3, 2)
    assert agg["place_mentions_per_1k"] == round(2 / 3, 2)  # GPE + LOC, not ORG


def test_census_top_components_mentions_and_units():
    out = ec.census(_entities(), [100, 100, 100])
    top = {t["component"]: t for t in out["top_components"]}
    assert top["harker"]["mentions"] == 3   # Mina Harker x2 + Jonathan Harker
    assert top["harker"]["units"] == 2
    assert top["renfield"]["mentions"] == 2
    assert top["renfield"]["units"] == 2
    assert top["jonathan"]["units"] == 1


def test_census_per_unit_rows():
    out = ec.census(_entities(), [100, 100, 100])
    u0 = out["per_unit"][0]
    assert u0 == {"unit": 0, "words": 100, "entities": 3, "person_mentions": 2,
                  "distinct_components": 3}  # mina, harker, jonathan


def test_census_stoplist_applied():
    ents = [[("PERSON", "Kepler Minds")]]
    out = ec.census(ents, [100], stoplist={"kepler", "minds"})
    assert out["aggregate"]["cast_size"] == 0
    assert out["aggregate"]["person_mentions"] == 1  # the mention still counts


def test_census_label_counts():
    out = ec.census(_entities(), [100, 100, 100])
    assert out["label_counts"] == {"PERSON": 5, "GPE": 1, "LOC": 1, "ORG": 1}


def test_census_empty():
    out = ec.census([], [])
    assert out["aggregate"]["cast_size"] == 0
    assert out["aggregate"]["person_mentions_per_1k"] is None


# --- unicode-safe name handling (War and Peace shakedown) --------------------------------

def test_person_components_folds_accents_instead_of_deleting():
    # The old [^A-Za-z] deletion produced "Kutzov"; NFKD folding must produce
    # "Kutuzov" (W&P shakedown: Kutúzov -> kutzov, Natásha -> "nat sha").
    assert ec.person_components("Kutúzov") == ["Kutuzov"]
    assert ec.person_components("Natásha Rostóva") == ["Natasha", "Rostova"]
    assert ec.person_components("Márya Dmítrievna") == ["Marya", "Dmitrievna"]


def test_person_components_keeps_nondecomposable_letters():
    # ø does not NFKD-decompose; it is a letter and must not be deleted
    assert ec.person_components("Møller") == ["Møller"]


def test_fold_marks():
    assert ec.fold_marks("Kutúzov") == "Kutuzov"
    assert ec.fold_marks("plain") == "plain"


def test_census_accented_names_count_as_one_component():
    ents = [[("PERSON", "Natásha")], [("PERSON", "Natásha")]]
    out = ec.census(ents, [100, 100])
    assert out["aggregate"]["cast_size"] == 1
    assert out["top_components"][0] == {"component": "natasha", "mentions": 2,
                                        "units": 2}


# --- deterministic capitalized-token census (NER false-negative fallback) ----------------

def test_capitalized_census_counts_mid_sentence_recurring_tokens():
    # the Natásha/Napoleon shape: principals invisible to NER (labeled
    # GPE/ORG) still recur as mid-sentence capitalized tokens
    units = [
        "Then Natásha smiled and watched Napoleon ride past the line.",
        "Later Natásha and Napoleon met again near Moscow before dark.",
    ]
    out = ec.capitalized_census(units)
    top = {t["token"]: t for t in out["top"]}
    assert top["natasha"] == {"token": "natasha", "mentions": 2, "units": 2}
    assert top["napoleon"] == {"token": "napoleon", "mentions": 2, "units": 2}
    assert "moscow" not in top        # one unit only: not recurring
    assert out["recurring_count"] == 2


def test_capitalized_census_filters_sentence_and_dialogue_starts():
    units = [
        "Suddenly the road bent. Suddenly it bent again, said Anna.",
        'She said, “Perhaps not now.” Suddenly all was quiet, said Anna.',
    ]
    out = ec.capitalized_census(units)
    tokens = {t["token"] for t in out["top"]}
    assert "suddenly" not in tokens   # only ever sentence-initial
    assert "perhaps" not in tokens    # dialogue opener
    assert "anna" in tokens           # mid-sentence in both units


def test_capitalized_census_stopwords_and_form_filters():
    units = [
        "He met Mr Darcy and the French envoy in LONDON with Natásha's aunt.",
        "She met Mr Darcy and the French envoy again with Natásha's uncle.",
    ]
    out = ec.capitalized_census(units)
    tokens = {t["token"] for t in out["top"]}
    assert "mr" not in tokens         # honorific stopword
    assert "french" not in tokens     # nationality stopword
    assert "london" not in tokens     # all-caps form dropped
    assert "darcy" in tokens
    assert "natasha" in tokens        # possessive stripped, accent folded


def test_capitalized_census_requires_two_units():
    units = ["He saw Kutúzov twice, and Kutúzov saw him.",
             "Nothing capitalized mid-sentence here at all."]
    out = ec.capitalized_census(units)
    assert out["recurring_count"] == 0
    assert out["top"] == []


def test_capitalized_census_caller_stoplist_folded():
    units = ["He saw Kutúzov there.", "She saw Kutúzov too."]
    out = ec.capitalized_census(units, stoplist={"kutuzov"})
    assert out["top"] == []
