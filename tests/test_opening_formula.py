"""Tests for opening_formula: within-text chapter-opening similarity census."""
from __future__ import annotations

from conftest import load_metric

of = load_metric("opening_formula")


def test_first_sentence_extraction():
    assert of.first_sentence("She ran. More text follows.") == "She ran."
    # documented approximation: terminal punctuation inside a closing quote
    # ends the sentence, so quoted exclamations cut short
    assert of.first_sentence('  "Stop!" he cried. Then quiet.') == '"Stop!"'


def test_first_sentence_fallback_without_terminator():
    text = "x" * 400
    assert of.first_sentence(text) == "x" * of.MAX_OPENING_CHARS


def test_first_sentence_not_cut_at_title_abbreviations():
    # shakedown: Dracula's "DR. SEWARD'S DIARY" headers were truncated to "DR."
    assert of.first_sentence("Dr. Seward spoke first. Then silence.") == \
        "Dr. Seward spoke first."
    assert of.first_sentence(
        "DR. SEWARD'S DIARY\n\nRenfield escaped again tonight. More text.") == \
        "DR. SEWARD'S DIARY\n\nRenfield escaped again tonight."
    assert of.first_sentence(
        "Mr. and Mrs. Harker walked to St. Paul's and rested. Then tea.") == \
        "Mr. and Mrs. Harker walked to St. Paul's and rested."


def test_first_sentence_not_cut_at_single_initials():
    assert of.first_sentence("J. S. Fletcher wrote many books. Nobody read them.") == \
        "J. S. Fletcher wrote many books."


def test_first_sentence_pronoun_i_still_ends_sentence():
    # "I" is a common English word, not an initial: the boundary stays
    assert of.first_sentence("So did I. Then we left.") == "So did I."


def test_first_sentence_quoted_abbreviation_still_ends_sentence():
    # a sentence that legitimately ends before a name: the period sits inside a
    # closing quote, so it is a real boundary, not an abbreviation period
    assert of.first_sentence('The plate read "Dr." Seward was out.') == \
        'The plate read "Dr."'


def test_seward_style_headers_no_longer_score_identical():
    # the Dracula artifact in miniature: same diary header, different chapters;
    # before the abbreviation guard all three openings were "DR." (1.0 pairs)
    units = [
        "DR. SEWARD'S DIARY\n\nRenfield was quiet in his cell today. More.",
        "DR. SEWARD'S DIARY\n\nLucy grows weaker with every passing hour. More.",
        "DR. SEWARD'S DIARY\n\nVan Helsing arrived on the morning train. More.",
    ]
    out = of.compute(units)
    assert out["aggregate"]["max"] < 1.0
    # the genuine epistolary formula signal remains visible in the census
    assert out["repeated_openers"][0]["opener"] == "dr seward's diary"
    assert out["repeated_openers"][0]["count"] == 3


def test_formulaic_openings_detected():
    units = [
        "The morning came cold and grey over the harbor. Ships moved slowly.",
        "The morning came cold and grey over the hills. Nothing stirred at all.",
        "Rain hammered the windows without pause. She waited by the door.",
    ]
    out = of.compute(units)
    agg = out["aggregate"]
    assert agg["n_units"] == 3
    assert agg["n_pairs"] == 3
    assert agg["max_pair"] == [0, 1]
    assert agg["max"] >= of.HIGH_SIMILARITY
    assert [p["pair"] for p in out["per_pair_high"]] == [[0, 1]]
    assert out["repeated_openers"] == [{"opener": "the morning came", "count": 2}]


def test_varied_openings_clean():
    units = [
        "Dawn broke over the ridge at last. Birds began.",
        "Nobody spoke at breakfast that day. The tea cooled.",
        "A letter arrived with the noon post. It was thin.",
    ]
    out = of.compute(units)
    assert out["per_pair_high"] == []
    assert out["repeated_openers"] == []
    assert out["aggregate"]["high_pair_rate"] == 0.0


def test_identical_openings_similarity_one():
    units = ["Same opening line here. Different middle one.",
             "Same opening line here. Different middle two."]
    out = of.compute(units)
    assert out["aggregate"]["max"] == 1.0
    assert out["aggregate"]["high_pair_rate"] == 1.0


def test_single_unit_no_pairs():
    out = of.compute(["Only one chapter. Nothing to compare."])
    assert out["aggregate"]["n_pairs"] == 0
    assert out["aggregate"]["mean"] is None


def test_openings_recorded_and_schema():
    out = of.compute(["Alpha beta gamma. Rest.", "Delta epsilon zeta. Rest."])
    assert out["schema"] == "opening_formula/1"
    assert out["openings"] == ["Alpha beta gamma.", "Delta epsilon zeta."]


# --- unicode-safe tokenization (War and Peace shakedown) ---------------------------------

def test_tokens_keep_accented_names_whole():
    # the old ASCII tokenizer split "Natásha" into "nat", "sha"
    assert of._tokens("Natásha smiled at Kutúzov.") == \
        ["natasha", "smiled", "at", "kutuzov"]


def test_accent_variants_compare_equal():
    # the same opening with and without combining marks must score 1.0
    out = of.compute([
        "Kutúzov advanced at dawn toward the river crossing. Then rain.",
        "Kutuzov advanced at dawn toward the river crossing. Then sun.",
    ])
    assert out["aggregate"]["max"] == 1.0
    assert out["per_pair_high"][0]["pair"] == [0, 1]


def test_repeated_openers_fold_accents():
    out = of.compute([
        "Natásha said nothing to him then. More.",
        "Natasha said nothing at all that day. More.",
    ])
    assert out["repeated_openers"] == [
        {"opener": "natasha said nothing", "count": 2}]
