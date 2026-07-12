"""Tests for the deterministic cast-based thread clustering (pure, no LLM)."""
from __future__ import annotations

from benchmarks.narrative_dynamics import clustering as cl


# --- name normalization ---------------------------------------------------------------

def test_norm_name_strips_honorifics_iteratively():
    assert cl.norm_name("Mr. Bennet") == "bennet"
    assert cl.norm_name("Lady Verinder") == "verinder"
    assert cl.norm_name("The Count Dracula") == "dracula"
    assert cl.norm_name("  Sergeant   Cuff ") == "cuff"


def test_norm_name_keeps_plain_names_and_case_folds():
    assert cl.norm_name("Mina Harker") == "mina harker"
    assert cl.norm_name("O'Brien-Smith") == "o'brien-smith"


def test_canon_alias_raw_form_wins_over_stripped_form():
    # 'mr bennet' and 'mrs bennet' both title-strip to 'bennet'; the raw-form
    # lookup keeps them distinct (the study's two-level rule).
    aliases = {"mr bennet": "mr bennet", "mrs bennet": "mrs bennet"}
    assert cl.canon("Mr. Bennet", aliases) == "mr bennet"
    assert cl.canon("Mrs. Bennet", aliases) == "mrs bennet"


def test_canon_stripped_form_alias():
    aliases = {"mina murray": "mina harker"}
    assert cl.canon("Mina Murray", aliases) == "mina harker"
    assert cl.canon("Miss Mina Murray", aliases) == "mina harker"


def test_canon_without_aliases_is_norm_name():
    assert cl.canon("Dr. Seward") == "seward"


# --- primitives -----------------------------------------------------------------------

def test_jaccard():
    assert cl.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert cl.jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert cl.jaccard(set(), {"a"}) == 0.0


def test_majority_profile_is_50_percent_membership():
    members = [{"cast": ["a", "b"]}, {"cast": ["a", "c"]}, {"cast": ["a", "b"]}]
    assert cl.majority_profile(members, "cast") == {"a", "b"}


def test_run_lengths():
    assert cl.run_lengths([0, 0, 1, 1, 1, 0]) == [2, 3, 1]
    assert cl.run_lengths([]) == []
    assert cl.run_lengths([2]) == [1]


def test_build_signatures_canonicalizes_and_adds_pov_token():
    sigs = cl.build_signatures(
        [{"pov": "Mina Murray", "principal_cast": ["Mina Murray", "Dr. Seward", ""]}],
        aliases={"mina murray": "mina harker"})
    assert sigs[0]["cast"] == ["mina harker", "seward"]
    assert sigs[0]["pov"] == "mina harker"
    assert "pov:mina harker" in sigs[0]["sig"]


# --- clustering -----------------------------------------------------------------------

def _rec(pov, cast):
    return {"pov": pov, "principal_cast": cast}


def _two_strand_book():
    """4 units of strand A, 6 of strand B, then a convergence unit."""
    a = [_rec("Jonathan", ["Jonathan", "Dracula"])] * 4
    b = [_rec("Mina", ["Mina", "Lucy"])] * 6
    conv = [_rec("Mina", ["Jonathan", "Mina", "Lucy"])]
    return cl.build_signatures(a + b + conv)


def test_cluster_two_strands_and_assignment():
    threads, assign, merges = cl.cluster(_two_strand_book())
    assert len(threads) == 2
    assert assign == [0] * 4 + [1] * 6 + [1]  # convergence unit joins strand B
    assert threads[0]["members"] == [0, 1, 2, 3]


def test_cluster_convergence_event_detected():
    _, _, merges = cl.cluster(_two_strand_book())
    assert merges == [(10, [0, 1])]  # the full-cast unit covers both profiles


def test_cluster_no_convergence_on_unestablished_thread():
    # thread B has a single unit when the mixed unit arrives: not established
    recs = ([_rec("Jonathan", ["Jonathan", "Dracula"])] * 2
            + [_rec("Mina", ["Mina", "Lucy"])]
            + [_rec("Mina", ["Jonathan", "Mina", "Lucy"])])
    _, _, merges = cl.cluster(cl.build_signatures(recs))
    assert merges == []


def test_cluster_below_theta_opens_new_thread():
    recs = [_rec("A", ["A", "B"]), _rec("C", ["C", "D"])]
    threads, assign, _ = cl.cluster(cl.build_signatures(recs))
    assert len(threads) == 2
    assert assign == [0, 1]


def test_cluster_tie_goes_to_most_recently_active_thread():
    # Two threads with identical profiles: a later unit matching both must join
    # the one whose last member is most recent.
    sigs = [
        {"cast": ["a"], "sig": ["a", "pov:a"], "pov": "a"},
        {"cast": ["b"], "sig": ["b", "pov:b"], "pov": "b"},  # opens thread 1
        {"cast": ["a"], "sig": ["a", "pov:a"], "pov": "a"},  # thread 0 active again
    ]
    # craft a 4th unit equally similar to both threads: impossible with distinct
    # profiles, so instead duplicate profiles across threads:
    sigs = [
        {"cast": ["a"], "sig": ["a"], "pov": ""},   # thread 0
        {"cast": ["b"], "sig": ["b"], "pov": ""},   # thread 1 (a vs b: below theta)
        {"cast": ["a"], "sig": ["a"], "pov": ""},   # joins thread 0 (most recent now 2)
        {"cast": ["a", "b"], "sig": ["a", "b"], "pov": ""},  # jaccard 0.5 with both
    ]
    threads, assign, _ = cl.cluster(sigs)
    assert assign[:3] == [0, 1, 0]
    assert assign[3] == 0  # tie on similarity, thread 0 was active more recently


def test_threshold_sensitivity_reports_all_thetas():
    sens = cl.threshold_sensitivity(_two_strand_book())
    assert set(sens) == {"0.2", "0.3", "0.4"}
    assert all(isinstance(v, int) for v in sens.values())


def test_single_pov_rotating_cast_fragments_at_high_theta():
    # The study's known edge (The Thirty-Nine Steps): one POV, supporting cast
    # rotates completely per episode. At 0.2 the pov: token unifies; at 0.4 the
    # book fragments.
    recs = [_rec("Hannay", ["Hannay", f"Guest{i}", f"Extra{i}"]) for i in range(6)]
    sigs = cl.build_signatures(recs)
    n_02 = len(cl.cluster(sigs, 0.2)[0])
    n_04 = len(cl.cluster(sigs, 0.4)[0])
    assert n_02 <= n_04
    assert n_02 == 1
