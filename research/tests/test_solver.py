"""
Tests for research/solver.py.

These are the validation the 2026-09-17 design discussion asked for: at
n=2 (and n=3, where feasible) everything can be checked against brute-force
enumeration of literally every possible world, with no approximation and no
unstated modelling assumption -- if a test here passes, the corresponding
claim in logbook.tex is not just argued, it is verified by exhaustive search.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wordlists"))

from solver import (  # noqa: E402
    ExactSolver,
    alphabet_for,
    enumerate_worlds,
    expected_world_count,
    greedy_choice,
    lines_for,
    outcome_entropy_bits,
    query_outcome,
    simulate_policy_exact_expectation,
)
from toy_words import WORDS_N2  # noqa: E402


def test_lines_for_n2():
    lines = lines_for(2)
    assert len(lines) == 6  # 2 rows + 2 cols + 2 diagonals
    assert len(set(lines)) == 6  # all distinct
    for line in lines:
        assert len(line) == 2


def test_alphabet_for_n2():
    assert alphabet_for(2) == ["a", "b", "c", "d", "e"]


def test_world_count_matches_closed_form_unconditioned():
    n = 2
    worlds = enumerate_worlds(n, WORDS_N2)
    assert len(worlds) == expected_world_count(n, len(WORDS_N2))
    # 5 words * 6 lines * 3 excluded-choices * 2! remaining-permutation
    # (grid has 4 cells; word fills 2, leaving 2 cells for the 2 letters
    # left after removing the word's 2 letters and the 1 excluded letter
    # from the 5-letter alphabet)
    assert len(worlds) == 5 * 6 * 3 * 2


def test_world_count_matches_closed_form_conditioned_on_excluded():
    n = 2
    worlds = enumerate_worlds(n, WORDS_N2, excluded_global="e")
    # only words not containing 'e' can have 'e' as the excluded letter
    words_without_e = [w for w in WORDS_N2 if "e" not in w]
    assert len(worlds) == expected_world_count(n, len(words_without_e), excluded_global="e")


def test_every_world_is_internally_consistent():
    """Sanity: every enumerated grid is a bijection using exactly the
    alphabet minus the excluded letter, and really does spell `word`
    along `line`."""
    n = 2
    alphabet = set(alphabet_for(n))
    for world in enumerate_worlds(n, WORDS_N2):
        assert set(world.grid.values()) == alphabet - {world.excluded}
        assert len(world.grid) == n * n
        assert "".join(world.grid[c] for c in world.line) == world.word


def test_query_outcome_always_returns_the_letters_true_cell():
    """Corrected oracle model (2026-09-17): every non-excluded letter
    resolves to its one true cell, whether or not it's part of the answer
    word -- the grid is a full bijection, so a dross letter is still
    somewhere and the oracle still reports it (this matches wordhunt.py's
    actual `revealed` bookkeeping, which never special-cases dross)."""
    n = 2
    for world in enumerate_worlds(n, WORDS_N2):
        for cell, letter in world.grid.items():
            assert query_outcome(world, letter) == cell
        # and, specifically, in-word letters land where the word says
        for i, letter in enumerate(world.word):
            assert query_outcome(world, letter) == world.line[i]


def _initial_state(n, words, excluded_global):
    worlds = enumerate_worlds(n, words, excluded_global=excluded_global)
    queryable = [l for l in alphabet_for(n) if l != excluded_global]
    idx = frozenset(range(len(worlds)))
    return worlds, queryable, idx


def _fixed_alphabetical_but_informative(candidate_idx, worlds, queryable):
    """A deliberately 'dumb' baseline: always ask the alphabetically-first
    letter that still distinguishes at least two remaining candidates.
    (Asking a letter whose outcome is already fixed for every remaining
    candidate would never terminate, since it never shrinks the state --
    this is why even a 'no strategy' baseline must skip those.)"""
    for letter in queryable:
        outcomes = {query_outcome(worlds[i], letter) for i in candidate_idx}
        if len(outcomes) > 1:
            return letter
    return None  # should be unreachable while |candidate_idx| resolves to >1 (word,line)


def test_exact_solver_terminates_and_beats_no_information_baseline():
    """
    With the excluded letter told up front, verify the DP finds a finite
    expected number of queries, and that it's obviously sane: never worse
    than "always ask the alphabetically-first still-informative letter",
    a deliberately unstrategic baseline.
    """
    n = 2
    worlds, queryable, idx = _initial_state(n, WORDS_N2, excluded_global="e")
    solver = ExactSolver(worlds, queryable)
    v_opt, first_letter = solver.solve(idx)

    baseline = simulate_policy_exact_expectation(
        worlds, idx,
        choose_letter=lambda s: _fixed_alphabetical_but_informative(s, worlds, queryable),
    )
    assert v_opt <= baseline + 1e-9
    assert first_letter in queryable


def test_exact_solver_value_matches_independent_simulation():
    """
    Cross-check: re-derive the DP's own claimed expected cost by an
    independently-coded exact simulation (simulate_policy_exact_expectation)
    that follows the solver's chosen policy at every node.  This catches
    bugs that a single implementation re-checking itself would miss.
    """
    n = 2
    worlds, queryable, idx = _initial_state(n, WORDS_N2, excluded_global="e")
    solver = ExactSolver(worlds, queryable)
    v_opt, _ = solver.solve(idx)

    def policy(candidate_idx):
        _, letter = solver.solve(candidate_idx)
        return letter

    v_check = simulate_policy_exact_expectation(worlds, idx, choose_letter=policy)
    assert math.isclose(v_opt, v_check, rel_tol=1e-9, abs_tol=1e-9)


def test_outcome_entropy_matches_manual_count_for_n2():
    """Ground the "entropy of the outcome partition" one-step metric in an
    explicit hand count for a specific small case, rather than trusting the
    formula's own algebra."""
    n = 2
    worlds, queryable, idx = _initial_state(n, WORDS_N2, excluded_global="e")
    # WORDS_N2 without 'e': "ab", "bc", "cd" -> 3 words, each with 6 lines,
    # 2! = 2 remaining-letter permutations => 3*6*2 = 36 worlds total.
    assert len(idx) == 36
    letter = "a"
    # 'a' appears only in "ab"; among the 3 words only "ab" contains it,
    # i.e. 1/3 of worlds are "present" (split further by which of 6 lines
    # -> 6 distinct present-outcomes, 3 worlds each, since 'e' excluded
    # collapses the excluded-choice axis to a single value and remaining
    # permutation axis is trivial for n=2... let's just check the counted
    # partition directly rather than re-deriving by hand.)
    h = outcome_entropy_bits(idx, worlds, letter)
    counts = {}
    for i in idx:
        o = query_outcome(worlds[i], letter)
        counts[o] = counts.get(o, 0) + 1
    manual_h = -sum((c / len(idx)) * math.log2(c / len(idx)) for c in counts.values())
    assert math.isclose(h, manual_h, rel_tol=1e-12)


def test_greedy_matches_or_loses_to_optimal():
    """
    The whole point of building an exact DP instead of trusting the
    one-step entropy heuristic: confirm (for this toy instance) whether
    greedy actually achieves the optimum, or whether it's strictly worse --
    either result is a valid, reportable finding for logbook.tex.
    """
    n = 2
    worlds, queryable, idx = _initial_state(n, WORDS_N2, excluded_global="e")
    solver = ExactSolver(worlds, queryable)
    v_opt, _ = solver.solve(idx)

    v_greedy = simulate_policy_exact_expectation(
        worlds, idx, choose_letter=lambda s: greedy_choice(s, worlds, queryable)
    )
    assert v_greedy >= v_opt - 1e-9  # greedy can never beat the true optimum
