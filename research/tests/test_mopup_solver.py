"""
Tests for MopUpSolver (solver.py) -- the corrected scoring objective from
the 2026-09-18 design discussion: the game does not end at logical
certainty, only once every cell of the true line has been individually
queried. Also tests the M2 veto (enumerate_worlds(..., reject_collisions=True))
and the "some (word,line) choices become unreachable under the veto" finding
from the same discussion.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wordlists"))

from solver import (  # noqa: E402
    ExactSolver,
    MopUpSolver,
    TagIdentifySolver,
    alphabet_for,
    enumerate_worlds,
    lines_for,
    simulate_mopup_bruteforce,
    simulate_mopup_policy_exact_expectation,
)
from toy_words import WORDS_N2  # noqa: E402

N2_EXCLUDED_SCENARIOS = list(alphabet_for(2))


def test_m2_veto_actually_removes_collisions():
    """enumerate_worlds(..., reject_collisions=True) should leave no world
    whose grid spells more than one dictionary word anywhere."""
    n = 2
    lines = lines_for(n)
    word_set = set(WORDS_N2)
    for excluded in N2_EXCLUDED_SCENARIOS:
        worlds = enumerate_worlds(n, WORDS_N2, excluded_global=excluded, reject_collisions=True)
        for w in worlds:
            hits = [
                "".join(w.grid[c] for c in line)
                for line in lines
                if "".join(w.grid[c] for c in line) in word_set
            ]
            assert len(hits) == 1, f"M2 world has {len(hits)} spelled words, expected exactly 1: {w}"


def test_some_word_line_choices_are_unreachable_under_m2():
    """
    Direct check of the 2026-09-18 conjecture: under the veto, some
    (word, line, excluded) combinations have ZERO surviving completions --
    no arrangement of the leftover letters avoids a second word appearing
    somewhere. Measured, not assumed: exactly 40/90 triples for this
    vocabulary (see logbook.tex).
    """
    n = 2
    alphabet = alphabet_for(n)
    lines = lines_for(n)
    total = 0
    impossible = 0
    for w in WORDS_N2:
        for line in lines:
            for excluded in [c for c in alphabet if c not in w]:
                total += 1
                worlds = enumerate_worlds(n, WORDS_N2, excluded_global=excluded, reject_collisions=True)
                if not any(wd.word == w and wd.line == line for wd in worlds):
                    impossible += 1
    assert total == 90
    assert impossible == 40, (
        f"expected the measured 40/90 impossible triples for WORDS_N2, got {impossible}/{total} "
        "-- if this changed, either the vocabulary or the veto logic changed; re-check logbook.tex"
    )


def _mopup_setup(excluded):
    n = 2
    worlds = enumerate_worlds(n, WORDS_N2, excluded_global=excluded, reject_collisions=True)
    queryable = [l for l in alphabet_for(n) if l != excluded]
    idx = frozenset(range(len(worlds)))
    return worlds, queryable, idx


def test_mopup_solver_terminates_and_matches_two_independent_checks():
    """
    The core validation: MopUpSolver's own DP value must match (a) an
    independently-coded recursive re-derivation
    (simulate_mopup_policy_exact_expectation) and (b) literal step-by-step
    play against every world as ground truth
    (simulate_mopup_bruteforce), which shares no formula with either
    solver. All three must agree exactly for every excluded-letter
    scenario, or the mop-up accounting has a bug.
    """
    for excluded in N2_EXCLUDED_SCENARIOS:
        worlds, queryable, idx = _mopup_setup(excluded)
        solver = MopUpSolver(worlds, queryable)
        v_opt, _ = solver.solve(idx)

        def policy(cidx, rev, _s=solver):
            _, letter = _s.solve(cidx, rev)
            return letter

        v_formula_check = simulate_mopup_policy_exact_expectation(worlds, idx, policy)
        v_bruteforce_check = simulate_mopup_bruteforce(worlds, idx, policy)

        assert math.isclose(v_opt, v_formula_check, rel_tol=1e-9, abs_tol=1e-9), excluded
        assert math.isclose(v_opt, v_bruteforce_check, rel_tol=1e-9, abs_tol=1e-9), excluded


def test_mopup_objective_is_strictly_more_expensive_than_tag_identify():
    """
    Mop-up cost is a real, positive tax over TAG certainty specifically:
    once (word,line) is deducible, mop-up still requires directly querying
    every remaining cell of that line, so E[mopup] >= E[tag-identify]
    always (deducing the tag is implied by, so happens no later than,
    completing the mandatory reveal). Measured tax at n=2 is exactly
    0.1818 in every excluded-letter scenario -- see logbook.tex.

    NOTE (found 2026-09-18, corrected the same day): this is NOT true of
    ExactSolver's GRID-uniqueness criterion, which was wrongly assumed to
    be an upper bound on mop-up cost. Grid uniqueness is a purely logical/
    deductive criterion (can be satisfied by elimination without directly
    querying every cell), so it is not operationally comparable to
    mop-up's literal per-cell requirement -- at n=2 grid-identify (2.09)
    is actually BELOW mopup (2.27), the opposite of what a naive "more
    cells needed = more queries" argument would suggest. Only
    TagIdentifySolver has a provable ordering against MopUpSolver.
    """
    for excluded in N2_EXCLUDED_SCENARIOS:
        worlds, queryable, idx = _mopup_setup(excluded)
        v_mopup, _ = MopUpSolver(worlds, queryable).solve(idx)
        v_tag, _ = TagIdentifySolver(worlds, queryable).solve(idx)
        assert v_mopup >= v_tag - 1e-9
        assert v_mopup - v_tag > 0.1, (excluded, v_mopup, v_tag)


def test_grid_identify_has_no_guaranteed_ordering_against_mopup():
    """
    Documents, rather than merely asserting, the 2026-09-18 finding that
    ExactSolver's grid-uniqueness value and MopUpSolver's value are NOT
    ordered in general (unlike TagIdentifySolver, which is always <=
    MopUpSolver). This vocabulary happens to show grid-identify BELOW
    mopup at n=2 for every excluded letter -- recorded here so a future
    change that "fixes" this by re-imposing an ordering assumption gets
    caught.
    """
    below_count = 0
    for excluded in N2_EXCLUDED_SCENARIOS:
        worlds, queryable, idx = _mopup_setup(excluded)
        v_mopup, _ = MopUpSolver(worlds, queryable).solve(idx)
        v_grid, _ = ExactSolver(worlds, queryable).solve(idx)
        if v_grid < v_mopup:
            below_count += 1
    assert below_count == len(N2_EXCLUDED_SCENARIOS)
