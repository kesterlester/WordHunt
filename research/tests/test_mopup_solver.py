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
    greedy_choice,
    lines_for,
    query_outcome,
    simulate_mopup_bruteforce,
    simulate_mopup_policy_exact_expectation,
    simulate_tag_policy_exact_expectation,
)
from toy_words import (  # noqa: E402
    WORDS_N2,
    WORDS_N2_ALL,
    WORDS_N2_DENSER,
    WORDS_N2_SPARSE,
)

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


def _mopup_cost_of_forcing(mopup, cidx, revealed_cells, letter):
    """Expected mop-up cost of forcing this specific letter as the next
    query, then playing optimally afterwards -- used to tell a genuine
    strategic difference from harmless tie-breaking between two letters
    that are actually equally good."""
    active = frozenset(i for i in cidx if not mopup._is_done(i, revealed_cells))
    n_total = len(cidx)
    n_active = len(active)
    buckets = {}
    for i in active:
        c = query_outcome(mopup.worlds[i], letter)
        buckets.setdefault(c, []).append(i)
    expected = 0.0
    for cell, idxs in buckets.items():
        new_rev = revealed_cells | {cell}
        v_sub, _ = mopup.solve(frozenset(idxs), new_rev)
        expected += (len(idxs) / n_active) * v_sub
    return (n_active / n_total) * (1.0 + expected)


def test_mopup_and_tag_identify_policies_genuinely_diverge():
    """
    2026-09-18 overnight finding: the mop-up-aware and tag-identify-only
    optimal POLICIES (not just their root values) are substantively
    different, at every state where both solvers still face a real
    choice -- not merely tied at a few states, and not merely different
    at the very first move (which happens to coincide for this
    vocabulary and was flagged in Section "mopup" of logbook.tex as not
    something to generalise from).

    Exhaustive over every state MopUpSolver's own recursion visits (its
    memo covers the whole reachable tree, since it tries every letter to
    find the min, not just the states on its optimal path) across all 5
    excluded-letter scenarios at n=2. Measured: 135 states where both
    solvers had a genuine choice, 65 disagreed, and -- checked here --
    every single disagreement is a real strategic difference (forcing
    the tag-optimal choice instead measurably costs more), none are
    ties.
    """
    n = 2
    alphabet = alphabet_for(n)
    meaningful = 0
    real_divergences = 0

    for excluded in alphabet:
        worlds, queryable, idx = _mopup_setup(excluded)
        mopup = MopUpSolver(worlds, queryable)
        mopup.solve(idx)
        tag = TagIdentifySolver(worlds, queryable)

        for (cidx, rev), (_, letter_mopup) in mopup._memo.items():
            if letter_mopup is None:
                continue
            _, letter_tag = tag.solve(cidx)
            if letter_tag is None:
                continue
            meaningful += 1
            if letter_tag == letter_mopup:
                continue
            gap = _mopup_cost_of_forcing(mopup, cidx, rev, letter_tag) - mopup._memo[(cidx, rev)][0]
            assert gap > 1e-9, (
                excluded, cidx, rev, letter_tag, letter_mopup,
                "disagreement should always be a genuine strategic difference, "
                "not a tie, for this vocabulary"
            )
            real_divergences += 1

    assert meaningful == 135
    assert real_divergences == 65


def _impossible_triple_rate(n, words):
    alphabet = alphabet_for(n)
    lines = lines_for(n)
    total = 0
    impossible = 0
    for w in words:
        for line in lines:
            for excluded in [c for c in alphabet if c not in w]:
                total += 1
                worlds = enumerate_worlds(n, words, excluded_global=excluded, reject_collisions=True)
                if not any(wd.word == w and wd.line == line for wd in worlds):
                    impossible += 1
    return impossible, total


def test_impossible_triple_rate_is_vocabulary_density_driven_not_n2_specific():
    """
    2026-09-18 overnight finding: the 44% M2-impossible-triple rate
    measured for WORDS_N2 (logbook.tex) is not a property of n=2's
    geometry -- it ranges from 7.4% (sparse, minimal-overlap vocabulary)
    to a degenerate 100% (a vocabulary where literally every string is
    "a word", so M2 is unsatisfiable by construction: every line of every
    grid trivially spells a word, so no grid can ever have exactly one).
    The measured 44% sits inside that range, not as an outlier.
    """
    sparse_impossible, sparse_total = _impossible_triple_rate(2, WORDS_N2_SPARSE)
    original_impossible, original_total = _impossible_triple_rate(2, WORDS_N2)
    denser_impossible, denser_total = _impossible_triple_rate(2, WORDS_N2_DENSER)
    all_impossible, all_total = _impossible_triple_rate(2, WORDS_N2_ALL)

    assert (sparse_impossible, sparse_total) == (4, 54)
    assert (original_impossible, original_total) == (40, 90)
    assert (denser_impossible, denser_total) == (26, 72)
    assert (all_impossible, all_total) == (all_total, all_total)  # 100%: M2 unsatisfiable

    # Not a strict total order across arbitrary vocabularies (denser's own
    # 36.1% < original's 44.0%, since "more words" isn't the same axis as
    # "more overlap") -- but the sparsest vocabulary is comfortably the
    # least impossible, and the degenerate "everything is a word" case is
    # comfortably the most, with the other two sitting strictly between.
    rates = [sparse_impossible / sparse_total, original_impossible / original_total,
             denser_impossible / denser_total]
    assert min(rates) == sparse_impossible / sparse_total
    assert all(r < 1.0 for r in rates)


def _mopup_greedy_policy(cidx, rev, worlds, queryable):
    """greedy_choice, adapted for use as a MopUpSolver-style policy: must
    exclude already-revealed letters (re-testing one is a guaranteed
    no-op and, worse, causes infinite recursion in the simulators -- see
    MopUpSolver's own docstring for why)."""
    any_i = next(iter(cidx))
    already = {worlds[any_i].grid[c] for c in rev}
    remaining = [l for l in queryable if l not in already]
    return greedy_choice(cidx, worlds, remaining)


def test_greedy_matches_optimal_for_certainty_but_not_for_real_objective_at_n2():
    """
    2026-09-18, logbook.tex Volume 2 Section "Optimal values": on this
    vocabulary, one-step maximum-entropy greedy selection happens to
    match the true optimum for the certainty objective C, but is
    measurably worse for the real objective T -- greedy does not see the
    double-duty value of a query landing on the true line. This is the
    result that replaces the old (retracted) "Finding 2", which compared
    greedy against grid-uniqueness termination, an objective with no
    natural meaning under M2.
    """
    for excluded in N2_EXCLUDED_SCENARIOS:
        worlds, queryable, idx = _mopup_setup(excluded)

        v_tag, _ = TagIdentifySolver(worlds, queryable).solve(idx)
        v_tag_greedy = simulate_tag_policy_exact_expectation(
            worlds, idx, lambda s: greedy_choice(s, worlds, queryable)
        )
        assert math.isclose(v_tag, v_tag_greedy, rel_tol=1e-9, abs_tol=1e-9), excluded

        v_mopup, _ = MopUpSolver(worlds, queryable).solve(idx)
        v_mopup_greedy = simulate_mopup_policy_exact_expectation(
            worlds, idx, lambda s, r: _mopup_greedy_policy(s, r, worlds, queryable)
        )
        assert v_mopup_greedy > v_mopup + 0.1, (excluded, v_mopup, v_mopup_greedy)
