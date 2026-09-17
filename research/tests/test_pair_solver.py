"""
RETRACTED 2026-09-17: the (word, line) sufficiency lemma this file set out
to validate is FALSE under the corrected oracle model (see logbook.tex,
"Corrections" section, and the oracle-model fix in solver.py). It held
under the earlier, wrong oracle (which returned no location at all for a
dross letter) because in that model a pair's outcome for any letter really
was fully determined by (word, line) alone. Under the corrected oracle
(every letter always resolves to its one true cell), a dross letter's
revealed cell depends on the specific dross permutation, which is NOT
determined by (word, line) alone and which itself becomes non-uniform
across surviving pairs once some cells have already been revealed. So
PairSolver, as implemented, is simply wrong post-fix -- it still silently
computes the old model's answer (see `pair_query_outcome`'s use of `None`).

These tests are kept, deliberately failing, as the documented record of the
retraction rather than deleted, per this project's stated practice of
logging corrections instead of quietly rewriting history. A correct
pair-based (i.e. not full-world) solver requires properly counting
consistent completions per pair given the revealed history so far -- a
real, nontrivial piece of design, not yet attempted. Until that exists,
`ExactSolver` on fully-enumerated worlds is the only trusted solver, which
caps exact solving at whatever `n` full-world enumeration can still reach
(n<=3; see logbook.tex on the n=4 dross-permutation factorial wall).
"""

import math
import sys
from pathlib import Path
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wordlists"))

from solver import (  # noqa: E402
    ExactSolver,
    PairSolver,
    alphabet_for,
    enumerate_pairs,
    enumerate_worlds,
)
from toy_words import WORDS_N2, WORDS_N3  # noqa: E402


def _compare(n, words, excluded):
    words_ok = [w for w in words if excluded not in w]
    if len(words_ok) < 2:
        return None  # trivial / not interesting

    worlds = enumerate_worlds(n, words, excluded_global=excluded)
    queryable = [l for l in alphabet_for(n) if l != excluded]
    world_solver = ExactSolver(worlds, queryable)
    v_world, _ = world_solver.solve(frozenset(range(len(worlds))))

    pairs = enumerate_pairs(n, words, excluded)
    pair_solver = PairSolver(pairs, queryable)
    v_pair, _ = pair_solver.solve(frozenset(range(len(pairs))))

    return v_world, v_pair


@pytest.mark.xfail(reason="(word,line) sufficiency lemma retracted 2026-09-17 "
                          "under the corrected oracle model; see module docstring",
                   strict=True)
def test_pair_solver_matches_full_world_solver_n2():
    n = 2
    checked = 0
    for excluded in alphabet_for(n):
        result = _compare(n, WORDS_N2, excluded)
        if result is None:
            continue
        v_world, v_pair = result
        assert math.isclose(v_world, v_pair, rel_tol=1e-9, abs_tol=1e-9), (
            f"excluded={excluded}: world-based={v_world} != pair-based={v_pair}"
        )
        checked += 1
    assert checked >= 1


@pytest.mark.skip(reason="Sufficiency lemma retracted (see module docstring) AND, "
                         "separately, ExactSolver.solve on full n=3 worlds no longer "
                         "reliably terminates in reasonable time under the corrected "
                         "oracle (>700k memoised states, >20s, not yet observed to "
                         "finish -- richer per-query outcome space than the old, wrong "
                         "oracle). Not safe to run unguarded in a test suite.")
def test_pair_solver_matches_full_world_solver_n3():
    n = 3
    checked = 0
    for excluded in alphabet_for(n):
        result = _compare(n, WORDS_N3, excluded)
        if result is None:
            continue
        v_world, v_pair = result
        assert math.isclose(v_world, v_pair, rel_tol=1e-9, abs_tol=1e-9), (
            f"excluded={excluded}: world-based={v_world} != pair-based={v_pair}"
        )
        checked += 1
    assert checked >= 1
