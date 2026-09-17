"""
Validates the (word, line) sufficiency LEMMA documented above PairSolver in
solver.py: that running the exact DP on (word, line) pairs alone gives
bit-for-bit identical results to running it on fully-enumerated worlds,
wherever both are computable (n=2, n=3).  Only once this passes is
PairSolver trusted to stand in for full-world enumeration at n=4+, where
full enumeration is combinatorially impossible (see logbook.tex).
"""

import math
import sys
from pathlib import Path

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
