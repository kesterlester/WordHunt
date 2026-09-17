"""
Exact solver + validator for the WordHunt oracle-query game, under an
explicit, fully-specified toy generative model, at board sizes small enough
that brute-force enumeration of every possible world is feasible.

This module is deliberately independent of wordhunt.py / wordhunt2.py (the
recreational tools) -- nothing here is imported by, or imports, either of
them. See research/logbook.tex for the write-up this code supports.

----------------------------------------------------------------------------
Board size n (n=5 is the real game; this module is exercised at n=2, 3, ...):

  - alphabet size  = n*n + 1   (n*n letters fill the grid, exactly one letter
                                 of the alphabet is entirely absent -- as in
                                 the real 5x5 game's 26-letter alphabet)
  - grid           = n x n
  - "lines"        = n rows + n columns + 2 diagonals = 2n + 2 candidate
                      word positions, each of length n
  - WORDS          = a small curated vocabulary of length-n strings, each
                      with n distinct characters drawn from the alphabet.
                      Need not be English -- just a fixed toy dictionary.

----------------------------------------------------------------------------
Model M1 ("no veto"):

  1. word      ~ Uniform(WORDS)
  2. line      ~ Uniform(LINES)
  3. excluded  ~ Uniform(ALPHABET \\ set(word))
  4. the remaining (n*n - n) letters are placed uniformly at random (a
     uniform random permutation) into the remaining (n*n - n) cells.
  5. (no rejection step -- M1 does not model the "dross letters
     accidentally spell a second valid word" veto raised in the design
     discussion of 2026-09-17; that is deferred to a future M2.)

`excluded` is revealed to the player before play starts (exactly as the
real game announces "which letter is missing"), so it is fixed per game
instance rather than treated as a live hypothesis.

Objective L1: minimise E[number of queries until (word, line) is uniquely
determined].

----------------------------------------------------------------------------
Player interaction model:

The player repeatedly picks an untested letter L (L != excluded). The
oracle deterministically reports:
  - the cell c = line[word.index(L)],   if L is part of the answer word
  - None (no location disclosed),        if L is not part of the answer word

This matches wordhunt.py's `revealed` / `excluded` bookkeeping: a "not in
the word" answer never discloses where that dross letter actually sits.
"""

from __future__ import annotations

import itertools
import math
import string
from dataclasses import dataclass

Cell = tuple[int, int]
Line = tuple[Cell, ...]


# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

def lines_for(n: int) -> list[Line]:
    """All 2n+2 candidate word positions for an n x n grid."""
    lines: list[Line] = []
    for r in range(n):
        lines.append(tuple((r, c) for c in range(n)))          # rows
    for c in range(n):
        lines.append(tuple((r, c) for r in range(n)))          # columns
    lines.append(tuple((i, i) for i in range(n)))               # TL->BR diag
    lines.append(tuple((n - 1 - i, i) for i in range(n)))       # BL->TR diag
    return lines


def alphabet_for(n: int) -> list[str]:
    """The n*n+1 letter toy alphabet, as the first that many lowercase letters."""
    size = n * n + 1
    if size > 26:
        raise ValueError(f"n={n} needs a {size}-letter alphabet; only 26 available")
    return list(string.ascii_lowercase[:size])


def validate_words(words: list[str], n: int, alphabet: list[str]) -> None:
    alpha_set = set(alphabet)
    for w in words:
        if len(w) != n:
            raise ValueError(f"word {w!r} has length {len(w)}, expected {n}")
        if len(set(w)) != n:
            raise ValueError(f"word {w!r} repeats a letter")
        if not set(w) <= alpha_set:
            raise ValueError(f"word {w!r} uses letters outside {alphabet}")


# ---------------------------------------------------------------------------
# World enumeration (the brute-force ground truth)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class World:
    """
    One fully-realised grid, tagged with the (word, line, excluded) that
    generated it.  Worlds are NOT deduplicated across generative paths: if
    two different (word, line, excluded, permutation) choices happen to
    produce the identical grid, both are kept as separate list entries, so
    that a grid reachable via more generative paths is correctly given more
    weight in the (implicitly uniform-per-entry) probability model.
    """
    word: str
    line: Line
    excluded: str
    grid: "dict[Cell, str]"


def spelled_words(grid: "dict[Cell, str]", lines: list[Line], word_set: set) -> "list[tuple[str, Line]]":
    """Every (word, line) pair such that `line` spells a member of
    `word_set` when read off `grid` -- normally exactly one (the designated
    answer); more than one means this grid is a "collision" (see
    enumerate_worlds's `reject_collisions`)."""
    hits = []
    for line in lines:
        s = "".join(grid[c] for c in line)
        if s in word_set:
            hits.append((s, line))
    return hits


def enumerate_worlds(
    n: int,
    words: list[str],
    excluded_global: str | None = None,
    reject_collisions: bool = False,
) -> list[World]:
    """
    Brute-force enumerate every world M1 can produce.

    If `excluded_global` is given, only worlds with that excluded letter are
    generated (this is the realistic case: the player is told the excluded
    letter before play starts, so it need not be left as a live hypothesis).

    If `reject_collisions` is True, applies the simplest veto policy from
    the design discussion of 2026-09-17 ("restart from step 1"): any grid
    where the dross letters happen to also spell a second dictionary word
    along some other line is discarded outright, rather than kept with an
    ambiguous designated answer.  Filtering the uniform base distribution
    this way is mathematically identical to resampling steps 1-4 from
    scratch until no collision occurs, so this is a legitimate (if the
    simplest possible) instance of a real M2, not just a data-cleaning
    step. See `collision_rate` for how often this actually triggers.
    """
    alphabet = alphabet_for(n)
    validate_words(words, n, alphabet)
    lines = lines_for(n)
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    word_set = set(words)

    worlds: list[World] = []
    for w in words:
        remaining_after_word = [ch for ch in alphabet if ch not in w]
        for line in lines:
            remaining_cells = [c for c in all_cells if c not in line]
            for excluded in remaining_after_word:
                if excluded_global is not None and excluded != excluded_global:
                    continue
                remaining_letters = [ch for ch in remaining_after_word if ch != excluded]
                assert len(remaining_cells) == len(remaining_letters)
                for perm in itertools.permutations(remaining_letters):
                    grid = dict(zip(line, w))
                    grid.update(zip(remaining_cells, perm))
                    if reject_collisions and len(spelled_words(grid, lines, word_set)) > 1:
                        continue
                    worlds.append(World(w, line, excluded, grid))
    return worlds


def collision_rate(n: int, words: list[str], excluded_global: str) -> float:
    """Fraction of raw-M1 worlds (no veto) whose grid spells more than one
    dictionary word somewhere -- i.e. how often the "designated answer" is
    actually ambiguous.  A direct, measured answer to "how significant a
    departure from the real game is M1 without a veto step?"."""
    lines = lines_for(n)
    word_set = set(words)
    worlds = enumerate_worlds(n, words, excluded_global=excluded_global)
    if not worlds:
        return 0.0
    collisions = sum(1 for w in worlds if len(spelled_words(w.grid, lines, word_set)) > 1)
    return collisions / len(worlds)


def expected_world_count(n: int, num_words: int, excluded_global: str | None = None) -> int:
    """Closed-form check on len(enumerate_worlds(...)) -- see test_solver.py."""
    alpha_size = n * n + 1
    num_lines = 2 * n + 2
    num_excluded_choices = 1 if excluded_global is not None else (alpha_size - n)
    remaining = alpha_size - n - 1
    return num_words * num_lines * num_excluded_choices * math.factorial(remaining)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
#
# CORRECTED 2026-09-17 (was wrong in the first commit on this branch): the
# grid is a full bijection over every non-excluded letter, so querying a
# letter L always resolves to L's one true cell -- whether or not L turns
# out to be part of the answer word.  It does NOT return "no location" for
# dross letters.  This matches the real game (querying a letter that isn't
# part of the answer still shows you where it sits) and matches
# wordhunt.py's actual `word_fits_position`, which treats every entry in
# `revealed` uniformly regardless of whether it's later found to be part of
# the answer word -- there is no separate "absent, no location" case there
# at all.  The previous version of this function silently modelled a much
# less informative (and wrong) oracle.
# ---------------------------------------------------------------------------

def query_outcome(world: World, letter: str) -> Cell:
    """The one cell where `letter` actually sits in this world's grid."""
    for cell, ch in world.grid.items():
        if ch == letter:
            return cell
    raise ValueError(f"{letter!r} not in this world's grid (is it the excluded letter?)")


def consistent_with_history(world: World, history: "list[tuple[str, Cell]]") -> bool:
    return all(query_outcome(world, letter) == outcome for letter, outcome in history)


# ---------------------------------------------------------------------------
# Exact DP solver (objective L1: minimise E[#queries to full determination])
# ---------------------------------------------------------------------------

class ExactSolver:
    """
    Memoized expectimax over the space of "candidate world index sets".

    V(S) = 0                                                     if |{(w.word,w.line) : w in S}| <= 1
    V(S) = 1 + min over letters L of  sum_{outcome} (|S_outcome|/|S|) * V(S_outcome)   otherwise

    Because every world in `worlds` is an equally-weighted sample under M1
    (see enumerate_worlds's docstring on non-deduplication), the fraction
    |S_outcome|/|S| IS the true conditional probability of that outcome --
    no separate weighting layer is needed, unlike the ad-hoc Zipf weighting
    used in the wordhunt2.py "posterior" stat.  That is the whole point of
    building this on top of exhaustive enumeration rather than reasoning
    about (word, position) pairs directly.
    """

    def __init__(self, worlds: list[World], queryable_letters: list[str]):
        self.worlds = worlds
        self.queryable_letters = queryable_letters
        self._memo: "dict[frozenset, tuple[float, str | None]]" = {}
        self._outcome_cache: "dict[tuple[int, str], Cell | None]" = {}

    def _outcome(self, idx: int, letter: str) -> "Cell | None":
        key = (idx, letter)
        cached = self._outcome_cache.get(key)
        if cached is not None or key in self._outcome_cache:
            return cached
        val = query_outcome(self.worlds[idx], letter)
        self._outcome_cache[key] = val
        return val

    def solve(self, candidate_idx: "frozenset[int]") -> "tuple[float, str | None]":
        cached = self._memo.get(candidate_idx)
        if cached is not None:
            return cached

        # Terminal iff the underlying GRID is uniquely determined -- NOT iff
        # the tagged "designated" (word,line) is uniquely determined.  Those
        # differ exactly when a grid happens to spell two dictionary words
        # on two different lines (a "collision", possible under M1's lack
        # of a veto step): two worlds can share an identical grid while
        # tagged with different designated answers, and no oracle query can
        # ever tell such worlds apart (the oracle is a function of the grid
        # alone).  Terminating on tagged pairs in that case is not merely
        # imprecise, it is a literal information-theoretic impossibility --
        # found in practice as an infinite loop in the DP (2026-09-17,
        # WORDS_N2 with excluded='e': "ab"@top-row and "cd"@bottom-row
        # produce the identical grid).  See logbook.tex.
        grids = {frozenset(self.worlds[i].grid.items()) for i in candidate_idx}
        if len(grids) <= 1:
            result = (0.0, None)
            self._memo[candidate_idx] = result
            return result

        n_total = len(candidate_idx)
        best_letter = None
        best_val = math.inf

        for letter in self.queryable_letters:
            buckets: "dict[Cell | None, list[int]]" = {}
            for i in candidate_idx:
                buckets.setdefault(self._outcome(i, letter), []).append(i)
            if len(buckets) <= 1:
                continue  # no information from this letter among remaining candidates

            expected = 0.0
            for idxs in buckets.values():
                sub = frozenset(idxs)
                v_sub, _ = self.solve(sub)
                expected += (len(idxs) / n_total) * v_sub
            total = 1.0 + expected
            if total < best_val - 1e-12:
                best_val = total
                best_letter = letter

        result = (best_val, best_letter)
        self._memo[candidate_idx] = result
        return result


# ---------------------------------------------------------------------------
# Greedy (one-step max information-gain) comparator
# ---------------------------------------------------------------------------

def outcome_entropy_bits(candidate_idx: "frozenset[int]", worlds: list[World], letter: str) -> float:
    """Shannon entropy (bits) of the outcome partition -- the one-step
    'value of information' metric from the wordhunt2.py design discussion,
    now computed against real enumerated worlds instead of assumed pairs."""
    n_total = len(candidate_idx)
    if n_total == 0:
        return 0.0
    counts: "dict[Cell | None, int]" = {}
    for i in candidate_idx:
        key = query_outcome(worlds[i], letter)
        counts[key] = counts.get(key, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n_total
        h -= p * math.log2(p)
    return h


def greedy_choice(candidate_idx: "frozenset[int]", worlds: list[World],
                   queryable_letters: list[str]) -> "str | None":
    """The one-step max-entropy letter -- NOT guaranteed optimal for L1;
    kept for comparison against ExactSolver's true optimum."""
    best_letter = None
    best_h = -1.0
    for letter in queryable_letters:
        h = outcome_entropy_bits(candidate_idx, worlds, letter)
        if h > best_h + 1e-12:
            best_h = h
            best_letter = letter
    return best_letter


def simulate_policy_exact_expectation(
    worlds: list[World],
    initial_idx: "frozenset[int]",
    choose_letter,
) -> float:
    """
    Exact expected number of queries a policy takes, averaged over every
    world in `initial_idx` as the (equally-likely) ground truth -- i.e. a
    full brute-force check, not a Monte Carlo estimate.  `choose_letter(S)`
    must return the next letter to query given current candidate set S.
    """
    memo: "dict[frozenset, float]" = {}

    def cost(candidate_idx: "frozenset[int]") -> float:
        cached = memo.get(candidate_idx)
        if cached is not None:
            return cached
        # See ExactSolver.solve: terminal iff the GRID is unique, not iff the
        # tagged (word,line) is unique -- a grid collision makes the latter
        # genuinely unknowable, not just hard.
        grids = {frozenset(worlds[i].grid.items()) for i in candidate_idx}
        if len(grids) <= 1:
            memo[candidate_idx] = 0.0
            return 0.0
        letter = choose_letter(candidate_idx)
        n_total = len(candidate_idx)
        buckets: "dict[Cell | None, list[int]]" = {}
        for i in candidate_idx:
            buckets.setdefault(query_outcome(worlds[i], letter), []).append(i)
        expected = 1.0
        for idxs in buckets.values():
            expected += (len(idxs) / n_total) * cost(frozenset(idxs))
        memo[candidate_idx] = expected
        return expected

    return cost(initial_idx)


# ---------------------------------------------------------------------------
# Pair-only solver: the (word, line) sufficiency lemma
# ---------------------------------------------------------------------------
#
# Full world enumeration is intractable beyond n=3 (see logbook.tex): the
# number of ways to permute the (n*n - n) dross letters into the remaining
# cells is (n*n-n-1)! -- 2 at n=2, 720 at n=3, but already ~4.8e8 at n=4.
#
# LEMMA: under M1, with `excluded` fixed, this factor is IDENTICAL for
# every (word, line) hypothesis (it only depends on n, not on which word or
# line), so it is a constant that cancels out of every relative-probability
# calculation. The exact DP can therefore be run directly on (word, line)
# pairs -- weighting each surviving pair equally -- without ever
# constructing a single full grid, and must give bit-for-bit identical
# results to the full-world solver wherever both are computable.  That
# equivalence is checked in tests/test_pair_solver.py at n=2 and n=3; only
# once that check passes is the pair-only solver trusted for n=4+.
# ---------------------------------------------------------------------------

Pair = "tuple[str, Line]"


def enumerate_pairs(n: int, words: list[str], excluded_global: str) -> list[Pair]:
    """Every (word, line) hypothesis consistent with `excluded_global`,
    each an equally-weighted equivalence class of full worlds (see LEMMA)."""
    lines = lines_for(n)
    return [(w, line) for w in words if excluded_global not in w for line in lines]


def pair_query_outcome(pair: Pair, letter: str) -> "Cell | None":
    word, line = pair
    if letter in word:
        return line[word.index(letter)]
    return None


class PairSolver:
    """Same recursion as ExactSolver, operating on (word, line) pairs
    directly instead of on fully-realised grids -- see the sufficiency
    LEMMA above.  Interface mirrors ExactSolver so tests can compare them
    directly."""

    def __init__(self, pairs: list[Pair], queryable_letters: list[str]):
        self.pairs = pairs
        self.queryable_letters = queryable_letters
        self._memo: "dict[frozenset, tuple[float, str | None]]" = {}

    def solve(self, candidate_idx: "frozenset[int]") -> "tuple[float, str | None]":
        cached = self._memo.get(candidate_idx)
        if cached is not None:
            return cached

        if len(candidate_idx) <= 1:
            result = (0.0, None)
            self._memo[candidate_idx] = result
            return result

        n_total = len(candidate_idx)
        best_letter = None
        best_val = math.inf

        for letter in self.queryable_letters:
            buckets: "dict[Cell | None, list[int]]" = {}
            for i in candidate_idx:
                buckets.setdefault(pair_query_outcome(self.pairs[i], letter), []).append(i)
            if len(buckets) <= 1:
                continue

            expected = 0.0
            for idxs in buckets.values():
                sub = frozenset(idxs)
                v_sub, _ = self.solve(sub)
                expected += (len(idxs) / n_total) * v_sub
            total = 1.0 + expected
            if total < best_val - 1e-12:
                best_val = total
                best_letter = letter

        result = (best_val, best_letter)
        self._memo[candidate_idx] = result
        return result


# ---------------------------------------------------------------------------
# MopUpSolver: the REAL scoring objective (2026-09-18 correction)
# ---------------------------------------------------------------------------
#
# Both ExactSolver and PairSolver above minimise E[queries until the
# hypothesis space collapses to certainty]. That is NOT what the real game
# scores. Per the 2026-09-18 correction: reaching logical certainty does not
# end the game. The oracle only declares victory once every cell of the
# (unique, under M2) true line has been individually revealed by a direct
# query -- so if you deduce the answer with 2 of its 5 letters still
# unqueried, you must spend 2 more turns querying exactly those letters
# (which will obviously hit) before the game ends and the turn count is
# reported. A query that happens to land on a true-line cell does double
# duty: it narrows the hypothesis space AND advances this mandatory
# mop-up, whereas a query landing on dross only does the former -- so the
# optimal policy for this objective can genuinely differ from the optimal
# policy for "reach certainty fastest".
#
# This requires tracking the revealed-cell set explicitly as DP state
# (candidate_idx alone is not sufficient: two different query histories can
# leave the same surviving hypothesis set behind while having incidentally
# revealed different amounts of whichever line turns out to be true). Cell
# identity (not which letter) is all that's needed: if a cell is already
# revealed, every surviving candidate is already forced to agree on its
# letter (that is what "surviving" / "consistent" means), so no extra
# bookkeeping of letters is needed alongside the cell set.
#
# Assumes M2 (collision-free) worlds as input -- built via
# enumerate_worlds(..., reject_collisions=True) -- so that "some candidate's
# own line is now fully revealed" is unambiguous (see logbook.tex,
# 2026-09-18 entry, on why raw M1 made this ill-posed).
# ---------------------------------------------------------------------------

class MopUpSolver:
    """
    Exact DP for E[query count at which the true line is fully revealed],
    under M2 (collision-free) worlds. State = (candidate_idx, revealed
    cells). See module-level comment above for why both are needed.
    """

    def __init__(self, worlds: list[World], queryable_letters: list[str]):
        self.worlds = worlds
        self.queryable_letters = queryable_letters
        self._memo: "dict[tuple[frozenset, frozenset], tuple[float, str | None]]" = {}

    def _is_done(self, world_idx: int, revealed_cells: "frozenset[Cell]") -> bool:
        return all(c in revealed_cells for c in self.worlds[world_idx].line)

    def solve(
        self,
        candidate_idx: "frozenset[int]",
        revealed_cells: "frozenset[Cell]" = frozenset(),
    ) -> "tuple[float, str | None]":
        key = (candidate_idx, revealed_cells)
        cached = self._memo.get(key)
        if cached is not None:
            return cached

        active = frozenset(i for i in candidate_idx if not self._is_done(i, revealed_cells))
        n_total = len(candidate_idx)
        if not active:
            # Every remaining candidate's own line is already fully revealed
            # -- whichever one is secretly true, the game has already ended.
            result = (0.0, None)
            self._memo[key] = result
            return result

        n_active = len(active)
        best_letter = None
        best_val = math.inf

        # A letter whose cell is already revealed is a guaranteed no-op --
        # every remaining candidate already agrees on its location (that is
        # what "still consistent" means), so re-querying it changes nothing
        # and recursing on it would call this exact (candidate_idx,
        # revealed_cells) state again before it is memoized, i.e. infinite
        # recursion, not just waste. Must be excluded, not merely
        # discouraged: found the hard way (RecursionError) rather than
        # reasoned out in advance.
        any_active = next(iter(active))
        already_revealed_letters = {self.worlds[any_active].grid[c] for c in revealed_cells}

        for letter in self.queryable_letters:
            if letter in already_revealed_letters:
                continue
            # Partition only the ACTIVE worlds -- worlds already done need
            # (and get) no further queries; their cost is already fixed and
            # accounted for by the (n_active/n_total) scaling below. Do NOT
            # skip a letter merely because it fails to discriminate between
            # active worlds (unlike ExactSolver/PairSolver): a letter that
            # lands on every active world's line at the same relative cell
            # can still advance mop-up for all of them without narrowing
            # anything, and that is real progress this objective must value.
            buckets: "dict[Cell, list[int]]" = {}
            for i in active:
                c = query_outcome(self.worlds[i], letter)
                buckets.setdefault(c, []).append(i)

            expected = 0.0
            for cell, idxs in buckets.items():
                new_revealed = revealed_cells | {cell}
                v_sub, _ = self.solve(frozenset(idxs), new_revealed)
                expected += (len(idxs) / n_active) * v_sub
            total = 1.0 + expected
            if total < best_val - 1e-12:
                best_val = total
                best_letter = letter

        result = ((n_active / n_total) * best_val, best_letter)
        self._memo[key] = result
        return result


def simulate_mopup_policy_exact_expectation(
    worlds: list[World],
    initial_idx: "frozenset[int]",
    choose_letter,
) -> float:
    """
    Independent brute-force check for MopUpSolver: exact expected query
    count under `choose_letter(candidate_idx, revealed_cells) -> letter`,
    averaged over every world in `initial_idx` as ground truth, stopping
    each one only when ITS OWN line is fully revealed (not merely when the
    hypothesis space collapses) -- mirrors the real scoring rule directly
    rather than reusing MopUpSolver's own recursion, so it can catch bugs
    a single implementation checking itself would miss.
    """
    memo: "dict[tuple[frozenset, frozenset], float]" = {}

    def cost(candidate_idx: "frozenset[int]", revealed_cells: "frozenset[Cell]") -> float:
        key = (candidate_idx, revealed_cells)
        cached = memo.get(key)
        if cached is not None:
            return cached
        active = frozenset(
            i for i in candidate_idx
            if not all(c in revealed_cells for c in worlds[i].line)
        )
        n_total = len(candidate_idx)
        if not active:
            memo[key] = 0.0
            return 0.0
        n_active = len(active)
        letter = choose_letter(candidate_idx, revealed_cells)
        buckets: "dict[Cell, list[int]]" = {}
        for i in active:
            buckets.setdefault(query_outcome(worlds[i], letter), []).append(i)
        expected = 1.0
        for cell, idxs in buckets.items():
            new_revealed = revealed_cells | {cell}
            expected += (len(idxs) / n_active) * cost(frozenset(idxs), new_revealed)
        val = (n_active / n_total) * expected
        memo[key] = val
        return val

    return cost(initial_idx, frozenset())


def simulate_mopup_bruteforce(
    worlds: list[World],
    initial_idx: "frozenset[int]",
    choose_letter,
) -> float:
    """
    A second, structurally-independent check for MopUpSolver, sharing no
    recursive formula with it or with simulate_mopup_policy_exact_expectation:
    literally play the policy out against each world in turn as if it were
    the one true hidden world, counting real turns until that world's own
    line is fully revealed, then average. This is the closest thing to
    "just play the game and see" available without a live oracle.
    """
    total_steps = 0
    for true_i in initial_idx:
        true_world = worlds[true_i]
        candidate_idx = initial_idx
        revealed: "frozenset[Cell]" = frozenset()
        steps = 0
        while not all(c in revealed for c in true_world.line):
            letter = choose_letter(candidate_idx, revealed)
            cell = query_outcome(true_world, letter)
            revealed = revealed | {cell}
            steps += 1
            candidate_idx = frozenset(
                i for i in candidate_idx if query_outcome(worlds[i], letter) == cell
            )
        total_steps += steps
    return total_steps / len(initial_idx)
