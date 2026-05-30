#!/usr/bin/env python3
"""
Word grid assistance tool.

Helps find the single 5-letter word hidden in a 5x5 grid where:
  - Every letter of the alphabet except one appears exactly once.
  - The word occupies one of 12 positions (5 rows, 5 cols, 2 diagonals).
  - The player learns one letter location per turn from an external oracle.

Coordinate convention (user-facing):
  col increases 1→5 left to right
  row increases 1→5 bottom to top
  So top-left corner of grid is (col=1, row=5).
  Oracle input format:  <letter> <col>,<row>   e.g.  m 3,4
"""

import csv
import os
import random
import sys
from collections import Counter

DICT_PATH = os.path.join(os.path.dirname(__file__), "dict_game.csv")

# ---------------------------------------------------------------------------
# 12 word positions in 0-based (matrix_row, matrix_col) where row 0 = top.
# ---------------------------------------------------------------------------
POSITIONS: list[tuple[tuple[int, int], ...]] = []
for _r in range(5):
    POSITIONS.append(tuple((_r, _c) for _c in range(5)))      # horizontal rows
for _c in range(5):
    POSITIONS.append(tuple((_r, _c) for _r in range(5)))      # vertical cols
POSITIONS.append(tuple((_i, _i) for _i in range(5)))          # TL -> BR diagonal
POSITIONS.append(tuple((4 - _i, _i) for _i in range(5)))      # BL -> TR diagonal

SORT_MODES = ["freq", "alpha", "speed"]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def user_to_matrix(user_col: int, user_row: int) -> tuple[int, int]:
    """Convert user (col, row) [1-based, row up] to 0-based (matrix_r, matrix_c)."""
    return (5 - user_row, user_col - 1)


def matrix_to_user(matrix_r: int, matrix_c: int) -> tuple[int, int]:
    """Convert 0-based (matrix_r, matrix_c) to user (col, row)."""
    return (matrix_c + 1, 5 - matrix_r)


# ---------------------------------------------------------------------------
# Word loading
# ---------------------------------------------------------------------------

def load_words(path: str = DICT_PATH) -> "tuple[list[str], dict[str, float]]":
    """Return (words, freq_dict) where freq_dict maps word -> Zipf frequency (0 if absent)."""
    words = []
    freq: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        has_freq = len(header) >= 2 and header[1].strip() != ""
        for row in reader:
            if row and row[0]:
                w = row[0].strip().lower()
                words.append(w)
                if has_freq and len(row) >= 2:
                    try:
                        freq[w] = float(row[1])
                    except ValueError:
                        freq[w] = 0.0
                else:
                    freq[w] = 0.0
    return words, freq


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def word_fits_position(
    word: str,
    pos: tuple[tuple[int, int], ...],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> bool:
    """Return True if word could be the hidden word at position pos."""
    if any(ch in excluded for ch in word):
        return False

    pos_index = {cell: i for i, cell in enumerate(pos)}

    for cell, letter in revealed.items():
        if cell in pos_index:
            if word[pos_index[cell]] != letter:
                return False
        else:
            # Grid letter is unique: if this letter is elsewhere, word can't have it.
            if letter in word:
                return False

    return True


def filter_words(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> list[str]:
    return [
        w for w in words
        if any(word_fits_position(w, pos, revealed, excluded) for pos in POSITIONS)
    ]


def find_fully_revealed(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> "str | None":
    """
    Return a word if ALL 5 cells of some valid position are revealed and
    spell that word.  By the game guarantee (only one valid word in the grid),
    this is definitively the answer regardless of how many words remain in the
    candidate list.
    """
    word_set = set(words)
    for pos in POSITIONS:
        if not all(cell in revealed for cell in pos):
            continue
        candidate = "".join(revealed[cell] for cell in pos)
        if candidate in word_set and not any(ch in excluded for ch in candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# F(L|G) — expected search-space size after asking the oracle about letter L
# ---------------------------------------------------------------------------
#
# Model: the true game state is (WTRUE, PTRUE), drawn uniformly from valid
# (word, position) pairs.  The oracle reports L's actual cell in the grid:
#
#   L ∈ WTRUE  →  u = PTRUE[ WTRUE.index(L) ]            (deterministic)
#   L ∉ WTRUE  →  L sits in one of the 20 non-word cells.
#                 Under a uniform prior over non-word arrangements, and given
#                 the already-revealed cells, u ~ Uniform( U(G) \ cells(PTRUE) )
#                 (any unrevealed cell not occupied by the word position).
#
# E[n(G') | ask L]
#   = (1/n(G)) × Σ_{(w_r,p_r)} E_u[ n(G with L at u) | w_r, p_r ]
#
# Exact when n(G) ≤ F_EXACT_THRESHOLD; Monte Carlo otherwise.
#
# MC estimator (one draw):
#   1. Sample reporter pair (w_r, p_r) uniformly from valid_pairs.
#   2. Draw oracle cell u:
#        L ∈ w_r  →  u = p_r[ w_r.index(L) ]
#        L ∉ w_r  →  u ~ Uniform( U(G) \ cells(p_r) )
#   3. Sample tester pair (w_t, p_t) uniformly from valid_pairs.
#   4. Score = word_fits(w_t, p_t, G with L at u).
#   Mean score × n(G)  ≈  E[n(G') | ask L].
#
# ---------------------------------------------------------------------------

F_EXACT_THRESHOLD = 2000   # n(G) above which MC is used
F_MC_SAMPLES      = 3000   # (reporter, tester) draws per letter in MC mode


def compute_F(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> "tuple[dict[str, float], bool]":
    """
    Return (F_dict, is_estimated).
    F_dict maps each active letter to E[n(G') | ask that letter].
    Smaller = more informative oracle query.
    """
    valid_pairs = [
        (w, pos)
        for w in words
        for pos in POSITIONS
        if word_fits_position(w, pos, revealed, excluded)
    ]
    n_G = len(valid_pairs)
    if n_G == 0:
        return {}, False

    U_G = [(r, c) for r in range(5) for c in range(5) if (r, c) not in revealed]
    if not U_G:
        return {}, False

    known_letters = set(revealed.values()) | excluded
    active_letters = {ch for w in words for ch in w} - known_letters

    estimated = n_G > F_EXACT_THRESHOLD
    result: dict[str, float] = {}

    if not estimated:
        # Exact: precompute n(G with L at u) for every u once per letter,
        # then sum over reporter pairs using the two-case oracle model.
        for L in active_letters:
            n_prime: dict = {}
            for u in U_G:
                new_rev = {**revealed, u: L}
                n_prime[u] = sum(
                    1 for w, pos in valid_pairs
                    if word_fits_position(w, pos, new_rev, excluded)
                )

            total = 0.0
            for w_r, p_r in valid_pairs:
                p_r_set = set(p_r)
                if L in w_r:
                    total += n_prime[p_r[w_r.index(L)]]
                else:
                    candidates = [u for u in U_G if u not in p_r_set]
                    if candidates:
                        total += sum(n_prime[u] for u in candidates) / len(candidates)

            result[L] = total / n_G

    else:
        # Monte Carlo: sample (reporter, tester) pairs with correct oracle model.
        rng = random.Random(hash(tuple(sorted(revealed.items()))))
        for L in active_letters:
            hits = 0
            effective = 0
            for _ in range(F_MC_SAMPLES):
                w_r, p_r = rng.choice(valid_pairs)
                if L in w_r:
                    u = p_r[w_r.index(L)]
                else:
                    candidates = [c for c in U_G if c not in set(p_r)]
                    if not candidates:
                        continue
                    u = rng.choice(candidates)
                w_t, p_t = rng.choice(valid_pairs)
                if word_fits_position(w_t, p_t, {**revealed, u: L}, excluded):
                    hits += 1
                effective += 1
            result[L] = (hits / effective * n_G) if effective > 0 else n_G

    return result, estimated


# ---------------------------------------------------------------------------
# Steps-to-completion
# ---------------------------------------------------------------------------

def min_steps(
    word: str,
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> int:
    """Minimum unknown cells across all valid positions for this word."""
    best = 6  # worse than worst case
    for pos in POSITIONS:
        if word_fits_position(word, pos, revealed, excluded):
            unknown = sum(1 for cell in pos if cell not in revealed)
            if unknown < best:
                best = unknown
    return best


def all_min_steps(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> dict[str, int]:
    """Pre-compute min_steps for every word in the list."""
    return {w: min_steps(w, revealed, excluded) for w in words}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def letter_freq_stats(
    words: list[str],
    revealed: dict[tuple[int, int], str],
) -> Counter:
    """Count how many words contain each unknown letter."""
    known = set(revealed.values())
    counter: Counter = Counter()
    for w in words:
        for ch in set(w):
            if ch not in known:
                counter[ch] += 1
    return counter


def letter_completion_stats(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    steps_map: dict[str, int],
) -> dict[str, float]:
    """Average min_steps across words containing each unknown letter."""
    known = set(revealed.values())
    buckets: dict[str, list[int]] = {}
    for w in words:
        s = steps_map[w]
        for ch in set(w):
            if ch not in known:
                buckets.setdefault(ch, []).append(s)
    return {ch: sum(v) / len(v) for ch, v in buckets.items()}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

MAX_WORD_DISPLAY = 200


def sorted_words(
    words: list[str],
    sort_mode: str,
    steps_map: dict[str, int],
    freq_dict: dict[str, float],
) -> list[str]:
    if sort_mode == "freq":
        # Highest Zipf first (most common); ties broken alphabetically.
        return sorted(words, key=lambda w: (-freq_dict.get(w, 0.0), w))
    if sort_mode == "speed":
        return sorted(words, key=lambda w: (steps_map[w], w))
    return sorted(words)  # alpha


def print_state(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
    sort_mode: str,
    freq_dict: dict[str, float],
) -> None:
    n = len(words)
    print(f"\nWords remaining: {n}  [sort: {sort_mode}]")

    if n == 0:
        print("  (none — check inputs for contradictions)")
        return

    steps_map = all_min_steps(words, revealed, excluded)
    display = sorted_words(words, sort_mode, steps_map, freq_dict)

    if sort_mode == "speed":
        annotated = [f"{w}({steps_map[w]})" for w in display[:MAX_WORD_DISPLAY]]
        label = "  (steps-to-complete shown in parens)"
    elif sort_mode == "freq":
        annotated = [
            f"{w}({freq_dict.get(w, 0.0):.1f})" if freq_dict.get(w, 0.0) > 0 else w
            for w in display[:MAX_WORD_DISPLAY]
        ]
        label = "  (Zipf freq shown in parens; 0 = not in corpus)"
    else:
        annotated = display[:MAX_WORD_DISPLAY]
        label = ""

    if n <= MAX_WORD_DISPLAY:
        print("  " + "  ".join(annotated) + label)
    else:
        print(f"  (showing first {MAX_WORD_DISPLAY} of {n})" + label)
        print("  " + "  ".join(annotated))

    # --- Frequency stats ---
    freq = letter_freq_stats(words, revealed)
    if freq:
        print("\n  Letter frequencies (unknown positions):")
        for letter, count in freq.most_common(5):
            pct = 100 * count / n
            print(f"    {letter.upper()}: in {count}/{n} words ({pct:.0f}%)")

    # --- Completion stats ---
    comp = letter_completion_stats(words, revealed, steps_map)
    if comp:
        print("\n  Avg steps to completion if letter chosen next:")
        for letter, avg in sorted(comp.items(), key=lambda kv: kv[1])[:5]:
            count = freq.get(letter, 0)
            print(f"    {letter.upper()}: avg {avg:.2f} steps  (in {count}/{n} words)")

    # --- F(L|G) metric ---
    F, estimated = compute_F(words, revealed, excluded)
    if F:
        tag = f"estimated, {F_MC_SAMPLES} samples" if estimated else "exact"
        print(f"\n  E[search space after asking] [{tag}] — smallest = ask this next:")
        for letter, exp_n in sorted(F.items(), key=lambda kv: kv[1])[:5]:
            print(f"    {letter.upper()}: {exp_n:.1f} pairs remaining on avg")


def print_grid(revealed: dict[tuple[int, int], str]) -> None:
    """Print grid with user-facing col/row labels (col left-right, row bottom-up)."""
    print("\n    col: 1 2 3 4 5")
    print("         ---------")
    for mr in range(5):           # matrix rows top to bottom
        user_row = 5 - mr         # label: 5 at top, 1 at bottom
        row_str = f"  row {user_row} | "
        for mc in range(5):
            cell = (mr, mc)
            row_str += (revealed[cell].upper() if cell in revealed else ".") + " "
        print(row_str)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def parse_cell(s: str) -> tuple[int, int]:
    """Parse 'col,row' (1-based, row up) into 0-based matrix (row, col)."""
    parts = s.strip().split(",")
    if len(parts) != 2:
        raise ValueError("Expected col,row (e.g. 3,4 means col 3 row 4)")
    user_col, user_row = int(parts[0]), int(parts[1])
    if not (1 <= user_col <= 5 and 1 <= user_row <= 5):
        raise ValueError("Col and row must each be between 1 and 5")
    return user_to_matrix(user_col, user_row)


def cell_label(cell: tuple[int, int]) -> str:
    """Format a matrix cell as user-facing 'col,row'."""
    col, row = matrix_to_user(*cell)
    return f"col {col}, row {row}"


def prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 54)
    print("  Word Grid Assistant")
    print("=" * 54)
    print("  Coordinates: col 1-5 left→right, row 1-5 bottom→up")
    print("  Oracle input:  <letter> <col>,<row>   e.g.  l 3,4")
    print("  Delete cell:   . <col>,<row>  or  del <col>,<row>")
    print("  Commands:  grid | mode | undo | quit")
    print("=" * 54 + "\n")

    words, freq_dict = load_words()
    print(f"Loaded {len(words):,} candidate words.")

    revealed: dict[tuple[int, int], str] = {}
    excluded: set[str] = set()
    sort_mode = SORT_MODES[0]
    history: list[dict[tuple[int, int], str]] = []   # undo stack

    # --- Excluded letter ---
    while True:
        exc = prompt("Which letter is excluded from the grid? ")
        if len(exc) == 1 and exc.isalpha():
            excluded.add(exc)
            break
        print("  Please enter a single letter (a-z).")

    all_words = filter_words(words, revealed, excluded)   # full list, post-exclusion
    words = all_words
    print_grid(revealed)
    print_state(words, revealed, excluded, sort_mode, freq_dict)

    # --- Main interaction loop ---
    while True:
        if not words:
            print("\nNo valid words remain — please check your inputs.")
            break
        if len(words) == 1:
            print(f"\n*** The word is: {words[0].upper()} ***")
            break
        solved = find_fully_revealed(words, revealed, excluded)
        if solved:
            print(f"\n*** {solved.upper()} is fully revealed in the grid — that's the word! ***")
            break

        print()
        inp = prompt("Oracle report (letter col,row) | . col,row | grid | mode | undo | quit : ")

        if inp in ("quit", "q", "exit"):
            break

        if inp == "grid":
            print_grid(revealed)
            continue

        if inp in ("mode", "sort"):
            idx = (SORT_MODES.index(sort_mode) + 1) % len(SORT_MODES)
            sort_mode = SORT_MODES[idx]
            print(f"  Sort mode -> {sort_mode}")
            print_grid(revealed)
            print_state(words, revealed, excluded, sort_mode, freq_dict)
            continue

        if inp == "undo":
            if not history:
                print("  Nothing to undo.")
            else:
                revealed = history.pop()
                words = filter_words(all_words, revealed, excluded)
                print("  Undone.")
                print_grid(revealed)
                print_state(words, revealed, excluded, sort_mode, freq_dict)
            continue

        parts = inp.split()
        if len(parts) != 2:
            print("  Format: <letter> <col>,<row>  e.g.  l 3,4  |  delete: . <col>,<row>")
            continue

        cmd, coord = parts[0], parts[1]

        # --- Delete command: ". col,row" or "del col,row" or "delete col,row" ---
        if cmd in (".", "del", "delete"):
            try:
                cell = parse_cell(coord)
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            if cell not in revealed:
                print(f"  {cell_label(cell)} is already empty.")
                continue
            history.append(dict(revealed))
            removed = revealed.pop(cell)
            print(f"  Removed {removed.upper()} from {cell_label(cell)}.")
            words = filter_words(all_words, revealed, excluded)
            print_grid(revealed)
            print_state(words, revealed, excluded, sort_mode, freq_dict)
            continue

        # --- Oracle letter report ---
        if not cmd.isalpha() or len(cmd) != 1:
            print("  Format: <letter> <col>,<row>  e.g.  l 3,4  |  delete: . <col>,<row>")
            continue

        letter = cmd
        if letter in excluded:
            print(f"  '{letter.upper()}' is the excluded letter — not in the grid.")
            continue

        try:
            cell = parse_cell(coord)
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        # Push undo snapshot before any mutation.
        history.append(dict(revealed))

        # Each letter appears once: remove its previous cell if it moved.
        old_cell = next((c for c, l in revealed.items() if l == letter), None)
        if old_cell is not None and old_cell != cell:
            print(f"  Note: {letter.upper()} moved from {cell_label(old_cell)} to {cell_label(cell)}.")
            del revealed[old_cell]

        # Warn if a different letter was already at this cell.
        if cell in revealed and revealed[cell] != letter:
            print(f"  Warning: {cell_label(cell)} was '{revealed[cell].upper()}', "
                  f"now overwritten with '{letter.upper()}'.")

        revealed[cell] = letter
        words = filter_words(all_words, revealed, excluded)
        print_grid(revealed)
        print_state(words, revealed, excluded, sort_mode, freq_dict)


if __name__ == "__main__":
    main()
