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
  Oracle input format:  <letter> <col>,<row>   e.g.  l 3,4
"""

import csv
import os
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

SORT_MODES = ["alpha", "speed"]


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

def load_words(path: str = DICT_PATH) -> list[str]:
    words = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row and row[0]:
                words.append(row[0].strip().lower())
    return words


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

MAX_WORD_DISPLAY = 40


def sorted_words(
    words: list[str],
    sort_mode: str,
    steps_map: dict[str, int],
) -> list[str]:
    if sort_mode == "speed":
        return sorted(words, key=lambda w: (steps_map[w], w))
    return sorted(words)  # alpha


def print_state(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
    sort_mode: str,
) -> None:
    n = len(words)
    print(f"\nWords remaining: {n}  [sort: {sort_mode}]")

    if n == 0:
        print("  (none — check inputs for contradictions)")
        return

    steps_map = all_min_steps(words, revealed, excluded)
    display = sorted_words(words, sort_mode, steps_map)

    if sort_mode == "speed":
        # Annotate each word with its step count
        annotated = [f"{w}({steps_map[w]})" for w in display[:MAX_WORD_DISPLAY]]
        label = "  (steps-to-complete shown in parens)"
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
    print("  Commands:  'grid'  'mode'  'quit'")
    print("=" * 54 + "\n")

    words = load_words()
    print(f"Loaded {len(words):,} candidate words.")

    revealed: dict[tuple[int, int], str] = {}
    excluded: set[str] = set()
    sort_mode = SORT_MODES[0]

    # --- Excluded letter ---
    while True:
        exc = prompt("Which letter is excluded from the grid? ")
        if len(exc) == 1 and exc.isalpha():
            excluded.add(exc)
            break
        print("  Please enter a single letter (a-z).")

    words = filter_words(words, revealed, excluded)
    print_grid(revealed)
    print_state(words, revealed, excluded, sort_mode)

    # --- Main interaction loop ---
    while True:
        if not words:
            print("\nNo valid words remain — please check your inputs.")
            break
        if len(words) == 1:
            print(f"\n*** The word is: {words[0].upper()} ***")
            break

        print()
        inp = prompt(f"Oracle report (letter col,row) | grid | mode | quit : ")

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
            print_state(words, revealed, excluded, sort_mode)
            continue

        parts = inp.split()
        if len(parts) != 2 or not parts[0].isalpha() or len(parts[0]) != 1:
            print("  Format: <letter> <col>,<row>  e.g.  l 3,4")
            continue

        letter = parts[0]
        if letter in excluded:
            print(f"  '{letter.upper()}' is the excluded letter — not in the grid.")
            continue

        try:
            cell = parse_cell(parts[1])
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        # Each letter appears once in the grid: remove any previous cell for this letter.
        old_cell = next((c for c, l in revealed.items() if l == letter), None)
        if old_cell is not None and old_cell != cell:
            print(f"  Note: {letter.upper()} moved from {cell_label(old_cell)} to {cell_label(cell)}.")
            del revealed[old_cell]

        # Warn if a different letter was already recorded at this cell.
        if cell in revealed and revealed[cell] != letter:
            print(f"  Warning: {cell_label(cell)} was '{revealed[cell].upper()}', "
                  f"now overwritten with '{letter.upper()}'.")

        revealed[cell] = letter
        words = filter_words(words, revealed, excluded)
        print_grid(revealed)
        print_state(words, revealed, excluded, sort_mode)


if __name__ == "__main__":
    main()
