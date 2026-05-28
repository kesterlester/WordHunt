#!/usr/bin/env python3
"""
Word grid assistance tool.

Helps find the single 5-letter word hidden in a 5x5 grid where:
  - Every letter of the alphabet except one appears exactly once.
  - The word occupies one of 12 positions (5 rows, 5 cols, 2 diagonals).
  - The player learns one letter location per turn from an external oracle.
"""

import csv
import os
import sys
from collections import Counter

DICT_PATH = os.path.join(os.path.dirname(__file__), "dict_game.csv")

# ---------------------------------------------------------------------------
# 12 word positions in the 5x5 grid, as tuples of (row, col) in 0-based index.
# Rows/cols are presented to the user as 1-5.
# ---------------------------------------------------------------------------
POSITIONS: list[tuple[tuple[int, int], ...]] = []
for _r in range(5):
    POSITIONS.append(tuple((_r, _c) for _c in range(5)))          # horizontal
for _c in range(5):
    POSITIONS.append(tuple((_r, _c) for _r in range(5)))          # vertical
POSITIONS.append(tuple((_i, _i) for _i in range(5)))              # TL -> BR diagonal
POSITIONS.append(tuple((4 - _i, _i) for _i in range(5)))          # BL -> TR diagonal


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
    # Word must not contain any excluded letter.
    if any(ch in excluded for ch in word):
        return False

    pos_index = {cell: i for i, cell in enumerate(pos)}

    for cell, letter in revealed.items():
        if cell in pos_index:
            # Cell is inside this position: word must have the right letter there.
            if word[pos_index[cell]] != letter:
                return False
        else:
            # Cell is outside this position: since every grid letter is unique,
            # the hidden word at pos cannot contain this letter at all.
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
# Statistics
# ---------------------------------------------------------------------------

def letter_stats(
    words: list[str],
    revealed: dict[tuple[int, int], str],
) -> Counter:
    """Count how many words contain each letter, ignoring already-known letters."""
    known = set(revealed.values())
    counter: Counter = Counter()
    for w in words:
        for ch in set(w):
            if ch not in known:
                counter[ch] += 1
    return counter


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

MAX_WORD_DISPLAY = 40


def print_state(
    words: list[str],
    revealed: dict[tuple[int, int], str],
    excluded: set[str],
) -> None:
    n = len(words)
    print(f"\nWords remaining: {n}")

    if n == 0:
        print("  (none — check inputs for contradictions)")
        return

    display = sorted(words)
    if n <= MAX_WORD_DISPLAY:
        print("  " + "  ".join(display))
    else:
        print(f"  (showing first {MAX_WORD_DISPLAY} of {n})")
        print("  " + "  ".join(display[:MAX_WORD_DISPLAY]))

    stats = letter_stats(words, revealed)
    if stats:
        print("\nTop letter frequencies (positions not yet fixed):")
        for letter, count in stats.most_common(5):
            pct = 100 * count / n
            print(f"  {letter.upper()}: in {count}/{n} words ({pct:.0f}%)")


def print_grid(revealed: dict[tuple[int, int], str]) -> None:
    """Print a simple ASCII representation of what is known about the grid."""
    print("\n  Grid (. = unknown):")
    for r in range(5):
        row_str = "  "
        for c in range(5):
            cell = (r, c)
            row_str += (revealed[cell].upper() if cell in revealed else ".") + " "
        print(row_str)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def parse_cell(s: str) -> tuple[int, int]:
    """Parse 'row,col' (1-based) into a 0-based (row, col) tuple."""
    parts = s.strip().split(",")
    if len(parts) != 2:
        raise ValueError("Expected row,col (e.g. 3,4)")
    r, c = int(parts[0]) - 1, int(parts[1]) - 1
    if not (0 <= r <= 4 and 0 <= c <= 4):
        raise ValueError("Row and col must each be between 1 and 5")
    return (r, c)


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
    print("=" * 50)
    print("  Word Grid Assistant")
    print("=" * 50)
    print("Grid: 5x5, rows/cols numbered 1-5 from top-left.")
    print("Enter oracle reports as:  <letter> <row>,<col>")
    print("Type 'quit' to exit, 'grid' to show the grid.\n")

    words = load_words()
    print(f"Loaded {len(words):,} candidate words.")

    revealed: dict[tuple[int, int], str] = {}
    excluded: set[str] = set()

    # --- Get excluded letter ---
    while True:
        exc = prompt("Which letter is excluded from the grid? ")
        if len(exc) == 1 and exc.isalpha():
            excluded.add(exc)
            break
        print("  Please enter a single letter (a-z).")

    words = filter_words(words, revealed, excluded)
    print_state(words, revealed, excluded)

    # --- Main interaction loop ---
    while True:
        if not words:
            print("\nNo valid words remain — please check your inputs.")
            break
        if len(words) == 1:
            print(f"\n*** The word is: {words[0].upper()} ***")
            break

        print()
        inp = prompt("Oracle report (letter row,col) or 'grid'/'quit': ")

        if inp in ("quit", "q", "exit"):
            break

        if inp == "grid":
            print_grid(revealed)
            continue

        parts = inp.split()
        if len(parts) != 2 or not parts[0].isalpha() or len(parts[0]) != 1:
            print("  Format: <letter> <row>,<col>  e.g.  l 3,4")
            continue

        letter = parts[0]
        if letter in excluded:
            print(f"  '{letter.upper()}' is the excluded letter — it cannot appear in the grid.")
            continue

        try:
            cell = parse_cell(parts[1])
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        if cell in revealed and revealed[cell] != letter:
            print(f"  Warning: cell {cell[0]+1},{cell[1]+1} was previously "
                  f"'{revealed[cell].upper()}', now overwritten with '{letter.upper()}'.")

        revealed[cell] = letter
        words = filter_words(words, revealed, excluded)
        print_grid(revealed)
        print_state(words, revealed, excluded)


if __name__ == "__main__":
    main()
