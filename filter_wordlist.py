#!/usr/bin/env python3
"""
Filter dict_raw.csv -> dict_game.csv.

Keeps only 5-letter words with no repeated letters, lowercased.
All metadata columns are preserved unchanged.
"""

import csv
import sys

INPUT = "dict_raw.csv"
OUTPUT = "dict_game.csv"


def is_valid_game_word(word: str) -> bool:
    return len(word) == 5 and word.isalpha() and len(set(word)) == 5


def main():
    kept = 0
    total = 0

    with open(INPUT, newline="", encoding="utf-8") as fin, \
         open(OUTPUT, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if not row:
                continue
            total += 1
            word = row[0].strip().lower()
            if is_valid_game_word(word):
                row[0] = word
                writer.writerow(row)
                kept += 1

    print(f"Read {total:,} words from {INPUT}.")
    print(f"Kept {kept:,} valid 5-letter no-repeat words -> {OUTPUT}.")


if __name__ == "__main__":
    main()
