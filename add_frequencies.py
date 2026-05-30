#!/usr/bin/env python3
"""
Enrich dict_raw.csv with Zipf-scale word frequencies from the wordfreq library.

Zipf scale: ~7 = extremely common ("the"), ~5 = common ("house"),
            ~3 = rare, 0 = not in corpus.

Reads and overwrites dict_raw.csv in place (column 2 = zipf_frequency).
After running this, re-run filter_wordlist.py to propagate to dict_game.csv.

Requires: wordfreq  (pip install wordfreq)
"""

import csv
import sys

try:
    from wordfreq import zipf_frequency
except ImportError:
    print("wordfreq not found.  Install it with:  pip install wordfreq", file=sys.stderr)
    sys.exit(1)

PATH = "dict_raw.csv"


def main() -> None:
    with open(PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("dict_raw.csv is empty.", file=sys.stderr)
        sys.exit(1)

    header = rows[0]
    if len(header) < 2:
        header.append("zipf_frequency")
    else:
        header[1] = "zipf_frequency"

    zeros = 0
    for row in rows[1:]:
        if not row:
            continue
        word = row[0].strip().lower()
        freq = zipf_frequency(word, "en")
        if len(row) < 2:
            row.append(f"{freq:.4f}")
        else:
            row[1] = f"{freq:.4f}"
        if freq == 0.0:
            zeros += 1

    with open(PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    total = len(rows) - 1
    print(f"Wrote Zipf frequencies for {total:,} words to {PATH}.")
    print(f"  {zeros:,} words had zero frequency (not in wordfreq corpus).")
    print("Now run:  python3 filter_wordlist.py")


if __name__ == "__main__":
    main()
