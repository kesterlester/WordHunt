#!/usr/bin/env python3
"""
Download a comprehensive English word list and save as dict_raw.csv.

Columns:
  1: word
  2: frequency  (empty; reserved for future use)
"""

import csv
import urllib.request
import sys

URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
OUTPUT = "dict_raw.csv"


def main():
    print(f"Downloading word list from {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    words = [w.strip().lower() for w in raw.splitlines() if w.strip()]
    print(f"Downloaded {len(words):,} words.")

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word in words:
            writer.writerow([word, ""])

    print(f"Saved {len(words):,} words to {OUTPUT}.")


if __name__ == "__main__":
    main()
