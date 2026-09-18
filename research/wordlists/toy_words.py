"""
Small curated toy vocabularies for research/solver.py, one per board size n.

These are NOT English dictionaries -- just fixed, arbitrary, hand-picked
lists of length-n strings (distinct letters, drawn from the n*n+1-letter toy
alphabet 'a', 'b', 'c', ...) chosen to have a little overlapping structure
(shared letters between words) so the solved examples aren't trivial.
"""

# n=2: alphabet {a,b,c,d,e} (5 letters).  All 5*4=20 distinct-letter
# 2-strings are "grammatical"; we designate 5 of them as the toy dictionary.
WORDS_N2 = ["ab", "bc", "cd", "de", "ea"]

# n=3: alphabet {a,...,j} (10 letters).
WORDS_N3 = ["abc", "cde", "efg", "ghi", "ijb", "bad", "dec", "fah"]

# Alternate n=2 vocabularies, used 2026-09-18 to check whether WORDS_N2's
# measured 44% M2-impossible-triple rate (logbook.tex) was specific to its
# densely-overlapping 5-word-cycle structure, or a property of n=2 itself.
# Confirmed: it's vocabulary-density-driven (7.4% -- 100% depending on
# vocabulary), not an n=2 artifact.
WORDS_N2_SPARSE = ["ab", "cd", "be"]              # minimal letter overlap -> 7.4% impossible
WORDS_N2_DENSER = ["ab", "cd", "ae", "bc"]        # more overlap -> 36.1% impossible
WORDS_N2_ALL = [a + b for a in "abcde" for b in "abcde" if a != b]  # every string is "a word" -> 100% impossible (M2 unsatisfiable by construction)
