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
