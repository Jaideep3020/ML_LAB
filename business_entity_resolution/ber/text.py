from __future__ import annotations

from collections import Counter
import re
import unicodedata

# Only conservative, explicitly documented expansions; never rewrite ambiguous St.
SUFFIX = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "pvt", "private", "sarl", "sas"}
EXPAND = {"rd": "road", "ave": "avenue", "blvd": "boulevard", "hwy": "highway"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    # Preserve combining marks (essential for Indic scripts).
    return " ".join("".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in text).split())


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def core_name(text):
    tokens = normalize(text).split()
    while len(tokens) > 1 and tokens[-1] in SUFFIX:
        tokens.pop()
    return " ".join(tokens)


def address(text):
    return " ".join(EXPAND.get(t, t) for t in normalize(text).split())


def grams(text, n=3):
    text = "_" + text.replace(" ", "_") + "_"
    return {text[i:i+n] for i in range(max(0, len(text)-n+1))}


def script(text):
    count = Counter(unicodedata.name(c, "UNKNOWN").split()[0] for c in text if c.isalpha())
    return count.most_common(1)[0][0] if count else "NONE"


def numeric(text):
    return set(re.findall(r"\d+", text))


def postal(text):
    return set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", text))


def fts_tokens(text, ngram=False):
    # Hex encoding makes tokenization independent of SQLite Unicode version.
    values = grams(text) if ngram else set(text.split())
    return " ".join("x" + t.encode("utf-8").hex() for t in sorted(values))


def views(name, addr):
    n, a = normalize(name), address(addr)
    return n, a, core_name(name), fold(n), script(name)
