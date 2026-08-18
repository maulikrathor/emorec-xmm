"""Encoding-artefact cleanup for MELD utterance text.

CONTRACT: `normalize_text` fixes ENCODING ARTEFACTS ONLY — it is not a general
text-cleaning or normalization pass. It does NOT strip punctuation, does NOT
lowercase, and does NOT remove non-alphanumeric characters. Punctuation
carries emotion signal ("You did WHAT?!" vs "You did what."), and the text
branch of this project is a RoBERTa contract — RoBERTa's tokenizer expects
natural, punctuated text, not a bag of words. Any future temptation to add
"cleaning" (stopword removal, casing, punctuation stripping) here is a
regression; put it in a model-specific step if it's ever needed at all.
"""

import re

# Curly quotes / dashes / ellipsis -> ASCII equivalents. These are the
# Windows-1252 smart-punctuation codepoints MELD's source CSVs actually
# contain (TRIBAL_KNOWLEDGE.md §1.3), decoded correctly as UTF-8/cp1252.
_UNICODE_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}

# When a cp1252-encoded byte in this range survives a latin-1 fallback decode
# (load_split_csv's retry path), it lands as a raw C1 control character
# (U+0091..U+0097) instead of its intended smart-quote/dash meaning. There is
# no reliable way to recover the intended glyph at this point, so these are
# stripped rather than mapped.
_STRAY_CONTROL_CHARS = "".join(chr(c) for c in range(0x91, 0x98))

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Fix Windows-1252/smart-quote encoding artefacts in `s`; nothing else.

    Steps, in order: (1) map curly quotes/dashes/ellipsis to ASCII
    equivalents, (2) strip stray \\x91-\\x97 control-range bytes left behind
    by a latin-1 fallback decode, (3) collapse runs of whitespace to a single
    space, (4) strip leading/trailing whitespace. Punctuation, case, and all
    non-alphanumeric characters are otherwise preserved exactly.
    """
    for bad, good in _UNICODE_REPLACEMENTS.items():
        s = s.replace(bad, good)
    for stray in _STRAY_CONTROL_CHARS:
        s = s.replace(stray, "")
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()
