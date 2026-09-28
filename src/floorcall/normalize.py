"""ASR-style text normalization, shared by every path text takes to the model.

Streaming ASR output usually has no casing and no punctuation. SwDA and CLINC have both. Train on
punctuated text and the model learns "ends with a period, so the turn is complete", a cue that
does not exist at inference time. So training, test, replay and live text all pass through
`normalize`, via the state packer (floorcall.state), which is the one place every state is built.

The rules:
- NFKC, then lowercase.
- Letters, digits and combining marks survive. Every other character becomes a space, never
  nothing, so "3.5" becomes "3 5" rather than the different number "35".
- An apostrophe survives only between two word characters ("don't", "it's"). Leading and
  trailing ones ("'quoted'", "dogs'") go. Contractions stay recognisable to the tokenizer, and
  quote marks cannot become a signal.
- Whitespace collapses to single spaces, trimmed.

Idempotent: normalize(normalize(x)) == normalize(x).
"""

import re
import unicodedata

_APOSTROPHES = str.maketrans(
    {
        chr(0x2019): "'",  # RIGHT SINGLE QUOTATION MARK, the usual curly apostrophe
        chr(0x2018): "'",  # LEFT SINGLE QUOTATION MARK
        chr(0x02BC): "'",  # MODIFIER LETTER APOSTROPHE
        chr(0x0060): "'",  # GRAVE ACCENT, typed as an apostrophe
    }
)
# An apostrophe with no word character on one side of it.
_LOOSE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")


def _keep(ch: str) -> bool:
    return ch == "'" or ch.isspace() or unicodedata.category(ch)[0] in "LNM"


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    s = "".join(ch if _keep(ch) else " " for ch in s)
    s = _LOOSE_APOSTROPHE.sub(" ", s)
    return " ".join(s.split())
