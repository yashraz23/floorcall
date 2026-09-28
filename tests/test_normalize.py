"""ASR-style normalization. Written before floorcall/normalize.py (CLAUDE.md §15).

The point of the normalizer is to remove cues that exist in written corpora but not in streaming
ASR output. The most dangerous one is sentence-final punctuation: a model trained on "okay." vs
"okay so" learns that a period means the turn is over, and live ASR never produces the period.
"""

import string

import pytest

from floorcall.normalize import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Yeah, but WHY?", "yeah but why"),
        ("I don't know.", "i don't know"),
        ("It's fine, it's fine!", "it's fine it's fine"),
        # word-internal apostrophes survive; quote marks around words do not
        ("'quoted' text", "quoted text"),
        ("the dogs' bowls", "the dogs bowls"),
        ("rock 'n' roll", "rock n roll"),
        # curly apostrophe is the same apostrophe
        (f"I don{chr(0x2019)}t", "i don't"),
        # punctuation inside a token becomes a space, never a join: 3.5 must not become 35
        ("3.5 percent", "3 5 percent"),
        ("e-mail me", "e mail me"),
        ("$50.00", "50 00"),
        # SwDA's interruption and abandonment markers are punctuation too
        ("so I was going to --", "so i was going to"),
        ("well, uh, / you know", "well uh you know"),
        # whitespace of every kind collapses
        ("  hello \t\n  world  ", "hello world"),
        ("", ""),
        ("?!...", ""),
        # letters outside ASCII are letters
        ("Café crème", "café crème"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


SAMPLES = [
    "Yeah, but WHY?",
    "I don't know.",
    "'quoted' text",
    "it's the dogs' bowls -- right?",
    "3.5% of $50.00",
    "  Mixed\tCASE \n text!  ",
    "don''t",
    "'''",
]


@pytest.mark.parametrize("raw", SAMPLES)
def test_idempotent(raw: str) -> None:
    once = normalize(raw)
    assert normalize(once) == once


@pytest.mark.parametrize("raw", SAMPLES)
def test_output_has_no_punctuation_but_internal_apostrophes(raw: str) -> None:
    out = normalize(raw)
    assert not (set(out) & (set(string.punctuation) - {"'"}))
    for token in out.split():
        assert not token.startswith("'")
        assert not token.endswith("'")


@pytest.mark.parametrize("raw", SAMPLES)
def test_output_is_lowercase_with_single_spaces(raw: str) -> None:
    out = normalize(raw)
    assert out == out.lower()
    assert "  " not in out
    assert out == out.strip()
