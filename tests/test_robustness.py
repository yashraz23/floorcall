"""The ASR noise model behind Table C. Written before floorcall/evaluate/robustness.py."""

import random
from typing import Any

import pytest

from floorcall.evaluate.robustness import asr_noise, noisy_row

VOCAB = ["the", "bank", "card", "money", "okay"]
TEXT = "I want to move money from my savings account to checking please"


def test_level_zero_only_normalizes() -> None:
    assert asr_noise(TEXT, 0.0, random.Random(0), VOCAB) == TEXT.lower()


def test_deterministic_for_a_seed() -> None:
    a = asr_noise(TEXT, 0.2, random.Random("row1:0.2"), VOCAB)
    b = asr_noise(TEXT, 0.2, random.Random("row1:0.2"), VOCAB)
    assert a == b


@pytest.mark.parametrize("level", [0.05, 0.1, 0.2])
def test_word_error_rate_tracks_the_level(level: float) -> None:
    # deletions + substitutions should hit about `level` of words; trailing drops add a little
    rng = random.Random(42)
    words = [f"w{i}" for i in range(20_000)]
    out = asr_noise(" ".join(words), level, rng, ["zzz"]).split()
    deleted = len(words) - len(out)
    substituted = out.count("zzz")
    rate = (deleted + substituted) / len(words)
    assert rate == pytest.approx(level, abs=0.01)


def test_only_the_users_words_are_noised() -> None:
    row: dict[str, Any] = {
        "id": "r1",
        "snapshot": {
            "agent_speaking": True,
            "user_partial": TEXT,
            "agent_last_utterance": "your card ending four four one seven was declined",
            "recent_turns": [
                {"speaker": "user", "text": TEXT},
                {"speaker": "agent", "text": "let me check that for you"},
            ],
        },
    }
    out = noisy_row(row, 0.5, seed=1, vocab=VOCAB)
    s = out["snapshot"]
    assert s["agent_last_utterance"] == row["snapshot"]["agent_last_utterance"]
    assert s["recent_turns"][1] == row["snapshot"]["recent_turns"][1]
    assert s["user_partial"] != TEXT.lower()
    assert row["snapshot"]["user_partial"] == TEXT, "the input row must not be modified"


def test_noise_is_per_row_and_level() -> None:
    base = {
        "id": "a",
        "snapshot": {
            "agent_speaking": False,
            "user_partial": TEXT,
            "agent_last_utterance": "",
            "recent_turns": [],
        },
    }
    other = {**base, "id": "b"}
    x = noisy_row(base, 0.3, seed=0, vocab=VOCAB)["snapshot"]["user_partial"]
    y = noisy_row(other, 0.3, seed=0, vocab=VOCAB)["snapshot"]["user_partial"]
    z = noisy_row(base, 0.3, seed=0, vocab=VOCAB)["snapshot"]["user_partial"]
    assert x == z
    assert x != y
