"""Table C: the same test sets, degraded to look like real ASR output.

The noise model, applied at a given level (roughly a word error rate):
- work in normalized text, as a recogniser emits it;
- each word is deleted with probability level/2, or replaced with probability level/2 by a word
  drawn uniformly from the most common train words (so substitutions are plausible words, not
  noise tokens);
- then, with probability `level`, the last one or two words go missing, as when a partial
  transcript has not caught up with the speaker.
Only the user's words are degraded: `user_partial` and the user's turns in the history. The agent's
lines come from our own TTS input, so a live pipeline knows them exactly.

Per row, the noise is seeded by (seed, row id, level), so every model sees the same degraded text.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from floorcall.normalize import normalize


def asr_noise(text: str, level: float, rng: random.Random, vocab: Sequence[str]) -> str:
    words = normalize(text).split()
    if level <= 0:
        return " ".join(words)
    out = []
    for w in words:
        r = rng.random()
        if r < level / 2:
            continue
        out.append(rng.choice(vocab) if r < level else w)
    if out and rng.random() < level:
        out = out[: len(out) - rng.randint(1, min(2, len(out)))]
    return " ".join(out)


def noisy_row(
    row: dict[str, Any], level: float, *, seed: int, vocab: Sequence[str]
) -> dict[str, Any]:
    """A deep copy of `row` with the user's words degraded."""
    rng = random.Random(f"{seed}:{row['id']}:{level}")
    out = copy.deepcopy(row)
    s = out["snapshot"]
    s["user_partial"] = asr_noise(s["user_partial"], level, rng, vocab)
    for turn in s["recent_turns"]:
        if turn["speaker"] == "user":
            turn["text"] = asr_noise(turn["text"], level, rng, vocab)
    return out


def train_vocab(texts: Iterable[str], size: int) -> list[str]:
    """The `size` most common normalized words, the pool substitutions are drawn from."""
    counts = Counter(w for t in texts for w in normalize(t).split())
    return [w for w, _ in counts.most_common(size)]
