"""FROZEN REFERENCE: the state packer as it was before the fast search (commit f4c7e87).

Only tests/test_packer_equivalence.py imports this file. The current packer (floorcall.state)
must produce exactly the same packed state as this one, row for row (DECISIONS.md D-026). Do not
edit or "fix" anything below; it is kept as it was.

---

The state packer: what the pipeline knows at decision time, fitted to the model's window.

Every state the model sees, in training, test, replay or live, is built here. That makes this the
one place to enforce the two rules every reported number depends on:

1. Text is ASR-normalized (floorcall.normalize), so written-corpus punctuation cannot leak.
2. The serialized state fits the token room its questions leave. Laya right-truncates an
   oversized state silently; a state that would be truncated is a bug (CLAUDE.md §5), so the
   packer shortens it deliberately, in priority order, and reports exactly what it dropped.

Priority when the budget is tight (CLAUDE.md §5):
  1. `user_partial`, whole. The words being decided about. If it alone overflows (a very long
     monologue), its *tail* is kept, since the end is what turn-completion hinges on, and
     `user_words_dropped` says so.
  2. `agent_last_utterance`, tail first. On a barge-in, what the agent had just said is what the
     user is reacting to.
  3. `recent_turns`, newest first, whole turns only. They stop at the first turn that does not fit,
     so the history stays contiguous.

Priority is strict: once a field is cut, every lower-priority field is left empty, even if a
scrap of it would fit in the tokens left over.

Field order in the serialized JSON is chronological (older context first, the user's words last).
The spec's example listed the fields in a different order. Order has no semantic meaning to the
packer, but it must be identical in training and inference, and chronological is the natural
reading order for an encoder.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from floorcall.normalize import normalize

log = logging.getLogger(__name__)

Speaker = Literal["user", "agent"]
TokenCounter = Callable[[str], int]

FIELD_ORDER = ("agent_speaking", "recent_turns", "agent_last_utterance", "user_partial")


class StateBudgetError(ValueError):
    """The budget cannot hold even an empty state. A configuration bug, not a packing problem."""


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in ("user", "agent"):
            raise ValueError(f"speaker must be 'user' or 'agent', got {self.speaker!r}")


@dataclass(frozen=True)
class Snapshot:
    """What the pipeline knows when a decision fires, before packing."""

    agent_speaking: bool
    user_partial: str
    agent_last_utterance: str = ""
    recent_turns: Sequence[Turn] = field(default_factory=tuple)  # oldest first


@dataclass(frozen=True)
class PackedState:
    state: dict[str, Any]
    n_tokens: int
    budget: int
    turns_kept: int
    turns_dropped: int
    agent_words_dropped: int
    user_words_dropped: int


def serialize(state: dict[str, Any]) -> str:
    """The exact text the model reads: laya.common.serialize_state is `json.dumps(state,
    ensure_ascii=False)` with default separators. tests/test_laya_adapter.py pins the two
    together."""
    return json.dumps(state, ensure_ascii=False)


def _build(
    agent_speaking: bool, turns: list[dict[str, str]], agent: str, user: str
) -> dict[str, Any]:
    return {
        "agent_speaking": agent_speaking,
        "recent_turns": turns,
        "agent_last_utterance": agent,
        "user_partial": user,
    }


def _longest_fitting_tail(words: list[str], fits: Callable[[str], bool]) -> int:
    """How many of `words`, counted from the end, can be kept. Binary search, then a linear
    step-down, since a BPE count is only nearly monotone in the number of words."""
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(" ".join(words[len(words) - mid :])):
            lo = mid
        else:
            hi = mid - 1
    while lo > 0 and not fits(" ".join(words[len(words) - lo :])):
        lo -= 1
    return lo


def pack_state(
    snapshot: Snapshot,
    *,
    budget: int,
    count_tokens: TokenCounter,
    normalize_text: bool = True,
    include_history: bool = True,
    include_agent: bool = True,
) -> PackedState:
    """Fit `snapshot` into `budget` tokens of serialized state.

    `budget` is the room the event's questions leave for the state, minus the configured safety
    margin (`LayaDecider.state_room`, `StateSettings.safety_margin_tokens`). `count_tokens` counts
    the serialized JSON; in production it is `LayaDecider.count_tokens`.

    `normalize_text`, `include_history` and `include_agent` exist for the Table D ablations only
    (StateSettings). Leaving a field out empties it but keeps its key, so the schema has one shape,
    and a field left out is not "cut": the fields below it still get their tokens.
    """
    clean = normalize if normalize_text else (lambda s: " ".join(s.split()))
    user_words = clean(snapshot.user_partial).split()
    agent_words = clean(snapshot.agent_last_utterance).split()
    turns = [{"speaker": t.speaker, "text": clean(t.text)} for t in snapshot.recent_turns]
    turns = [t for t in turns if t["text"]]
    speaking = bool(snapshot.agent_speaking)

    def size(state: dict[str, Any]) -> int:
        return count_tokens(serialize(state))

    if size(_build(speaking, [], "", "")) > budget:
        raise StateBudgetError(f"budget {budget} cannot hold an empty state")

    # 1. the user's words, whole if possible, else their tail
    n_user = _longest_fitting_tail(
        user_words, lambda u: size(_build(speaking, [], "", u)) <= budget
    )
    user = " ".join(user_words[len(user_words) - n_user :])

    # Strict priority: a field gets no tokens while a higher-priority one is cut. Greedy filling
    # would hand the scraps left after a cut to the next field, e.g. a one-word agent fragment
    # beside a truncated user turn, which is noise rather than context.
    user_cut = n_user < len(user_words)

    agent_pool = agent_words if include_agent else []
    turn_pool = turns if include_history else []

    # 2. what the agent last said, tail first
    n_agent = (
        0
        if user_cut
        else _longest_fitting_tail(
            agent_pool, lambda a: size(_build(speaking, [], a, user)) <= budget
        )
    )
    agent = " ".join(agent_pool[len(agent_pool) - n_agent :])
    agent_cut = user_cut or n_agent < len(agent_pool)

    # 3. history, newest first, whole turns, contiguous
    kept: list[dict[str, str]] = []
    for turn in [] if agent_cut else reversed(turn_pool):
        trial = [turn, *kept]
        if size(_build(speaking, trial, agent, user)) > budget:
            break
        kept = trial

    state = _build(speaking, kept, agent, user)
    n_tokens = size(state)
    # The steps above only ever accept a candidate after measuring it, so this cannot fire unless
    # the counter is not a function of its input. It is checked rather than assumed because an
    # oversized state would be truncated silently downstream.
    if n_tokens > budget:
        raise AssertionError(f"packer produced {n_tokens} tokens for a budget of {budget}")
    packed = PackedState(
        state=state,
        n_tokens=n_tokens,
        budget=budget,
        turns_kept=len(kept),
        turns_dropped=len(turns) - len(kept),
        agent_words_dropped=len(agent_words) - n_agent,
        user_words_dropped=len(user_words) - n_user,
    )
    log.debug(
        "packed state: %d/%d tokens, turns %d kept %d dropped, agent words dropped %d, "
        "user words dropped %d",
        packed.n_tokens,
        packed.budget,
        packed.turns_kept,
        packed.turns_dropped,
        packed.agent_words_dropped,
        packed.user_words_dropped,
    )
    return packed
