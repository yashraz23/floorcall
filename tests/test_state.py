"""The state packer. Written before floorcall/state.py (CLAUDE.md §15).

These tests use `len` (characters of the serialized JSON) as the token counter. It is strictly
monotone and deterministic, so the tests pin down the packer's priority rules without loading a
tokenizer. The real counter is `LayaDecider.count_tokens`; tests/test_laya_adapter.py checks the
packer against it.
"""

import json
import random

import pytest

from floorcall.state import (
    FIELD_ORDER,
    PackedState,
    Snapshot,
    StateBudgetError,
    Turn,
    pack_state,
    serialize,
)

AGENT = "your card ending four four one seven was declined because the merchant flagged it"
USER = "yeah but why"
TURNS = (
    Turn("user", "I tried to pay at the grocery store."),
    Turn("agent", "I can help with that. Let me check your card."),
    Turn("user", "Okay, thanks!"),
    Turn("agent", "One moment please."),
)


def snap(**kw: object) -> Snapshot:
    base: dict[str, object] = {
        "agent_speaking": True,
        "user_partial": USER,
        "agent_last_utterance": AGENT,
        "recent_turns": TURNS,
    }
    base.update(kw)
    return Snapshot(**base)  # type: ignore[arg-type]


def full_size(s: Snapshot) -> int:
    return len(serialize(pack_state(s, budget=10_000, count_tokens=len).state))


def test_everything_fits() -> None:
    p = pack_state(snap(), budget=10_000, count_tokens=len)
    assert p.turns_dropped == 0
    assert p.agent_words_dropped == 0
    assert p.user_words_dropped == 0
    assert p.state["user_partial"] == "yeah but why"
    assert p.state["agent_last_utterance"] == AGENT
    assert [t["text"] for t in p.state["recent_turns"]] == [
        "i tried to pay at the grocery store",
        "i can help with that let me check your card",
        "okay thanks",
        "one moment please",
    ]
    assert p.n_tokens == len(serialize(p.state))


def test_field_order_is_chronological() -> None:
    p = pack_state(snap(), budget=10_000, count_tokens=len)
    assert tuple(p.state) == FIELD_ORDER
    assert FIELD_ORDER == ("agent_speaking", "recent_turns", "agent_last_utterance", "user_partial")


def test_serialize_matches_laya() -> None:
    # laya.common.serialize_state is json.dumps(state, ensure_ascii=False) with default
    # separators; the packer must count exactly the text the model will read.
    p = pack_state(snap(user_partial="café"), budget=10_000, count_tokens=len)
    assert serialize(p.state) == json.dumps(p.state, ensure_ascii=False)


def test_oldest_turns_go_first() -> None:
    size = full_size(snap())
    p = pack_state(snap(), budget=size - 1, count_tokens=len)
    assert p.turns_dropped >= 1
    kept = [t["text"] for t in p.state["recent_turns"]]
    # what survives is the newest contiguous run, still oldest-first
    assert (
        kept
        == ["i can help with that let me check your card", "okay thanks", "one moment please"][
            -len(kept) :
        ]
    )
    assert p.agent_words_dropped == 0


def test_agent_utterance_keeps_its_tail() -> None:
    no_turns = full_size(snap(recent_turns=()))
    p = pack_state(snap(), budget=no_turns - 10, count_tokens=len)
    assert p.state["recent_turns"] == []
    assert p.turns_dropped == len(TURNS)
    assert p.agent_words_dropped > 0
    kept = p.state["agent_last_utterance"].split()
    assert kept == AGENT.split()[-len(kept) :]
    assert p.state["user_partial"] == "yeah but why"


def test_user_partial_is_cut_last() -> None:
    bare = len(
        serialize(
            {
                "agent_speaking": True,
                "recent_turns": [],
                "agent_last_utterance": "",
                "user_partial": "yeah but why",
            }
        )
    )
    p = pack_state(snap(), budget=bare, count_tokens=len)
    assert p.state["user_partial"] == "yeah but why"
    assert p.state["agent_last_utterance"] == ""
    assert p.user_words_dropped == 0


def test_user_partial_keeps_its_tail_when_it_alone_overflows() -> None:
    long_user = " ".join(f"word{i}" for i in range(100))
    p = pack_state(snap(user_partial=long_user), budget=200, count_tokens=len)
    assert p.user_words_dropped > 0
    kept = p.state["user_partial"].split()
    assert kept == long_user.split()[-len(kept) :]
    assert p.state["agent_last_utterance"] == ""
    assert p.state["recent_turns"] == []
    assert p.n_tokens <= 200


def test_priority_is_strict_no_scraps_after_a_cut() -> None:
    # The agent's next word is 60 characters, so cutting it leaves room for a short turn that a
    # greedy packer would add. A cut field means every lower-priority field stays empty.
    long_word = "x" * 60
    s = snap(agent_last_utterance=f"{long_word} tail", recent_turns=(Turn("user", "ok"),))
    tail_only = len(
        serialize(
            pack_state(
                snap(agent_last_utterance="tail", recent_turns=()), budget=10_000, count_tokens=len
            ).state
        )
    )
    p = pack_state(s, budget=tail_only + 50, count_tokens=len)
    assert p.state["agent_last_utterance"] == "tail"
    assert p.state["recent_turns"] == []
    assert p.turns_dropped == 1


def test_budget_below_the_empty_skeleton_raises() -> None:
    with pytest.raises(StateBudgetError):
        pack_state(snap(), budget=10, count_tokens=len)


@pytest.mark.parametrize("seed", range(25))
def test_never_exceeds_budget(seed: int) -> None:
    rng = random.Random(seed)
    words = [
        "i",
        "want",
        "to",
        "move",
        "money",
        "from",
        "my",
        "savings",
        "to",
        "checking",
        "account",
        "please",
        "now",
    ]
    turns = tuple(
        Turn(rng.choice(["user", "agent"]), " ".join(rng.choices(words, k=rng.randint(1, 15))))
        for _ in range(rng.randint(0, 8))
    )
    s = snap(
        user_partial=" ".join(rng.choices(words, k=rng.randint(1, 40))),
        agent_last_utterance=" ".join(rng.choices(words, k=rng.randint(0, 30))),
        recent_turns=turns,
    )
    skeleton = len(
        serialize(
            {
                "agent_speaking": True,
                "recent_turns": [],
                "agent_last_utterance": "",
                "user_partial": "",
            }
        )
    )
    budget = rng.randint(skeleton, full_size(s) + 20)
    p = pack_state(s, budget=budget, count_tokens=len)
    assert isinstance(p, PackedState)
    assert p.n_tokens == len(serialize(p.state)) <= budget


def test_normalizes_every_text_field() -> None:
    p = pack_state(
        snap(user_partial="Yeah, but WHY?", agent_last_utterance="Hold on."),
        budget=10_000,
        count_tokens=len,
    )
    assert p.state["user_partial"] == "yeah but why"
    assert p.state["agent_last_utterance"] == "hold on"
    assert all(t["text"] == t["text"].lower() for t in p.state["recent_turns"])


def test_normalization_can_be_switched_off_for_the_ablation() -> None:
    p = pack_state(
        snap(user_partial="Yeah, but WHY?"), budget=10_000, count_tokens=len, normalize_text=False
    )
    assert p.state["user_partial"] == "Yeah, but WHY?"


def test_turns_empty_after_normalization_are_dropped() -> None:
    p = pack_state(
        snap(recent_turns=(Turn("user", "..."), Turn("agent", "okay"))),
        budget=10_000,
        count_tokens=len,
    )
    assert p.state["recent_turns"] == [{"speaker": "agent", "text": "okay"}]


def test_rejects_unknown_speaker() -> None:
    with pytest.raises(ValueError, match="speaker"):
        Turn("caller", "hello")  # type: ignore[arg-type]
