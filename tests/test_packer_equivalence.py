"""The fast packer against the frozen reference, with the real tokenizer (DECISIONS.md D-026).

The full check ran over all 29,807 frozen test rows with 0 differences. This test re-runs it on a
spread of 600 rows, so any change to the search that alters a packed state fails here.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.evaluate.dataset import load_test, snapshot_of
from floorcall.model.laya_adapter import LayaDecider
from floorcall.state import pack_state

sys.path.insert(0, str(Path(__file__).parent))
import reference_packer as ref


@pytest.fixture(scope="module")
def decider() -> Iterator[LayaDecider]:
    yield LayaDecider(LayaSettings(**{**get_settings().laya.model_dump(), "device": "cpu"}))


@pytest.mark.model
@pytest.mark.parametrize("decision", ["turn_complete", "barge_in", "route"])
def test_fast_packer_matches_the_reference(decider: LayaDecider, decision: str) -> None:
    s = get_settings()
    test = load_test(s, decision)
    budget = decider.state_room(questions.questions_for(test.event)) - s.state.safety_margin_tokens
    for r in test.rows[:: max(1, len(test.rows) // 200)][:200]:
        snap = snapshot_of(r)
        # the reference module has its own Snapshot and Turn types
        ref_snap = ref.Snapshot(
            agent_speaking=snap.agent_speaking,
            user_partial=snap.user_partial,
            agent_last_utterance=snap.agent_last_utterance,
            recent_turns=tuple(ref.Turn(t.speaker, t.text) for t in snap.recent_turns),
        )
        a = ref.pack_state(ref_snap, budget=budget, count_tokens=decider.count_tokens)
        b = pack_state(snap, budget=budget, count_tokens=decider.count_tokens)
        old = (a.state, a.n_tokens, a.turns_kept, a.agent_words_dropped, a.user_words_dropped)
        new = (b.state, b.n_tokens, b.turns_kept, b.agent_words_dropped, b.user_words_dropped)
        assert old == new, r["id"]
