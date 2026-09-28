"""The Decider, against a fake model: packing budget, temperatures, one call vs sequential."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from floorcall import questions
from floorcall.config import StateSettings
from floorcall.decider import Decider
from floorcall.evaluate.scoring import softmax
from floorcall.model.laya_adapter import QuestionLogits, TimedLogits, option_labels
from floorcall.state import Snapshot

ROOM = 300


@dataclass
class FakeModel:
    """Scores every option of every question with fixed logits; records how it was called."""

    calls: list[list[str]] = field(default_factory=list)
    shipped_t: float = 2.0

    def state_room(self, qs: Any) -> int:
        return ROOM

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def effective_temperature(self, qtype: Any, n_options: int) -> float:
        return self.shipped_t

    def logits_one(self, state: Any, qs: Any, *, check: bool = True) -> TimedLogits:
        self.calls.append(list(qs))
        out = {}
        for qid, q in qs.items():
            labels = option_labels(q)
            out[qid] = QuestionLogits(qid, q["type"], labels, np.linspace(0.0, 1.0, len(labels)))
        return TimedLogits(out, encode_ms=1.0, forward_ms=2.0, input_tokens=10, graphed=False)


SNAP = Snapshot(
    agent_speaking=False, user_partial="I want to move money", agent_last_utterance="hi"
)


def test_one_batched_call_per_event() -> None:
    m = FakeModel()
    d = Decider(m, StateSettings())
    out = d.decide(questions.Event.USER_PAUSE, SNAP)
    assert m.calls == [["turn_complete", "route", "escalate"]]
    assert set(out.probabilities) == {"turn_complete", "route", "escalate"}
    assert out.timings.calls == 1
    assert out.timings.model_ms == pytest.approx(3.0)


def test_sequential_makes_one_call_per_question() -> None:
    m = FakeModel()
    out = Decider(m, StateSettings()).decide(
        questions.Event.USER_SPEECH_DURING_AGENT, SNAP, sequential=True
    )
    assert m.calls == [["barge_in"], ["escalate"]]
    assert out.timings.calls == 2
    assert out.timings.model_ms == pytest.approx(6.0)


def test_stock_uses_the_checkpoint_temperature() -> None:
    out = Decider(FakeModel(shipped_t=2.0), StateSettings()).decide(
        questions.Event.USER_PAUSE, SNAP
    )
    expected = softmax(np.array([0.0, 1.0]) / 2.0)
    assert out.p("turn_complete", "true") == pytest.approx(expected[1])


def test_fitted_temperatures_are_per_decision() -> None:
    temps = {"turn_complete": 0.5, "route": 1.0, "escalate": 4.0}
    out = Decider(FakeModel(), StateSettings(), temps).decide(questions.Event.USER_PAUSE, SNAP)
    assert out.p("turn_complete", "true") == pytest.approx(softmax(np.array([0.0, 1.0]) / 0.5)[1])
    assert out.p("escalate", "true") == pytest.approx(softmax(np.array([0.0, 1.0]) / 4.0)[1])
    for dist in out.probabilities.values():
        assert sum(dist.values()) == pytest.approx(1.0)


def test_budget_is_room_minus_margin_and_state_is_normalized() -> None:
    d = Decider(FakeModel(), StateSettings(safety_margin_tokens=8))
    out = d.decide(questions.Event.USER_PAUSE, SNAP)
    assert out.packed.budget == ROOM - 8
    assert out.packed.state["user_partial"] == "i want to move money"


def test_ablation_switches_reach_the_packer() -> None:
    d = Decider(FakeModel(), StateSettings(include_agent=False, normalize=False))
    out = d.decide(questions.Event.USER_PAUSE, SNAP)
    assert out.packed.state["agent_last_utterance"] == ""
    assert out.packed.state["user_partial"] == "I want to move money"


def test_every_question_has_its_labels_in_order() -> None:
    out = Decider(FakeModel(), StateSettings()).decide(questions.Event.USER_PAUSE, SNAP)
    assert tuple(out.probabilities["route"]) == questions.ROUTE_LABELS
