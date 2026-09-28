"""The latency harness's bookkeeping, with a fake model (no GPU, no checkpoint)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from floorcall import questions
from floorcall.config import StateSettings, get_settings
from floorcall.decider import Decider
from floorcall.evaluate import latency
from floorcall.model.laya_adapter import QuestionLogits, TimedLogits, option_labels


@dataclass
class FakeModel:
    device: str = "cuda"
    fall_back_after: int | None = None  # simulate Laya's silent CPU fallback
    calls: int = 0
    seen: list[str] = field(default_factory=list)

    def state_room(self, qs: Any) -> int:
        return 400

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def effective_temperature(self, qtype: Any, n: int) -> float:
        return 1.0

    def logits_one(self, state: Any, qs: Any, *, check: bool = True) -> TimedLogits:
        self.calls += 1
        if self.fall_back_after is not None and self.calls > self.fall_back_after:
            self.device = "cpu"
        out = {
            qid: QuestionLogits(qid, q["type"], option_labels(q), np.zeros(len(option_labels(q))))
            for qid, q in qs.items()
        }
        return TimedLogits(out, encode_ms=0.5, forward_ms=1.5, input_tokens=5, graphed=False)


ROW = latency.Row("t", "test", "cuda", False, questions.Event.USER_PAUSE, False)


def test_percentiles() -> None:
    p = latency.percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert p["p99"] == pytest.approx(99.01)
    assert p["max"] == 100


def test_inputs_are_fixed_and_mix_decisions() -> None:
    s = get_settings()
    a = latency.inputs(s, questions.Event.USER_PAUSE)
    b = latency.inputs(s, questions.Event.USER_PAUSE)
    assert a == b
    assert len(a) == latency.N_INPUTS
    assert len(latency.inputs(s, questions.Event.USER_SPEECH_DURING_AGENT)) == latency.N_INPUTS


def test_measure_reports_every_component() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    out = latency.measure(Decider(FakeModel(), StateSettings()), ROW, snaps, warmup=3, iters=20)
    assert set(out) == {"total_ms", "pack_ms", "encode_ms", "forward_ms", "state_tokens_mean"}
    assert out["forward_ms"]["p50"] == pytest.approx(1.5)
    # The fake reports a forward time without spending it, so the wall-clock total is only
    # bounded below by the parts that are really timed here: packing.
    assert out["total_ms"]["p50"] >= out["pack_ms"]["p50"] > 0


def test_a_gpu_row_refuses_a_cpu_model() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    with pytest.raises(RuntimeError, match="D-010"):
        latency.measure(Decider(FakeModel(device="cpu"), StateSettings()), ROW, snaps, 1, 2)


def test_a_silent_fallback_during_timing_is_caught() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    model = FakeModel(fall_back_after=5)
    with pytest.raises(RuntimeError, match="left cuda"):
        latency.measure(Decider(model, StateSettings()), ROW, snaps, warmup=2, iters=10)
