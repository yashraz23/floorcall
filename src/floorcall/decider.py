"""The decision layer's one entry point: an event and a snapshot in, calibrated probabilities out.

    decider = Decider.load(settings)                     # checkpoint + its calibration file
    decision = decider.decide(Event.USER_PAUSE, snapshot)
    decision.probabilities["turn_complete"]["true"]      # calibrated p(turn complete)

Every consumer goes through here: the Pipecat processor, replay mode, the latency harness, and the
evaluation of a fine-tuned checkpoint. That keeps packing, the forward call and the temperatures in
one place, so what is measured is what is served.

Temperatures: floorcall fits one per decision on the calib split (evaluate.calibration; D-021) and
stores them beside the checkpoint. A checkpoint without that file, i.e. the stock one, uses Laya's
own shipped temperatures, so the stock model behaves exactly as Laya serves it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from floorcall import questions
from floorcall.config import Settings, StateSettings
from floorcall.evaluate.calibration import load_calibration
from floorcall.state import PackedState, Snapshot, pack_state


class LogitsSource(Protocol):
    """What the Decider needs from a model. LayaDecider implements it."""

    def state_room(self, questions: Mapping[str, Mapping[str, Any]]) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    def effective_temperature(self, qtype: Any, n_options: int) -> float: ...
    def logits_one(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        check: bool = ...,
    ) -> Any: ...


@dataclass(frozen=True)
class Timings:
    pack_ms: float
    encode_ms: float
    forward_ms: float
    calls: int  # forward calls made: 1 batched, or one per question

    @property
    def model_ms(self) -> float:
        return self.encode_ms + self.forward_ms


@dataclass(frozen=True)
class Decision:
    event: questions.Event
    probabilities: dict[str, dict[str, float]]  # qid -> option label -> p
    packed: PackedState
    timings: Timings
    total_ms: float  # wall clock for the whole decide() call

    def p(self, qid: str, label: str) -> float:
        return self.probabilities[qid][label]


class Decider:
    def __init__(
        self,
        model: LogitsSource,
        state: StateSettings,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        self.model = model
        self.state = state
        self.temperatures = dict(temperatures) if temperatures is not None else None
        self._budgets: dict[questions.Event, int] = {}

    @classmethod
    def load(cls, settings: Settings) -> Decider:
        from floorcall.model.laya_adapter import LayaDecider

        model = LayaDecider(settings.laya)
        cal = load_calibration(Path(settings.laya.checkpoint))
        return cls(model, settings.state, cal.temperatures if cal else None)

    def budget(self, event: questions.Event) -> int:
        if event not in self._budgets:
            room = self.model.state_room(questions.questions_for(event))
            self._budgets[event] = room - self.state.safety_margin_tokens
        return self._budgets[event]

    def temperature(self, qid: str, qtype: Any, n_options: int) -> float:
        if self.temperatures is None:
            return self.model.effective_temperature(qtype, n_options)
        return self.temperatures[qid]

    def pack(self, event: questions.Event, snapshot: Snapshot) -> PackedState:
        return pack_state(
            snapshot,
            budget=self.budget(event),
            count_tokens=self.model.count_tokens,
            normalize_text=self.state.normalize,
            include_history=self.state.include_history,
            include_agent=self.state.include_agent,
        )

    def decide(
        self, event: questions.Event, snapshot: Snapshot, *, sequential: bool = False
    ) -> Decision:
        """Pack, score every question of `event`, apply temperatures.

        `sequential=True` makes one forward call per question instead of one batched call. It
        exists for the Table B comparison, not for serving.
        """
        t0 = time.perf_counter()
        packed = self.pack(event, snapshot)
        t_pack = (time.perf_counter() - t0) * 1000.0
        qs = questions.questions_for(event)
        # The packer measured the state with the model's own tokenizer against the smallest room
        # the event's questions leave, so re-checking it here would only re-tokenize.
        groups = [{qid: q} for qid, q in qs.items()] if sequential else [qs]
        encode_ms = forward_ms = 0.0
        probs: dict[str, dict[str, float]] = {}
        for group in groups:
            out = self.model.logits_one(packed.state, group, check=False)
            encode_ms += out.encode_ms
            forward_ms += out.forward_ms
            for qid, ql in out.logits.items():
                z = np.asarray(ql.logits, dtype=np.float64) / self.temperature(
                    qid, ql.qtype, len(ql.labels)
                )
                e = np.exp(z - z.max())
                p = e / e.sum()
                probs[qid] = {lab: float(v) for lab, v in zip(ql.labels, p, strict=True)}
        return Decision(
            event=event,
            probabilities=probs,
            packed=packed,
            timings=Timings(
                pack_ms=t_pack, encode_ms=encode_ms, forward_ms=forward_ms, calls=len(groups)
            ),
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )
