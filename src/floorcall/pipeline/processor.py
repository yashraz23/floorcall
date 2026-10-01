"""The decision processor: one pipeline event in, one decision event out (CLAUDE.md §7).

    processor = DecisionProcessor(Decider.load(settings), served.policy, device="cpu")
    ev = processor.on_user_pause(snapshot, silence_ms=300)
    ev.action, ev.route, ev.probabilities, ev.compute_ms

It holds no framework (DECISIONS.md D-045). Replay drives it directly, and the Space will too. In
v2, a Pipecat FrameProcessor will wrap it: VAD and transcript frames in, decision events out. Each
event makes one batched forward pass (Decider.decide). The policy (floorcall.policy) then turns the
calibrated probabilities into an action.

The one call that makes no forward pass is `after_silence`, the safety-net timer. The user has
said nothing since the pause decision, so the state is unchanged, and the policy is re-applied to
the same probabilities with the longer silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from floorcall import questions
from floorcall.config import PolicySettings
from floorcall.decider import Decision
from floorcall.policy import on_user_pause, on_user_speech_during_agent
from floorcall.questions import BARGE_IN, ESCALATE, ROUTE, TURN_COMPLETE, Event
from floorcall.state import Snapshot


class DecisionSource(Protocol):
    def decide(self, event: Event, snapshot: Snapshot) -> Decision: ...


@dataclass(frozen=True)
class DecisionEvent:
    """What the processor emits for one event, with everything a UI or log needs."""

    event: Event
    questions: tuple[str, ...]
    probabilities: dict[str, dict[str, float]]  # qid -> option label -> calibrated p
    action: str  # a policy.PauseAction or policy.BargeAction value
    escalate: bool
    route: str | None
    reason: str
    # Measured wall-clock ms of the forward pass on `device`; None for the safety-net timer, which
    # makes no pass.
    compute_ms: float | None
    device: str
    state_tokens: int


class DecisionProcessor:
    def __init__(self, decider: DecisionSource, policy: PolicySettings, *, device: str) -> None:
        self.decider = decider
        self.policy = policy
        self.device = device

    def on_user_pause(self, snapshot: Snapshot, *, silence_ms: float) -> DecisionEvent:
        """VAD heard `silence_ms` of silence while the user held the floor."""
        d = self.decider.decide(Event.USER_PAUSE, snapshot)
        return self._pause(d.probabilities, silence_ms, d.total_ms, d.packed.n_tokens)

    def after_silence(self, pause: DecisionEvent, *, silence_ms: float) -> DecisionEvent:
        """The safety-net timer: the policy again, on the pause's probabilities, after more
        silence. No forward pass."""
        if pause.event is not Event.USER_PAUSE:
            raise ValueError("after_silence follows a user_pause decision")
        return self._pause(pause.probabilities, silence_ms, None, pause.state_tokens)

    def on_user_speech_during_agent(self, snapshot: Snapshot) -> DecisionEvent:
        """The user spoke while the agent's speech was playing."""
        d = self.decider.decide(Event.USER_SPEECH_DURING_AGENT, snapshot)
        out = on_user_speech_during_agent(
            d.probabilities[BARGE_IN], d.p(ESCALATE, "true"), cfg=self.policy
        )
        return DecisionEvent(
            event=Event.USER_SPEECH_DURING_AGENT,
            questions=questions.EVENT_QUESTIONS[Event.USER_SPEECH_DURING_AGENT],
            probabilities=d.probabilities,
            action=out.action.value,
            escalate=out.escalate,
            route=None,
            reason=out.reason,
            compute_ms=d.total_ms,
            device=self.device,
            state_tokens=d.packed.n_tokens,
        )

    def _pause(
        self,
        probs: dict[str, dict[str, float]],
        silence_ms: float,
        compute_ms: float | None,
        state_tokens: int,
    ) -> DecisionEvent:
        out = on_user_pause(
            probs[TURN_COMPLETE]["true"],
            probs[ROUTE],
            probs[ESCALATE]["true"],
            silence_ms=silence_ms,
            cfg=self.policy,
        )
        return DecisionEvent(
            event=Event.USER_PAUSE,
            questions=questions.EVENT_QUESTIONS[Event.USER_PAUSE],
            probabilities=probs,
            action=out.action.value,
            escalate=out.escalate,
            route=out.route,
            reason=out.reason,
            compute_ms=compute_ms,
            device=self.device,
            state_tokens=state_tokens,
        )
