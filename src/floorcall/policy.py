"""The policy: turns calibrated probabilities into actions.

The model says how likely things are; the policy decides what to do about it. Every threshold is
a point on a tradeoff curve, lives in config.PolicySettings, and is chosen on the calibration split,
never on test. Because the probabilities are calibrated, choosing a threshold means choosing an
error rate on purpose instead of guessing a timeout.

Design decisions, each one a question an interviewer can ask:

- **Thresholds, not argmax.** "Stop talking" is `p(interruption) >= theta_interrupt`, and "out of
  scope" is `p(out_of_scope) >= theta_oos`. Neither also requires being the argmax. An
  argmax-then-threshold rule caps the sweep at the argmax boundary. A pure threshold makes each
  theta one dial, and its sweep is exactly the tradeoff curve the README plots.
- **Escalation and routing act only when the agent responds.** While the user still holds the
  floor, handing off would cut them off mid-sentence. The escalation is acted on at the next
  response, whether from a completed turn or from the silence safety net.
- **An escalation signal during agent speech stops the agent.** Asking for a human while the agent
  talks is a claim on the floor, whatever `barge_in` says.
- **The silence safety net.** Whatever p(turn_complete) says, the agent responds after
  `max_wait_ms` of silence. A text model cannot hear prosody, and it must never leave a user
  waiting indefinitely.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from floorcall.config import PolicySettings
from floorcall.questions import BACKCHANNEL, BARGE_IN_LABELS, INTERRUPTION, OUT_OF_SCOPE


class PauseAction(StrEnum):
    RESPOND = "respond"
    RESPOND_TIMEOUT = "respond_after_timeout"
    KEEP_LISTENING = "keep_listening"


class BargeAction(StrEnum):
    STOP_AND_LISTEN = "stop_and_listen"
    KEEP_TALKING = "keep_talking"
    IGNORE = "ignore"


@dataclass(frozen=True)
class PauseDecision:
    action: PauseAction
    escalate: bool
    route: str | None  # an intent, OUT_OF_SCOPE, or None while listening
    reason: str


@dataclass(frozen=True)
class BargeDecision:
    action: BargeAction
    escalate: bool
    reason: str


def _check_p(name: str, p: float) -> None:
    if math.isnan(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {p}")


def _route(route_probs: Mapping[str, float], cfg: PolicySettings) -> str:
    if OUT_OF_SCOPE not in route_probs:
        raise ValueError(f"route probabilities must include {OUT_OF_SCOPE!r}")
    in_scope = {k: v for k, v in route_probs.items() if k != OUT_OF_SCOPE}
    if not in_scope:
        raise ValueError("route probabilities must include at least one in-scope intent")
    for k, v in route_probs.items():
        _check_p(f"p({k})", v)
    if route_probs[OUT_OF_SCOPE] >= cfg.theta_oos:
        return OUT_OF_SCOPE
    return max(in_scope, key=in_scope.__getitem__)


def on_user_pause(
    p_turn_complete: float,
    route_probs: Mapping[str, float],
    p_escalate: float,
    *,
    silence_ms: float,
    cfg: PolicySettings,
) -> PauseDecision:
    """VAD heard silence while the user holds the floor."""
    _check_p("p(turn_complete)", p_turn_complete)
    _check_p("p(escalate)", p_escalate)
    route = _route(route_probs, cfg)

    if p_turn_complete >= cfg.theta_yield:
        action = PauseAction.RESPOND
        reason = f"p(turn_complete)={p_turn_complete:.2f} >= {cfg.theta_yield}"
    elif silence_ms >= cfg.max_wait_ms:
        action = PauseAction.RESPOND_TIMEOUT
        reason = f"silence {silence_ms:.0f} ms >= max_wait {cfg.max_wait_ms} ms"
    else:
        return PauseDecision(
            action=PauseAction.KEEP_LISTENING,
            escalate=False,
            route=None,
            reason=(
                f"p(turn_complete)={p_turn_complete:.2f} < {cfg.theta_yield}, "
                f"silence {silence_ms:.0f} ms < {cfg.max_wait_ms} ms"
            ),
        )

    escalate = p_escalate >= cfg.theta_escalate
    if escalate:
        reason += f"; p(escalate)={p_escalate:.2f} >= {cfg.theta_escalate}: hand off"
    return PauseDecision(action=action, escalate=escalate, route=route, reason=reason)


def on_user_speech_during_agent(
    barge_probs: Mapping[str, float],
    p_escalate: float,
    *,
    cfg: PolicySettings,
) -> BargeDecision:
    """User audio arrived while the agent's TTS was playing."""
    if set(barge_probs) != set(BARGE_IN_LABELS):
        raise ValueError(f"barge_in probabilities must be exactly {BARGE_IN_LABELS}")
    for k, v in barge_probs.items():
        _check_p(f"p({k})", v)
    _check_p("p(escalate)", p_escalate)

    p_int = barge_probs[INTERRUPTION]
    escalate = p_escalate >= cfg.theta_escalate
    if escalate:
        return BargeDecision(
            action=BargeAction.STOP_AND_LISTEN,
            escalate=True,
            reason=f"p(escalate)={p_escalate:.2f} >= {cfg.theta_escalate}: stop and hand off",
        )
    if p_int >= cfg.theta_interrupt:
        return BargeDecision(
            action=BargeAction.STOP_AND_LISTEN,
            escalate=False,
            reason=f"p(interruption)={p_int:.2f} >= {cfg.theta_interrupt}",
        )
    top = max(barge_probs, key=barge_probs.__getitem__)
    if top in (BACKCHANNEL, INTERRUPTION):
        return BargeDecision(
            action=BargeAction.KEEP_TALKING,
            escalate=False,
            reason=f"{top} (p={barge_probs[top]:.2f}); p(interruption) {p_int:.2f} < {cfg.theta_interrupt}",
        )
    return BargeDecision(
        action=BargeAction.IGNORE,
        escalate=False,
        reason=f"noise (p={barge_probs[top]:.2f})",
    )
