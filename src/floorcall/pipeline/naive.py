"""The naive baseline agent: a silence timeout, and stop on any user speech (CLAUDE.md §13).

This is what a voice agent does with no decision layer. It answers once the user has been silent
for `silence_ms` (800 ms, a typical end-of-turn timeout; DECISIONS.md D-045), and it stops talking
the moment VAD hears the user. So it cuts users off when they pause mid-thought for longer than
the timeout, and it stops for every "uh-huh". It has no router, so it cannot tell an
out-of-scope request, and no escalation.

It returns the policy's own decision types, so replay compares the two agents action for action.
"""

from __future__ import annotations

from floorcall.policy import BargeAction, BargeDecision, PauseAction, PauseDecision


class NaiveAgent:
    name = "naive"

    def __init__(self, silence_ms: int) -> None:
        self.silence_ms = silence_ms

    def on_user_pause(self, *, silence_ms: float) -> PauseDecision:
        if silence_ms >= self.silence_ms:
            return PauseDecision(
                action=PauseAction.RESPOND_TIMEOUT,
                escalate=False,
                route=None,
                reason=f"silence {silence_ms:.0f} ms >= timeout {self.silence_ms} ms",
            )
        return PauseDecision(
            action=PauseAction.KEEP_LISTENING,
            escalate=False,
            route=None,
            reason=f"silence {silence_ms:.0f} ms < timeout {self.silence_ms} ms",
        )

    def on_user_speech_during_agent(self) -> BargeDecision:
        return BargeDecision(
            action=BargeAction.STOP_AND_LISTEN,
            escalate=False,
            reason="any user speech stops the agent",
        )
