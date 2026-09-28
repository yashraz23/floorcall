"""The policy: probabilities in, actions out. Written before floorcall/policy.py (CLAUDE.md §15)."""

import pytest

from floorcall.config import PolicySettings
from floorcall.policy import (
    BargeAction,
    PauseAction,
    on_user_pause,
    on_user_speech_during_agent,
)
from floorcall.questions import BACKCHANNEL, INTERRUPTION, NOISE, OUT_OF_SCOPE

CFG = PolicySettings(
    theta_yield=0.6, max_wait_ms=1500, theta_escalate=0.7, theta_oos=0.5, theta_interrupt=0.6
)
ROUTE = {"balance": 0.7, "transfer": 0.1, OUT_OF_SCOPE: 0.2}


def barge(b: float, i: float, n: float) -> dict[str, float]:
    return {BACKCHANNEL: b, INTERRUPTION: i, NOISE: n}


# -- user_pause ------------------------------------------------------------------------------


def test_responds_when_turn_is_complete() -> None:
    d = on_user_pause(0.8, ROUTE, 0.1, silence_ms=300, cfg=CFG)
    assert d.action is PauseAction.RESPOND
    assert d.route == "balance"
    assert not d.escalate


def test_threshold_is_inclusive() -> None:
    assert on_user_pause(0.6, ROUTE, 0.1, silence_ms=0, cfg=CFG).action is PauseAction.RESPOND


def test_keeps_listening_when_turn_is_incomplete() -> None:
    d = on_user_pause(0.3, ROUTE, 0.1, silence_ms=300, cfg=CFG)
    assert d.action is PauseAction.KEEP_LISTENING
    assert d.route is None


def test_safety_net_responds_after_max_wait() -> None:
    d = on_user_pause(0.3, ROUTE, 0.1, silence_ms=1500, cfg=CFG)
    assert d.action is PauseAction.RESPOND_TIMEOUT
    assert d.route == "balance"


def test_no_escalation_while_the_user_still_holds_the_floor() -> None:
    # Handing off mid-sentence would cut the user off. The escalation is acted on when the agent
    # next responds: a completed turn, or the safety net.
    d = on_user_pause(0.3, ROUTE, 0.95, silence_ms=300, cfg=CFG)
    assert d.action is PauseAction.KEEP_LISTENING
    assert not d.escalate


def test_escalates_when_responding() -> None:
    d = on_user_pause(0.8, ROUTE, 0.7, silence_ms=300, cfg=CFG)
    assert d.escalate


def test_out_of_scope_is_a_threshold_not_an_argmax() -> None:
    # p(oos) = 0.35 is not the argmax, but it clears a threshold of 0.3
    cfg = CFG.model_copy(update={"theta_oos": 0.3})
    probs = {"balance": 0.6, "transfer": 0.05, OUT_OF_SCOPE: 0.35}
    assert on_user_pause(0.9, probs, 0.0, silence_ms=0, cfg=cfg).route == OUT_OF_SCOPE


def test_below_the_oos_threshold_routes_to_the_best_in_scope_intent() -> None:
    # oos is the argmax at 0.6 but under a 0.9 threshold: the best in-scope intent wins. This is
    # what makes theta_oos a single dial trading false rejections against missed out-of-scope.
    cfg = CFG.model_copy(update={"theta_oos": 0.9})
    probs = {"balance": 0.1, "transfer": 0.3, OUT_OF_SCOPE: 0.6}
    assert on_user_pause(0.9, probs, 0.0, silence_ms=0, cfg=cfg).route == "transfer"


def test_route_probs_need_out_of_scope_and_an_intent() -> None:
    with pytest.raises(ValueError, match="out_of_scope"):
        on_user_pause(0.9, {"balance": 1.0}, 0.0, silence_ms=0, cfg=CFG)
    with pytest.raises(ValueError, match="in-scope"):
        on_user_pause(0.9, {OUT_OF_SCOPE: 1.0}, 0.0, silence_ms=0, cfg=CFG)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_probabilities(bad: float) -> None:
    with pytest.raises(ValueError):
        on_user_pause(bad, ROUTE, 0.1, silence_ms=0, cfg=CFG)


# -- user_speech_during_agent ----------------------------------------------------------------


def test_stops_for_an_interruption() -> None:
    d = on_user_speech_during_agent(barge(0.1, 0.8, 0.1), 0.1, cfg=CFG)
    assert d.action is BargeAction.STOP_AND_LISTEN


def test_keeps_talking_through_a_backchannel() -> None:
    d = on_user_speech_during_agent(barge(0.8, 0.1, 0.1), 0.1, cfg=CFG)
    assert d.action is BargeAction.KEEP_TALKING


def test_ignores_noise() -> None:
    d = on_user_speech_during_agent(barge(0.1, 0.1, 0.8), 0.1, cfg=CFG)
    assert d.action is BargeAction.IGNORE


def test_interruption_below_threshold_does_not_stop() -> None:
    # argmax is interruption at 0.5, but the deployer set 0.6: the agent keeps talking
    d = on_user_speech_during_agent(barge(0.3, 0.5, 0.2), 0.1, cfg=CFG)
    assert d.action is BargeAction.KEEP_TALKING


def test_interruption_is_a_threshold_not_an_argmax() -> None:
    cfg = CFG.model_copy(update={"theta_interrupt": 0.3})
    d = on_user_speech_during_agent(barge(0.6, 0.35, 0.05), 0.1, cfg=cfg)
    assert d.action is BargeAction.STOP_AND_LISTEN


def test_an_escalation_request_stops_the_agent() -> None:
    # "just get me a human" while the agent talks is a floor claim whatever barge_in says
    d = on_user_speech_during_agent(barge(0.7, 0.2, 0.1), 0.9, cfg=CFG)
    assert d.action is BargeAction.STOP_AND_LISTEN
    assert d.escalate


def test_barge_probs_must_be_exactly_the_three_labels() -> None:
    with pytest.raises(ValueError):
        on_user_speech_during_agent({BACKCHANNEL: 0.5, INTERRUPTION: 0.5}, 0.0, cfg=CFG)


@pytest.mark.parametrize("p_int", [0.05, 0.3, 0.55, 0.61, 0.9])
def test_raising_theta_interrupt_never_adds_a_stop(p_int: float) -> None:
    # The theta_interrupt sweep is only a tradeoff curve if stopping is monotone in the threshold.
    rest = (1 - p_int) / 2
    stops = [
        on_user_speech_during_agent(
            barge(rest, p_int, rest), 0.0, cfg=CFG.model_copy(update={"theta_interrupt": t / 20})
        ).action
        is BargeAction.STOP_AND_LISTEN
        for t in range(21)
    ]
    # once it stops stopping as theta rises, it never starts again
    assert stops == sorted(stops, reverse=True)
