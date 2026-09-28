"""Tradeoff curves and operating-point choice. Written before floorcall/evaluate/curves.py."""

import numpy as np
import pytest

from floorcall.evaluate.curves import choose_threshold, interrupt_curve, yield_curve

LABELS = ("backchannel", "interruption", "noise")
THETAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def test_interrupt_curve_hand_computed() -> None:
    # 2 interruptions (p 0.9, 0.4), 2 backchannels (p 0.6, 0.1), 1 noise (p 0.3)
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.3])
    y = np.array([1, 1, 0, 0, 2])
    c = interrupt_curve(p, y, LABELS, THETAS)
    # theta 0.5: stops on 0.9 and 0.6 -> false stops 1/3 non-interruptions, misses 1/2 interruptions
    i = 2
    assert c["false_stop"][i] == pytest.approx(1 / 3)
    assert c["false_stop_backchannel"][i] == pytest.approx(1 / 2)
    assert c["missed"][i] == pytest.approx(1 / 2)
    # theta 0: stops on everything; theta 1: on nothing (p < 1 everywhere)
    assert c["false_stop"][0] == 1.0 and c["missed"][0] == 0.0
    assert c["false_stop"][-1] == 0.0 and c["missed"][-1] == 1.0


def test_yield_curve_hand_computed() -> None:
    # complete turns (y=1) at p 0.8, 0.3; incomplete (y=0) at p 0.6, 0.2
    p = np.array([0.8, 0.3, 0.6, 0.2])
    y = np.array([1, 1, 0, 0])
    c = yield_curve(p, y, THETAS, vad_pause_ms=300, max_wait_ms=2000)
    i = 2  # theta 0.5: responds to 0.8 and 0.6
    assert c["premature"][i] == pytest.approx(1 / 2)  # one of two incomplete turns answered
    assert c["fallback"][i] == pytest.approx(1 / 2)  # one of two complete turns waits
    assert c["added_delay_ms"][i] == pytest.approx(0.5 * (2000 - 300))


def test_curves_are_monotone_in_theta() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = rng.integers(0, 3, 500)
    thetas = np.linspace(0, 1, 101)
    c = interrupt_curve(p, y, LABELS, thetas)
    assert np.all(np.diff(c["false_stop"]) <= 0)
    assert np.all(np.diff(c["missed"]) >= 0)


def test_choose_threshold_takes_the_smallest_theta_meeting_the_limit() -> None:
    # false stops fall as theta rises; the smallest theta at or under 5% keeps the most stops
    thetas = np.array([0.1, 0.2, 0.3, 0.4])
    rate = np.array([0.30, 0.08, 0.05, 0.01])
    choice = choose_threshold(thetas, rate, limit=0.05)
    assert choice.theta == 0.3
    assert choice.feasible


def test_choose_threshold_reports_an_unreachable_limit() -> None:
    choice = choose_threshold(np.array([0.1, 0.9]), np.array([0.5, 0.2]), limit=0.05)
    assert not choice.feasible
    assert choice.theta == 0.9
