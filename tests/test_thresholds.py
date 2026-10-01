"""The calib-chosen threshold (D-035) and scoring with it."""

import numpy as np
import pytest

from floorcall.evaluate.scoring import score
from floorcall.evaluate.thresholds import GRID, choose_threshold


def test_the_grid_is_0_to_1_in_steps_of_0005() -> None:
    assert GRID[0] == 0.0 and GRID[-1] == 1.0 and len(GRID) == 201


def test_a_separable_set_gets_a_threshold_inside_the_gap() -> None:
    # negatives at 0.2, positives at 0.6: every theta in (0.2, 0.6] is perfect (macro-F1 1)
    p = np.array([0.2, 0.2, 0.2, 0.6, 0.6])
    y = np.array([0, 0, 0, 1, 1])
    c = choose_threshold(p, y)
    assert c.macro_f1 == 1.0
    tied = [t for t in GRID if 0.2 < t <= 0.6]
    assert c.tied == len(tied)
    assert c.theta == tied[(len(tied) - 1) // 2]  # the median of the plateau, not its edge
    assert 0.2 < c.theta <= 0.6


def test_a_prior_shifted_model_is_rescued_by_its_threshold() -> None:
    # a model that ranks well but whose probabilities all sit below 0.5 (trained on a lower prior)
    p = np.array([0.05, 0.1, 0.12, 0.3, 0.35, 0.4])
    y = np.array([0, 0, 0, 1, 1, 1])
    c = choose_threshold(p, y)
    assert c.macro_f1 == 1.0 and 0.12 < c.theta <= 0.3
    argmax = score(np.stack([1 - p, p], 1), y, np.zeros(6, bool), ("false", "true"), n_bins=15)
    at_theta = score(
        np.stack([1 - p, p], 1), y, np.zeros(6, bool), ("false", "true"), n_bins=15,
        threshold=c.theta,
    )  # fmt: skip
    assert argmax["accuracy"] == 0.5 and at_theta["accuracy"] == 1.0
    assert at_theta["threshold"] == c.theta and "threshold" not in argmax
    # ECE and Brier describe the probabilities; the threshold leaves them alone
    assert at_theta["ece"] == argmax["ece"] and at_theta["brier"] == argmax["brier"]


def test_a_threshold_needs_a_true_false_decision() -> None:
    probs = np.array([[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="binary"):
        score(probs, np.array([2]), np.array([False]), ("a", "b", "c"), n_bins=15, threshold=0.5)


def test_choose_threshold_checks_its_input() -> None:
    with pytest.raises(ValueError):
        choose_threshold(np.array([0.1, 0.2]), np.array([0]))


# -- theta_oos (D-045): scored on the routing policy's own output --------------------------------


def test_oos_threshold_is_the_middle_of_the_best_plateau() -> None:
    from floorcall.evaluate.thresholds import RULE_OOS, choose_oos_threshold

    # labels: a, b, oos. Every row is right exactly when 0.3 < theta <= 0.4.
    probs = np.array(
        [[0.6, 0.1, 0.3], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.5, 0.1, 0.4]], dtype=np.float64
    )
    y = np.array([0, 1, 2, 2])
    c = choose_oos_threshold(probs, y, oos=2)
    assert c.macro_f1 == pytest.approx(1.0)
    assert c.tied == 20  # 0.305 ... 0.400
    assert c.theta == pytest.approx(0.35)
    assert c.rule == RULE_OOS and c.n == 4


def test_route_with_oos_matches_the_policy() -> None:
    from floorcall.config import PolicySettings
    from floorcall.evaluate.thresholds import route_with_oos
    from floorcall.policy import _route

    rng = np.random.default_rng(0)
    labels = ["transfer", "balance", "out_of_scope"]
    probs = rng.dirichlet(np.ones(3), size=200)
    probs[:5] = [0.4, 0.4, 0.2]  # an in-scope tie: both take the first
    for theta in (0.0, 0.25, 0.5, 0.9, 1.0):
        got = route_with_oos(probs, 2, theta)
        cfg = PolicySettings(theta_oos=theta)
        want = [labels.index(_route(dict(zip(labels, p, strict=True)), cfg)) for p in probs]
        assert got.tolist() == want
