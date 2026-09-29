"""Scoring glue: softmax with temperature, and the shape of a Table A row."""

import numpy as np
import pytest

from floorcall.evaluate.scoring import score, softmax


def test_softmax_rows_sum_to_one_and_temperature_softens() -> None:
    z = np.array([[2.0, 0.0, -1.0]])
    p1, p2 = softmax(z, 1.0), softmax(z, 2.0)
    assert p1.sum() == pytest.approx(1.0)
    assert p2.max() < p1.max()
    assert softmax(np.array([[1000.0, 0.0]]))[0, 0] == pytest.approx(1.0)  # no overflow


def test_score_row() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1, 1])
    hard = np.array([False, False, True, True])
    row = score(probs, y, hard, ("false", "true"), n_bins=15)
    assert row["n"] == 4
    assert row["accuracy"] == 0.75
    assert row["hard_n"] == 2
    assert row["hard_accuracy"] == 0.5
    assert row["confusion"] == [[1, 0], [1, 2]]
    assert row["per_class"]["true"] == {"support": 3, "precision": 1.0, "recall": 2 / 3}
    lo, hi = row["accuracy_ci95"]
    assert lo < 0.75 < hi


def test_score_without_a_hard_subset() -> None:
    row = score(np.array([[0.9, 0.1]]), np.array([0]), np.array([False]), ("a", "b"), n_bins=15)
    assert row["hard_accuracy"] is None


def test_score_with_bootstrap_gives_every_metric_an_interval() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p1 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 200), 0.01, 0.99)
    probs = np.stack([1 - p1, p1], axis=1)
    hard = np.arange(200) % 2 == 0
    row = score(probs, y, hard, ("false", "true"), n_bins=15, bootstrap=(500, 7))
    assert set(row["ci95"]) == {"accuracy", "macro_f1", "ece", "brier", "hard_accuracy"}
    for name in ("accuracy", "macro_f1", "brier", "hard_accuracy"):
        lo, hi = row["ci95"][name]
        assert lo <= row[name] <= hi
    assert row["bootstrap"]["samples"] == 500 and row["bootstrap"]["seed"] == 7
    assert score(probs, y, hard, ("false", "true"), n_bins=15, bootstrap=(500, 7)) == row
    assert "ci95" not in score(probs, y, hard, ("false", "true"), n_bins=15)
