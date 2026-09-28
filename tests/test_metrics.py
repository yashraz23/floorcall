"""Metrics. Written before floorcall/evaluate/metrics.py (CLAUDE.md §15).

ECE and Brier are checked against examples worked by hand in the comments, not against another
library: the point is that the numbers in Table A mean what the README says they mean.
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from floorcall.evaluate.metrics import (
    accuracy,
    brier,
    ece,
    macro_f1,
    reliability_bins,
    wilson_interval,
)


def test_accuracy() -> None:
    assert accuracy([0, 1, 2, 1], [0, 1, 1, 1]) == 0.75


def test_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_ece_hand_computed() -> None:
    # 15 bins. 0.95 falls in (14/15, 1]; 0.65 falls in (9/15, 10/15].
    #   high bin: 2 items, mean conf 0.95, accuracy 1/2 -> |0.5 - 0.95| = 0.45, weight 2/4
    #   low bin:  2 items, mean conf 0.65, accuracy 2/2 -> |1.0 - 0.65| = 0.35, weight 2/4
    #   ECE = 0.5 * 0.45 + 0.5 * 0.35 = 0.40
    conf = np.array([0.95, 0.95, 0.65, 0.65])
    correct = np.array([1, 0, 1, 1])
    assert ece(conf, correct, n_bins=15) == pytest.approx(0.40)


def test_ece_one_bin_is_the_calibration_gap() -> None:
    # one bin: |mean accuracy - mean confidence| = |0.75 - 0.80| = 0.05
    conf = np.array([0.95, 0.95, 0.65, 0.65])
    correct = np.array([1, 0, 1, 1])
    assert ece(conf, correct, n_bins=1) == pytest.approx(0.05)


def test_ece_is_zero_when_calibrated() -> None:
    conf = np.full(10, 0.7)
    correct = np.array([1] * 7 + [0] * 3)
    assert ece(conf, correct, n_bins=15) == pytest.approx(0.0)


def test_ece_bin_edges_follow_laya() -> None:
    # Laya's ece_score puts 0.0 in the first bin and uses (lo, hi] everywhere else, so a
    # confidence of exactly 1.0 lands in the last bin and nothing is dropped.
    conf = np.array([0.0, 1.0])
    correct = np.array([0, 1])
    bins = reliability_bins(conf, correct, n_bins=4)
    assert sum(b.count for b in bins) == 2
    assert bins[0].count == 1
    assert bins[-1].count == 1


def test_reliability_bins_hand_computed() -> None:
    conf = np.array([0.95, 0.95, 0.65, 0.65])
    correct = np.array([1, 0, 1, 1])
    occupied = [b for b in reliability_bins(conf, correct, n_bins=15) if b.count]
    assert [b.count for b in occupied] == [2, 2]
    assert [b.mean_confidence for b in occupied] == pytest.approx([0.65, 0.95])
    assert [b.accuracy for b in occupied] == [1.0, 0.5]


def test_brier_hand_computed() -> None:
    # row 1, gold 0: (0.7-1)^2 + 0.2^2 + 0.1^2           = 0.09 + 0.04 + 0.01 = 0.14
    # row 2, gold 1: 0.1^2 + (0.1-1)^2 + 0.8^2           = 0.01 + 0.81 + 0.64 = 1.46
    # mean = 0.80
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert brier(probs, np.array([0, 1])) == pytest.approx(0.80)


def test_brier_binary_is_twice_the_scalar_form() -> None:
    # The README reports the multi-class form; for two classes it is 2 * (p_true - y)^2.
    p_true = np.array([0.9, 0.2, 0.6])
    y = np.array([1, 0, 0])
    probs = np.stack([1 - p_true, p_true], axis=1)
    assert brier(probs, y) == pytest.approx(2 * np.mean((p_true - y) ** 2))


def test_brier_perfect_and_worst() -> None:
    assert brier(np.array([[1.0, 0.0]]), np.array([0])) == 0.0
    assert brier(np.array([[0.0, 1.0]]), np.array([0])) == 2.0


def test_brier_rejects_non_distributions() -> None:
    with pytest.raises(ValueError):
        brier(np.array([[0.5, 0.6]]), np.array([0]))


def test_macro_f1_hand_computed() -> None:
    # class 0: tp=1 fp=0 fn=1 -> P=1, R=0.5, F1=2/3
    # class 1: tp=2 fp=1 fn=0 -> P=2/3, R=1, F1=0.8
    # macro = (2/3 + 0.8) / 2 = 0.7333...
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1]) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_counts_a_class_never_predicted_as_zero() -> None:
    # class 2 has support but is never predicted: F1 = 0, and it drags the average down
    assert macro_f1([0, 1, 2], [0, 1, 1], labels=[0, 1, 2]) == pytest.approx((1 + 2 / 3 + 0) / 3)


@pytest.mark.parametrize("seed", range(5))
def test_macro_f1_agrees_with_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 4, 300))
    ours = macro_f1(y, p, labels=[0, 1, 2, 3])
    theirs = f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    assert ours == pytest.approx(theirs)


def test_wilson_interval_hand_computed() -> None:
    # 8/10 at z=1.96: centre (0.8 + 1.96^2/20) / (1 + 1.96^2/10) = 0.71674,
    # half-width 1.96 * sqrt(0.8*0.2/10 + 1.96^2/400) / 1.38416 = 0.22658
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.49016, abs=1e-4)
    assert hi == pytest.approx(0.94332, abs=1e-4)


def test_wilson_interval_stays_in_range_at_the_extremes() -> None:
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson_interval(20, 20)
    assert hi == pytest.approx(1.0) and 0.8 < lo < 1
