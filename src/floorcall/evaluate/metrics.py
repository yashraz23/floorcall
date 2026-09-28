"""Quality and calibration metrics for Table A, implemented here rather than imported.

Definitions, so that every number in the README means exactly one thing:

- **accuracy**: fraction of rows whose argmax equals the gold label.
- **macro-F1**: unweighted mean of per-class F1 over the given labels. A class with support that
  is never predicted scores 0, which is the point: a model that ignores a minority class cannot
  hide behind the majority. A label with neither support nor predictions is left out of the mean,
  as sklearn does with `zero_division=0` (cross-checked in tests).
- **ECE**: expected calibration error with equal-width bins over the confidence of the reported
  answer, max(p). Bin edges follow `laya.common.ece_score`: [0, 1/n], then (lo, hi]. The count is
  15 bins by default (config), so ECE here is comparable to the figures Laya publishes.
- **Brier**: the multi-class form, mean over rows of sum over classes of (p_k - y_k)^2, range
  [0, 2]. For a binary decision it is twice the familiar (p - y)^2. Laya's notebook uses the same
  form, so the numbers compare.
- **Wilson interval**: the score interval for a proportion. It stays inside [0, 1] and behaves at
  small n, which matters for the hand-labelled D4 test set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

IntArray = Sequence[int] | npt.NDArray[np.integer]
FloatArray = npt.NDArray[np.floating]


def _as_int(a: IntArray) -> npt.NDArray[np.int64]:
    return np.asarray(a, dtype=np.int64)


def accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    t, p = _as_int(y_true), _as_int(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("accuracy of an empty set is undefined")
    return float((t == p).mean())


def macro_f1(y_true: IntArray, y_pred: IntArray, *, labels: Sequence[int]) -> float:
    t, p = _as_int(y_true), _as_int(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: {t.shape} vs {p.shape}")
    scores = []
    for c in labels:
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        if tp + fp + fn == 0:
            continue  # no support and never predicted: undefined, left out of the mean
        scores.append(2 * tp / (2 * tp + fp + fn))
    if not scores:
        raise ValueError("macro-F1 is undefined: no label has support or predictions")
    return float(np.mean(scores))


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_confidence: float  # nan when empty
    accuracy: float  # nan when empty


def reliability_bins(confidence: FloatArray, correct: IntArray, *, n_bins: int) -> list[Bin]:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"length mismatch: {conf.shape} vs {corr.shape}")
    if conf.size and (conf.min() < 0 or conf.max() > 1):
        raise ValueError("confidences must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(pairwise(edges)):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        n = int(sel.sum())
        out.append(
            Bin(
                lo=float(lo),
                hi=float(hi),
                count=n,
                mean_confidence=float(conf[sel].mean()) if n else math.nan,
                accuracy=float(corr[sel].mean()) if n else math.nan,
            )
        )
    return out


def ece(confidence: FloatArray, correct: IntArray, *, n_bins: int) -> float:
    bins = reliability_bins(confidence, correct, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("ECE of an empty set is undefined")
    return float(
        sum(b.count / total * abs(b.accuracy - b.mean_confidence) for b in bins if b.count)
    )


def brier(probs: FloatArray, y_true: IntArray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = _as_int(y_true)
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        raise ValueError(f"probs must be (n, k) with n = len(y), got {p.shape} and {y.shape}")
    if (p < -1e-9).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("each row of probs must be a probability distribution")
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0, got {successes}/{n}")
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
