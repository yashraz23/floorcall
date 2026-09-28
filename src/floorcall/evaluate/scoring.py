"""Turn a matrix of probabilities into a Table A row."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall.evaluate.metrics import accuracy, brier, ece, macro_f1, wilson_interval


def softmax(logits: npt.NDArray[np.float64], temperature: float = 1.0) -> npt.NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: npt.NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return out


def score(
    probs: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    hard: npt.NDArray[np.bool_],
    labels: tuple[str, ...],
    *,
    n_bins: int,
) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.int64)
    k = len(labels)
    per_class = {}
    for c, name in enumerate(labels):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        per_class[name] = {
            "support": int((y == c).sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    lo, hi = wilson_interval(int(correct.sum()), len(y))
    out: dict[str, Any] = {
        "n": len(y),
        "accuracy": accuracy(y, pred),
        "accuracy_ci95": [lo, hi],
        "macro_f1": macro_f1(y, pred, labels=list(range(k))),
        "ece": ece(conf, correct, n_bins=n_bins),
        "brier": brier(probs, y),
        "mean_confidence": float(conf.mean()),
        "per_class": per_class,
        "confusion": np.bincount(y * k + pred, minlength=k * k).reshape(k, k).tolist(),
        "hard_n": int(hard.sum()),
        "hard_accuracy": accuracy(y[hard], pred[hard]) if hard.any() else None,
    }
    return out
