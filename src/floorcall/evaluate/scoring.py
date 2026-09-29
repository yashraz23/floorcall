"""Turn a matrix of probabilities into a Table A row."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall.evaluate.metrics import (
    accuracy,
    bootstrap_interval,
    brier,
    ece,
    macro_f1,
    wilson_interval,
)


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
    bootstrap: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Every Table A metric. `bootstrap` = (samples, seed) adds a percentile-bootstrap 95%
    interval for each metric, resampling rows (the hard subset's accuracy resamples hard rows)."""
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
    if bootstrap is not None:
        samples, seed = bootstrap
        cls = list(range(k))
        stats = {
            "accuracy": lambda i: accuracy(y[i], pred[i]),
            "macro_f1": lambda i: macro_f1(y[i], pred[i], labels=cls),
            "ece": lambda i: ece(conf[i], correct[i], n_bins=n_bins),
            "brier": lambda i: brier(probs[i], y[i]),
        }
        ci = {
            name: list(bootstrap_interval(f, len(y), samples=samples, seed=seed))
            for name, f in stats.items()
        }
        if hard.any():
            yh, ph = y[hard], pred[hard]
            ci["hard_accuracy"] = list(
                bootstrap_interval(
                    lambda i: accuracy(yh[i], ph[i]), len(yh), samples=samples, seed=seed
                )
            )
        out["ci95"] = ci
        out["bootstrap"] = {
            "samples": samples,
            "seed": seed,
            "method": "percentile interval; rows resampled with replacement",
        }
    return out
