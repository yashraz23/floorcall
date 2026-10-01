"""A binary decision's threshold, chosen on calib by macro-F1 (DECISIONS.md D-035).

p(positive) >= theta predicts the positive class. theta is the grid point (0.000 to 1.000 in steps
of 0.005) with the highest macro-F1 on the calib split; when several tie, the median of them is
taken (the lower middle one when their count is even), so the choice sits inside a plateau rather
than on its edge. It is never chosen on test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall.evaluate.metrics import macro_f1

GRID = np.round(np.linspace(0.0, 1.0, 201), 3)
RULE = "max macro-F1 on calib over thresholds 0.000-1.000 step 0.005; median of ties"


@dataclass(frozen=True)
class ThresholdChoice:
    theta: float
    macro_f1: float  # on the split it was chosen on
    n: int
    tied: int  # grid points sharing the maximum
    rule: str = RULE

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def choose_threshold(
    p_positive: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    grid: npt.NDArray[np.floating] = GRID,
) -> ThresholdChoice:
    """The macro-F1-maximising threshold for labels y in {0, 1} (1 = positive)."""
    p = np.asarray(p_positive, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if p.shape != labels.shape or p.size == 0:
        raise ValueError(f"need matching, non-empty arrays, got {p.shape} and {labels.shape}")
    scores = np.array([macro_f1(labels, (p >= t).astype(np.int64), labels=[0, 1]) for t in grid])
    best = float(scores.max())
    tied = np.asarray(grid)[scores >= best - 1e-12]
    return ThresholdChoice(
        theta=float(tied[(len(tied) - 1) // 2]), macro_f1=best, n=int(p.size), tied=len(tied)
    )


RULE_OOS = (
    "max 16-label macro-F1 of the routing policy's output (out_of_scope if p(out_of_scope) >= "
    "theta, else the in-scope argmax) on calib over thresholds 0.000-1.000 step 0.005; "
    "median of ties"
)


def route_with_oos(
    probs: npt.NDArray[np.floating], oos: int, theta: float
) -> npt.NDArray[np.int64]:
    """policy.on_user_pause's route, vectorised: `oos` when p(oos) >= theta, else the in-scope
    argmax (the first one on a tie, as max() over the policy's dict picks)."""
    p = np.asarray(probs, dtype=np.float64)
    in_scope = p.copy()
    in_scope[:, oos] = -np.inf
    return np.where(p[:, oos] >= theta, oos, in_scope.argmax(axis=1)).astype(np.int64)


def choose_oos_threshold(
    probs: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    oos: int,
    grid: npt.NDArray[np.floating] = GRID,
) -> ThresholdChoice:
    """theta_oos by the rule theta_escalate uses (D-045), scored on what the policy would route."""
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if p.ndim != 2 or p.shape[0] != labels.shape[0] or p.size == 0:
        raise ValueError(
            f"need (n, k) probabilities and n labels, got {p.shape} and {labels.shape}"
        )
    every = list(range(p.shape[1]))
    scores = np.array([macro_f1(labels, route_with_oos(p, oos, t), labels=every) for t in grid])
    best = float(scores.max())
    tied = np.asarray(grid)[scores >= best - 1e-12]
    return ThresholdChoice(
        theta=float(tied[(len(tied) - 1) // 2]),
        macro_f1=best,
        n=int(labels.size),
        tied=len(tied),
        rule=RULE_OOS,
    )
