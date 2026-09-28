"""Temperature scaling, one temperature per decision, fitted on the calibration split.

Why per decision and not Laya's own scheme: Laya keeps one temperature per question *type* and
option count (`temperature_by_options`, e.g. `noul:2`). D1 turn_complete and D4 escalate are both
two-option `noul` questions, so under Laya's scheme they would share one temperature, fitted on a
mixture of two unrelated tasks. floorcall therefore applies its own temperatures to the raw logits,
one per decision (DECISIONS.md D-021).

The fit: for T > 0, p = softmax(z / T). Negative log-likelihood on the calib split, as a function of
beta = 1/T, is convex (log-sum-exp minus a linear term), so it has a single minimum. A golden-section
search over log(beta) finds it deterministically. It is bounded to [t_min, t_max]: on data the model
already separates perfectly, NLL keeps falling as T -> 0, so the unbounded optimum is infinite
sharpening. A fit that ends on a bound is flagged `at_bound` and should be reported, not trusted.

Temperatures are fitted on calib only. Never on test (CLAUDE.md §15), and never on the training
rows, where a fine-tuned model is near-certain and near-right, so the fit measures memorisation
rather than calibration (the Laya notebook makes the same point).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

CALIBRATION_FILE = "floorcall_calibration.json"
_GOLDEN = (math.sqrt(5) - 1) / 2


def nll(logits: npt.NDArray[np.floating], y: npt.NDArray[np.integer], temperature: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
    return float((lse - z[np.arange(len(y)), np.asarray(y)]).mean())


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float  # at T = 1
    nll_after: float
    n: int
    at_bound: bool


def fit_temperature(
    logits: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    *,
    t_min: float,
    t_max: float,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> TemperatureFit:
    if len(y) == 0:
        raise ValueError("cannot fit a temperature on no rows")
    lo, hi = math.log(1 / t_max), math.log(1 / t_min)  # u = log(beta)

    def f(u: float) -> float:
        return nll(logits, y, 1 / math.exp(u))

    a, b = lo, hi
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    # When the minimum lies beyond an end of the interval, the search converges onto that end.
    u = (a + b) / 2
    temperature = 1 / math.exp(u)
    at_bound = abs(u - lo) < 1e-3 or abs(u - hi) < 1e-3
    return TemperatureFit(
        temperature=temperature,
        nll_before=nll(logits, y, 1.0),
        nll_after=nll(logits, y, temperature),
        n=len(y),
        at_bound=at_bound,
    )


@dataclass(frozen=True)
class Calibration:
    """Per-decision temperatures for one checkpoint, with where they came from."""

    temperatures: dict[str, float]
    fits: dict[str, dict[str, Any]]
    calib_sha256: dict[str, str]
    checkpoint: str
    code: str


def save_calibration(directory: Path, cal: Calibration) -> Path:
    path = directory / CALIBRATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(asdict(cal), f, indent=2)
        f.write("\n")
    return path


def load_calibration(directory: Path) -> Calibration | None:
    path = directory / CALIBRATION_FILE
    if not path.exists():
        return None
    return Calibration(**json.loads(path.read_text(encoding="utf-8")))
