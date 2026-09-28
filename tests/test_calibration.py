"""Temperature scaling, per decision. Written before floorcall/evaluate/calibration.py.

The synthetic data has a known answer: labels are drawn from softmax(z), and the "model" reports
z * 3, i.e. it is overconfident by exactly a temperature of 3. The fit must recover it.
"""

from pathlib import Path

import numpy as np
import pytest

from floorcall.evaluate.calibration import (
    Calibration,
    fit_temperature,
    load_calibration,
    nll,
    save_calibration,
)
from floorcall.evaluate.metrics import ece
from floorcall.evaluate.scoring import softmax


def synthetic(
    n: int, k: int, overconfidence: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, k))
    p = softmax(z)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z * overconfidence, y


def test_recovers_a_known_temperature() -> None:
    logits, y = synthetic(20_000, 3, overconfidence=3.0)
    fit = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert fit.temperature == pytest.approx(3.0, rel=0.08)
    assert not fit.at_bound


def test_a_calibrated_model_keeps_temperature_one() -> None:
    logits, y = synthetic(20_000, 2, overconfidence=1.0, seed=1)
    fit = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert fit.temperature == pytest.approx(1.0, rel=0.08)


def test_underconfidence_gives_a_temperature_below_one() -> None:
    logits, y = synthetic(20_000, 4, overconfidence=0.5, seed=2)
    fit = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert fit.temperature == pytest.approx(0.5, rel=0.1)


def test_fit_lowers_nll_and_ece() -> None:
    logits, y = synthetic(10_000, 3, overconfidence=3.0, seed=3)
    fit = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert fit.nll_after < fit.nll_before
    assert fit.nll_before == pytest.approx(nll(logits, y, 1.0))
    before = softmax(logits)
    after = softmax(logits, fit.temperature)
    ece_before = ece(before.max(1), (before.argmax(1) == y).astype(int), n_bins=15)
    ece_after = ece(after.max(1), (after.argmax(1) == y).astype(int), n_bins=15)
    assert ece_after < ece_before / 3


def test_separable_data_is_flagged_not_trusted() -> None:
    # every row right by a wide margin: NLL keeps falling as T -> 0, so the fit runs to the bound
    logits = np.array([[5.0, -5.0], [-5.0, 5.0]] * 50)
    y = np.array([0, 1] * 50)
    fit = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert fit.at_bound
    assert fit.temperature == pytest.approx(0.25, rel=1e-3)


def test_deterministic() -> None:
    logits, y = synthetic(2_000, 3, overconfidence=2.0, seed=4)
    a = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    b = fit_temperature(logits, y, t_min=0.25, t_max=10.0)
    assert a == b


def test_calibration_file_round_trip(tmp_path: Path) -> None:
    cal = Calibration(
        temperatures={"turn_complete": 1.7, "route": 0.9},
        fits={"turn_complete": {"n": 10}, "route": {"n": 20}},
        calib_sha256={"turn_complete": "ab", "route": "cd"},
        checkpoint="ckpt",
        code="abc1234",
    )
    save_calibration(tmp_path, cal)
    assert load_calibration(tmp_path) == cal
    assert load_calibration(tmp_path / "missing") is None
