"""Inference-precision parity (D-039), on synthetic probabilities: no GPU."""

import json
from pathlib import Path

import numpy as np
import pytest

from floorcall.config import get_settings
from floorcall.evaluate import latency
from floorcall.evaluate.precision import parity_stats, passing, verdict

REF = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45], [0.3, 0.7]])


def test_identical_probabilities_agree_fully() -> None:
    st = parity_stats(REF, REF.copy(), theta=0.5)
    assert st["argmax_agreement"] == 1.0 and st["argmax_disagreements"] == 0
    assert st["max_abs_prob_diff"] == 0.0 and st["threshold_agreement"] == 1.0


def test_a_flip_near_the_boundary_is_counted() -> None:
    other = REF.copy()
    other[2] = [0.45, 0.55]  # row 2 crosses from "false" to "true"
    st = parity_stats(REF, other, theta=0.5)
    assert st["argmax_agreement"] == 0.75 and st["argmax_disagreements"] == 1
    assert st["max_abs_prob_diff"] == pytest.approx(0.10)
    assert st["threshold_agreement"] == 0.75


def test_the_verdict_takes_the_worst_set() -> None:
    stats = {
        "a/calib": {"argmax_agreement": 0.999, "max_abs_prob_diff": 0.01},
        "b/dev": {"argmax_agreement": 0.994, "max_abs_prob_diff": 0.02},
    }
    v = verdict(stats, 0.995)
    assert not v["passes"] and v["worst_set"] == "b/dev" and v["max_abs_prob_diff"] == 0.02
    assert verdict({"a/calib": stats["a/calib"]}, 0.995)["passes"]


def test_only_a_passing_precision_gets_latency_rows(tmp_path: Path) -> None:
    path = tmp_path / "parity.json"
    assert passing("fp32", path) and not passing("bf16", path)  # fp32 is the reference
    path.write_text(
        json.dumps(
            {
                "precisions": {
                    "bf16": {"verdict": {"passes": True}},
                    "fp16": {"verdict": {"passes": False}},
                }
            }
        )
    )
    assert passing("bf16", path) and not passing("fp16", path)


def test_the_harness_refuses_an_unchecked_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    import floorcall.evaluate.precision as precision

    monkeypatch.setattr(precision, "RESULTS", Path("does-not-exist"))
    monkeypatch.setattr(
        precision, "passing", lambda p, path=None: p == "fp32"
    )  # nothing but the reference has passed
    with pytest.raises(ValueError, match="parity"):
        latency.run(get_settings(), "gpu", precisions=("fp32", "fp16"))
