"""Operating points and the data behind the curves: `uv run floorcall eval curves [--checkpoint]`.

For one model (the stock checkpoint, or a trained run with its calibration):
- theta_interrupt and theta_yield are chosen on the **calib** split, against the targets in
  PolicySettings, and then applied to the **test** split, which reports what they actually do;
- the full test curves are stored for the figures;
- reliability bins are stored for every decision, before and after temperature.

Writes results/curves/<model>.json. floorcall.evaluate.figures renders it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.data.build import processed_file, test_file
from floorcall.evaluate.calibration import load_calibration
from floorcall.evaluate.checkpoint import logits_for
from floorcall.evaluate.curves import choose_threshold, interrupt_curve, yield_curve
from floorcall.evaluate.dataset import EvalSet, load_processed, load_test
from floorcall.evaluate.metrics import ece, reliability_bins
from floorcall.evaluate.scoring import softmax
from floorcall.provenance import git_head

RESULTS = REPO_ROOT / "results" / "curves"
THETAS = np.round(np.linspace(0.0, 1.0, 201), 3)
DECISIONS = ("turn_complete", "barge_in", "route", "escalate")


def _available(settings: Settings, decision: str) -> bool:
    return (settings.paths.test_frozen / test_file(decision)).exists() and (
        settings.paths.data_processed / processed_file(decision, "calib")
    ).exists()


def _curve_json(c: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: [round(float(x), 5) for x in v] for k, v in c.items()}


def _reliability(probs: np.ndarray, y: np.ndarray, n_bins: int) -> dict[str, Any]:
    conf, correct = probs.max(1), (probs.argmax(1) == y).astype(int)
    return {
        "ece": ece(conf, correct, n_bins=n_bins),
        "bins": [
            {
                "lo": b.lo,
                "hi": b.hi,
                "count": b.count,
                "confidence": b.mean_confidence,
                "accuracy": b.accuracy,
            }
            for b in reliability_bins(conf, correct, n_bins=n_bins)
            if b.count
        ],
    }


def run_curves(
    settings: Settings, *, checkpoint: Path | None = None, device: str = "cuda"
) -> dict[str, Any]:
    from floorcall.model.laya_adapter import LayaDecider

    code = git_head()
    if checkpoint is None:
        model = "stock_laya"
        decider = LayaDecider(
            LayaSettings(**{**settings.laya.model_dump(), "device": device, "cuda_graphs": False})
        )
        cal = None
    else:
        model = "finetuned_temp"
        decider = LayaDecider(
            LayaSettings(
                checkpoint=str(checkpoint), revision=None, device=device, cuda_graphs=False
            )
        )
        cal = load_calibration(checkpoint)
        if cal is None:
            raise FileNotFoundError(f"{checkpoint} is not calibrated")

    def temperature(data: EvalSet) -> float:
        if cal is not None:
            return cal.temperatures[data.decision]
        qtype: Literal["noul", "choice"] = "noul" if data.labels == ("false", "true") else "choice"
        return decider.effective_temperature(qtype, len(data.labels))

    pol = settings.policy
    out: dict[str, Any] = {
        "model": model,
        "precision": decider.precision,
        "checkpoint": str(checkpoint) if checkpoint else decider.checkpoint,
        "code": code,
    }
    splits: dict[str, dict[str, tuple[np.ndarray, EvalSet]]] = {}
    for decision in DECISIONS:
        if not _available(settings, decision):
            continue
        calib, test = load_processed(settings, decision, "calib"), load_test(settings, decision)
        splits[decision] = {
            "calib": (logits_for(decider, calib, settings.state), calib),
            "test": (logits_for(decider, test, settings.state), test),
        }

    if "barge_in" in splits:
        (zc, c), (zt, t) = splits["barge_in"]["calib"], splits["barge_in"]["test"]
        i = c.labels.index("interruption")
        pc, pt = softmax(zc, temperature(c))[:, i], softmax(zt, temperature(t))[:, i]
        choice = choose_threshold(
            THETAS,
            interrupt_curve(pc, c.y, c.labels, THETAS)["false_stop"],
            limit=pol.target_false_stop_rate,
        )
        test_curve = interrupt_curve(pt, t.y, t.labels, THETAS)
        at = int(np.flatnonzero(choice.theta == THETAS)[0])
        out["interrupt"] = {
            "target_false_stop_rate": pol.target_false_stop_rate,
            "theta": choice.theta,
            "feasible_on_calib": choice.feasible,
            "calib_false_stop": choice.rate,
            "test_at_theta": {k: float(v[at]) for k, v in test_curve.items()},
            "test_curve": _curve_json(test_curve),
        }

    if "turn_complete" in splits:
        (zc, c), (zt, t) = splits["turn_complete"]["calib"], splits["turn_complete"]["test"]
        pc, pt = softmax(zc, temperature(c))[:, 1], softmax(zt, temperature(t))[:, 1]
        kw = {"vad_pause_ms": pol.vad_pause_ms, "max_wait_ms": pol.max_wait_ms}
        choice = choose_threshold(
            THETAS, yield_curve(pc, c.y, THETAS, **kw)["premature"], limit=pol.target_premature_rate
        )
        test_curve = yield_curve(pt, t.y, THETAS, **kw)
        at = int(np.flatnonzero(choice.theta == THETAS)[0])
        out["yield"] = {
            "target_premature_rate": pol.target_premature_rate,
            **kw,
            "theta": choice.theta,
            "feasible_on_calib": choice.feasible,
            "calib_premature": choice.rate,
            "test_at_theta": {k: float(v[at]) for k, v in test_curve.items()},
            "test_curve": _curve_json(test_curve),
        }

    out["reliability"] = {}
    for decision, s in splits.items():
        z, t = s["test"]
        out["reliability"][decision] = {
            "before": _reliability(softmax(z, 1.0), t.y, settings.eval.ece_bins),
            "after": _reliability(softmax(z, temperature(t)), t.y, settings.eval.ece_bins),
            "temperature": temperature(t),
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{model}.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return out
