"""Do lower-precision forwards give the same answers? (DECISIONS.md D-039)

    uv run floorcall eval precision-parity --checkpoint checkpoints/main-r2

The fine-tuned checkpoint scores calib and the dev split (carved from train as in training), never
test, once per precision: "fp32" (no autocast, the reference), "bf16" and "fp16" (autocast over the
fp32 weights, Laya's two serving modes). Probabilities are the served ones, with the checkpoint's
calibrated temperatures. Against fp32, per decision and split: argmax agreement, the largest and the
mean absolute probability difference, and for decisions with a calib threshold, agreement of the
thresholded decision too. A precision passes when every argmax agreement reaches
`EvalSettings.precision_min_agreement`; only a passing precision may get latency rows. Each pass
runs under the latency thermal rules: it starts at or below the start temperature, and nvidia-smi
watches it, stopping it at the abort temperature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.evaluate.calibration import load_calibration
from floorcall.evaluate.checkpoint import DECISIONS, logits_for
from floorcall.evaluate.dataset import EvalSet, _make, load_processed
from floorcall.evaluate.latency import GpuProbe, summarise_telemetry, wait_until_cool
from floorcall.evaluate.scoring import softmax
from floorcall.provenance import git_head
from floorcall.train.data import carve_dev

RESULTS = REPO_ROOT / "results" / "precision"
REFERENCE = "fp32"


def parity_stats(
    ref: npt.NDArray[np.float64],
    other: npt.NDArray[np.float64],
    theta: float | None = None,
    positive: int = 1,
) -> dict[str, Any]:
    """How far `other`'s probabilities are from the reference's, row by row."""
    if ref.shape != other.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {other.shape}")
    diff = np.abs(ref - other)
    out: dict[str, Any] = {
        "n": int(ref.shape[0]),
        "argmax_agreement": float((ref.argmax(1) == other.argmax(1)).mean()),
        "argmax_disagreements": int((ref.argmax(1) != other.argmax(1)).sum()),
        "max_abs_prob_diff": float(diff.max()),
        "mean_abs_prob_diff": float(diff.mean()),
    }
    if theta is not None:
        a, b = ref[:, positive] >= theta, other[:, positive] >= theta
        out["threshold"] = theta
        out["threshold_agreement"] = float((a == b).mean())
    return out


def verdict(stats: dict[str, dict[str, Any]], min_agreement: float) -> dict[str, Any]:
    """Whether one precision passes: every set's argmax agreement at or above the bar."""
    worst_key = min(stats, key=lambda k: stats[k]["argmax_agreement"])
    worst = stats[worst_key]["argmax_agreement"]
    return {
        "passes": worst >= min_agreement,
        "min_agreement_required": min_agreement,
        "worst_argmax_agreement": worst,
        "worst_set": worst_key,
        "max_abs_prob_diff": max(s["max_abs_prob_diff"] for s in stats.values()),
    }


def eval_sets(settings: Settings) -> dict[str, EvalSet]:
    """calib and dev for every decision: `<decision>/calib`, `<decision>/dev`."""
    out = {}
    for d in DECISIONS:
        out[f"{d}/calib"] = load_processed(settings, d, "calib")
        rows = load_processed(settings, d, "train").rows
        _, dev = carve_dev(
            rows, fraction=settings.train.dev_fraction, seed=settings.train.seed, task=d
        )
        out[f"{d}/dev"] = _make(d, "dev", dev)
    return out


def run_parity(
    settings: Settings, checkpoint: Path, precisions: tuple[str, ...] = ("fp32", "bf16", "fp16")
) -> dict[str, Any]:
    from floorcall.model.laya_adapter import LayaDecider

    ev = settings.eval
    code = git_head()
    cal = load_calibration(checkpoint)
    if cal is None:
        raise FileNotFoundError(f"{checkpoint} is not calibrated")
    decider = LayaDecider(
        LayaSettings(checkpoint=str(checkpoint), revision=None, device="cuda", cuda_graphs=False)
    )
    sets = eval_sets(settings)
    probs: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    telemetry: dict[str, Any] = {}
    for p in precisions:
        gate = wait_until_cool(
            ev.latency_start_max_temp_c,
            poll_s=ev.latency_cooldown_poll_s,
            max_wait_s=ev.latency_cooldown_max_s,
        )
        decider.set_precision(p)
        probs[p] = {}
        with GpuProbe(ev.latency_telemetry_ms, ev.latency_abort_temp_c) as probe:
            probe.phase("timed")
            for key, data in sets.items():
                probe.check()  # stops the pass at the abort temperature
                z = logits_for(decider, data, settings.state)
                probs[p][key] = softmax(z, cal.temperatures[data.decision])
            probe.phase("idle")
        telemetry[p] = {"thermal_gate": gate, **summarise_telemetry(probe.samples)["timed"]}
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "code": code,
        "reference": REFERENCE,
        "sets": {k: len(v.rows) for k, v in sets.items()},
        "telemetry": telemetry,
        "precisions": {},
    }
    for p in precisions:
        if p == REFERENCE:
            continue
        stats = {}
        for key, data in sets.items():
            choice = cal.thresholds.get(data.decision)
            stats[key] = parity_stats(
                probs[REFERENCE][key],
                probs[p][key],
                theta=None if choice is None else choice["theta"],
                positive=data.labels.index("true") if "true" in data.labels else 1,
            )
        report["precisions"][p] = {
            "verdict": verdict(stats, ev.precision_min_agreement),
            "sets": stats,
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "parity.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def passing(precision: str, path: Path = RESULTS / "parity.json") -> bool:
    """Whether a precision may get latency rows: the reference always, others only if they passed."""
    if precision == REFERENCE:
        return True
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    entry = report["precisions"].get(precision)
    return bool(entry and entry["verdict"]["passes"])
