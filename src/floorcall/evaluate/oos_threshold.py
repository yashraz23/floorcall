"""theta_oos, chosen on D3 calib (DECISIONS.md D-045): `uv run floorcall eval oos-threshold`.

The routing policy says `out_of_scope` when p(out_of_scope) >= theta_oos, and otherwise the most
likely in-scope intent. theta_oos is the grid point that maximises the 16-label macro-F1 of that
output on the calib split, under the checkpoint's calib temperature for route. This is the same
rule as theta_escalate, and it is never chosen on test.

Writes results/thresholds/route_oos.json, which replay and the Space read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from floorcall.config import REPO_ROOT, Settings
from floorcall.data.build import processed_file
from floorcall.data.freeze import sha256_file
from floorcall.evaluate.checkpoint import _calibration, _load, logits_for, trained_ablation
from floorcall.evaluate.dataset import load_processed
from floorcall.evaluate.metrics import macro_f1
from floorcall.evaluate.scoring import softmax
from floorcall.evaluate.thresholds import choose_oos_threshold, route_with_oos
from floorcall.provenance import git_head
from floorcall.questions import OUT_OF_SCOPE

RESULTS = REPO_ROOT / "results" / "thresholds"


def run_oos_threshold(
    settings: Settings, checkpoint: Path, *, device: str = "cpu", out_dir: Path = RESULTS
) -> dict[str, Any]:
    if trained_ablation(checkpoint) != "full":
        raise ValueError(f"{checkpoint} is an ablation run: theta_oos is for the served checkpoint")
    cal = _calibration(checkpoint)
    calib_path = settings.paths.data_processed / processed_file("route", "calib")
    digest = sha256_file(calib_path)
    if digest != cal.calib_sha256["route"]:
        raise ValueError("the route calib file is not the one this checkpoint was calibrated on")
    calib = load_processed(settings, "route", "calib")
    decider = _load(checkpoint, device)
    temperature = cal.temperatures["route"]
    probs = softmax(logits_for(decider, calib, settings.state), temperature)
    oos = calib.labels.index(OUT_OF_SCOPE)
    choice = choose_oos_threshold(probs, calib.y, oos)
    every = list(range(len(calib.labels)))
    at_default = macro_f1(calib.y, route_with_oos(probs, oos, 0.5), labels=every)
    out = {
        "decision": "route",
        "threshold": "theta_oos",
        "split": "calib",
        "calib_file": calib_path.name,
        "calib_sha256": digest,
        "checkpoint": checkpoint.as_posix(),
        "trained_at": cal.code,
        "code": git_head(),
        "device": device,
        "temperature": temperature,
        "labels": list(calib.labels),
        "choice": choice.to_json(),
        "calib_macro_f1_at_0.5": at_default,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "route_oos.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return out
