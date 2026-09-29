"""Train, save and calibrate one multi-task floorcall checkpoint.

    uv run floorcall train run --out checkpoints/<name>
    uv run floorcall train calibrate --checkpoint checkpoints/<name>

Order of operations, and why:
  1. Refuse unless all four test sets are frozen and intact. A test set built after training could
     be shaped, even unintentionally, by what the model does (CLAUDE.md §10.2; DECISIONS.md D-009).
  2. Build items from the processed *train* files only, hashing each file used.
  3. Train (floorcall.train.loop), saving a rolling checkpoint after every epoch.
  4. Save the final checkpoint in Laya's layout, then reload it from disk and fit one temperature
     per decision on the *calib* split's logits. Fitting on the reloaded fp16 checkpoint rather than
     the fp32 weights in memory means the temperatures match exactly what is served.
  5. Write run.json: settings, data hashes, test manifest, base checkpoint, code commit, history.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import LayaSettings, Settings
from floorcall.data.build import processed_file, test_file
from floorcall.data.freeze import sha256_file, verify
from floorcall.evaluate.calibration import Calibration, fit_temperature, save_calibration
from floorcall.evaluate.dataset import load_processed
from floorcall.evaluate.scoring import softmax
from floorcall.evaluate.thresholds import choose_threshold
from floorcall.provenance import git_head
from floorcall.train.data import build_items, event_budget

DECISIONS = ("turn_complete", "barge_in", "route", "escalate")


class TrainingNotAllowedError(RuntimeError):
    """A precondition for training does not hold."""


def assert_test_sets_frozen(settings: Settings) -> dict[str, str]:
    """All four test sets frozen and matching MANIFEST.sha256, or refuse."""
    manifest = verify(settings.paths.test_frozen)
    missing = [d for d in DECISIONS if test_file(d) not in manifest]
    if missing:
        raise TrainingNotAllowedError(
            f"test sets not frozen yet: {missing}. Every test set is frozen before the first "
            "training run (CLAUDE.md §10.2, DECISIONS.md D-009); for D4 that means Yash's labels "
            "(`floorcall label escalate`, then `floorcall data freeze-escalate`)."
        )
    return manifest


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def calibrate(settings: Settings, checkpoint: Path, *, device: str = "cuda") -> Calibration:
    """Fit one temperature per decision on calib logits of the checkpoint as saved."""
    from floorcall.model.laya_adapter import LayaDecider

    decider = LayaDecider(
        LayaSettings(checkpoint=str(checkpoint), revision=None, device=device, cuda_graphs=False)
    )
    temps: dict[str, float] = {}
    thresholds: dict[str, dict[str, Any]] = {}
    fits: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for decision in DECISIONS:
        path = settings.paths.data_processed / processed_file(decision, "calib")
        if not path.exists():
            raise FileNotFoundError(f"no calib split for {decision}: {path}")
        calib = load_processed(settings, decision, "calib")
        budget = event_budget(decider, calib.event, settings.state)
        states = calib.packed_states(
            state=settings.state, budget=budget, count_tokens=decider.count_tokens
        )
        rows = decider.logits_batch(states, questions.questions_by_id(decision))
        logits = np.stack([r[decision].logits for r in rows])
        fit = fit_temperature(
            logits,
            calib.y,
            t_min=settings.eval.temperature_min,
            t_max=settings.eval.temperature_max,
        )
        temps[decision] = fit.temperature
        fits[decision] = {
            "n": fit.n,
            "nll_before": fit.nll_before,
            "nll_after": fit.nll_after,
            "at_bound": fit.at_bound,
        }
        hashes[decision] = sha256_file(path)
        if decision in settings.eval.threshold_decisions:
            p = softmax(logits, fit.temperature)[:, calib.labels.index("true")]
            thresholds[decision] = choose_threshold(p, calib.y).to_json()
    cal = Calibration(
        temperatures=temps,
        fits=fits,
        calib_sha256=hashes,
        checkpoint=str(checkpoint),
        code=git_head(),
        thresholds=thresholds,
    )
    save_calibration(checkpoint, cal)
    return cal


def _train(
    settings: Settings, out_dir: Path, tasks: list[str], code: str, manifest: dict[str, str]
) -> dict[str, Any]:
    """Build items, train, save. The model lives only inside this call, so its GPU memory is freed
    before calibration reloads the checkpoint from disk."""
    import torch

    from floorcall.evaluate.dataset import _make
    from floorcall.model.laya_adapter import LayaDecider
    from floorcall.train.data import carve_dev
    from floorcall.train.loop import best_epoch, train_model

    cfg = settings.train
    decider = LayaDecider(LayaSettings(**{**settings.laya.model_dump(), "device": "cuda"}))
    if decider.device != "cuda":
        raise TrainingNotAllowedError(f"model is on {decider.device}, not cuda (D-010)")

    log_path = out_dir / "train_log.jsonl"

    def log(rec: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps(rec, default=str), flush=True)

    items_by_task = {}
    dev_items_by_task = {}
    dev_rows: dict[str, int] = {}
    data_sha256 = {}
    for task in tasks:
        path = settings.paths.data_processed / processed_file(task, "train")
        if not path.exists():
            raise TrainingNotAllowedError(
                f"no training rows for {task} ({path}). D4's are the prompt-v3 labels of the "
                "train pool (D-035): `floorcall data escalate-llm-label --split train`."
            )
        data_sha256[task] = sha256_file(path)
        data = load_processed(settings, task, "train")
        train_rows, dev = carve_dev(data.rows, fraction=cfg.dev_fraction, seed=cfg.seed, task=task)
        dev_rows[task] = len(dev)
        log({"building_items": task, "rows": len(train_rows), "dev_rows": len(dev)})
        items_by_task[task] = build_items(
            decider, _make(task, "train", train_rows), settings.state,
            label_smoothing=cfg.label_smoothing,
        )  # fmt: skip
        # dev targets stay one-hot: its cross-entropy is what chooses the checkpoint
        dev_items_by_task[task] = build_items(decider, _make(task, "dev", dev), settings.state)

    meta = {
        "code": code,
        "base_checkpoint": decider.checkpoint,
        "base_revision": decider.revision,
        "tasks": tasks,
        "train_sha256": data_sha256,
        "test_manifest": manifest,
        "dev": {
            "fraction": cfg.dev_fraction,
            "rows": dev_rows,
            "rule": (
                "conversations held out of each task's train rows, seeded; the kept checkpoint "
                "is the epoch with the lowest dev cross-entropy at T = 1, averaged over tasks"
            ),
        },
    }

    def save_epoch(ep: dict[str, Any]) -> None:
        n = ep["epoch"]
        decider.save_checkpoint(
            out_dir / f"checkpoint_epoch{n}",
            {**meta, "epoch": n, "dev_ce_by_task": ep.get("dev_ce_by_task")},
        )

    decider.enable_gradient_checkpointing()
    t0 = time.time()
    history = train_model(
        decider.model,
        items_by_task,
        cfg,
        collate=decider.collate,
        reward_fn=partial(decider.proper_reward, w_sph=cfg.w_sph, w_rps=cfg.w_rps),
        device=torch.device("cuda"),
        rows_per_epoch={t: cfg.rows_per_epoch[t] for t in tasks},
        dev_items_by_task=dev_items_by_task,
        log=log,
        on_epoch_end=save_epoch,
    )
    # The run's checkpoint is the best epoch's, copied to the run root, where calibration and
    # evaluation read it; every epoch's stays beside it.
    best = best_epoch(history.epochs)
    shutil.copytree(out_dir / f"checkpoint_epoch{best}", out_dir, dirs_exist_ok=True)
    log({"best_epoch": best, "dev_ce_macro": [e["dev_ce_macro"] for e in history.epochs]})
    return {
        **meta,
        "best_epoch": best,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seconds": time.time() - t0,
        "settings": settings.model_dump(mode="json"),
        "history": {"epochs": history.epochs, "updates": history.updates},
    }


def run_training(settings: Settings, out_dir: Path, *, tasks: Sequence[str] | None = None) -> Path:
    import torch

    code = git_head()
    manifest = assert_test_sets_frozen(settings)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; every run gets its own directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _train(settings, out_dir, list(tasks or settings.train.rows_per_epoch), code, manifest)
    _write_json(out_dir / "run.json", run)
    torch.cuda.empty_cache()
    cal = calibrate(settings, out_dir)
    with (out_dir / "train_log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"calibration": cal.temperatures, "fits": cal.fits}) + "\n")
    return out_dir
