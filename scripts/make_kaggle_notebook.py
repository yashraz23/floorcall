"""Write kaggle/floorcall_table_d.ipynb (docs/runbook-kaggle.md). Edit this, not the notebook.

uv run python scripts/make_kaggle_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n").splitlines(True),
        }
    )


md(
    """
# floorcall · Table D ablations on Kaggle

One **arm** per session: `full`, `no_history`, `no_agent` or `no_normalize`. Each arm trains the
multi-task checkpoint with the committed recipe and the same seed. It keeps the epoch with the
lowest dev loss, calibrates on calib and scores Table D once on the frozen test sets. See
`docs/runbook-kaggle.md` and `docs/DECISIONS.md` (D-035 amendment 1, D-037).

Settings: **Accelerator: GPU T4 x2** (one GPU is used) or P100. **Internet: on** (pip and the pinned
Laya checkpoint). Run it with **Save Version → Save & Run All** so it runs headless for up to 12 hours.

Stop at the first failed cell and report it. Do not work around it.
"""
)
code(
    """
# --- the only cell to edit ---------------------------------------------------------------------
ARM = "full"        # full | no_history | no_agent | no_normalize
EPOCHS = 2          # every Kaggle arm, full and ablations alike (DECISIONS.md D-038)
SMOKE_ONLY = False  # True: stop after the smoke run and the ETA
BUNDLE = "/kaggle/input/datasets/yashraz/floorcall-table-d"  # the dataset's input folder
WORK = "/kaggle/working/floorcall"
"""
)
code(
    """
# Copy the bundle to WORK and verify it end to end (scripts/kaggle_restore.py, standard library
# only). Kaggle unpacks every .jsonl.gz on upload: each file's decompressed content must match
# BUNDLE.json, and its .gz is rebuilt byte for byte, so the frozen-test manifest and every test
# hash hold exactly as they do locally. Any mismatch stops here.
import json, os, shutil, subprocess, sys
from pathlib import Path

assert ARM in ("full", "no_history", "no_agent", "no_normalize"), ARM
root = sorted(Path(BUNDLE).rglob("BUNDLE.json"), key=lambda p: len(p.parts))[0].parent
subprocess.run(
    [sys.executable, str(root / "scripts" / "kaggle_restore.py"), "--input", BUNDLE, "--work", WORK],
    check=True,
)
bundle = json.loads(Path(WORK, "BUNDLE.json").read_text())
"""
)
code(
    """
# Pinned dependencies next to Kaggle's own CUDA torch (never replaced), then floorcall itself,
# editable, so its paths resolve inside WORK.
pins = [f"{name}=={v}" for name, v in bundle["pinned"].items()]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", WORK], check=True)
"""
)
code(
    """
# The environment this arm runs in. T4 and P100 (compute capability < 8) have no fast bf16, so
# training runs in fp16 with dynamic loss scaling; Laya also serves them in fp16.
import torch

cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| capability", cap)
env = {"FLOORCALL_CODE": bundle["commit"] + "-kaggle"}
if cap[0] < 8:
    env["FLOORCALL_TRAIN__AMP_DTYPE"] = "fp16"
os.environ.update(env)
print(env)
subprocess.run(["nvidia-smi"], check=False)


def floorcall(*args, extra=None):
    subprocess.run(["floorcall", *args], check=True, cwd=WORK, env={**os.environ, **(extra or {})})
"""
)
code(
    """
# Smoke run: the whole path (dev split, per-epoch checkpoints, best-epoch copy, calibration) on 600
# train rows per task and 64 rows per epoch. Minutes. It is the first run of this code on this
# GPU; if it fails, stop here.
tasks = ("turn_complete", "barge_in", "route", "escalate")
smoke = {
    "FLOORCALL_TRAIN__EPOCHS": "2",
    "FLOORCALL_TRAIN__MAX_TRAIN_ROWS_PER_TASK": "600",
    "FLOORCALL_TRAIN__ROWS_PER_EPOCH": json.dumps({t: 64 for t in tasks}),
}
shutil.rmtree("/kaggle/working/smoke", ignore_errors=True)
floorcall("train", "run", "--out", "/kaggle/working/smoke", "--ablation", ARM, extra=smoke)
log = [json.loads(l) for l in open("/kaggle/working/smoke/train_log.jsonl")]
rate = [r["rows_per_s"] for r in log if "update" in r][-1]
sys.path.insert(0, f"{WORK}/src")
from floorcall.config import get_settings

cfg = get_settings().train
epochs = EPOCHS or cfg.epochs
rows = sum(cfg.rows_per_epoch.values()) * epochs
print(f"smoke ok: {rate:.1f} rows/s -> this arm needs about {rows / rate / 3600:.1f} h of training "
      f"for {epochs} epochs, plus item building and calibration (Kaggle stops a session at 12 h)")
shutil.rmtree("/kaggle/working/smoke")
"""
)
code(
    """
# The arm: train (the best epoch by dev loss is kept), calibrate, then Table D once on test.
if not SMOKE_ONLY:
    out_dir = f"/kaggle/working/runs/{ARM}"
    floorcall("train", "run", "--out", out_dir, "--ablation", ARM,
              extra={"FLOORCALL_TRAIN__EPOCHS": str(EPOCHS)} if EPOCHS else None)
    floorcall("eval", "ablation", "--checkpoint", out_dir)
"""
)
code(
    """
# What to download: small files only (the checkpoints stay in this version's output).
if not SMOKE_ONLY:
    out = Path("/kaggle/working/outputs") / ARM
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(f"/kaggle/working/runs/{ARM}")
    for name in ("run.json", "train_log.jsonl", "floorcall_calibration.json"):
        shutil.copy2(run_dir / name, out / name)
    for p in Path(WORK, "results", "table_d").glob("*.json"):
        shutil.copy2(p, out / p.name)
    for p in Path(WORK, "runs", "eval", "table_d").glob("*.npz"):  # per-row predictions
        shutil.copy2(p, out / p.name)
    shutil.make_archive(str(out), "zip", out)
    print("download /kaggle/working/outputs/" + ARM + ".zip:", sorted(p.name for p in out.iterdir()))
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = Path(__file__).resolve().parents[1] / "kaggle" / "floorcall_table_d.ipynb"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("wrote", path, len(cells), "cells")
