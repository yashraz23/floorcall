"""The Hugging Face release of r2 (DECISIONS.md D-048): the model card and a folder to upload.

    uv run floorcall release card     # re-render release/model_card.md's tables from results/
    uv run floorcall release stage    # dist/hf/<repo>/: weights, config, calibration and the card

Nothing here uploads. The card's tables come from committed result files through the same
renderer as the README, so the card cannot carry a number the README does not.
tests/test_release.py fails when the card differs from what they render.

The staged folder holds exactly RELEASE_FILES, the card and Laya's licence text: the weights,
Laya's configs and the tokenizer that loading needs, and the calibration. No training, calib or test data, no run logs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from floorcall.config import REPO_ROOT
from floorcall.evaluate import report

# Yash's Hugging Face account (`hf auth whoami`) and the model's repo name. The card's loading
# example names it too; tests/test_release.py keeps the two the same.
HF_REPO_ID = "enz23/floorcall"
RELEASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "main-r2"
CARD = REPO_ROOT / "release" / "model_card.md"
# Laya's licence, byte for byte from github.com/NandhaKishorM/laya (LICENSE at 136910c). Apache-2.0
# §4(a) asks that a derivative ship with a copy of it (D-048). Staged under the same name.
LAYA_LICENCE = REPO_ROOT / "release" / "LICENSE-laya-Apache-2.0.txt"
STAGE = REPO_ROOT / "dist" / "hf"

# Relative to the checkpoint. The tokenizer is ModernBERT-large's (Apache-2.0), as Laya ships it;
# laya.Agent needs it to load the folder.
RELEASE_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "floorcall_calibration.json",
)

CARD_SECTIONS = {
    name: report.SECTIONS[name]
    for name in (
        "table-a",
        "table-b",
        "table-b-env",
        "table-c",
        "table-d",
        "table-d-paired",
        "operating-points",
    )
}


def render_card(text: str) -> str:
    return report.render_readme(text, CARD_SECTIONS)


def write_card() -> bool:
    old = CARD.read_text(encoding="utf-8")
    new = render_card(old)
    if new != old:
        with CARD.open("w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return new != old


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(checkpoint: Path = RELEASE_CHECKPOINT) -> list[str]:
    """Reasons the folder is not the model Table A measured; empty when it is."""
    problems = []
    missing = [f for f in RELEASE_FILES if not (checkpoint / f).exists()]
    if missing:
        return [f"{checkpoint} lacks {', '.join(missing)}"]
    cal = json.loads((checkpoint / "floorcall_calibration.json").read_text(encoding="utf-8"))
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    for decision in report.DECISION_NAMES:
        row = json.loads(
            (report.RESULTS / f"{decision}.finetuned_temp.json").read_text(encoding="utf-8")
        )
        if row["trained_at"] != run["code"]:
            problems.append(f"{decision}: Table A was scored on a run from {row['trained_at']}")
        if row["temperature"] != cal["temperatures"][decision]:
            problems.append(f"{decision}: Table A's temperature is not the calibration file's")
    return problems


def check_card() -> list[str]:
    card = CARD.read_text(encoding="utf-8")
    if render_card(card) != card:
        return ["the model card is out of date: run `floorcall release card`"]
    return []


def stage(out: Path | None = None, checkpoint: Path = RELEASE_CHECKPOINT) -> dict[str, str]:
    """Copy the release files and the card into a fresh folder; return each file's sha256."""
    problems = check_checkpoint(checkpoint) + check_card()
    if problems:
        raise ValueError("; ".join(problems))
    target = out or STAGE / HF_REPO_ID.split("/")[1]
    if target.exists():
        shutil.rmtree(target)
    for rel in RELEASE_FILES:
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint / rel, target / rel)
    shutil.copy2(CARD, target / "README.md")
    shutil.copy2(LAYA_LICENCE, target / LAYA_LICENCE.name)
    return {
        p.relative_to(target).as_posix(): _sha256(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
