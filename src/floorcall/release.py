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
import subprocess
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


# The Space (D-050): space/app.py and floorcall.space, staged with exactly the data they read.
SPACE_SRC = REPO_ROOT / "space"
SPACE_REPO_ID = "enz23/floorcall"  # a Space; the model is a separate repo of the same name
GRADIO_VERSION = "6.29.1"  # the Space's SDK; Hugging Face installs it, floorcall never imports it
GITHUB_REPO = "https://github.com/yashraz23/floorcall"
SPACE_THRESHOLDS = (  # what pipeline.operating.served_policy reads, nothing more
    "curves/finetuned_temp.json",
    "table_a/escalate.finetuned_temp_threshold.json",
    "thresholds/route_oos.json",
)
SPACE_FIGURES = (
    ("tradeoff_interrupt.light.png", "θ_interrupt: false stops against missed interruptions"),
    ("tradeoff_yield.light.png", "θ_yield: premature responses against added delay"),
    (
        "reliability_finetuned_temp.light.png",
        "Reliability, fine-tuned, before and after temperature",
    ),
)


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def space_requirements(code_sha: str) -> str:
    """The Space's environment: the locked runtime dependencies with CPU torch, as CI installs
    them, then floorcall itself from the public repository at one commit."""
    export = subprocess.run(
        ["uv", "export", "--frozen", "--no-default-groups", "--group", "cpu", "--no-hashes",
         "--no-emit-project", "--no-annotate"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    pins = [ln for ln in export.splitlines() if ln and not ln.startswith("#")]
    return (
        "\n".join(
            [
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                *pins,
                f"floorcall @ git+{GITHUB_REPO}@{code_sha}",
            ]
        )
        + "\n"
    )


def stage_space(out: Path | None = None, *, require_pushed: bool = True) -> dict[str, str]:
    """Stage the Space into dist/hf/space/: app, README, requirements and the data it reads.

    The Space installs floorcall from the public repository at this checkout's commit, so the
    tree must be clean and, unless `require_pushed` is False (a local trial), the commit pushed.
    The model is pinned to its current revision on the Hub.
    """
    from huggingface_hub import HfApi

    from floorcall import space

    if _git("status", "--porcelain"):
        raise ValueError("the working tree is not clean: the Space would not match its commit")
    sha = _git("rev-parse", "HEAD")
    if require_pushed and not _git("branch", "-r", "--contains", sha):
        raise ValueError(f"{sha[:7]} is not pushed: the Space installs floorcall from GitHub")
    revision = HfApi().model_info(HF_REPO_ID).sha
    target = out or STAGE / "space"
    if target.exists():
        shutil.rmtree(target)
    data = target / "data"
    (data / "scripts").mkdir(parents=True)
    shutil.copy2(SPACE_SRC / "app.py", target / "app.py")
    readme = (SPACE_SRC / "README.md").read_text(encoding="utf-8")
    readme = readme.format(gradio_version=GRADIO_VERSION, model_id=HF_REPO_ID, code_sha=sha[:7])
    (target / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    (target / "requirements.txt").write_text(
        space_requirements(sha), encoding="utf-8", newline="\n"
    )
    results = REPO_ROOT / "results"
    shutil.copy2(results / "replay" / "replay.json", data / "replay.json")
    for script in sorted((REPO_ROOT / "demo" / "scripts").glob("*.json")):
        shutil.copy2(script, data / "scripts" / script.name)
    for rel in SPACE_THRESHOLDS:
        (data / "results" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results / rel, data / "results" / rel)
    (data / "figures").mkdir()
    for name, _ in SPACE_FIGURES:
        shutil.copy2(results / "figures" / name, data / "figures" / name)
    (data / "results.md").write_text(
        space.results_markdown() + "\n", encoding="utf-8", newline="\n"
    )
    config = {
        "model_id": HF_REPO_ID,
        "model_revision": revision,
        "code_sha": sha,
        "figures": [list(f) for f in SPACE_FIGURES],
    }
    (data / "space.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        p.relative_to(target).as_posix(): _sha256(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }


def stage_space_static(out: Path | None = None, *, require_pushed: bool = True) -> dict[str, str]:
    """The free Space (D-050): Replay and Results as one static page, no server, no model.

    Hosting a Gradio Space now needs a paid Hugging Face plan; a static Space does not. The page is
    built from the same committed files and the same views as the Gradio app (stage_space).
    """
    from floorcall import space

    if _git("status", "--porcelain"):
        raise ValueError("the working tree is not clean: the Space would not match its commit")
    sha = _git("rev-parse", "HEAD")
    if require_pushed and not _git("branch", "-r", "--contains", sha):
        raise ValueError(f"{sha[:7]} is not pushed: the page links the code at this commit")
    target = out or STAGE / "space"
    if target.exists():
        shutil.rmtree(target)
    (target / "figures").mkdir(parents=True)
    results = REPO_ROOT / "results"
    replay = json.loads((results / "replay" / "replay.json").read_text(encoding="utf-8"))
    scripts = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((REPO_ROOT / "demo" / "scripts").glob("*.json"))
    }
    page = space.static_page(
        replay, scripts, space.results_markdown(), SPACE_FIGURES, HF_REPO_ID, sha
    )
    (target / "index.html").write_text(page, encoding="utf-8", newline="\n")
    readme = (SPACE_SRC / "README.static.md").read_text(encoding="utf-8")
    readme = readme.format(model_id=HF_REPO_ID, code_sha=sha[:7])
    (target / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    for name, _ in SPACE_FIGURES:
        shutil.copy2(results / "figures" / name, target / "figures" / name)
    return {
        p.relative_to(target).as_posix(): _sha256(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
