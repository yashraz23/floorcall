"""Build the Kaggle dataset for the Table D ablations (docs/runbook-kaggle.md, DECISIONS.md D-037).

    uv run python scripts/kaggle_bundle.py

Writes dist/kaggle/floorcall-table-d/, ready for `kaggle datasets create -p <dir>`:

- floorcall/ : the committed source tree at HEAD (`git archive`, so nothing uncommitted), plus the
  processed train and calib files for all four decisions and the frozen test sets. Nothing else:
  no raw corpora, no hand-label files, no LLM labels, no checkpoints, no keys.
- floorcall/BUNDLE.json : the commit, when it was built, the pinned versions the notebook installs,
  and the sha256 of every data file, which the notebook re-checks before training.
- dataset-metadata.json : for the Kaggle CLI. Set its id to your Kaggle username first.

Refuses on a dirty tree: the bundle's commit is what every Kaggle result will cite as its code.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dist" / "kaggle" / "floorcall-table-d"
DECISIONS = ("turn_complete", "barge_in", "route", "escalate")
# Installed on Kaggle next to its own CUDA torch (never replaced: its build matches its GPUs).
PINNED = (
    "laya",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "pydantic",
    "pydantic-settings",
    "typer",
    "rich",
    "scikit-learn",
    "numpy",
    "pandas",
    "pyarrow",
    "httpx",
    "matplotlib",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("the working tree has uncommitted changes: commit first, the bundle cites HEAD")
    commit = git("rev-parse", "--short", "HEAD").strip()
    if OUT.exists():
        shutil.rmtree(OUT)
    tree = OUT / "floorcall"
    tree.mkdir(parents=True)

    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", "src", "scripts", "pyproject.toml", "README.md"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter="data")

    sys.path.insert(0, str(REPO / "src"))
    from floorcall.config import get_settings
    from floorcall.data.build import processed_file

    s = get_settings()
    files: dict[str, str] = {}
    for d in DECISIONS:
        for split in ("train", "calib"):
            src = s.paths.data_processed / processed_file(d, split)
            dst = tree / "data" / "processed" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files[f"data/processed/{src.name}"] = sha256(dst)
    frozen = tree / "data" / "test_frozen"
    shutil.copytree(s.paths.test_frozen, frozen)
    for p in sorted(frozen.iterdir()):
        files[f"data/test_frozen/{p.name}"] = sha256(p)

    bundle = {
        "commit": commit,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pinned": {name: version(name) for name in PINNED},
        "local_torch": version("torch"),
        "files": files,
    }
    (tree / "BUNDLE.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    meta = {
        "title": "floorcall Table D bundle",
        "id": "YOUR-KAGGLE-USERNAME/floorcall-table-d",
        # derived from SwDA (CC BY-NC-SA 3.0), CLINC150 (CC BY 3.0) and Customer Support on
        # Twitter (CC BY-NC-SA 4.0): keep the dataset private
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB) from commit {commit}")


if __name__ == "__main__":
    main()
