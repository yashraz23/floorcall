"""Build the Kaggle dataset for the Table D ablations (docs/runbook-kaggle.md, DECISIONS.md D-037).

    uv run python scripts/kaggle_bundle.py

Writes dist/kaggle/<DATASET_SLUG>/, ready for `kaggle datasets create -p <dir>`:

- floorcall/ : the committed source tree at HEAD (`git archive`, so nothing uncommitted), plus the
  processed train and calib files for all four decisions and the frozen test sets. Nothing else:
  no raw corpora, no hand-label files, no LLM labels, no checkpoints, no keys.
- floorcall/BUNDLE.json : the commit, when it was built, the pinned versions the notebook installs,
  and the sha256 of every data file, which the notebook re-checks before training.
- dataset-metadata.json : for the Kaggle CLI, id `<KAGGLE_OWNER>/<DATASET_SLUG>`.

Refuses on a dirty tree: the bundle's commit is what every Kaggle result will cite as its code.
Refuses, and deletes the bundle, if any file in it looks like a secret (D-038): a secret-looking
file name, a known key pattern, or any value from the local .env, searched in raw bytes and inside
the gzipped data. Secret values are never printed. The dataset must be created private: never pass
--public (docs/runbook-kaggle.md).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The Kaggle dataset: the owner is Yash's Kaggle username; the id is "<owner>/<slug>".
KAGGLE_OWNER = "yashraz"
DATASET_SLUG = "floorcall-table-d"
OUT = REPO / "dist" / "kaggle" / DATASET_SLUG
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


SECRET_FILE_NAMES = re.compile(
    r"(^\.env|^kaggle\.json$|^id_(rsa|ed25519)|credentials|\.(pem|key|p12|pfx)$)", re.IGNORECASE
)
SECRET_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rb"sk-or-v1-[0-9a-f]{20,}",  # OpenRouter
        rb"sk-ant-[A-Za-z0-9_-]{20,}",  # Anthropic
        rb"gsk_[A-Za-z0-9]{40,}",  # Groq
        rb"hf_[A-Za-z0-9]{30,}",  # Hugging Face
        rb"AKIA[0-9A-Z]{16}",  # AWS
        rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        rb"(?m)^[A-Z][A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*\s*=\s*\S{8,}",  # a .env-style line
    )
)


def local_secret_values(env_file: Path) -> list[bytes]:
    """The values in the local .env (never printed), to prove none of them reached the bundle."""
    if not env_file.exists():
        return []
    values = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if len(value) >= 8:
                values.append(value.encode())
    return values


def find_secrets(root: Path, secret_values: list[bytes]) -> list[str]:
    """What in `root` looks like a secret: file names, key patterns, local .env values."""
    problems = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if SECRET_FILE_NAMES.search(p.name):
            problems.append(f"{rel}: a secret-looking file name")
            continue
        data = p.read_bytes()
        if p.suffix == ".gz":
            data = gzip.decompress(data)
        problems += [f"{rel}: matches a key pattern" for pat in SECRET_PATTERNS if pat.search(data)]
        if any(v in data for v in secret_values):
            problems.append(f"{rel}: contains a value from the local .env")
    return problems


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def file_entry(path: Path) -> dict[str, str]:
    """The bytes' sha256; for a .gz, also its decompressed content's (D-042): Kaggle unpacks every
    .jsonl.gz on upload, and scripts/kaggle_restore.py verifies the content and rebuilds the .gz."""
    data = path.read_bytes()
    entry = {"sha256": hashlib.sha256(data).hexdigest()}
    if path.name.endswith(".gz"):
        entry["content_sha256"] = hashlib.sha256(gzip.decompress(data)).hexdigest()
    return entry


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
    files: dict[str, dict[str, str]] = {}
    for d in DECISIONS:
        for split in ("train", "calib"):
            src = s.paths.data_processed / processed_file(d, split)
            dst = tree / "data" / "processed" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files[f"data/processed/{src.name}"] = file_entry(dst)
    frozen = tree / "data" / "test_frozen"
    shutil.copytree(s.paths.test_frozen, frozen)
    for p in sorted(frozen.iterdir()):
        files[f"data/test_frozen/{p.name}"] = file_entry(p)

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
        "id": f"{KAGGLE_OWNER}/{DATASET_SLUG}",
        # derived from SwDA (CC BY-NC-SA 3.0), CLINC150 (CC BY 3.0) and Customer Support on
        # Twitter (CC BY-NC-SA 4.0): keep the dataset private
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    problems = find_secrets(OUT, local_secret_values(REPO / ".env"))
    if problems:
        shutil.rmtree(OUT)
        listing = "\n  ".join(problems)
        sys.exit(f"refusing the bundle, it may hold a secret (deleted):\n  {listing}")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB, {n} files) from commit {commit}")
    print("secret scan: clean (file names, key patterns, local .env values; gz data included)")
    kaggle = 'uvx --from "kaggle>=1.8" kaggle'
    print(
        f"first upload, PRIVATE: {kaggle} datasets create -p dist/kaggle/{DATASET_SLUG} --dir-mode zip"
    )
    print(
        f"new version: {kaggle} datasets version -p dist/kaggle/{DATASET_SLUG} "
        f'-m "bundle {commit}" --dir-mode zip'
    )
    print("(never --public; then check the dataset page shows Private)")


if __name__ == "__main__":
    main()
