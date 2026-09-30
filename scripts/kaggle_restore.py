"""Restore and verify the floorcall bundle on Kaggle (docs/runbook-kaggle.md). Standard library only.

    python kaggle_restore.py --input /kaggle/input/datasets/yashraz/floorcall-table-d \
        --work /kaggle/working/floorcall

Kaggle unpacks uploads: it decompresses every `.jsonl.gz` into a plain `.jsonl`, and the bundle may
land at the input root or one folder down. This finds BUNDLE.json under `--input`, copies the tree
to `--work`, then checks every data file listed there:

1. Its **content** (the decompressed bytes) must match `content_sha256`, whether Kaggle left the
   file gzipped or unpacked it.
2. The `.gz` is rebuilt in `--work` with floorcall's deterministic gzip (mtime 0, no filename,
   level 9) and must match the original file's `sha256` **byte for byte**. Then the frozen-test
   manifest and every result's `test_sha256` hold on Kaggle exactly as they do locally.

Any mismatch or missing file stops with a non-zero exit. That includes content that is right but
recompresses to different bytes, which a different zlib could cause; the bundle is never used on
bytes it could not verify.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path


def gzip_bytes(raw: bytes) -> bytes:
    """floorcall.data.freeze.gzip_bytes, copied: this runs before floorcall is installed. A test
    checks the two agree."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_bundle(root: Path) -> Path:
    """The folder holding BUNDLE.json: `root` itself or somewhere below it."""
    hits = sorted(root.rglob("BUNDLE.json"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no BUNDLE.json under {root}")
    return hits[0].parent


def restore(bundle_root: Path, work: Path) -> tuple[list[str], list[str]]:
    """Copy `bundle_root` to `work` and restore every listed data file there. Returns the problems
    (empty when everything verified), and one report line per file."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bundle_root, work)
    bundle = json.loads((work / "BUNDLE.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    report: list[str] = []
    for rel, entry in sorted(bundle["files"].items()):
        target = work / rel
        if not rel.endswith(".gz"):
            if not target.exists():
                problems.append(f"{rel}: missing")
            elif sha256(target.read_bytes()) != entry["sha256"]:
                problems.append(f"{rel}: sha256 differs")
            else:
                report.append(f"{rel}: verified")
            continue
        unpacked = work / rel.removesuffix(".gz")
        if target.exists():
            raw, how = gzip.decompress(target.read_bytes()), "uploaded gzipped"
        elif unpacked.exists():
            raw, how = unpacked.read_bytes(), "rebuilt from Kaggle's unpacked copy"
        else:
            problems.append(f"{rel}: missing, gzipped or not")
            continue
        if sha256(raw) != entry["content_sha256"]:
            problems.append(f"{rel}: content sha256 differs")
            continue
        gz = gzip_bytes(raw) if how.startswith("rebuilt") else target.read_bytes()
        if sha256(gz) != entry["sha256"]:
            problems.append(
                f"{rel}: content verified, but this Python's gzip rebuilds different bytes "
                "(a different zlib); the manifest and test hashes cannot be checked"
            )
            continue
        target.write_bytes(gz)
        if unpacked.exists():
            unpacked.unlink()  # the loaders read the .gz; leave nothing ambiguous beside it
        report.append(f"{rel}: verified, {how}")
    return problems, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    root = find_bundle(args.input)
    problems, report = restore(root, args.work)
    commit = json.loads((args.work / "BUNDLE.json").read_text(encoding="utf-8"))["commit"]
    print(f"bundle at {root}, commit {commit}")
    for line in report:
        print("  " + line)
    if problems:
        print("VERIFICATION FAILED:")
        for line in problems:
            print("  " + line)
        return 1
    print(f"all {len(report)} files verified; restored to {args.work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
