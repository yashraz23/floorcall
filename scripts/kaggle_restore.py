"""Restore and verify the floorcall bundle on Kaggle (docs/runbook-kaggle.md). Standard library only.

    python kaggle_restore.py --input /kaggle/input/datasets/yashraz/floorcall-table-d \
        --work /kaggle/working/floorcall

Kaggle unpacks uploads: it decompresses `.gz` files, and the bundle may land at the input root or
one folder down. So the bundle ships every `.jsonl.gz` as `.jsonl.gz.bin` (D-042). This finds
BUNDLE.json under `--input`, copies the tree to `--work`, and restores every data file listed
there to its `.gz` name, verified:

1. **Original bytes** (the `.gz.bin` as shipped, or a `.gz`) must match the recorded `sha256`
   exactly. No recompression is involved, so no zlib version matters.
2. Only if Kaggle unpacked a file anyway does its **decompressed content** have to match
   `content_sha256`. It is then re-gzipped with floorcall's deterministic gzip and must still
   match `sha256` byte for byte. A different zlib can rebuild different bytes (CPython 3.14's
   zlib-ng does, measured); that is a stop, never a pass.

Either way, the frozen-test manifest and every result's `test_sha256` hold on Kaggle exactly as
they do locally. Any mismatch or missing file stops with a non-zero exit.
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

GZIP_MAGIC = b"\x1f\x8b"


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


def _restore_gz(work: Path, rel: str, entry: dict[str, str]) -> tuple[str | None, str]:
    """(problem or None, how) for one gzipped data file, written to `work/rel` when verified."""
    target = work / rel
    candidates = [work / f"{rel}.bin", target, work / rel.removesuffix(".gz")]
    present = [p for p in candidates if p.exists()]
    if not present:
        return f"{rel}: missing (no .gz.bin, .gz or unpacked copy)", ""
    data = present[0].read_bytes()
    if data[:2] == GZIP_MAGIC:  # the original gzip bytes, however Kaggle named them
        if sha256(data) != entry["sha256"]:
            return f"{rel}: sha256 differs", ""
        gz, how = data, f"original bytes ({present[0].name})"
    else:  # Kaggle decompressed it
        if sha256(data) != entry["content_sha256"]:
            return f"{rel}: content sha256 differs", ""
        gz = gzip_bytes(data)
        if sha256(gz) != entry["sha256"]:
            return (
                f"{rel}: content verified, but this Python's gzip rebuilds different bytes (a "
                "different zlib), so the manifest and test hashes cannot be checked",
                "",
            )
        how = f"rebuilt byte for byte from Kaggle's unpacked {present[0].name}"
    for p in present:
        p.unlink()  # leave nothing ambiguous beside the .gz the loaders read
    target.write_bytes(gz)
    return None, how


def restore(bundle_root: Path, work: Path) -> tuple[list[str], list[str]]:
    """Copy `bundle_root` to `work` and restore every listed data file there.

    Returns the problems (empty when everything verified) and one report line per file.
    """
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bundle_root, work)
    bundle = json.loads((work / "BUNDLE.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    report: list[str] = []
    for rel, entry in sorted(bundle["files"].items()):
        if rel.endswith(".gz"):
            problem, how = _restore_gz(work, rel, entry)
        else:
            target = work / rel
            if not target.exists():
                problem, how = f"{rel}: missing", ""
            elif sha256(target.read_bytes()) != entry["sha256"]:
                problem, how = f"{rel}: sha256 differs", ""
            else:
                problem, how = None, "original bytes"
        if problem:
            problems.append(problem)
        else:
            report.append(f"{rel}: verified, {how}")
    return problems, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore and verify the floorcall bundle on Kaggle.")
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
