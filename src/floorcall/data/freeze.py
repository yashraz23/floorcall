"""Frozen, hashed test sets (CLAUDE.md §10.2).

A test set is written once, hashed, and committed. After that it can only be re-verified, never
edited. A change is a new versioned file (`.v2`), recorded in DECISIONS.md.

The bytes are deterministic: gzip with mtime 0 and no embedded filename, LF newlines, and
json.dumps with a fixed key order. Rebuilding from the same source with the same code reproduces
the same SHA256. `freeze` exploits that: if the target file already exists, the rebuild must match
it byte for byte, or the build stops. A frozen set is therefore also a reproducibility check on the
builder.

`data/test_frozen/MANIFEST.sha256` is in `sha256sum -c` format. `MANIFEST.json` carries
provenance and class balance for each file. tests/test_frozen_data.py verifies both on every CI run.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

MANIFEST = "MANIFEST.sha256"
MANIFEST_JSON = "MANIFEST.json"


class FrozenSetChangedError(RuntimeError):
    """A rebuild produced different bytes for an already-frozen test set."""


def jsonl_gz_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    buf = io.BytesIO()
    # filename="" and mtime=0 keep the gzip header free of anything but the data
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows (overwriting). For train and calib files, which are rebuilt freely."""
    data = jsonl_gz_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(root: Path) -> dict[str, str]:
    path = root / MANIFEST
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    return out


def _write_manifest(root: Path, entries: Mapping[str, str]) -> None:
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
    with (root / MANIFEST).open("w", encoding="utf-8", newline="\n") as f:
        f.write(lines)


def _read_manifest_json(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def freeze(
    root: Path, name: str, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> tuple[str, bool]:
    """Freeze `rows` as `root/name`. Returns (sha256, newly_written).

    If `name` is already frozen, the rebuilt bytes must be identical, or FrozenSetChangedError is
    raised: a changed test set is a new version, never an edit.
    """
    data = jsonl_gz_bytes(rows)
    digest = hashlib.sha256(data).hexdigest()
    path = root / name
    manifest = _read_manifest(root)
    if path.exists() or name in manifest:
        on_disk = sha256_file(path) if path.exists() else None
        if on_disk != digest or manifest.get(name) != digest:
            raise FrozenSetChangedError(
                f"{name} is frozen (manifest {manifest.get(name)}, file {on_disk}) and the rebuild "
                f"gives {digest}. A frozen test set is never edited: write a new version "
                "(e.g. .v2) and record why in docs/DECISIONS.md."
            )
        return digest, False
    root.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest[name] = digest
    _write_manifest(root, manifest)
    info = _read_manifest_json(root)
    info[name] = {"sha256": digest, "rows": len(rows), **meta}
    with (root / MANIFEST_JSON).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return digest, True


def verify(root: Path) -> dict[str, str]:
    """Check every frozen file against the manifest, and that nothing unlisted is present."""
    manifest = _read_manifest(root)
    present = {p.name for p in root.glob("*.jsonl.gz")}
    unlisted = present - set(manifest)
    if unlisted:
        raise FrozenSetChangedError(f"files in {root} not in {MANIFEST}: {sorted(unlisted)}")
    for name, digest in manifest.items():
        path = root / name
        if not path.exists():
            raise FrozenSetChangedError(f"{name} is in {MANIFEST} but missing")
        actual = sha256_file(path)
        if actual != digest:
            raise FrozenSetChangedError(f"{name}: manifest {digest}, file {actual}")
    return manifest
