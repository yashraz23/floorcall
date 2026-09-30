"""The Kaggle bundle: its secret scan (D-038), and restoring it on Kaggle (D-042)."""

import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from floorcall.config import REPO_ROOT


def bundle_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "kaggle_bundle", REPO_ROOT / "scripts" / "kaggle_bundle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_scan_catches_every_kind_of_leak(tmp_path: Path) -> None:
    kb = bundle_module()
    (tmp_path / "ok.txt").write_text("nothing to see: OPENROUTER_API_KEY is read from .env\n")
    (tmp_path / ".env").write_text("X=1\n")
    (tmp_path / "kaggle.json").write_text("{}")
    (tmp_path / "pattern.py").write_text('KEY = "sk-or-v1-' + "ab" * 32 + '"\n')
    (tmp_path / "envline.txt").write_text("OPENROUTER_API_KEY=abcdefghijklmnop\n")
    secret = b"a-local-secret-value-123"
    # gzip is recognised by content, so a shipped .gz.bin is scanned inside too
    (tmp_path / "rows.jsonl.gz.bin").write_bytes(gzip.compress(b'{"text": "' + secret + b'"}'))
    found = kb.find_secrets(tmp_path, [secret])
    flagged = {line.split(":", 1)[0] for line in found}
    assert flagged == {".env", "kaggle.json", "pattern.py", "envline.txt", "rows.jsonl.gz.bin"}
    assert all(secret.decode() not in line for line in found)  # a value is never echoed


def test_local_env_values_are_read_without_short_or_commented_ones(tmp_path: Path) -> None:
    kb = bundle_module()
    env = tmp_path / ".env"
    env.write_text(
        '# COMMENT=ignoredvalue123\nA="quoted-secret-value"\nB=short\nC=plain-secret-value\n'
    )
    assert kb.local_secret_values(env) == [b"quoted-secret-value", b"plain-secret-value"]
    assert kb.local_secret_values(tmp_path / "missing") == []


# -- restoring on Kaggle (D-042): Kaggle unpacks .gz files on upload -----------------------------

ROWS = b'{"id": 1}\n{"id": 2}\n'


def restore_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "kaggle_restore", REPO_ROOT / "scripts" / "kaggle_restore.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(root: Path) -> bytes:
    """A two-file bundle as kaggle_bundle.py writes it (a .gz shipped as .gz.bin); returns the gz."""
    from floorcall.data.freeze import gzip_bytes

    kb = bundle_module()
    gz = gzip_bytes(ROWS)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "a.jsonl.gz").write_bytes(gz)
    (root / "MANIFEST.json").write_text("{}")
    files = {
        "data/processed/a.jsonl.gz": kb.file_entry(processed / "a.jsonl.gz"),
        "MANIFEST.json": kb.file_entry(root / "MANIFEST.json"),
    }
    (processed / "a.jsonl.gz").rename(processed / "a.jsonl.gz.bin")
    (root / "BUNDLE.json").write_text(json.dumps({"commit": "abc", "files": files}))
    return gz


def unpack_like_kaggle(root: Path) -> None:
    processed = root / "data" / "processed"
    (processed / "a.jsonl").write_bytes(
        gzip.decompress((processed / "a.jsonl.gz.bin").read_bytes())
    )
    (processed / "a.jsonl.gz.bin").unlink()


def test_the_restore_gzip_is_floorcalls() -> None:
    from floorcall.data.freeze import gzip_bytes

    raw = "héllo\n".encode() * 100
    assert restore_module().gzip_bytes(raw) == gzip_bytes(raw)


def test_shipped_bytes_are_verified_and_renamed_back(tmp_path: Path) -> None:
    kr = restore_module()
    upload = tmp_path / "input" / "datasets" / "yashraz" / "floorcall-table-d"
    upload.mkdir(parents=True)
    gz = make_bundle(upload)
    assert kr.find_bundle(tmp_path / "input") == upload
    problems, report = kr.restore(upload, tmp_path / "work")
    assert problems == [] and any("original bytes (a.jsonl.gz.bin)" in r for r in report)
    restored = tmp_path / "work" / "data" / "processed"
    assert (restored / "a.jsonl.gz").read_bytes() == gz
    assert not (restored / "a.jsonl.gz.bin").exists()


def test_a_file_kaggle_unpacked_is_rebuilt_byte_for_byte(tmp_path: Path) -> None:
    kr = restore_module()
    (tmp_path / "in" / "floorcall").mkdir(parents=True)  # one folder down
    gz = make_bundle(tmp_path / "in" / "floorcall")
    unpack_like_kaggle(tmp_path / "in" / "floorcall")
    problems, report = kr.restore(kr.find_bundle(tmp_path / "in"), tmp_path / "work")
    assert problems == [] and any("rebuilt byte for byte" in r for r in report)
    assert (tmp_path / "work" / "data" / "processed" / "a.jsonl.gz").read_bytes() == gz


def test_a_different_zlib_cannot_pass_but_the_shipped_bytes_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kr = restore_module()
    monkeypatch.setattr(kr, "gzip_bytes", lambda raw: gzip.compress(raw, compresslevel=1))
    make_bundle(tmp_path / "shipped")
    assert kr.restore(tmp_path / "shipped", tmp_path / "w1")[0] == []  # no recompression needed
    make_bundle(tmp_path / "unpacked")
    unpack_like_kaggle(tmp_path / "unpacked")
    problems, _ = kr.restore(tmp_path / "unpacked", tmp_path / "w2")
    assert any("different zlib" in p for p in problems)


def test_changed_or_missing_files_stop_the_restore(tmp_path: Path) -> None:
    kr = restore_module()
    make_bundle(tmp_path / "in")
    processed = tmp_path / "in" / "data" / "processed"
    shipped = processed / "a.jsonl.gz.bin"
    shipped.write_bytes(gzip.compress(ROWS + b"tampered"))
    assert any("sha256 differs" in p for p in kr.restore(tmp_path / "in", tmp_path / "w1")[0])
    shipped.unlink()
    (processed / "a.jsonl").write_bytes(ROWS + b"tampered")  # unpacked by Kaggle, and changed
    problems, _ = kr.restore(tmp_path / "in", tmp_path / "w2")
    assert any("content sha256 differs" in p for p in problems)
    (processed / "a.jsonl").unlink()
    assert any("missing" in p for p in kr.restore(tmp_path / "in", tmp_path / "w3")[0])
