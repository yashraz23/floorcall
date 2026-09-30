"""The Kaggle bundle's secret scan (D-038): nothing that looks like a key may leave the laptop."""

import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType

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
    (tmp_path / "rows.jsonl.gz").write_bytes(gzip.compress(b'{"text": "' + secret + b'"}'))
    found = kb.find_secrets(tmp_path, [secret])
    flagged = {line.split(":", 1)[0] for line in found}
    assert flagged == {".env", "kaggle.json", "pattern.py", "envline.txt", "rows.jsonl.gz"}
    assert all(secret.decode() not in line for line in found)  # a value is never echoed


def test_local_env_values_are_read_without_short_or_commented_ones(tmp_path: Path) -> None:
    kb = bundle_module()
    env = tmp_path / ".env"
    env.write_text(
        '# COMMENT=ignoredvalue123\nA="quoted-secret-value"\nB=short\nC=plain-secret-value\n'
    )
    assert kb.local_secret_values(env) == [b"quoted-secret-value", b"plain-secret-value"]
    assert kb.local_secret_values(tmp_path / "missing") == []


# -- restoring on Kaggle (D-042): Kaggle unpacks every .jsonl.gz on upload ----------------------


def restore_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "kaggle_restore", REPO_ROOT / "scripts" / "kaggle_restore.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(root: Path) -> bytes:
    """A two-file bundle as kaggle_bundle.py writes it; returns the .gz bytes."""
    from floorcall.data.freeze import gzip_bytes

    kb = bundle_module()
    gz = gzip_bytes(b'{"id": 1}\n{"id": 2}\n')
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "processed" / "a.jsonl.gz").write_bytes(gz)
    (root / "MANIFEST.json").write_text("{}")
    files = {
        "data/processed/a.jsonl.gz": kb.file_entry(root / "data" / "processed" / "a.jsonl.gz"),
        "MANIFEST.json": kb.file_entry(root / "MANIFEST.json"),
    }
    (root / "BUNDLE.json").write_text(json.dumps({"commit": "abc", "files": files}))
    return gz


def test_the_restore_gzip_is_floorcalls() -> None:
    from floorcall.data.freeze import gzip_bytes

    raw = "héllo\n".encode() * 100
    assert restore_module().gzip_bytes(raw) == gzip_bytes(raw)


def test_a_bundle_kaggle_unpacked_is_restored_byte_for_byte(tmp_path: Path) -> None:
    kr = restore_module()
    upload = tmp_path / "input" / "datasets" / "yashraz" / "floorcall-table-d"
    upload.mkdir(parents=True)
    gz = make_bundle(upload)
    gz_path = upload / "data" / "processed" / "a.jsonl.gz"  # what Kaggle does to it
    (upload / "data" / "processed" / "a.jsonl").write_bytes(gzip.decompress(gz_path.read_bytes()))
    gz_path.unlink()
    assert kr.find_bundle(tmp_path / "input") == upload
    problems, report = kr.restore(upload, tmp_path / "work")
    assert problems == [] and any("rebuilt from Kaggle's unpacked copy" in r for r in report)
    restored = tmp_path / "work" / "data" / "processed"
    assert (restored / "a.jsonl.gz").read_bytes() == gz  # the original bytes, exactly
    assert not (restored / "a.jsonl").exists()


def test_a_bundle_left_gzipped_verifies_too(tmp_path: Path) -> None:
    kr = restore_module()
    (tmp_path / "in" / "floorcall").mkdir(parents=True)  # one folder down
    make_bundle(tmp_path / "in" / "floorcall")
    root = kr.find_bundle(tmp_path / "in")
    problems, report = kr.restore(root, tmp_path / "work")
    assert problems == [] and any("uploaded gzipped" in r for r in report)


def test_changed_or_missing_files_stop_the_restore(tmp_path: Path) -> None:
    kr = restore_module()
    make_bundle(tmp_path / "in")
    processed = tmp_path / "in" / "data" / "processed"
    (processed / "a.jsonl.gz").unlink()
    (processed / "a.jsonl").write_bytes(b'{"id": 1}\n{"id": 3}\n')  # tampered content
    problems, _ = kr.restore(tmp_path / "in", tmp_path / "w1")
    assert any("content sha256 differs" in p for p in problems)
    (processed / "a.jsonl").unlink()
    problems, _ = kr.restore(tmp_path / "in", tmp_path / "w2")
    assert any("missing" in p for p in problems)
