"""The Kaggle bundle's secret scan (D-038): nothing that looks like a key may leave the laptop."""

import gzip
import importlib.util
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
