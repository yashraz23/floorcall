"""Freezing machinery. The frozen sets themselves are checked in tests/test_frozen_data.py."""

from pathlib import Path

import pytest

from floorcall.data.freeze import (
    FrozenSetChangedError,
    freeze,
    jsonl_gz_bytes,
    read_jsonl_gz,
    sha256_file,
    verify,
)

ROWS = [{"id": "a", "label": "true", "text": "café"}, {"id": "b", "label": "false", "text": "x"}]


def test_bytes_are_deterministic() -> None:
    assert jsonl_gz_bytes(ROWS) == jsonl_gz_bytes([dict(r) for r in ROWS])


def test_round_trip(tmp_path: Path) -> None:
    freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {"source": "unit"})
    assert read_jsonl_gz(tmp_path / "t.v1.jsonl.gz") == ROWS


def test_refreezing_identical_rows_is_a_no_op(tmp_path: Path) -> None:
    d1, new1 = freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {})
    d2, new2 = freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {})
    assert (d1, new1, new2) == (d2, True, False)


def test_a_frozen_set_cannot_be_edited(tmp_path: Path) -> None:
    freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {})
    with pytest.raises(FrozenSetChangedError, match="new version"):
        freeze(tmp_path, "t.v1.jsonl.gz", [*ROWS, {"id": "c"}], {})


def test_manifest_is_sha256sum_format(tmp_path: Path) -> None:
    digest, _ = freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {})
    line = (tmp_path / "MANIFEST.sha256").read_bytes().decode()
    assert line == f"{digest}  t.v1.jsonl.gz\n"
    assert sha256_file(tmp_path / "t.v1.jsonl.gz") == digest


def test_verify_catches_tampering_and_strays(tmp_path: Path) -> None:
    freeze(tmp_path, "t.v1.jsonl.gz", ROWS, {})
    verify(tmp_path)
    (tmp_path / "stray.jsonl.gz").write_bytes(jsonl_gz_bytes(ROWS))
    with pytest.raises(FrozenSetChangedError, match="not in"):
        verify(tmp_path)
    (tmp_path / "stray.jsonl.gz").unlink()
    (tmp_path / "t.v1.jsonl.gz").write_bytes(jsonl_gz_bytes(ROWS[:1]))
    with pytest.raises(FrozenSetChangedError):
        verify(tmp_path)
