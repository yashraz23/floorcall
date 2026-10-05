"""D4's real Twitter text is private (D-049): what is public, what is restorable, and how it fails."""

from pathlib import Path

import pytest

from floorcall.config import REPO_ROOT, Settings, get_settings
from floorcall.data import labelling, private
from floorcall.data.freeze import FrozenSetChangedError, freeze, sha256_file, verify
from floorcall.evaluate.dataset import load_test


def settings_at(tmp_path: Path) -> Settings:
    s = get_settings()
    paths = s.paths.model_copy(
        update={"labels": tmp_path / "labels", "test_frozen": tmp_path / "frozen"}
    )
    return s.model_copy(update={"paths": paths})


def test_every_private_file_is_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for path in private.private_paths(get_settings()):
        assert path.relative_to(REPO_ROOT).as_posix() in ignored


def test_the_label_manifest_names_the_two_private_label_files() -> None:
    s = get_settings()
    want = private.expected_hashes(s)
    assert {p.name for p in want} == {
        private.CANDIDATES,
        private.TRAIN_SAMPLE,
        *private.TEST_FILES,
    }
    for path, digest in want.items():
        if path.exists():  # a restored or archive copy must match
            assert sha256_file(path) == digest, path.name


def test_a_missing_private_file_says_how_to_restore_it(tmp_path: Path) -> None:
    with pytest.raises(private.PrivateDataMissingError, match="restore-escalate"):
        labelling.load_candidates(tmp_path / private.CANDIDATES)
    with pytest.raises(FileNotFoundError) as e:
        labelling.load_candidates(tmp_path / "some_other_pool.jsonl")
    assert not isinstance(e.value, private.PrivateDataMissingError)


def test_a_missing_d4_test_set_says_how_to_restore_it(tmp_path: Path) -> None:
    with pytest.raises(private.PrivateDataMissingError, match="restore-escalate"):
        load_test(settings_at(tmp_path), "escalate")


def test_verify_lets_only_the_named_files_be_absent(tmp_path: Path) -> None:
    freeze(tmp_path, "a.test.v1.jsonl.gz", [{"id": "1"}], {})
    freeze(tmp_path, "b.test.v1.jsonl.gz", [{"id": "2"}], {})
    (tmp_path / "b.test.v1.jsonl.gz").unlink()
    with pytest.raises(FrozenSetChangedError, match="missing"):
        verify(tmp_path)
    assert set(verify(tmp_path, optional={"b.test.v1.jsonl.gz"})) == {
        "a.test.v1.jsonl.gz",
        "b.test.v1.jsonl.gz",
    }
    (tmp_path / "b.test.v1.jsonl.gz").write_bytes(b"tampered")
    with pytest.raises(FrozenSetChangedError):
        verify(tmp_path, optional={"b.test.v1.jsonl.gz"})  # present: still checked


def test_restoring_writes_only_verified_bytes_and_never_overwrites(tmp_path: Path) -> None:
    import hashlib

    data = b'{"id": "x"}\n'
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / "f.jsonl"
    with pytest.raises(ValueError, match="manifest says"):
        private._put(path, b"other", digest)
    assert not path.exists()
    assert private._put(path, data, digest) == "restored, verified"
    assert private._put(path, data, digest) == "already present, verified"
    path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="left untouched"):
        private._put(path, data, digest)
    assert path.read_bytes() == b"edited"
