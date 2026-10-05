"""The committed test sets: intact, well-formed, real, and disjoint from training.

Runs in CI on the files in data/test_frozen/. The disjointness check against train and calib needs
the processed files (`uv run floorcall data build`) and is skipped where they are absent.
"""

from pathlib import Path
from typing import Any

import pytest

from floorcall import questions
from floorcall.config import get_settings
from floorcall.data.build import processed_file
from floorcall.data.freeze import read_jsonl_gz, verify
from floorcall.data.private import TEST_FILES as PRIVATE_TEST_FILES

FROZEN = get_settings().paths.test_frozen
PROCESSED = get_settings().paths.data_processed

LABELS = {
    "turn_complete": {"true", "false"},
    "barge_in": set(questions.BARGE_IN_LABELS),
    "route": set(questions.ROUTE_LABELS),
    "escalate": {"true", "false"},
}
EVENTS = {
    "turn_complete": questions.Event.USER_PAUSE,
    "barge_in": questions.Event.USER_SPEECH_DURING_AGENT,
    "route": questions.Event.USER_PAUSE,
}
# Synthetic rows may train D4; they never enter a test set (CLAUDE.md §10.3).
REAL_SOURCES = {"swda", "clinc150", "twcs"}


def frozen_files() -> list[Path]:
    return sorted(FROZEN.glob("*.test.v*.jsonl.gz"))


def test_manifest_matches_every_file() -> None:
    # D4's files hold real Twitter text and may be absent from the public copy (D-049)
    manifest = verify(FROZEN, optional=PRIVATE_TEST_FILES)
    absent = {n for n in PRIVATE_TEST_FILES if not (FROZEN / n).exists()}
    assert set(manifest) - absent == {p.name for p in frozen_files()}
    assert manifest, "no frozen test sets"


@pytest.fixture(scope="module", params=frozen_files(), ids=lambda p: p.name)
def frozen(request: pytest.FixtureRequest) -> tuple[str, list[dict[str, Any]]]:
    path: Path = request.param
    return path.name.split(".")[0], read_jsonl_gz(path)


def test_rows_are_well_formed(frozen: tuple[str, list[dict[str, Any]]]) -> None:
    decision, rows = frozen
    assert rows
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        assert r["decision"] == decision
        assert r["split"] == "test"
        assert r["label"] in LABELS[decision], r["id"]
        if decision in EVENTS:
            assert r["event"] == EVENTS[decision].value
        s = r["snapshot"]
        assert set(s) == {"agent_speaking", "user_partial", "agent_last_utterance", "recent_turns"}
        assert isinstance(s["agent_speaking"], bool)
        assert all(t["speaker"] in ("user", "agent") for t in s["recent_turns"])


def test_no_synthetic_rows(frozen: tuple[str, list[dict[str, Any]]]) -> None:
    _, rows = frozen
    assert {r["source"] for r in rows} <= REAL_SOURCES


def test_every_class_is_present(frozen: tuple[str, list[dict[str, Any]]]) -> None:
    decision, rows = frozen
    assert {r["label"] for r in rows} == LABELS[decision]


def test_disjoint_from_train_and_calib(frozen: tuple[str, list[dict[str, Any]]]) -> None:
    decision, rows = frozen
    paths = [PROCESSED / processed_file(decision, s) for s in ("train", "calib")]
    if not all(p.exists() for p in paths):
        pytest.skip("processed train/calib not built here (uv run floorcall data build)")
    test_groups = {r["group"] for r in rows}
    for p in paths:
        other = {r["group"] for r in read_jsonl_gz(p)}
        assert not (test_groups & other), f"{p.name} shares groups with the test set"
