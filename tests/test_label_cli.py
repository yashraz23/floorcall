"""The labelling tool, driven end to end with scripted keypresses on a throwaway pool."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from floorcall import cli
from floorcall.config import get_settings
from floorcall.data import labelling
from floorcall.data.escalate import Candidate


def cand(i: int, split: str) -> Candidate:
    return Candidate(
        id=f"c{i}",
        group=f"twcs:{i}",
        split=split,  # type: ignore[arg-type]
        company="BofA_Help",
        prefiltered=bool(i % 2),
        user_partial=f"message {i}",
        agent_last_utterance="" if i % 3 else "an agent reply",
        recent_turns=[],
    )


@pytest.fixture
def labels_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("FLOORCALL_PATHS__LABELS", str(tmp_path))
    get_settings.cache_clear()
    pool = [cand(0, "test"), cand(1, "test"), cand(2, "test"), cand(3, "calib")]
    labelling.save_candidates(tmp_path / cli.ESCALATE_CANDIDATES, pool)
    yield tmp_path
    get_settings.cache_clear()


def run(keys: str, *args: str) -> str:
    result = CliRunner().invoke(cli.app, ["label", "escalate", *args], input=keys)
    assert result.exit_code == 0, result.output
    return result.output


def test_keys_become_labels(labels_dir: Path) -> None:
    run("yns")  # c0 yes, c1 no, c2 skip, then the test queue is empty
    labels = labelling.load_labels(labels_dir / cli.ESCALATE_LABELS)
    assert {k: v["label"] for k, v in labels.items()} == {"c0": "true", "c1": "false", "c2": "skip"}
    assert all(v["labeller"] == "yash" and v["guidelines"] == "v1" for v in labels.values())


def test_undo_and_unknown_keys(labels_dir: Path) -> None:
    run("yxunq")  # c0 yes, x ignored, undo c0, c0 no, quit
    labels = labelling.load_labels(labels_dir / cli.ESCALATE_LABELS)
    assert {k: v["label"] for k, v in labels.items()} == {"c0": "false"}


def test_resumes_where_it_stopped_and_keeps_splits_apart(labels_dir: Path) -> None:
    run("yq")
    run("nq")  # resumes at c1
    run("y", "--split", "calib")
    labels = labelling.load_labels(labels_dir / cli.ESCALATE_LABELS)
    assert {k: v["label"] for k, v in labels.items()} == {"c0": "true", "c1": "false", "c3": "true"}


def test_labeller_sees_no_sampling_information(labels_dir: Path) -> None:
    # (pytest names the tmp dir after the test, and the output prints that path, so this test's
    # name must not contain any of the words it checks for)
    shown = run("q")
    for leak in ("cue/", "plain/", "prefiltered", "stratum"):
        assert leak not in shown
