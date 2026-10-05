"""The model card's numbers come from results files, and the upload folder holds only the release."""

import json
from pathlib import Path

import pytest

from floorcall import release
from floorcall.evaluate import report


def test_the_model_card_matches_the_committed_results() -> None:
    text = release.CARD.read_text(encoding="utf-8")
    assert release.render_card(text) == text, (
        "release/model_card.md differs from results/: run `uv run floorcall release card`"
    )


def test_the_card_declares_the_release_licence_and_base_model() -> None:
    head = release.CARD.read_text(encoding="utf-8").split("---")[1]
    assert "license: cc-by-nc-sa-4.0" in head  # D-048
    assert "base_model: convaiinnovations/laya" in head


def test_the_card_loads_the_repo_it_is_uploaded_to() -> None:
    card = release.CARD.read_text(encoding="utf-8")
    assert f'snapshot_download("{release.HF_REPO_ID}")' in card


def test_laya_licence_text_is_apache_2() -> None:
    text = release.LAYA_LICENCE.read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_a_paired_table_d_file_fills_its_cell(tmp_path: Path) -> None:
    diff = {"difference": -0.0123, "ci95": [-0.02, -0.004], "share_a_not_better": 1.0}
    payload = {"differences": {"macro_f1": diff}}
    (tmp_path / "route.no_history_vs_full.json").write_text(json.dumps(payload))
    row = next(
        line
        for line in report.table_d_paired(tmp_path).splitlines()
        if "without recent_turns" in line
    )
    assert "-0.012 [-0.020, -0.004]" in row
    assert "TODO" in row  # the other decisions have no file


def _fake_checkpoint(root: Path) -> Path:
    ckpt = root / "ckpt"
    for rel in release.RELEASE_FILES:
        (ckpt / rel).parent.mkdir(parents=True, exist_ok=True)
        (ckpt / rel).write_text("x")
    (ckpt / "run.json").write_text("{}")
    (ckpt / "train_log.jsonl").write_text("{}")
    (ckpt / "checkpoint_epoch1").mkdir()
    (ckpt / "checkpoint_epoch1" / "model.safetensors").write_text("x")
    return ckpt


def test_staging_copies_only_the_release_files_and_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _fake_checkpoint(tmp_path)
    monkeypatch.setattr(release, "check_checkpoint", lambda checkpoint: [])
    monkeypatch.setattr(release, "check_card", list)
    hashes = release.stage(tmp_path / "out", checkpoint=ckpt)
    assert set(hashes) == {*release.RELEASE_FILES, "README.md", release.LAYA_LICENCE.name}


def test_staging_refuses_a_checkpoint_table_a_did_not_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _fake_checkpoint(tmp_path)
    monkeypatch.setattr(release, "check_checkpoint", lambda checkpoint: ["wrong run"])
    monkeypatch.setattr(release, "check_card", list)
    with pytest.raises(ValueError, match="wrong run"):
        release.stage(tmp_path / "out", checkpoint=ckpt)
    assert not (tmp_path / "out").exists()
