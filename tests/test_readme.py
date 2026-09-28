"""README numbers come from results files, never from a keyboard."""

import json
from pathlib import Path

from floorcall.evaluate import report


def test_readme_matches_the_committed_results() -> None:
    text = report.README.read_text(encoding="utf-8")
    assert report.render_readme(text) == text, (
        "README.md differs from results/: run `uv run floorcall eval readme`"
    )


def test_missing_results_render_as_todo(tmp_path: Path) -> None:
    table = report.table_a(tmp_path)
    assert "TODO" in table
    assert not any(ch.isdigit() for line in table.splitlines()[2:] for ch in line.split("|", 3)[3])


def test_a_results_file_fills_its_row(tmp_path: Path) -> None:
    metrics = {
        "accuracy": 0.5,
        "accuracy_ci95": [0.4, 0.6],
        "macro_f1": 0.45,
        "ece": 0.1,
        "brier": 0.5,
        "hard_accuracy": None,
        "hard_n": 0,
    }
    (tmp_path / "route.majority.json").write_text(json.dumps({"metrics": metrics}))
    row = next(
        line for line in report.table_a(tmp_path).splitlines() if "D3 route | majority" in line
    )
    assert "0.500 [0.400, 0.600] | 0.450 | 0.100 | 0.500 | n/a" in row
