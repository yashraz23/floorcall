"""The report PDF (D-051): every number from a committed file, TODO when one is missing."""

import json
from datetime import UTC, datetime
from pathlib import Path

from floorcall import report_pdf


def _text(blocks: list[report_pdf.Block]) -> str:
    return "\n".join(b.text + " ".join(b.items) + json.dumps(b.rows) for b in blocks)


def test_the_report_has_no_todo_on_the_committed_results() -> None:
    text = _text(report_pdf.content("0" * 40, "2026-10-05"))
    assert report_pdf.TODO not in text
    for heading in ("Summary", "Data", "Results", "Replay", "Limitations", "Reproducing"):
        assert heading in text


def test_a_missing_number_prints_as_todo(tmp_path: Path) -> None:
    assert report_pdf._num(tmp_path / "absent.json", "metrics", "macro_f1") == report_pdf.TODO
    (tmp_path / "row.json").write_text(json.dumps({"metrics": {"macro_f1": 0.5}}))
    assert report_pdf._num(tmp_path / "row.json", "metrics", "macro_f1") == "0.500"
    assert report_pdf._num(tmp_path / "row.json", "metrics", "ece") == report_pdf.TODO
    assert report_pdf._ci(tmp_path / "row.json", "macro_f1") == "0.500"  # no interval stored


def test_the_summary_numbers_are_the_results_files(tmp_path: Path) -> None:
    row = json.loads(
        (report_pdf.RESULTS / "table_a" / "barge_in.finetuned_temp.json").read_text("utf-8")
    )
    text = _text(report_pdf.content("0" * 40, "2026-10-05"))
    assert f"D2 barge_in: macro-F1 {row['metrics']['macro_f1']:.3f}" in text


def test_the_same_commit_gives_the_same_bytes(tmp_path: Path) -> None:
    blocks = report_pdf.content("0" * 40, "2026-10-05")
    when = datetime(2026, 10, 5, tzinfo=UTC)
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    pages = report_pdf.render(blocks, a, when)
    report_pdf.render(blocks, b, when)
    assert a.read_bytes()[:5] == b"%PDF-" and pages >= 5
    assert a.read_bytes() == b.read_bytes()
