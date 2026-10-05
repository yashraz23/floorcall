"""Guideline v2 relabel (D-033): the verbatim guideline, the seeded sample, and the blind tool."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from floorcall import cli
from floorcall.config import REPO_ROOT, get_settings
from floorcall.data import labelling
from floorcall.data.escalate import STRATA, Candidate, stratum

# Yash's guideline v2, as he gave it.
GUIDELINE_V2_LINES = [
    "Escalate (y) if any of: (1) Asks for a human, a call, or a manager, or says they can't reach or "
    "get help from support. (2) Says they already raised it (called, DM'd, visited, told a rep) and "
    "it's still unresolved or past a promised time. (3) Says they'll leave, close the account, switch "
    "banks, or take it public/viral.",
    "Not escalate (n): venting or insults with no prior contact and no threat to leave; a first "
    'report of a problem, even an angry one; questions, thanks, "sent you a DM"; fraud or phishing '
    "reports; general sarcasm about the company.",
    "If unsure: apply the three rules literally. If none clearly fits, n.",
]


def test_guideline_v2_is_verbatim_in_the_code_and_the_doc() -> None:
    assert labelling.GUIDELINE_V2.split("\n") == GUIDELINE_V2_LINES
    doc_lines = (
        (REPO_ROOT / "docs" / "labelling-escalate.md").read_text(encoding="utf-8").split("\n")
    )
    for line in GUIDELINE_V2_LINES:
        assert line in doc_lines


def test_v2_labels_have_no_skip() -> None:
    assert labelling.label_record("a", "true", "yash", guidelines="v2")["guidelines"] == "v2"
    with pytest.raises(ValueError, match="v2"):
        labelling.label_record("a", "skip", "yash", guidelines="v2")


# -- the sample ------------------------------------------------------------------------------


def cand(i: int, split: str, cue: bool, reply: bool) -> Candidate:
    return Candidate(
        id=f"c{i:04d}",
        group=f"twcs:{i}",
        split=split,  # type: ignore[arg-type]
        company="BofA_Help",
        prefiltered=cue,
        user_partial=f"message {i}",
        agent_last_utterance="an agent reply" if reply else "",
        recent_turns=[],
    )


def pool(per_cell: int = 30) -> list[Candidate]:
    out, i = [], 0
    for split in ("test", "calib"):
        for cue in (True, False):
            for reply in (True, False):
                for _ in range(per_cell):
                    out.append(cand(i, split, cue, reply))
                    i += 1
    return out


def v1_labels(cands: list[Candidate], flip: bool = False) -> dict[str, dict[str, Any]]:
    out = {}
    for n, c in enumerate(cands):
        label = "skip" if n % 10 == 0 else ("true" if (n % 3 == 0) != flip else "false")
        out[c.id] = {"id": c.id, "label": label}
    return out


SIZES = {"test": 40, "calib": 20}


def test_allocation_is_proportional_and_exact() -> None:
    assert labelling.allocate({"a": 50, "b": 50}, 20) == {"a": 10, "b": 10}
    got = labelling.allocate({"a": 49, "b": 50, "c": 47, "d": 49}, 100)
    assert sum(got.values()) == 100 and got == {"a": 25, "b": 26, "c": 24, "d": 25}
    with pytest.raises(ValueError):
        labelling.allocate({"a": 3}, 4)


def test_the_sample_is_stratified_from_the_v1_eval_sets() -> None:
    cands = pool()
    labels = v1_labels(cands)
    sample = labelling.draw_relabel_sample(cands, labels, SIZES, seed=7)
    by_id = {c.id: c for c in cands}
    for split, n in SIZES.items():
        ids = sample["ids"][split]
        assert len(ids) == len(set(ids)) == n
        assert all(by_id[i].split == split and labels[i]["label"] != "skip" for i in ids)
        assert Counter(stratum(by_id[i]) for i in ids) == Counter(sample["allocation"][split])
        assert set(sample["allocation"][split]) == set(STRATA)
    assert sorted(sample["order"]) == sorted(sample["ids"]["test"] + sample["ids"]["calib"])


def test_the_sample_depends_on_the_seed_and_not_on_label_values() -> None:
    cands = pool()
    a = labelling.draw_relabel_sample(cands, v1_labels(cands), SIZES, seed=7)
    assert labelling.draw_relabel_sample(cands, v1_labels(cands), SIZES, seed=7) == a
    assert labelling.draw_relabel_sample(cands, v1_labels(cands, flip=True), SIZES, seed=7) == a
    assert labelling.draw_relabel_sample(cands, v1_labels(cands), SIZES, seed=8) != a


def test_a_drawn_sample_is_never_redrawn(tmp_path: Path) -> None:
    cands = pool()
    path = tmp_path / "sample.json"
    a = labelling.draw_relabel_sample(cands, v1_labels(cands), SIZES, seed=7)
    assert labelling.save_relabel_sample(path, a) is True
    assert labelling.save_relabel_sample(path, a) is False  # same sample: nothing to do
    b = labelling.draw_relabel_sample(cands, v1_labels(cands), SIZES, seed=8)
    with pytest.raises(FileExistsError):
        labelling.save_relabel_sample(path, b)


# -- the blind tool --------------------------------------------------------------------------


@pytest.fixture
def labels_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("FLOORCALL_PATHS__LABELS", str(tmp_path))
    get_settings.cache_clear()
    cands = pool(per_cell=5)
    labelling.save_candidates(tmp_path / cli.ESCALATE_CANDIDATES, cands)
    v1 = {c.id: {"id": c.id, "label": "true", "labeller": "V1-SENTINEL"} for c in cands}
    (tmp_path / cli.ESCALATE_LABELS).write_text(
        "".join(json.dumps(r) + "\n" for r in v1.values()), encoding="utf-8"
    )
    sample = labelling.draw_relabel_sample(cands, v1, {"test": 4, "calib": 2}, seed=3)
    labelling.save_relabel_sample(tmp_path / labelling.RELABEL_SAMPLE_FILE, sample)
    yield tmp_path
    get_settings.cache_clear()


def relabel(keys: str) -> str:
    result = CliRunner().invoke(cli.app, ["label", "escalate-relabel"], input=keys)
    assert result.exit_code == 0, result.output
    return result.output


def v2(labels_dir: Path) -> dict[str, dict[str, Any]]:
    return labelling.load_labels(labels_dir / labelling.LABELS_V2_FILE)


def test_relabel_follows_the_shuffled_order_and_writes_v2_only(labels_dir: Path) -> None:
    v1_before = (labels_dir / cli.ESCALATE_LABELS).read_bytes()
    order = json.loads((labels_dir / labelling.RELABEL_SAMPLE_FILE).read_text())["order"]
    relabel("ynsyq")  # y, n, s ignored (v2 has no skip), y, quit
    got = v2(labels_dir)
    assert [got[i]["label"] for i in order[:3]] == ["true", "false", "true"]
    assert all(r["guidelines"] == "v2" and r["labeller"] == "yash" for r in got.values())
    assert (labels_dir / cli.ESCALATE_LABELS).read_bytes() == v1_before


def test_relabel_resumes_and_undoes(labels_dir: Path) -> None:
    order = json.loads((labels_dir / labelling.RELABEL_SAMPLE_FILE).read_text())["order"]
    relabel("yq")
    relabel("yunq")  # resumes at the second message, labels it y, undoes, labels it n
    got = v2(labels_dir)
    assert {i: r["label"] for i, r in got.items()} == {order[0]: "true", order[1]: "false"}


def test_relabel_shows_the_guideline_and_hides_everything_else(labels_dir: Path) -> None:
    # (pytest names the tmp dir after the test and the output prints that path, so this test's
    # name must avoid every word it checks for)
    shown = " ".join(relabel("q").replace("│", " ").replace("|", " ").split())  # panel borders
    assert "guideline v2" in shown and "If none clearly fits, n." in shown
    for leak in ("V1-SENTINEL", "calib", "cue/", "plain/", "prefiltered", "stratum", "skip"):
        assert leak not in shown


def test_relabel_finishes_when_every_message_is_labelled(labels_dir: Path) -> None:
    assert "All 6 messages are labelled" in relabel("nnnnnn")
    assert len(v2(labels_dir)) == 6


# -- the v2 eval sets ------------------------------------------------------------------------


def test_d4_is_evaluated_on_v2_and_the_rest_on_v1() -> None:
    from floorcall.data.build import test_file

    assert test_file("escalate") == "escalate.test.v2.jsonl.gz"
    assert test_file("turn_complete") == "turn_complete.test.v1.jsonl.gz"
    assert test_file("escalate", "v1") == "escalate.test.v1.jsonl.gz"


def test_d4_v2_sets_rebuild_from_the_committed_labels() -> None:
    import hashlib

    from floorcall.data import private

    if private.missing(get_settings()):
        pytest.skip("D4's Twitter text is private (D-049): `floorcall data restore-escalate`")

    from floorcall.data.build import cards_dir, escalate_eval_rows, test_file
    from floorcall.data.freeze import jsonl_gz_bytes

    s = get_settings()
    test, calib = escalate_eval_rows(s, "v2")
    frozen = (s.paths.test_frozen / test_file("escalate")).read_bytes()
    assert jsonl_gz_bytes(sorted(test, key=lambda r: r["id"])) == frozen
    card = json.loads((cards_dir(s) / "escalate.json").read_text(encoding="utf-8"))
    assert card["guidelines"] == "v2"
    assert hashlib.sha256(jsonl_gz_bytes(calib)).hexdigest() == card["calib_sha256"]
    sample = json.loads((s.paths.labels / labelling.RELABEL_SAMPLE_FILE).read_text("utf-8"))
    assert sorted(r["id"] for r in test) == sample["ids"]["test"]
    assert sorted(r["id"] for r in calib) == sample["ids"]["calib"]
    assert {r["guidelines"] for r in test + calib} == {"v2"}


def test_guideline_versions_never_mix_in_one_set() -> None:
    c = cand(1, "test", cue=True, reply=False)
    with pytest.raises(ValueError, match="guidelines v1"):
        labelling.to_rows(
            [c], {c.id: {"label": "true", "guidelines": "v1"}}, "test", guidelines="v2"
        )


# -- the primary D4 model's training sample (D-034) ------------------------------------------


def test_the_train_sample_is_seeded_and_train_only() -> None:
    train = [c for c in pool() if c.split == "test"]  # reuse the fixture pool, relabelled train
    train = [Candidate(**{**c.to_json(), "split": "train"}) for c in train]
    a = labelling.draw_train_sample(train, set(), n=50, seed=1)
    assert len(a) == len({c.id for c in a}) == 50
    assert labelling.draw_train_sample(train, set(), n=50, seed=1) == a
    assert labelling.draw_train_sample(train, set(), n=50, seed=2) != a
    assert [c.id for c in a] != sorted(c.id for c in a)  # shown shuffled


def test_the_train_sample_refuses_eval_threads_and_other_splits() -> None:
    cands = pool()
    train = [Candidate(**{**c.to_json(), "split": "train"}) for c in cands[:20]]
    with pytest.raises(ValueError, match="eval set"):
        labelling.draw_train_sample(train, {train[3].group}, n=5, seed=1)
    with pytest.raises(ValueError, match="not train-split"):
        labelling.draw_train_sample(cands[:20], set(), n=5, seed=1)


def test_train_labelling_saves_every_key_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOORCALL_PATHS__LABELS", str(tmp_path))
    get_settings.cache_clear()
    try:
        sample = [Candidate(**{**c.to_json(), "split": "train"}) for c in pool(per_cell=1)]
        labelling.save_candidates(tmp_path / labelling.TRAIN_SAMPLE_FILE, sample)
        for keys in ("yq", "nyq"):  # stop after one, then resume at the second
            result = CliRunner().invoke(cli.app, ["label", "escalate-train"], input=keys)
            assert result.exit_code == 0, result.output
        got = labelling.load_labels(tmp_path / labelling.TRAIN_LABELS_FILE)
        assert [got[c.id]["label"] for c in sample[:3]] == ["true", "false", "true"]
        assert {r["guidelines"] for r in got.values()} == {"v2"}
    finally:
        get_settings.cache_clear()
