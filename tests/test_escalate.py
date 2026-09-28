"""D4 candidate pool and hand-label handling.

tests/fixtures/twcs/twcs.csv:
  1  customer -> BofA   "my card was declined twice today"            opener
  2  BofA               "Sorry to hear that. Please DM us ... ^MG url"
  3  customer -> BofA   "I already DMed you and nobody is helping!!"  reply, escalation cue
  4/5  Amazon thread                                                    not a bank: excluded
  6  customer -> PayPal "Hola no funciona ..." (+9 PayPal reply)        not English: excluded
  7  customer -> Chase  "thanks, that fixed it"  (+8 Chase reply)       opener
"""

from pathlib import Path

import pytest

from floorcall.data import escalate, labelling
from floorcall.data.escalate import Candidate

CSV = Path(__file__).parent / "fixtures" / "twcs" / "twcs.csv"
FRACTIONS = {"train": 0.7, "calib": 0.1, "test": 0.2}


@pytest.mark.parametrize(
    ("raw", "agent", "clean"),
    [
        (
            "@111 Sorry to hear that. Please DM us your zip &amp; name. ^MG https://t.co/x",
            True,
            "Sorry to hear that. Please DM us your zip & name.",
        ),
        ("@444 Glad to help! ^JK", True, "Glad to help!"),
        ("Try clearing your cache. ^Clarissa", True, "Try clearing your cache."),
        ("@BofA_Help this is *terrible", False, "this is *terrible"),  # customer text keeps it
        (
            "@BofA_Help I already DMed you and nobody is helping!!",
            False,
            "I already DMed you and nobody is helping!!",
        ),
        ("call me at __phone__ please", False, "call me at please"),
        ("@a @b   spaced   out", False, "spaced out"),
    ],
)
def test_clean_tweet(raw: str, agent: bool, clean: str) -> None:
    assert escalate.clean_tweet(raw, agent=agent) == clean


def test_looks_english() -> None:
    assert escalate.looks_english("my card was declined twice today")
    assert not escalate.looks_english("Hola no funciona mi cuenta por favor")
    assert not escalate.looks_english("")


@pytest.fixture(scope="module")
def cands() -> dict[str, Candidate]:
    out = escalate.build_candidates(CSV, seed=0, fractions=FRACTIONS, max_history=8)
    return {c.id: c for c in out}


def test_only_english_customer_messages_in_bank_threads(cands: dict[str, Candidate]) -> None:
    assert set(cands) == {"twcs-1", "twcs-3", "twcs-7"}


def test_context_and_roles(cands: dict[str, Candidate]) -> None:
    reply = cands["twcs-3"]
    assert reply.company == "BofA_Help"
    assert reply.agent_last_utterance == "Sorry to hear that. Please DM us your zip & name."
    assert reply.recent_turns == [{"speaker": "user", "text": "my card was declined twice today"}]
    opener = cands["twcs-1"]
    assert opener.agent_last_utterance == ""
    assert opener.recent_turns == []


def test_a_thread_is_one_group_in_one_split(cands: dict[str, Candidate]) -> None:
    assert cands["twcs-1"].group == cands["twcs-3"].group == "twcs:1"
    assert cands["twcs-1"].split == cands["twcs-3"].split


def test_cue_flag_is_recorded(cands: dict[str, Candidate]) -> None:
    assert cands["twcs-3"].prefiltered
    assert not cands["twcs-7"].prefiltered
    assert escalate.stratum(cands["twcs-3"]) == "cue/reply"
    assert escalate.stratum(cands["twcs-7"]) == "plain/opener"


def fake(i: int, split: str, cue: bool, reply: bool, thread: int | None = None) -> Candidate:
    return Candidate(
        id=f"c{i}",
        group=f"twcs:{thread if thread is not None else i}",
        split=split,  # type: ignore[arg-type]
        company="BofA_Help",
        prefiltered=cue,
        user_partial="message",
        agent_last_utterance="reply" if reply else "",
        recent_turns=[],
    )


def test_selection_is_stratified_and_one_per_thread() -> None:
    pool = [
        fake(i, "test", cue=bool(i % 2), reply=bool((i // 2) % 2), thread=i // 8)
        for i in range(800)
    ]
    chosen = escalate.select_for_labelling(pool, split="test", n=40, seed=1)
    assert len(chosen) == 40
    assert len({c.group for c in chosen}) == 40
    counts = {s: sum(escalate.stratum(c) == s for c in chosen) for s in escalate.STRATA}
    assert counts == dict.fromkeys(escalate.STRATA, 10)


def test_selection_refuses_an_underfilled_stratum() -> None:
    pool = [fake(i, "test", cue=False, reply=False) for i in range(100)]
    with pytest.raises(ValueError, match="need"):
        escalate.select_for_labelling(pool, split="test", n=8, seed=1)


# -- labels ----------------------------------------------------------------------------------


def test_label_round_trip_and_skips(tmp_path: Path) -> None:
    pool = [
        fake(0, "test", True, True),
        fake(1, "test", False, False),
        fake(2, "calib", True, False),
    ]
    path = tmp_path / "labels.jsonl"
    labels = {
        "c0": labelling.label_record("c0", "true", "yash"),
        "c1": labelling.label_record("c1", "skip", "yash"),
        "c2": labelling.label_record("c2", "false", "yash"),
    }
    labelling.save_labels(path, labels, [c.id for c in pool])
    loaded = labelling.load_labels(path)
    assert loaded == labels
    test_rows = labelling.to_rows(pool, loaded, "test")
    assert [r["id"] for r in test_rows] == ["c0"]  # the skip is dropped
    assert test_rows[0]["label"] == "true"
    assert test_rows[0]["kind"] == "cue/reply"
    assert test_rows[0]["hard"] is False
    assert [r["id"] for r in labelling.to_rows(pool, loaded, "calib")] == ["c2"]


def test_label_values_are_checked() -> None:
    with pytest.raises(ValueError):
        labelling.label_record("c0", "maybe", "yash")


def test_labels_from_other_guidelines_are_refused() -> None:
    pool = [fake(0, "test", True, True)]
    rec = labelling.label_record("c0", "true", "yash")
    rec["guidelines"] = "v0"
    with pytest.raises(ValueError, match="guidelines"):
        labelling.to_rows(pool, {"c0": rec}, "test")


def test_the_pool_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    labelling.save_candidates(path, [fake(0, "test", True, True)])
    assert [c.id for c in labelling.load_candidates(path)] == ["c0"]
    with pytest.raises(FileExistsError):
        labelling.save_candidates(path, [fake(1, "test", True, True)])
