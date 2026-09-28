"""Conversation-disjoint splits. Written before floorcall/data/splits.py (CLAUDE.md §10.1, §15).

Splitting SwDA by utterance would put the same two speakers, the same topic and near-duplicate
backchannels on both sides of the train/test line. Every number in Table A depends on these tests.
"""

import pytest

from floorcall.data.splits import (
    SPLITS,
    SplitLeakError,
    assert_disjoint,
    assign_split,
    assign_splits,
)

FRACTIONS = {"train": 0.7, "calib": 0.1, "test": 0.2}


def test_deterministic() -> None:
    ids = [f"conv{i}" for i in range(500)]
    assert assign_splits(ids, seed=1, fractions=FRACTIONS) == assign_splits(
        ids, seed=1, fractions=FRACTIONS
    )


def test_seed_changes_the_assignment() -> None:
    ids = [f"conv{i}" for i in range(500)]
    a = assign_splits(ids, seed=1, fractions=FRACTIONS)
    b = assign_splits(ids, seed=2, fractions=FRACTIONS)
    assert sum(a[i] != b[i] for i in ids) > 100


def test_a_group_ignores_every_other_group() -> None:
    # Adding conversations must never move an existing one to another split: a rebuilt dataset
    # with more data cannot silently leak an old test conversation into training.
    small = assign_splits([f"conv{i}" for i in range(100)], seed=7, fractions=FRACTIONS)
    large = assign_splits([f"conv{i}" for i in range(10_000)], seed=7, fractions=FRACTIONS)
    assert all(large[k] == v for k, v in small.items())
    assert all(assign_split(k, seed=7, fractions=FRACTIONS) == v for k, v in small.items())


def test_fractions_are_respected_in_aggregate() -> None:
    ids = [f"conv{i}" for i in range(20_000)]
    a = assign_splits(ids, seed=3, fractions=FRACTIONS)
    for name, frac in FRACTIONS.items():
        share = sum(v == name for v in a.values()) / len(ids)
        assert abs(share - frac) < 0.015, (name, share)


def test_only_known_splits() -> None:
    a = assign_splits([f"c{i}" for i in range(1000)], seed=0, fractions=FRACTIONS)
    assert set(a.values()) <= set(SPLITS)


@pytest.mark.parametrize(
    "bad",
    [
        {"train": 0.7, "calib": 0.1, "test": 0.1},  # sums to 0.9
        {"train": 0.8, "test": 0.2},  # missing calib
        {"train": 0.7, "calib": 0.1, "test": 0.2, "dev": 0.0},  # unknown split
        {"train": 1.1, "calib": -0.1, "test": 0.0},  # negative
    ],
)
def test_rejects_bad_fractions(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        assign_split("x", seed=0, fractions=bad)


def test_assert_disjoint_passes_on_group_assigned_rows() -> None:
    groups = assign_splits([f"conv{i}" for i in range(50)], seed=0, fractions=FRACTIONS)
    rows = [
        {"conversation_id": g, "split": s, "utt": u} for g, s in groups.items() for u in range(5)
    ]
    assert_disjoint(rows, group_key="conversation_id")


def test_assert_disjoint_catches_a_leak() -> None:
    rows = [
        {"conversation_id": "conv1", "split": "train"},
        {"conversation_id": "conv1", "split": "train"},
        {"conversation_id": "conv2", "split": "test"},
        {"conversation_id": "conv1", "split": "test"},  # the leak
    ]
    with pytest.raises(SplitLeakError, match="conv1"):
        assert_disjoint(rows, group_key="conversation_id")


def test_assert_disjoint_rejects_unknown_split_names() -> None:
    with pytest.raises(ValueError, match="dev"):
        assert_disjoint([{"conversation_id": "a", "split": "dev"}], group_key="conversation_id")
