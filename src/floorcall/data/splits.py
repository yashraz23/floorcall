"""Group-disjoint splitting: every row from one conversation lands in the same split.

Splitting SwDA by utterance would put the same two speakers, the same topic, and the
conversation's own backchannels on both sides of the train/test line, and Table A would measure
memorisation. So the unit of splitting is the group (a SwDA conversation, a Twitter thread), and
`assert_disjoint` is called on every built dataset.

A group's split is a pure function of (seed, group id): the first 8 bytes of
sha256("{seed}:{id}"), read as a uniform number in [0, 1) and cut at the cumulative fractions.
A shuffle-and-slice split moves groups whenever the group list changes. This one never does:
rebuilding with more data cannot migrate an old test conversation into training. The price is that
fractions hold in aggregate, not exactly, which is irrelevant at hundreds of groups.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

Split = Literal["train", "calib", "test"]
SPLITS: tuple[Split, ...] = ("train", "calib", "test")


class SplitLeakError(AssertionError):
    """A group appears in more than one split."""


def _check_fractions(fractions: Mapping[str, float]) -> None:
    if set(fractions) != set(SPLITS):
        raise ValueError(f"fractions must name exactly {SPLITS}, got {sorted(fractions)}")
    if any(v < 0 for v in fractions.values()):
        raise ValueError(f"fractions must be non-negative, got {dict(fractions)}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {total}")


def _unit_interval(seed: int, group_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(group_id: str, *, seed: int, fractions: Mapping[str, float]) -> Split:
    _check_fractions(fractions)
    u = _unit_interval(seed, group_id)
    edge = 0.0
    for name in SPLITS:
        edge += fractions[name]
        if u < edge:
            return name
    return SPLITS[-1]  # u within float error of 1.0


def assign_splits(
    group_ids: Iterable[str], *, seed: int, fractions: Mapping[str, float]
) -> dict[str, Split]:
    _check_fractions(fractions)
    return {g: assign_split(g, seed=seed, fractions=fractions) for g in group_ids}


def assert_disjoint(
    rows: Iterable[Mapping[str, Any]], *, group_key: str, split_key: str = "split"
) -> None:
    """Raise SplitLeakError if any group appears in more than one split."""
    seen: dict[Any, Split] = {}
    for row in rows:
        split = row[split_key]
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        group = row[group_key]
        first = seen.setdefault(group, cast(Split, split))
        if first != split:
            raise SplitLeakError(f"{group_key}={group!r} appears in both {first!r} and {split!r}")
