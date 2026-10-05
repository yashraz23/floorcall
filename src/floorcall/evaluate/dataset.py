"""Evaluation sets: frozen test rows (or processed train/calib rows) as arrays ready to score."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall import questions
from floorcall.config import Settings, StateSettings
from floorcall.data.build import processed_file, test_file
from floorcall.data.freeze import read_jsonl_gz
from floorcall.state import Snapshot, Turn, pack_state


def option_labels(decision: str) -> tuple[str, ...]:
    """Class order for a decision: the order the model scores options in."""
    q = questions.ALL_QUESTIONS[decision]
    if q["type"] == "noul":
        return ("false", "true")
    return tuple(q["criteria"].keys())


def snapshot_of(row: dict[str, Any]) -> Snapshot:
    s = row["snapshot"]
    return Snapshot(
        agent_speaking=s["agent_speaking"],
        user_partial=s["user_partial"],
        agent_last_utterance=s["agent_last_utterance"],
        recent_turns=tuple(Turn(t["speaker"], t["text"]) for t in s["recent_turns"]),
    )


@dataclass
class EvalSet:
    decision: str
    split: str
    labels: tuple[str, ...]
    rows: list[dict[str, Any]]
    y: npt.NDArray[np.int64]
    hard: npt.NDArray[np.bool_]

    @property
    def event(self) -> questions.Event:
        (event,) = {r["event"] for r in self.rows}
        return questions.Event(event)

    def packed_states(
        self, *, state: StateSettings, budget: int, count_tokens: Callable[[str], int]
    ) -> list[dict[str, Any]]:
        """Every row packed as Decider.pack packs it, with the same StateSettings."""
        return [
            pack_state(
                snapshot_of(r),
                budget=budget,
                count_tokens=count_tokens,
                normalize_text=state.normalize,
                include_history=state.include_history,
                include_agent=state.include_agent,
            ).state
            for r in self.rows
        ]


def _make(decision: str, split: str, rows: Sequence[dict[str, Any]]) -> EvalSet:
    labels = option_labels(decision)
    index = {lab: i for i, lab in enumerate(labels)}
    return EvalSet(
        decision=decision,
        split=split,
        labels=labels,
        rows=list(rows),
        y=np.array([index[r["label"]] for r in rows], dtype=np.int64),
        hard=np.array([bool(r["hard"]) for r in rows], dtype=np.bool_),
    )


def load_test(settings: Settings, decision: str) -> EvalSet:
    from floorcall.data.private import require

    path = require(settings.paths.test_frozen / test_file(decision), settings)  # D4: D-049
    return _make(decision, "test", read_jsonl_gz(path))


def load_processed(settings: Settings, decision: str, split: str) -> EvalSet:
    path = settings.paths.data_processed / processed_file(decision, split)
    return _make(decision, split, read_jsonl_gz(path))


def class_prior(train: EvalSet) -> npt.NDArray[np.float64]:
    counts = np.bincount(train.y, minlength=len(train.labels)).astype(np.float64)
    prior: npt.NDArray[np.float64] = counts / counts.sum()
    return prior
