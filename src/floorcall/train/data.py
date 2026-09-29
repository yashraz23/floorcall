"""Training rows: every processed train row, packed exactly as it will be served, as Laya items.

A state is packed with its event's token budget and the same StateSettings (normalization,
ablation switches) that evaluation and serving use, so the model trains on the input distribution
it is tested on. Token ids are held as int32 arrays; as Python int lists, ~95k rows of up to 512
ids would take well over a gigabyte.

Each epoch draws a fixed-size mixture per task (`TrainSettings.rows_per_epoch`): the raw train sets
differ by 30x (D1 49k rows, D3 1.75k), and without a mixture the multi-task checkpoint would be a D1
model that also saw some routing. A task with more rows than its quota is subsampled without
replacement each epoch (so every epoch sees different rows); a task with fewer is repeated whole
and topped up with a sample. Seeded, so a run is reproducible.

A task named in `balance` (D4, D-035) splits its quota evenly across its classes, and each class
is drawn by the same rule. D4's training pool follows the natural mix while its calib and test sets
were drawn in equal strata; temperature scaling cannot move a prior, so the balance is set here.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from floorcall import questions
from floorcall.config import StateSettings
from floorcall.evaluate.dataset import EvalSet

Item = dict[str, Any]


class ItemBuilder(Protocol):
    def state_room(self, questions: Mapping[str, Mapping[str, Any]]) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    def training_item(
        self, state: Mapping[str, Any], qid: str, qdef: Mapping[str, Any], target: Sequence[float]
    ) -> Item: ...


def event_budget(builder: ItemBuilder, event: questions.Event, state: StateSettings) -> int:
    return builder.state_room(questions.questions_for(event)) - state.safety_margin_tokens


def build_items(
    builder: ItemBuilder,
    data: EvalSet,
    state: StateSettings,
    *,
    label_smoothing: float = 0.0,
    progress: Callable[[int, int], None] | None = None,
) -> list[Item]:
    """One training item per row: the packed state, this decision's question, a gold distribution.

    The gold is one-hot (optionally smoothed): SwDA, CLINC and hand labels are hard labels, and the
    proper-scoring-rule reward accepts either.
    """
    decision = data.decision
    qdef = questions.questions_by_id(decision)[decision]
    k = len(data.labels)
    budget = event_budget(builder, data.event, state)
    states = data.packed_states(state=state, budget=budget, count_tokens=builder.count_tokens)
    items = []
    for i, (st, y) in enumerate(zip(states, data.y, strict=True)):
        target = [label_smoothing / k] * k
        target[int(y)] += 1.0 - label_smoothing
        item = builder.training_item(st, decision, qdef, target)
        item["ids"] = np.asarray(item["ids"], dtype=np.int32)
        item["decision"] = decision
        items.append(item)
        if progress and (i + 1) % 5000 == 0:
            progress(i + 1, len(states))
    return items


@dataclass(frozen=True)
class MixtureReport:
    rows: dict[str, int]
    repeats: dict[str, float]  # quota / available: > 1 means rows are repeated
    by_class: dict[str, dict[int, int]]  # balanced tasks only: rows drawn per class


def _draw(pool: Sequence[Item], quota: int, rng: random.Random) -> list[Item]:
    """`quota` rows: the pool repeated whole as often as it fits, topped up with a sample."""
    whole, rest = divmod(quota, len(pool))
    return list(pool) * whole + rng.sample(list(pool), rest)


def epoch_mixture(
    items_by_task: Mapping[str, Sequence[Item]],
    rows_per_epoch: Mapping[str, int],
    rng: random.Random,
    *,
    balance: Sequence[str] = (),
) -> tuple[list[Item], MixtureReport]:
    """One epoch's rows: each task's quota (class-balanced for tasks in `balance`), shuffled."""
    missing = set(rows_per_epoch) - set(items_by_task)
    if missing:
        raise ValueError(f"no training rows for {sorted(missing)}")
    out: list[Item] = []
    rows, repeats, by_class = {}, {}, {}
    for task in sorted(rows_per_epoch):
        pool = list(items_by_task[task])
        quota = rows_per_epoch[task]
        if not pool or quota <= 0:
            raise ValueError(f"{task}: {len(pool)} rows, quota {quota}")
        if task in balance:
            classes: dict[int, list[Item]] = {}
            for item in pool:
                classes.setdefault(int(item["label"]), []).append(item)
            share, extra = divmod(quota, len(classes))
            counts = {c: share + (i < extra) for i, c in enumerate(sorted(classes))}
            chosen = [x for c in sorted(classes) for x in _draw(classes[c], counts[c], rng)]
            by_class[task] = counts
            repeats[task] = max(counts[c] / len(classes[c]) for c in classes)
        else:
            chosen = _draw(pool, quota, rng)
            repeats[task] = quota / len(pool)
        out += chosen
        rows[task] = len(chosen)
    rng.shuffle(out)
    return out, MixtureReport(rows=rows, repeats=repeats, by_class=by_class)
