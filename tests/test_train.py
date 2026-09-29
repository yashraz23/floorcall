"""Training code, tested without training anything real.

The loop runs on a tiny stand-in with the decision model's forward signature, on a synthetic task
that can only be solved by reading the state. No optimizer step touches Laya's weights here: the
real checkpoint is only saved and reloaded (untrained), and training itself is refused until all
four test sets are frozen, which is also tested.
"""

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from floorcall.config import Settings, TrainSettings
from floorcall.data.build import test_file as frozen_name
from floorcall.data.freeze import freeze
from floorcall.train.data import epoch_mixture
from floorcall.train.loop import train_model
from floorcall.train.run import TrainingNotAllowedError, assert_test_sets_frozen

K = 3  # options per question


class _Encoder(nn.Module):
    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)


class Tiny(nn.Module):
    """Laya's DecisionModel forward signature, in a few hundred parameters. Parameter names start
    with "encoder." like the real model's, so the loop's two learning rates are exercised."""

    def __init__(self, vocab: int = 32, d: int = 16) -> None:
        super().__init__()
        self.encoder = _Encoder(vocab, d)
        self.proj = nn.Linear(d, d)
        self.act_head = nn.Linear(d, 2)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):  # type: ignore[no-untyped-def]
        h = self.encoder.emb(input_ids)
        mask = attention_mask[..., None].float()
        ctx = (h * mask).sum(1) / mask.sum(1)
        m = torch.gather(h, 1, marker_pos[..., None].expand(-1, -1, h.size(-1)))
        logits = (self.proj(m) * ctx[:, None, :]).sum(-1).float()
        return logits.masked_fill(~marker_mask, -1e4), self.act_head(ctx)


def item(label: int, task: str) -> dict[str, Any]:
    # markers are tokens 1..K at positions 0..K-1; the state token 10+label carries the answer
    ids = np.array([1, 2, 3, 10 + label, 20], dtype=np.int32)
    target = [0.0] * K
    target[label] = 1.0
    return {
        "ids": ids,
        "markers": [0, 1, 2],
        "qtype": 0,
        "target": target,
        "label": label,
        "decision": task,
    }


def collate(items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    n, length = len(items), max(len(it["ids"]) for it in items)
    ids = torch.zeros((n, length), dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.as_tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": torch.tensor([it["markers"] for it in items]),
        "marker_mask": torch.ones((n, K), dtype=torch.bool),
        "target": torch.tensor([it["target"] for it in items]),
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def log_score(
    q: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return (target * torch.log(q.clamp_min(1e-12))).sum(-1)


CFG = TrainSettings(
    epochs=4,
    micro_batch=4,
    grad_accum=2,
    group_size=4,
    lr_encoder=5e-2,
    lr_head=5e-2,
    lr_min=1e-3,
    log_every_updates=1,
    rows_per_epoch={"a": 48, "b": 16},
)
DATA = {
    "a": [item(i % K, "a") for i in range(30)],
    "b": [item((i + 1) % K, "b") for i in range(9)],  # fewer rows than its quota: repeated
}


def run(seed: int = 0) -> tuple[Tiny, Any, list[int]]:
    torch.manual_seed(seed)
    model = Tiny()
    epochs_seen: list[int] = []
    hist = train_model(
        model,
        DATA,
        CFG,
        collate=collate,
        reward_fn=log_score,
        device=torch.device("cpu"),
        on_epoch_end=epochs_seen.append,
    )
    return model, hist, epochs_seen


def test_the_loop_learns_the_synthetic_task() -> None:
    model, hist, _ = run()
    first, last = hist.epochs[0]["mean_ce_by_task"], hist.epochs[-1]["mean_ce_by_task"]
    assert last["a"] < first["a"] / 2
    assert last["b"] < first["b"] / 2
    batch = collate([item(lab, "a") for lab in range(K)])
    with torch.no_grad():
        logits, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
    assert logits.argmax(-1).tolist() == [0, 1, 2]


def test_quotas_updates_and_callbacks() -> None:
    _, hist, epochs_seen = run()
    assert epochs_seen == [1, 2, 3, 4]
    for ep in hist.epochs:
        assert ep["rows"] == {"a": 48, "b": 16}
        assert ep["repeats"]["b"] == pytest.approx(16 / 9)
    # 64 rows per epoch, 8 rows per update: 8 updates per epoch
    assert hist.updates[-1]["update"] == 4 * 8
    assert hist.updates[-1]["sigma"] == pytest.approx(CFG.sigma_end)
    assert hist.updates[0]["sigma"] == pytest.approx(CFG.sigma_start)


def test_deterministic_for_a_seed() -> None:
    _, h1, _ = run(seed=3)
    _, h2, _ = run(seed=3)
    assert [u["loss"] for u in h1.updates] == [u["loss"] for u in h2.updates]


def test_mixture_samples_without_replacement_when_there_is_enough() -> None:
    rows, rep = epoch_mixture({"a": DATA["a"]}, {"a": 20}, random.Random(0))
    assert len(rows) == 20
    assert len({id(r) for r in rows}) == 20
    assert rep.repeats["a"] == pytest.approx(20 / 30)


def test_a_balanced_task_draws_its_classes_evenly() -> None:
    items = [{"label": 1} for _ in range(9)] + [{"label": 0} for _ in range(51)]
    rows, rep = epoch_mixture({"d": items}, {"d": 50}, random.Random(0), balance=("d",))
    assert Counter(r["label"] for r in rows) == {0: 25, 1: 25}
    assert rep.by_class == {"d": {0: 25, 1: 25}} and rep.repeats["d"] == pytest.approx(25 / 9)
    # 25 draws from 9 positives: every one twice, 7 of them a third time
    assert sorted(Counter(id(r) for r in rows if r["label"] == 1).values()) == [2, 2] + [3] * 7
    assert len({id(r) for r in rows if r["label"] == 0}) == 25  # negatives: no repeats
    _, odd = epoch_mixture({"d": items}, {"d": 51}, random.Random(0), balance=("d",))
    assert odd.by_class == {"d": {0: 26, 1: 25}}


def test_d4_is_class_balanced_by_default() -> None:
    assert TrainSettings().balance_classes == ("escalate",)


def test_mixture_refuses_a_task_without_rows() -> None:
    with pytest.raises(ValueError, match="escalate"):
        epoch_mixture({"a": DATA["a"]}, {"a": 10, "escalate": 5}, random.Random(0))


# -- the guard -------------------------------------------------------------------------------


def settings_with_frozen(tmp_path: Path, decisions: Sequence[str]) -> Settings:
    for d in decisions:
        freeze(tmp_path, frozen_name(d), [{"id": d}], {})  # the version d is evaluated on
    s = Settings()
    return s.model_copy(update={"paths": s.paths.model_copy(update={"test_frozen": tmp_path})})


def test_training_is_refused_until_d4_is_frozen(tmp_path: Path) -> None:
    s = settings_with_frozen(tmp_path, ["turn_complete", "barge_in", "route"])
    with pytest.raises(TrainingNotAllowedError, match="escalate"):
        assert_test_sets_frozen(s)


def test_training_is_allowed_with_all_four(tmp_path: Path) -> None:
    s = settings_with_frozen(tmp_path, ["turn_complete", "barge_in", "route", "escalate"])
    assert len(assert_test_sets_frozen(s)) == 4


def test_training_is_refused_if_a_frozen_set_was_tampered_with(tmp_path: Path) -> None:
    s = settings_with_frozen(tmp_path, ["turn_complete", "barge_in", "route", "escalate"])
    (tmp_path / "route.test.v1.jsonl.gz").write_bytes(b"edited")
    with pytest.raises(Exception, match="route"):
        assert_test_sets_frozen(s)


def test_the_real_repo_refuses_training_today() -> None:
    # data/test_frozen has D1-D3 only until Yash's D4 labels are frozen
    from floorcall.config import get_settings
    from floorcall.data.build import test_file

    s = get_settings()
    if (s.paths.test_frozen / test_file("escalate")).exists():
        pytest.skip("D4 is frozen: training is allowed now")
    with pytest.raises(TrainingNotAllowedError):
        assert_test_sets_frozen(s)
