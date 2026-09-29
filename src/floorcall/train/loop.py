"""The RLCD training loop (Laya's notebook, one GPU), over plain callables so it can be tested.

`train_model` knows nothing about Laya or floorcall's data: it takes a torch module with the
decision model's forward signature, the items, a collate function and a reward function. The
orchestration (floorcall.train.run) supplies the real ones; tests supply a tiny random model.

Per update: `micro_batch` rows per forward, `grad_accum` forwards per optimizer step (64 rows, the
notebook's effective batch). AdamW with separate learning rates for the encoder and the head, cosine
decay to `lr_min` over the whole run, gradient clipping at 1.0, bf16 autocast. The exploration
noise sigma anneals linearly from `sigma_start` in the first epoch to `sigma_end` in the last; with
`rl_weight` 0 (D-035 amendment 1) it only shapes the reported reward.

Every logged update carries the pre-clip gradient norm (mean and max over the window) and the
cross-entropy per task. After every epoch, cross-entropy at T = 1 is scored on the dev items when
they are given, and the epoch's record goes to `on_epoch_end`.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from floorcall.config import TrainSettings
from floorcall.train.data import Item, epoch_mixture
from floorcall.train.rlcd import RewardFn, rlcd_loss, sigma_for_epoch

Collate = Callable[[Sequence[Item]], dict[str, torch.Tensor]]


@dataclass
class History:
    updates: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)


def _param_groups(model: torch.nn.Module, cfg: TrainSettings) -> list[dict[str, Any]]:
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad
    ]
    return [{"params": enc, "lr": cfg.lr_encoder}, {"params": head, "lr": cfg.lr_head}]


def _forward(model: torch.nn.Module, b: Mapping[str, torch.Tensor]) -> Any:
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


def _row_ce(logits: torch.Tensor, b: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logp = torch.log_softmax(logits.float().masked_fill(~b["marker_mask"], -1e4), -1)
    return -(b["target"] * logp).sum(-1)


def dev_ce_by_task(
    model: torch.nn.Module,
    items_by_task: Mapping[str, Sequence[Item]],
    *,
    collate: Collate,
    device: torch.device,
    batch_size: int,
    amp: torch.dtype,
) -> dict[str, float]:
    """Mean cross-entropy at T = 1 per task, in eval mode, without gradients."""
    was_training = model.training
    model.eval()
    out = {}
    with torch.no_grad():
        for task in sorted(items_by_task):
            items = items_by_task[task]
            total = 0.0
            for lo in range(0, len(items), batch_size):
                b = {k: v.to(device) for k, v in collate(items[lo : lo + batch_size]).items()}
                with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
                    logits, _ = _forward(model, b)
                total += float(_row_ce(logits, b).sum())
            out[task] = total / len(items)
    model.train(was_training)
    return out


def train_model(
    model: torch.nn.Module,
    items_by_task: Mapping[str, Sequence[Item]],
    cfg: TrainSettings,
    *,
    collate: Collate,
    reward_fn: RewardFn,
    device: torch.device,
    rows_per_epoch: Mapping[str, int] | None = None,
    dev_items_by_task: Mapping[str, Sequence[Item]] | None = None,
    log: Callable[[dict[str, Any]], None] = lambda _: None,
    on_epoch_end: Callable[[dict[str, Any]], None] = lambda _: None,
) -> History:
    quotas = dict(rows_per_epoch if rows_per_epoch is not None else cfg.rows_per_epoch)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    amp = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    rows_per_update = cfg.micro_batch * cfg.grad_accum
    updates_per_epoch = math.ceil(sum(quotas.values()) / rows_per_update)
    total_updates = updates_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_updates), eta_min=cfg.lr_min
    )
    model.train()
    history = History()
    update = 0
    t_start = time.perf_counter()

    def fresh() -> dict[str, Any]:
        return {"loss": 0.0, "ce": 0.0, "rl": 0.0, "reward": 0.0, "n": 0, "norms": [], "task": {}}

    for epoch in range(cfg.epochs):
        sigma = sigma_for_epoch(epoch, cfg.epochs, cfg.sigma_start, cfg.sigma_end)
        rows, mix = epoch_mixture(items_by_task, quotas, rng, balance=cfg.balance_classes)
        opt.zero_grad(set_to_none=True)
        acc = fresh()
        task_ce: dict[str, list[float]] = {}
        micro_in_update = 0
        t_epoch = time.perf_counter()
        for lo in range(0, len(rows), cfg.micro_batch):
            chunk = rows[lo : lo + cfg.micro_batch]
            b = {k: v.to(device) for k, v in collate(chunk).items()}
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
                logits, act = _forward(model, b)
            step = rlcd_loss(
                logits,
                b["target"],
                b["marker_mask"],
                b["qtype"],
                sigma=sigma,
                group_size=cfg.group_size,
                reward_fn=reward_fn,
                ce_weight=cfg.ce_weight,
                rl_weight=cfg.rl_weight,
                generator=gen,
            )
            # act.sum() * 0 keeps the unused action head in the graph, as the notebook does.
            ((step.loss + 0.0 * act.sum()) / cfg.grad_accum).backward()
            micro_in_update += 1
            for key in ("ce", "rl", "reward"):
                acc[key] += getattr(step, key)
            acc["loss"] += float(step.loss.detach())
            acc["n"] += 1
            with (
                torch.no_grad()
            ):  # per-row CE, so mixed-task batches are credited to the right task
                row_ce = _row_ce(logits, b).tolist()
            for it, ce in zip(chunk, row_ce, strict=True):
                task_ce.setdefault(it["decision"], []).append(ce)
                acc["task"].setdefault(it["decision"], []).append(ce)
            last = lo + cfg.micro_batch >= len(rows)
            if micro_in_update == cfg.grad_accum or last:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                acc["norms"].append(float(norm))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                micro_in_update = 0
                update += 1
                if update % cfg.log_every_updates == 0 or last:
                    n = max(1, acc["n"])
                    rec = {
                        "update": update,
                        "epoch": epoch + 1,
                        "loss": acc["loss"] / n,
                        "ce": acc["ce"] / n,
                        "rl": acc["rl"] / n,
                        "reward": acc["reward"] / n,
                        "ce_by_task": {t: sum(v) / len(v) for t, v in sorted(acc["task"].items())},
                        "grad_norm": sum(acc["norms"]) / len(acc["norms"]),  # before clipping
                        "grad_norm_max": max(acc["norms"]),
                        "lr_encoder": sched.get_last_lr()[0],
                        "sigma": sigma,
                        "rows_per_s": (lo + len(chunk)) / (time.perf_counter() - t_epoch),
                    }
                    history.updates.append(rec)
                    log(rec)
                    acc = fresh()
        ep: dict[str, Any] = {
            "epoch": epoch + 1,
            "rows": mix.rows,
            "repeats": mix.repeats,
            "rows_by_class": mix.by_class,
            "sigma": sigma,
            "seconds": time.perf_counter() - t_epoch,
            "mean_ce_by_task": {t: sum(v) / len(v) for t, v in task_ce.items()},
        }
        if dev_items_by_task:
            dev = dev_ce_by_task(
                model,
                dev_items_by_task,
                collate=collate,
                device=device,
                batch_size=cfg.micro_batch * 4,
                amp=amp,
            )
            ep["dev_ce_by_task"] = dev
            ep["dev_ce_macro"] = sum(dev.values()) / len(dev)
        history.epochs.append(ep)
        log({"epoch_end": ep, "elapsed_s": time.perf_counter() - t_start})
        on_epoch_end(ep)
    model.eval()
    return history


def best_epoch(epochs: Sequence[Mapping[str, Any]]) -> int:
    """The epoch with the lowest macro dev cross-entropy; the earliest on a tie."""
    scored = [e for e in epochs if "dev_ce_macro" in e]
    if not scored:
        raise ValueError("no epoch was scored on a dev split")
    return int(min(scored, key=lambda e: (e["dev_ce_macro"], e["epoch"]))["epoch"])
