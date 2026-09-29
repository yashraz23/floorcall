"""Why the first full training run broke in epoch 2 (DECISIONS.md D-035 amendment 1).

    uv run python scripts/diagnose_r1.py

Needs the stopped run's last checkpoint, checkpoints/main/checkpoint_latest (the end of epoch 3;
gitignored, kept locally as evidence). Reads train rows only, never calib or test. It measures:

- pre-clip gradient norms of the cross-entropy term and of the policy-gradient term, separately,
  per task and per sigma, on the stock weights and on the epoch-3 weights: the median over six
  micro-batches of eight train rows;
- the shape of D1's predictions on 400 train rows: accuracy, the share predicted "true", mean
  confidence and quantiles of p(true).

Writes results/training/r1_stopped/diagnosis.json.
"""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from floorcall import questions
from floorcall.config import REPO_ROOT, LayaSettings, get_settings
from floorcall.evaluate.dataset import EvalSet, _make, load_processed
from floorcall.model.laya_adapter import LayaDecider
from floorcall.train.data import build_items, event_budget
from floorcall.train.rlcd import rlcd_loss

S = get_settings()
TASKS = ("turn_complete", "barge_in", "route", "escalate")
SIGMAS = (0.4, 0.3, 0.2, 0.1)
MICRO, N_MICRO = 8, 6
EPOCH3 = REPO_ROOT / "checkpoints" / "main" / "checkpoint_latest"
OUT = REPO_ROOT / "results" / "training" / "r1_stopped" / "diagnosis.json"


def train_subset(task: str, n: int, seed: int) -> EvalSet:
    data = load_processed(S, task, "train")
    idx = random.Random(f"{seed}:{task}").sample(range(len(data.rows)), n)
    return _make(task, "train", [data.rows[i] for i in idx])


def grad_norm(model: torch.nn.Module) -> float:
    return float(
        sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None)
        ** 0.5
    )


def gradient_norms(decider: LayaDecider) -> dict[str, dict[str, float]]:
    cfg = S.train
    model = decider.model
    decider.enable_gradient_checkpointing()
    model.train()
    reward_fn = partial(decider.proper_reward, w_sph=cfg.w_sph, w_rps=cfg.w_rps)
    gen = torch.Generator(device="cuda").manual_seed(0)
    out = {}
    for task in TASKS:
        items = build_items(decider, train_subset(task, MICRO * N_MICRO, 1), S.state)
        norms: dict[str, list[float]] = {"ce": [], **{f"pg@{sg}": [] for sg in SIGMAS}}
        for m in range(N_MICRO):
            b = {
                k: v.cuda() for k, v in decider.collate(items[m * MICRO : (m + 1) * MICRO]).items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(
                    b["input_ids"],
                    b["attention_mask"],
                    b["marker_pos"],
                    b["marker_mask"],
                    b["qtype"],
                )
            model.zero_grad(set_to_none=True)
            logp = torch.log_softmax(logits.float().masked_fill(~b["marker_mask"], -1e4), -1)
            (-(b["target"] * logp).sum(-1).mean() + 0.0 * act.sum()).backward(retain_graph=True)
            norms["ce"].append(grad_norm(model))
            for sg in SIGMAS:
                pg = rlcd_loss(
                    logits, b["target"], b["marker_mask"], b["qtype"], sigma=sg,
                    group_size=cfg.group_size, reward_fn=reward_fn, ce_weight=0.0, rl_weight=1.0,
                    generator=gen,
                )  # fmt: skip
                model.zero_grad(set_to_none=True)
                (pg.loss + 0.0 * act.sum()).backward(retain_graph=True)
                norms[f"pg@{sg}"].append(grad_norm(model))
            model.zero_grad(set_to_none=True)
        out[task] = {k: float(np.median(v)) for k, v in norms.items()}
    return out


def d1_predictions(decider: LayaDecider) -> dict[str, Any]:
    data = train_subset("turn_complete", 400, 2)
    budget = event_budget(decider, data.event, S.state)
    states = data.packed_states(state=S.state, budget=budget, count_tokens=decider.count_tokens)
    rows = decider.logits_batch(states, questions.questions_by_id("turn_complete"))
    z = np.stack([r["turn_complete"].logits for r in rows])
    p = np.exp(z - z.max(1, keepdims=True))
    p_true = (p / p.sum(1, keepdims=True))[:, data.labels.index("true")]
    pred = (p_true >= 0.5).astype(int)
    return {
        "n": int(p_true.size),
        "gold_true_share": float(data.y.mean()),
        "pred_true_share": float(pred.mean()),
        "accuracy": float((pred == data.y).mean()),
        "mean_confidence": float(np.maximum(p_true, 1 - p_true).mean()),
        "p_true_quantiles_5_25_50_75_95": [
            float(q) for q in np.quantile(p_true, [0.05, 0.25, 0.5, 0.75, 0.95])
        ],
    }


def main() -> None:
    report: dict[str, Any] = {
        "split": "train rows only",
        "micro_batch": MICRO,
        "micro_batches": N_MICRO,
    }
    for name, lay in (
        ("stock", LayaSettings(**{**S.laya.model_dump(), "device": "cuda", "cuda_graphs": False})),
        (
            "epoch3",
            LayaSettings(checkpoint=str(EPOCH3), revision=None, device="cuda", cuda_graphs=False),
        ),
    ):
        decider = LayaDecider(lay)
        decider.model.eval()
        report[name] = {
            "d1_predictions": d1_predictions(decider),
            "grad_norms": gradient_norms(decider),
        }
        del decider
        torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with Path(OUT).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
