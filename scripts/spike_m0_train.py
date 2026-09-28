"""Milestone 0: does RLCD fine-tuning of the 421M checkpoint fit in 12 GB on this laptop?

    uv run python scripts/spike_m0_train.py

Runs real optimizer updates of the notebook's objective on real CLINC150 banking rows, with half
the states padded with conversation history to the full 512-token row, so peak memory is measured
at the longest input training will see. Nothing is saved: the weights this produces are thrown
away. Writes results/spike_m0_train.json.
"""

from __future__ import annotations

import json
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.model.laya_adapter import LayaDecider
from floorcall.normalize import normalize
from floorcall.train.rlcd import rlcd_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "spike_m0_train.json"
CLINC = ROOT / "data" / "raw" / "clinc150"


def clinc_banking_rows() -> list[tuple[str, str]]:
    import re

    names: dict[int, str] = {}
    for line in (CLINC / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+'(\d+)': (\S+)", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    df = pd.read_parquet(CLINC / "plus_train.parquet")
    df["name"] = df.intent.map(names)
    keep = df[df.name.isin([*questions.BANKING_INTENTS, "oos"])]
    return [
        (normalize(t), questions.OUT_OF_SCOPE if n == "oos" else n)
        for t, n in zip(keep.text, keep.name, strict=True)
    ]


def build_items(
    d: LayaDecider, rows: list[tuple[str, str]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    qdef = questions.questions_by_id(questions.ROUTE)[questions.ROUTE]
    labels = list(questions.ROUTE_LABELS)
    room = d.state_room({questions.ROUTE: qdef})
    filler = [t for t, _ in rows]
    items = []
    for i, (text, label) in enumerate(rng.sample(rows, n)):
        state: dict[str, Any] = {
            "agent_speaking": False,
            "recent_turns": [],
            "agent_last_utterance": "how can i help you today",
            "user_partial": text,
        }
        if i % 2 == 0:  # pad to the budget: worst-case memory
            while True:
                turn = {"speaker": rng.choice(["user", "agent"]), "text": rng.choice(filler)}
                trial = {**state, "recent_turns": [*state["recent_turns"], turn]}
                if d.count_tokens(d.serialize(trial)) > room:
                    break
                state = trial
        target = [1.0 if lab == label else 0.0 for lab in labels]
        items.append(d.training_item(state, questions.ROUTE, qdef, target))
    return items


def main() -> None:
    cfg = get_settings()
    t = cfg.train
    rng = random.Random(t.seed)
    torch.manual_seed(t.seed)

    d = LayaDecider(LayaSettings(**{**cfg.laya.model_dump(), "device": "cuda"}))
    assert d.device == "cuda", d.device
    items = build_items(d, clinc_banking_rows(), n=512, rng=rng)
    lengths = sorted(len(it["ids"]) for it in items)

    d.enable_gradient_checkpointing()
    model = d.model
    model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": t.lr_encoder}, {"params": head, "lr": t.lr_head}],
        weight_decay=t.weight_decay,
    )
    reward = partial(d.proper_reward, w_sph=t.w_sph, w_rps=t.w_rps)

    # Longest rows first, so the very first micro-batch is the memory worst case.
    items.sort(key=lambda it: -len(it["ids"]))
    torch.cuda.reset_peak_memory_stats()
    log: list[dict[str, float]] = []
    t_start = time.perf_counter()
    n_micro = 0
    for update in range(len(items) // (t.micro_batch * t.grad_accum)):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        stats = []
        for a in range(t.grad_accum):
            lo = (update * t.grad_accum + a) * t.micro_batch
            b = {k: v.to("cuda") for k, v in d.collate(items[lo : lo + t.micro_batch]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(
                    b["input_ids"],
                    b["attention_mask"],
                    b["marker_pos"],
                    b["marker_mask"],
                    b["qtype"],
                )
            step = rlcd_loss(
                logits,
                b["target"],
                b["marker_mask"],
                b["qtype"],
                sigma=t.sigma_start,
                group_size=t.group_size,
                reward_fn=reward,
                ce_weight=t.ce_weight,
            )
            ((step.loss + 0.0 * act.sum()) / t.grad_accum).backward()
            stats.append(step)
            n_micro += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad_norm)
        opt.step()
        torch.cuda.synchronize()
        log.append(
            {
                "update": update,
                "seconds": time.perf_counter() - t0,
                "ce": sum(s.ce for s in stats) / len(stats),
                "reward": sum(s.reward for s in stats) / len(stats),
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )
        print(
            f"update {update}: {log[-1]['seconds']:.1f}s  ce={log[-1]['ce']:.3f}  reward={log[-1]['reward']:.3f}  peak={log[-1]['peak_alloc_gb']:.2f} GB"
        )

    total = time.perf_counter() - t_start
    result = {
        "note": "Memory/throughput check only. Weights discarded; not a trained model.",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "rows": len(items),
        "row_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
        "micro_batch": t.micro_batch,
        "grad_accum": t.grad_accum,
        "group_size": t.group_size,
        "amp": "bf16",
        "gradient_checkpointing": "encoder + head",
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "rows_per_second": n_micro * t.micro_batch / total,
        "updates": log,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "updates"}, indent=2))


if __name__ == "__main__":
    main()
