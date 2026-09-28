"""Table A baselines: majority class and stock Laya, on the frozen test sets.

    uv run floorcall eval baselines [--decision turn_complete|barge_in|route|escalate|all]

Writes results/table_a/<decision>.<model>.json (committed: every Table A cell comes from one of
these files) and runs/eval/<decision>.<model>.npz with per-row logits (gitignored).

- **majority**: predicts the most common class of the train split, with the train class prior as
  its probabilities. D4 has no train split of hand labels, so its prior comes from calib. Never
  from test: taking the majority from the test set would be peeking. Using the prior, not a
  one-hot, gives a meaningful Brier score and ECE.
- **stock_laya**: the pinned checkpoint zero-shot, with its own shipped temperatures (including the
  clamped `choice:11+` bucket that sharpens D3). Every state is packed with its event's budget,
  exactly as it would be served.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.data.build import processed_file, test_file
from floorcall.data.freeze import sha256_file
from floorcall.evaluate.dataset import EvalSet, class_prior, load_processed, load_test
from floorcall.evaluate.scoring import score, softmax
from floorcall.provenance import git_head

RESULTS = REPO_ROOT / "results" / "table_a"
RUNS = REPO_ROOT / "runs" / "eval"
DECISIONS = ("turn_complete", "barge_in", "route", "escalate")


def _write(decision: str, model: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{decision}.{model}.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


# Read once, when this module is imported: the code a run executes is what was loaded then.
CODE = git_head()


def _provenance(settings: Settings, test: EvalSet) -> dict[str, Any]:
    return {
        "decision": test.decision,
        "test_file": test_file(test.decision),
        "test_sha256": sha256_file(settings.paths.test_frozen / test_file(test.decision)),
        "code": CODE,
        "labels": list(test.labels),
    }


def run_majority(settings: Settings, decision: str) -> dict[str, Any]:
    test = load_test(settings, decision)
    # D4 has no train split of hand labels; its prior comes from calib, which is still not test.
    prior_split = "train"
    if not (settings.paths.data_processed / processed_file(decision, "train")).exists():
        prior_split = "calib"
    prior = class_prior(load_processed(settings, decision, prior_split))
    probs = np.tile(prior, (len(test.y), 1))
    payload = {
        **_provenance(settings, test),
        "model": "majority",
        "prior_split": prior_split,
        "prior": {lab: float(p) for lab, p in zip(test.labels, prior, strict=True)},
        "metrics": score(probs, test.y, test.hard, test.labels, n_bins=settings.eval.ece_bins),
    }
    _write(decision, "majority", payload)
    return payload


def run_stock_laya(
    settings: Settings, decision: str, *, device: str | None = None
) -> dict[str, Any]:
    from floorcall.model.laya_adapter import LayaDecider

    test = load_test(settings, decision)
    decider = LayaDecider(LayaSettings(**{**settings.laya.model_dump(), "device": device}))
    event = test.event
    qdef = questions.questions_by_id(decision)
    budget = (
        decider.state_room(questions.questions_for(event)) - settings.state.safety_margin_tokens
    )
    states = test.packed_states(budget=budget, count_tokens=decider.count_tokens)
    t0 = time.perf_counter()
    per_row = decider.logits_batch(states, qdef, batch_size=32)
    seconds = time.perf_counter() - t0
    logits = np.stack([r[decision].logits for r in per_row])
    qtype = per_row[0][decision].qtype
    temperature = decider.effective_temperature(qtype, len(test.labels))
    probs = softmax(logits, temperature)

    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RUNS / f"{decision}.stock_laya.npz",
        logits=logits,
        y=test.y,
        hard=test.hard,
        ids=np.array([r["id"] for r in test.rows]),
    )
    payload = {
        **_provenance(settings, test),
        "model": "stock_laya",
        "checkpoint": decider.checkpoint,
        "revision": decider.revision,
        "device": decider.device,
        "autocast": decider.autocast_dtype,
        "event": event.value,
        "state_budget_tokens": budget,
        "temperature": temperature,
        "seconds": seconds,
        "metrics": score(probs, test.y, test.hard, test.labels, n_bins=settings.eval.ece_bins),
    }
    _write(decision, "stock_laya", payload)
    return payload
