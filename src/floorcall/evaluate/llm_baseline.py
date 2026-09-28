"""The prompted-LLM baseline (CLAUDE.md §12, Tables A and B), via OpenRouter (D-030, D-031).

    uv run floorcall eval llm-baseline     # Table A row, every decision
    uv run floorcall eval llm-latency      # Table B row

The LLM sees exactly what Laya sees: the state packed by the same packer with the same event
budget, and each question's instructions and option descriptions from floorcall.questions. It
states a probability per option under a strict JSON schema. Token log-probabilities are not used
(provider support varies behind OpenRouter, and the answer is a JSON object, not one option
token), so its ECE and Brier score measure *stated* confidence, and the README says so.

Table A: D1 and D2 are scored on fixed, evenly spaced subsamples (`LLMSettings.baseline_rows`), to
stay inside the $5 cap; D3 and D4 are scored whole. The n is in every results file, and the
accuracy interval reflects it. An answer that cannot be parsed counts as uniform, and the count of
such answers is reported.

Table B: the user_pause event, all three questions in one prompt, over the same 50 fixed inputs
and protocol as every other row (50 warmup, `latency_iters` timed calls, cache bypassed). A call is
timed end to end: packing, building the prompt, the network, the answer, and parsing.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.decider import Decider
from floorcall.evaluate.baselines import RESULTS as RESULTS_A
from floorcall.evaluate.baselines import _provenance
from floorcall.evaluate.dataset import EvalSet, load_test
from floorcall.evaluate.latency import RESULTS as RESULTS_B
from floorcall.evaluate.latency import environment, inputs, percentiles
from floorcall.evaluate.scoring import score
from floorcall.llm.client import Ledger, LLMClient, LLMError
from floorcall.llm.prompts import (
    BASELINE_VERSION,
    baseline_messages,
    baseline_schema,
    read_probabilities,
)
from floorcall.provenance import git_head

DECISIONS = ("turn_complete", "barge_in", "route", "escalate")


def _client(settings: Settings) -> tuple[LLMClient, Ledger]:
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env, which git ignores)")
    ledger = Ledger(REPO_ROOT / "runs" / "llm" / "ledger.sqlite", settings.llm)
    return LLMClient(settings.llm, settings.openrouter_api_key.get_secret_value(), ledger), ledger


def _packer(settings: Settings) -> Decider:
    """A Decider used only to pack states: the same tokenizer and budgets Laya gets. No forward."""
    from floorcall.model.laya_adapter import LayaDecider

    model = LayaDecider(LayaSettings(**{**settings.laya.model_dump(), "device": "cpu"}))
    return Decider(model, settings.state)


def subsample(data: EvalSet, n: int | None) -> EvalSet:
    if n is None or n >= len(data.rows):
        return data
    idx = np.linspace(0, len(data.rows) - 1, n).round().astype(int)
    return EvalSet(
        data.decision,
        data.split,
        data.labels,
        [data.rows[i] for i in idx],
        data.y[idx],
        data.hard[idx],
    )


def run_table_a(settings: Settings) -> list[dict[str, Any]]:
    from floorcall.evaluate.dataset import snapshot_of

    code = git_head()
    client, ledger = _client(settings)
    packer = _packer(settings)
    model = settings.llm.baseline_model
    out = []
    for decision in DECISIONS:
        full = load_test(settings, decision)
        test = subsample(full, settings.llm.baseline_rows.get(decision))
        qs = questions.questions_by_id(decision)
        states = [packer.pack(test.event, snapshot_of(r)).state for r in test.rows]
        spent_before = ledger.spent()

        def ask(
            state: dict[str, Any], qs: dict[str, Any] = qs, decision: str = decision
        ) -> tuple[list[float], bool]:
            try:
                res = client.chat_json(
                    model=model,
                    messages=baseline_messages(state, qs),
                    schema=baseline_schema(qs),
                    purpose=f"baseline-{decision}",
                )
            except LLMError:
                k = len(questions.ALL_QUESTIONS[decision]["criteria"])
                return [1.0 / k] * k, False
            return read_probabilities(res.data.get(decision, {}), list(qs[decision]["criteria"]))

        with ThreadPoolExecutor(max_workers=settings.llm.concurrency) as pool:
            answers = list(pool.map(ask, states))
        probs = np.array([p for p, _ in answers])
        payload = {
            **_provenance(settings, test),
            "code": code,
            "model": "prompted_llm",
            "llm": model,
            "prompt": BASELINE_VERSION,
            "n_test_rows": len(full.rows),
            "subsampled": len(test.rows) < len(full.rows),
            "invalid_answers": sum(1 for _, ok in answers if not ok),
            "probabilities": "stated by the model, not token log-probabilities",
            "cost_usd": ledger.spent() - spent_before,
            "metrics": score(probs, test.y, test.hard, test.labels, n_bins=settings.eval.ece_bins),
        }
        path = RESULTS_A / f"{decision}.prompted_llm.json"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        out.append(payload)
    client.close()
    return out


def run_latency(settings: Settings) -> dict[str, Any]:
    code = git_head()
    client, ledger = _client(settings)
    packer = _packer(settings)
    event = questions.Event.USER_PAUSE
    qs = questions.questions_for(event)
    snaps = inputs(settings, event)
    schema = baseline_schema(qs)
    model = settings.llm.baseline_model
    spent_before = ledger.spent()

    def call(i: int) -> float:
        t0 = time.perf_counter()
        state = packer.pack(event, snaps[i % len(snaps)]).state
        res = client.chat_json(
            model=model,
            messages=baseline_messages(state, qs),
            schema=schema,
            purpose="latency",
            use_cache=False,
        )
        for qid, q in qs.items():
            read_probabilities(res.data.get(qid, {}), list(q["criteria"]))
        return (time.perf_counter() - t0) * 1000.0

    for i in range(settings.eval.latency_warmup):
        call(i)
    timed, failures = [], 0
    for i in range(settings.eval.latency_iters):
        try:
            timed.append(call(i))
        except LLMError:
            failures += 1
    budget = settings.eval.gpu_p99_budget_ms
    total = percentiles(timed)
    payload = {
        "row": "prompted_llm",
        "label": f"prompted LLM ({model} via OpenRouter), user_pause, 3 questions in 1 call",
        "device": "remote (OpenRouter)",
        "event": event.value,
        "questions": list(qs),
        "warmup": settings.eval.latency_warmup,
        "iterations": settings.eval.latency_iters,
        "failed_calls": failures,
        "inputs": len(snaps),
        "llm": model,
        "prompt": BASELINE_VERSION,
        "code": code,
        "environment": environment(),
        # compared with the inline-decision budget floorcall is held to on a GPU
        "budget_p99_ms": budget,
        "fits_budget": total["p99"] <= budget,
        # One number: packing, prompt, network and parsing are all inside total_ms, and there is
        # no local forward pass to separate out. The README table prints n/a for the parts.
        "total_ms": total,
        "cost_usd": ledger.spent() - spent_before,
    }
    with (RESULTS_B / "prompted_llm.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    client.close()
    return payload
