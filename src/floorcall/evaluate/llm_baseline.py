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


def estimate_spend(settings: Settings, per_decision: int = 20) -> dict[str, Any]:
    """What the Table A and Table B runs would cost (D-041), from a pilot on **calib** rows only.

    `per_decision` evenly spaced calib rows per decision, plus as many user_pause prompts with all
    three questions (the latency prompt), are answered through the pinned endpoint. OpenRouter's
    reported cost per call, averaged, is multiplied by the test rows and by the latency calls. The
    pilot is itself spent, and counted.
    """
    from floorcall.evaluate.dataset import load_processed, snapshot_of

    client, ledger = _client(settings)
    packer = _packer(settings)
    model = settings.llm.baseline_model
    before = ledger.spent()
    per_call: dict[str, float] = {}
    invalid: dict[str, int] = {}
    plan: dict[str, int] = {}

    def pilot(part: str, event: questions.Event, qs: dict[str, Any], rows: list[Any]) -> None:
        # the ledger bills every call, empty answers included; the cache is bypassed so no call
        # looks free
        spent0, bad = ledger.spent(), 0
        for r in rows:
            try:
                client.chat_json(
                    model=model,
                    messages=baseline_messages(packer.pack(event, snapshot_of(r)).state, qs),
                    schema=baseline_schema(qs),
                    purpose="baseline-estimate",
                    use_cache=False,
                )
            except LLMError:
                bad += 1
        per_call[part] = (ledger.spent() - spent0) / len(rows)
        invalid[part] = bad

    for decision in DECISIONS:
        calib = subsample(load_processed(settings, decision, "calib"), per_decision)
        pilot(decision, calib.event, questions.questions_by_id(decision), calib.rows)
        plan[decision] = len(load_test(settings, decision).rows)
    event = questions.Event.USER_PAUSE
    calib = subsample(load_processed(settings, "turn_complete", "calib"), per_decision)
    pilot("latency", event, questions.questions_for(event), calib.rows)
    plan["latency"] = settings.eval.latency_warmup + settings.eval.latency_iters
    client.close()
    projected = {k: per_call[k] * n for k, n in plan.items()}
    return {
        "model": model,
        "pin": settings.llm.provider_pins[model].endpoint,
        "pilot_calls_per_part": per_decision,
        "pilot_cost_usd": ledger.spent() - before,
        "cost_per_call_usd": per_call,
        "pilot_invalid_answers": invalid,
        "calls": plan,
        "projected_usd": projected,
        "projected_total_usd": sum(projected.values()),
        "spent_so_far_usd": ledger.spent(),
        "stop_usd": ledger.stop_usd,
    }


def _payload_a(
    settings: Settings,
    test: EvalSet,
    full: EvalSet,
    probs: np.ndarray,
    invalid: int,
    code: str,
    cost: float,
) -> dict[str, Any]:
    model = settings.llm.baseline_model
    return {
        **_provenance(settings, test),
        "code": code,
        "model": "prompted_llm",
        "llm": model,
        "prompt": BASELINE_VERSION,
        "provider_pin": settings.llm.provider_pins[model].endpoint,
        "n_test_rows": len(full.rows),
        "subsampled": len(test.rows) < len(full.rows),
        "invalid_answers": invalid,
        "probabilities": "stated by the model, not token log-probabilities",
        "cost_usd": cost,
        "metrics": score(
            probs,
            test.y,
            test.hard,
            test.labels,
            n_bins=settings.eval.ece_bins,
            bootstrap=(settings.eval.bootstrap_samples, settings.eval.bootstrap_seed),
        ),
    }


def _write_a(decision: str, payload: dict[str, Any], test: EvalSet, probs: np.ndarray) -> None:
    from floorcall.evaluate.checkpoint import save_predictions

    # per-row answers kept (D-036); log-probabilities, so softmax at T = 1 gives them back
    save_predictions(f"{decision}.prompted_llm", np.log(np.clip(probs, 1e-12, 1.0)), test)
    path = RESULTS_A / f"{decision}.prompted_llm.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump({**payload, "temperature": 1.0}, f, indent=2)
        f.write("\n")


def _states(
    settings: Settings, packer: Decider, decision: str
) -> tuple[EvalSet, EvalSet, list[Any]]:
    from floorcall.evaluate.dataset import snapshot_of

    full = load_test(settings, decision)
    test = subsample(full, settings.llm.baseline_rows.get(decision))
    states = [packer.pack(test.event, snapshot_of(r)).state for r in test.rows]
    return full, test, states


def run_table_a(settings: Settings) -> list[dict[str, Any]]:
    code = git_head()
    client, ledger = _client(settings)
    packer = _packer(settings)
    model = settings.llm.baseline_model
    out = []
    for decision in DECISIONS:
        full, test, states = _states(settings, packer, decision)
        qs = questions.questions_by_id(decision)
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
        invalid = sum(1 for _, ok in answers if not ok)
        payload = _payload_a(
            settings, test, full, probs, invalid, code, ledger.spent() - spent_before
        )
        _write_a(decision, payload, test, probs)
        out.append(payload)
    client.close()
    return out


def recover_table_a(settings: Settings) -> dict[str, list[str]]:
    """Rebuild each row's per-row answers from the response cache only (D-041): no API call.

    The Table A pass predates D-036's rule to save per-row predictions. Every successful answer is
    cached under its exact request; a failed one never was, and stays uniform and invalid, as it
    was counted. The rebuilt answers are used only if they reproduce the written row's metrics and
    invalid count exactly; then the row is rewritten with bootstrap intervals and its answers are
    saved. The row keeps its code, cost and every other field.
    """
    client, _ = _client(settings)
    packer = _packer(settings)
    model = settings.llm.baseline_model
    problems: dict[str, list[str]] = {}
    for decision in DECISIONS:
        full, test, states = _states(settings, packer, decision)
        qs = questions.questions_by_id(decision)
        labels = list(qs[decision]["criteria"])
        answers = []
        for state in states:
            hit = client.cached(model, baseline_messages(state, qs), baseline_schema(qs))
            if hit is None:
                answers.append(([1.0 / len(labels)] * len(labels), False))
            else:
                answers.append(read_probabilities(hit.get(decision, {}), labels))
        probs = np.array([p for p, _ in answers])
        invalid = sum(1 for _, ok in answers if not ok)
        path = RESULTS_A / f"{decision}.prompted_llm.json"
        written = json.loads(path.read_text(encoding="utf-8"))
        again = _payload_a(
            settings, test, full, probs, invalid, written["code"], written["cost_usd"]
        )
        found = []
        if invalid != written["invalid_answers"]:
            found.append(f"invalid answers {invalid} vs {written['invalid_answers']}")
        for key in ("accuracy", "macro_f1", "ece", "brier", "confusion", "hard_accuracy"):
            if again["metrics"][key] != written["metrics"][key]:
                found.append(f"{key}: {again['metrics'][key]} vs {written['metrics'][key]}")
        problems[decision] = found
        if not found:
            _write_a(decision, {**written, "metrics": again["metrics"]}, test, probs)
    client.close()
    return problems


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
        "label": f"prompted LLM ({model} via OpenRouter), user_pause, 3 questions in 1 call: "
        "network latency, request to parsed answer",
        "provider_pin": settings.llm.provider_pins[model].endpoint,
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
