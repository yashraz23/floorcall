"""Milestone 0 spike: does stock Laya run here, how fast, and what does it say zero-shot?

    uv run python scripts/spike_m0.py            # GPU + CPU
    uv run python scripts/spike_m0.py --no-cpu   # GPU only

Writes results/spike_m0.json. The 20 states are hand-written probes for plumbing and a first
qualitative look. They are NOT a test set: they were written by the person reading the output,
so no accuracy is computed from them and none is reported anywhere. The latency numbers here are
a quick look (fewer iterations than CLAUDE.md §12 requires); Table B comes from
floorcall.evaluate.latency, not from this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.model.laya_adapter import LayaDecider

console = Console()
OUT = Path(__file__).resolve().parents[1] / "results" / "spike_m0.json"

AGENT_MID = "your card ending four four one seven was declined because the"

# (event, user_partial, agent_speaking, agent_last_utterance, what the writer intended)
# Text is already in ASR form (lowercase, no punctuation), as floorcall.normalize will produce.
PROBES: list[tuple[questions.Event, str, bool, str, str]] = [
    (
        questions.Event.USER_PAUSE,
        "i need to check the balance on my savings account",
        False,
        "how can i help you today",
        "complete; balance",
    ),
    (
        questions.Event.USER_PAUSE,
        "i want to transfer some money from my",
        False,
        "how can i help you today",
        "incomplete; transfer",
    ),
    (
        questions.Event.USER_PAUSE,
        "so um i was wondering if",
        False,
        "how can i help you today",
        "incomplete",
    ),
    (
        questions.Event.USER_PAUSE,
        "whats the weather going to be like tomorrow",
        False,
        "how can i help you today",
        "complete; out_of_scope",
    ),
    (
        questions.Event.USER_PAUSE,
        "can i just talk to a real person please",
        False,
        "i can help with that let me check",
        "complete; escalate",
    ),
    (
        questions.Event.USER_PAUSE,
        "this is the third time ive called and nobody has fixed it",
        False,
        "i understand let me look into it",
        "complete; escalate (frustration)",
    ),
    (
        questions.Event.USER_PAUSE,
        "someone used my card for a purchase i never made",
        False,
        "how can i help you today",
        "complete; report_fraud",
    ),
    (questions.Event.USER_PAUSE, "and then the charge showed up on", False, "go on", "incomplete"),
    (
        questions.Event.USER_PAUSE,
        "when is my electric bill due",
        False,
        "how can i help you today",
        "complete; bill_due",
    ),
    (
        questions.Event.USER_PAUSE,
        "whats the routing number for my checking account",
        False,
        "how can i help you today",
        "complete; routing",
    ),
    (
        questions.Event.USER_PAUSE,
        "can you book me a table for two at eight",
        False,
        "how can i help you today",
        "complete; out_of_scope",
    ),
    (
        questions.Event.USER_PAUSE,
        "id like to change my pin and",
        False,
        "how can i help you today",
        "incomplete; pin_change",
    ),
    (questions.Event.USER_SPEECH_DURING_AGENT, "uh huh", True, AGENT_MID, "backchannel"),
    (questions.Event.USER_SPEECH_DURING_AGENT, "right", True, AGENT_MID, "backchannel"),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "yeah",
        True,
        AGENT_MID,
        "backchannel (hard: surface token)",
    ),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "yeah but thats not what i asked",
        True,
        AGENT_MID,
        "interruption (hard: surface token)",
    ),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "wait no i said checking not savings",
        True,
        AGENT_MID,
        "interruption",
    ),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "hold on can you repeat the last part",
        True,
        AGENT_MID,
        "interruption",
    ),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "honey can you grab the door",
        True,
        AGENT_MID,
        "noise (talk to someone else)",
    ),
    (
        questions.Event.USER_SPEECH_DURING_AGENT,
        "i want a human now",
        True,
        AGENT_MID,
        "interruption; escalate",
    ),
]


def make_state(user_partial: str, agent_speaking: bool, agent_last: str) -> dict[str, Any]:
    return {
        "agent_speaking": agent_speaking,
        "recent_turns": [],
        "agent_last_utterance": agent_last,
        "user_partial": user_partial,
    }


def summarize(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]

    return {
        "n": len(s),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(s),
    }


def time_event(
    d: LayaDecider, event: questions.Event, warmup: int, iters: int, sequential: bool
) -> dict[str, float]:
    state = make_state("yeah but thats not what i asked", True, AGENT_MID)
    qs = questions.questions_for(event)
    call = d.predict_sequential if sequential else d.predict
    for _ in range(warmup):
        call(state, qs)
    return summarize([call(state, qs).latency_ms for _ in range(iters)])


def run_probes(d: LayaDecider) -> list[dict[str, Any]]:
    table = Table(title=f"stock Laya zero-shot on 20 hand-written probes ({d.device})")
    for col in ("intended", "user_partial", "answers"):
        table.add_column(col, overflow="fold")
    rows = []
    for event, text, speaking, agent_last, intended in PROBES:
        pred = d.predict(make_state(text, speaking, agent_last), questions.questions_for(event))
        answers = {
            qid: {
                "label": a.label,
                "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()},
            }
            for qid, a in pred.answers.items()
        }
        shown = "; ".join(
            f"{qid}={a.label} ({a.answer_confidence:.2f})"
            if a.qtype == "choice"
            else f"{qid}: p(true)={a.probabilities['true']:.2f}"
            for qid, a in pred.answers.items()
        )
        table.add_row(intended, text, shown)
        rows.append(
            {"event": event.value, "user_partial": text, "intended": intended, "answers": answers}
        )
    console.print(table)
    return rows


def bench(device: str, warmup: int, iters: int) -> dict[str, Any]:
    d = LayaDecider(LayaSettings(**{**get_settings().laya.model_dump(), "device": device}))
    if d.device != device:
        raise RuntimeError(
            f"asked for {device} but the model is on {d.device} (DECISIONS.md D-010)"
        )
    out: dict[str, Any] = {"device": d.device, "autocast": d.autocast_dtype}
    for event in questions.Event:
        out[event.value] = {
            "batched": time_event(d, event, warmup, iters, sequential=False),
            "sequential": time_event(d, event, warmup, iters, sequential=True),
        }
    if d.device != device:
        raise RuntimeError(f"model moved to {d.device} during timing (DECISIONS.md D-010)")
    return {"decider": d, "timing": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    result: dict[str, Any] = {
        "note": "Spike numbers: quick look only. Not Table B. Probes are hand-written, not a test set.",
        "env": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "warmup": args.warmup,
            "iters": args.iters,
        },
    }

    gpu = bench("cuda", args.warmup, args.iters)
    decider: LayaDecider = gpu["decider"]
    result["revision"] = decider.revision
    result["state_room"] = {
        e.value: decider.state_room(questions.questions_for(e)) for e in questions.Event
    }
    result["probes"] = run_probes(decider)
    result["latency"] = {"cuda": gpu["timing"]}
    del decider, gpu
    torch.cuda.empty_cache()

    if not args.no_cpu:
        result["latency"]["cpu"] = bench("cpu", max(5, args.warmup // 4), max(50, args.iters // 4))[
            "timing"
        ]

    table = Table(title="batch-1 latency, ms (spike: quick look, not Table B)")
    for col in ("device", "event", "mode", "p50", "p95", "p99"):
        table.add_column(col)
    for dev, timing in result["latency"].items():
        for event in questions.Event:
            for mode in ("batched", "sequential"):
                s = timing[event.value][mode]
                table.add_row(
                    dev, event.value, mode, f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}"
                )
    console.print(table)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    console.print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
