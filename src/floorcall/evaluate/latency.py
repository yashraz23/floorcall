"""Table B: batch-1 decision latency (CLAUDE.md §12).

    uv run floorcall eval latency [--rows gpu|cpu|all]

Protocol: batch 1 (real time is batch 1), `EvalSettings.latency_warmup` untimed iterations, then
`EvalSettings.latency_iters` timed ones, over a fixed input set of 50 real snapshots per event,
drawn at even spacing from the frozen test sets. The warmup cycles through every input, so every
CUDA-graph bucket is captured before timing starts. What is timed is the whole `Decider.decide`
call (packing, tokenizing, the forward pass, temperatures), because that is what sits on the
pipeline's critical path, and each component is reported beside it.

The device the weights are on is checked before and after timing. Laya falls back to CPU silently,
and a "GPU" row must never be a CPU number (DECISIONS.md D-010).

Latency does not depend on the weights, only on the architecture and the input shapes, so the stock
checkpoint gives the same numbers as a fine-tuned one. The harness takes any checkpoint, and is
rerun on the fine-tuned one before release.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.decider import Decider
from floorcall.evaluate.dataset import load_test, snapshot_of
from floorcall.provenance import git_head
from floorcall.state import Snapshot

RESULTS = REPO_ROOT / "results" / "table_b"
N_INPUTS = 50


@dataclass(frozen=True)
class Row:
    name: str
    label: str
    device: str
    graphs: bool
    event: questions.Event
    sequential: bool


PAUSE, BARGE = questions.Event.USER_PAUSE, questions.Event.USER_SPEECH_DURING_AGENT
ROWS: tuple[Row, ...] = (
    Row(
        "gpu_graphed_pause",
        "GPU, CUDA graphs: user_pause (3 questions, 1 call)",
        "cuda",
        True,
        PAUSE,
        False,
    ),
    Row(
        "gpu_graphed_pause_seq",
        "GPU, CUDA graphs: user_pause (3 questions, 3 calls)",
        "cuda",
        True,
        PAUSE,
        True,
    ),
    Row(
        "gpu_graphed_barge",
        "GPU, CUDA graphs: user_speech_during_agent (2 questions, 1 call)",
        "cuda",
        True,
        BARGE,
        False,
    ),
    Row(
        "gpu_eager_pause",
        "GPU, eager: user_pause (3 questions, 1 call)",
        "cuda",
        False,
        PAUSE,
        False,
    ),
    Row(
        "gpu_eager_pause_seq",
        "GPU, eager: user_pause (3 questions, 3 calls)",
        "cuda",
        False,
        PAUSE,
        True,
    ),
    Row(
        "gpu_eager_barge",
        "GPU, eager: user_speech_during_agent (2 questions, 1 call)",
        "cuda",
        False,
        BARGE,
        False,
    ),
    Row("cpu_pause", "CPU: user_pause (3 questions, 1 call)", "cpu", False, PAUSE, False),
    Row("cpu_pause_seq", "CPU: user_pause (3 questions, 3 calls)", "cpu", False, PAUSE, True),
    Row(
        "cpu_barge",
        "CPU: user_speech_during_agent (2 questions, 1 call)",
        "cpu",
        False,
        BARGE,
        False,
    ),
)


def inputs(settings: Settings, event: questions.Event, n: int = N_INPUTS) -> list[Snapshot]:
    """A fixed input set: evenly spaced test rows. user_pause mixes D1 and D3 states."""
    decisions = ("turn_complete", "route") if event is PAUSE else ("barge_in",)
    per = n // len(decisions)
    out: list[Snapshot] = []
    for d in decisions:
        rows = load_test(settings, d).rows
        step = max(1, len(rows) // per)
        out += [snapshot_of(r) for r in rows[::step][:per]]
    return out


def percentiles(xs: Sequence[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict[str, Any]:
    import torch

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "laya": version("laya"),
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_driver": _run(
            ["nvidia-smi", "--query-gpu=driver_version,power.limit", "--format=csv,noheader"]
        ),
    }
    if sys.platform == "win32":
        env["on_ac_power"] = (
            _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Battery).BatteryStatus",
                ]
            )
            == "2"
        )
        env["power_plan"] = _run(["powercfg", "/getactivescheme"])
    return env


def measure(
    decider: Decider, row: Row, snaps: Sequence[Snapshot], warmup: int, iters: int
) -> dict[str, Any]:
    device = getattr(decider.model, "device", None)
    if device != row.device:
        raise RuntimeError(f"{row.name}: model is on {device}, row needs {row.device} (D-010)")
    for i in range(warmup):
        decider.decide(row.event, snaps[i % len(snaps)], sequential=row.sequential)
    total, pack, encode, forward = [], [], [], []
    tokens = []
    for i in range(iters):
        d = decider.decide(row.event, snaps[i % len(snaps)], sequential=row.sequential)
        total.append(d.total_ms)
        pack.append(d.timings.pack_ms)
        encode.append(d.timings.encode_ms)
        forward.append(d.timings.forward_ms)
        tokens.append(d.packed.n_tokens)
    if getattr(decider.model, "device", None) != row.device:
        raise RuntimeError(f"{row.name}: model left {row.device} during timing (D-010)")
    return {
        "total_ms": percentiles(total),
        "pack_ms": percentiles(pack),
        "encode_ms": percentiles(encode),
        "forward_ms": percentiles(forward),
        "state_tokens_mean": float(np.mean(tokens)),
    }


def run(settings: Settings, which: str = "all") -> list[dict[str, Any]]:
    from floorcall.model.laya_adapter import LayaDecider

    selected = [r for r in ROWS if which == "all" or r.name.startswith(which)]
    # Provenance is read when the run starts: the code executing is what was loaded then, and
    # edits to files during a long run do not change it.
    code = git_head()
    env = environment()
    out = []
    for device in ("cuda", "cpu"):
        rows = [r for r in selected if r.device == device]
        if not rows:
            continue
        model = LayaDecider(LayaSettings(**{**settings.laya.model_dump(), "device": device}))
        decider = Decider(model, settings.state)
        for row in rows:
            if row.graphs:
                model.enable_cuda_graphs(settings.laya.graph_bucket_tokens)
            else:
                model.disable_cuda_graphs()
            snaps = inputs(settings, row.event)
            summary = measure(
                decider, row, snaps, settings.eval.latency_warmup, settings.eval.latency_iters
            )
            budget = (
                settings.eval.gpu_p99_budget_ms
                if device == "cuda"
                else settings.eval.cpu_p99_budget_ms
            )
            payload = {
                "row": row.name,
                "label": row.label,
                "device": model.device,
                "cuda_graphs": model.cuda_graphs,
                "autocast": model.autocast_dtype,
                "event": row.event.value,
                "questions": list(questions.EVENT_QUESTIONS[row.event]),
                "sequential": row.sequential,
                "warmup": settings.eval.latency_warmup,
                "iterations": settings.eval.latency_iters,
                "inputs": len(snaps),
                "checkpoint": model.checkpoint,
                "revision": model.revision,
                "code": code,
                "environment": env,
                "budget_p99_ms": budget,
                "fits_budget": summary["total_ms"]["p99"] <= budget,
                **summary,
            }
            RESULTS.mkdir(parents=True, exist_ok=True)
            with (RESULTS / f"{row.name}.json").open("w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            out.append(payload)
        model.disable_cuda_graphs()
        del decider, model
    return out
