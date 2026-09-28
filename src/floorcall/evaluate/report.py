"""README tables, rendered from committed result files. Nothing in them is typed by hand.

    uv run floorcall eval readme     # rewrite the marked sections of README.md

Each table lives between `<!-- name:start -->` and `<!-- name:end -->`. A cell whose results file
does not exist renders as TODO, never as an estimate. tests/test_readme.py fails when README.md
differs from what these functions render, so a number cannot be edited in by hand without the
file behind it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from floorcall.config import REPO_ROOT

RESULTS = REPO_ROOT / "results" / "table_a"
RESULTS_B = REPO_ROOT / "results" / "table_b"
README = REPO_ROOT / "README.md"

DECISION_NAMES = {
    "turn_complete": "D1 turn_complete",
    "barge_in": "D2 barge_in",
    "route": "D3 route",
    "escalate": "D4 escalate",
}
# (results file model id, row label), in the order CLAUDE.md §12 lists Table A's rows
MODELS = (
    ("majority", "majority class (train prior)"),
    ("stock_laya", "stock Laya, zero-shot"),
    ("finetuned", "fine-tuned"),
    ("finetuned_temp", "fine-tuned + temperature"),
)


def _load(decision: str, model: str, root: Path) -> dict[str, Any] | None:
    path = root / f"{decision}.{model}.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def table_a(root: Path = RESULTS) -> str:
    lines = [
        "| Decision | Model | Accuracy [95% CI] | Macro-F1 | ECE | Brier | Hard-subset acc. (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for decision, name in DECISION_NAMES.items():
        for model, label in MODELS:
            r = _load(decision, model, root)
            if r is None:
                lines.append(f"| {name} | {label} | TODO | TODO | TODO | TODO | TODO |")
                continue
            m = r["metrics"]
            lo, hi = m["accuracy_ci95"]
            hard = (
                "n/a" if m["hard_accuracy"] is None else f"{m['hard_accuracy']:.3f} ({m['hard_n']})"
            )
            lines.append(
                f"| {name} | {label} | {m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{_f(m['macro_f1'])} | {_f(m['ece'])} | {_f(m['brier'])} | {hard} |"
            )
    return "\n".join(lines)


# Table B rows: (results file stem, label). Order follows CLAUDE.md §12. Rows with no file yet
# (baselines not built) render as TODO.
LATENCY_ROWS = (
    ("gpu_graphed_pause", "GPU, CUDA graphs: user_pause, 3 questions in 1 call"),
    ("gpu_graphed_pause_seq", "GPU, CUDA graphs: user_pause, 3 questions in 3 calls"),
    ("gpu_graphed_barge", "GPU, CUDA graphs: user_speech_during_agent, 2 questions in 1 call"),
    ("gpu_eager_pause", "GPU, eager: user_pause, 3 questions in 1 call"),
    ("gpu_eager_pause_seq", "GPU, eager: user_pause, 3 questions in 3 calls"),
    ("gpu_eager_barge", "GPU, eager: user_speech_during_agent, 2 questions in 1 call"),
    ("cpu_pause", "CPU: user_pause, 3 questions in 1 call"),
    ("cpu_pause_seq", "CPU: user_pause, 3 questions in 3 calls"),
    ("cpu_barge", "CPU: user_speech_during_agent, 2 questions in 1 call"),
    ("livekit_cpu", "LiveKit text turn detector, CPU (D1 only)"),
    ("prompted_llm", "prompted LLM, end to end, network included"),
)


def table_b(root: Path = RESULTS_B) -> str:
    lines = [
        "| Path | p50 ms | p95 ms | p99 ms | of which forward, p50 | of which packing, p50 | "
        "Fits budget (p99) |",
        "|---|---|---|---|---|---|---|",
    ]
    for stem, label in LATENCY_ROWS:
        path = root / f"{stem}.json"
        if not path.exists():
            lines.append(f"| {label} | TODO | TODO | TODO | TODO | TODO | TODO |")
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        t = r["total_ms"]
        verdict = "yes" if r["fits_budget"] else "no"
        lines.append(
            f"| {label} | {t['p50']:.1f} | {t['p95']:.1f} | {t['p99']:.1f} | "
            f"{r['forward_ms']['p50']:.1f} | {r['pack_ms']['p50']:.1f} | "
            f"{verdict} (≤ {r['budget_p99_ms']:.0f} ms) |"
        )
    return "\n".join(lines)


def _latency_env(root: Path = RESULTS_B) -> str:
    files = sorted(root.glob("*.json"))
    if not files:
        return "Measured environment: TODO."
    env = json.loads(files[0].read_text(encoding="utf-8"))
    e, first = env["environment"], env
    return (
        f"Measured on {e.get('gpu')} (driver, power limit: {e.get('gpu_driver')}) and "
        f"{e.get('cpu')} with {e.get('torch_threads')} torch threads; on AC power: "
        f"{e.get('on_ac_power')}; torch {e.get('torch')}, laya {e.get('laya')}. "
        f"Batch 1, {first['warmup']} warmup and {first['iterations']} timed iterations over "
        f"{first['inputs']} fixed inputs per event; every timed call is a full `Decider.decide` "
        "(packing, tokenizing, forward, temperatures)."
    )


SECTIONS = {"table-a": table_a, "table-b": table_b, "table-b-env": _latency_env}


def _filler(body: str) -> Callable[[re.Match[str]], str]:
    def fill(m: re.Match[str]) -> str:
        return m.group(1) + body + m.group(2)

    return fill


def render_readme(text: str) -> str:
    """Every marked section, each rendered from its own results directory."""
    for name, fn in SECTIONS.items():
        pattern = re.compile(
            rf"(<!-- {name}:start -->\n).*?(\n<!-- {name}:end -->)", flags=re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"README has no {name} markers")
        text = pattern.sub(_filler(fn()), text)
    return text


def write_readme() -> bool:
    old = README.read_text(encoding="utf-8")
    new = render_readme(old)
    if new != old:
        with README.open("w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return new != old
