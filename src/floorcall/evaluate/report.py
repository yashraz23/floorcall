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


SECTIONS = {"table-a": table_a}


def _filler(body: str) -> Callable[[re.Match[str]], str]:
    def fill(m: re.Match[str]) -> str:
        return m.group(1) + body + m.group(2)

    return fill


def render_readme(text: str, root: Path = RESULTS) -> str:
    for name, fn in SECTIONS.items():
        pattern = re.compile(
            rf"(<!-- {name}:start -->\n).*?(\n<!-- {name}:end -->)", flags=re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"README has no {name} markers")
        text = pattern.sub(_filler(fn(root)), text)
    return text


def write_readme() -> bool:
    old = README.read_text(encoding="utf-8")
    new = render_readme(old)
    if new != old:
        with README.open("w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return new != old
