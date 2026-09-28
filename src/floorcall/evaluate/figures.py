"""Figures for the README, rendered from results/curves/*.json: `uv run floorcall eval figures`.

Colors follow the entity, never its rank: stock Laya is always categorical slot 1, the fine-tuned
model slot 2 (the reference palette of the dataviz method; validated with its script, see
DECISIONS.md D-027). Every figure is rendered twice, light and dark, from each mode's own palette
steps, and the README serves them through <picture>. One axis per chart; 2 px lines; markers of
at least 8 px with a surface-colored ring; hairline solid gridlines; text in ink colors, never in
a series color.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from floorcall.config import REPO_ROOT

CURVES = REPO_ROOT / "results" / "curves"
FIGURES = REPO_ROOT / "results" / "figures"

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926"),
    },
}
# entity -> fixed categorical slot
MODELS = {"stock_laya": (0, "stock Laya"), "finetuned_temp": (1, "fine-tuned + temperature")}
DECISION_TITLES = {
    "turn_complete": "D1 turn_complete",
    "barge_in": "D2 barge_in",
    "route": "D3 route",
    "escalate": "D4 escalate",
}


def _load() -> dict[str, dict[str, Any]]:
    out = {}
    for model in MODELS:
        path = CURVES / f"{model}.json"
        if path.exists():
            out[model] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _style(ax: Axes, t: dict[str, Any]) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=t["grid"], linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def _figure(t: dict[str, Any], size: tuple[float, float]) -> tuple[Figure, Any]:
    fig, axes = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def _legend(ax: Axes, t: dict[str, Any], loc: Literal["upper right", "upper left"]) -> None:
    leg = ax.legend(loc=loc, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def _point(ax: Axes, x: float, y: float, color: str, t: dict[str, Any], label: str) -> None:
    ax.plot(
        [x],
        [y],
        "o",
        color=color,
        markersize=7,
        markeredgecolor=t["surface"],
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=9, color=t["ink2"]
    )


def tradeoff_interrupt(data: dict[str, dict[str, Any]], t: dict[str, Any]) -> Figure:
    fig, ax = _figure(t, (6.4, 4.4))
    for model, d in data.items():
        if "interrupt" not in d:
            continue
        slot, name = MODELS[model]
        c, color = d["interrupt"]["test_curve"], t["series"][slot]
        ax.plot(
            c["false_stop"],
            c["missed"],
            color=color,
            linewidth=2,
            solid_capstyle="round",
            label=name,
        )
        at = d["interrupt"]["test_at_theta"]
        _point(ax, at["false_stop"], at["missed"], color, t, f"θ = {d['interrupt']['theta']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False stops: share of backchannels and noise the agent stops for")
    ax.set_ylabel("Missed interruptions: share talked over")
    ax.set_title(
        "θ_interrupt tradeoff on the D2 test set (point: θ chosen on calib)",
        fontsize=10,
        loc="left",
    )
    _style(ax, t)
    _legend(ax, t, "upper right")
    fig.tight_layout()
    return fig


def tradeoff_yield(data: dict[str, dict[str, Any]], t: dict[str, Any]) -> Figure:
    fig, ax = _figure(t, (6.4, 4.4))
    top = 0.0
    for model, d in data.items():
        if "yield" not in d:
            continue
        slot, name = MODELS[model]
        c, color = d["yield"]["test_curve"], t["series"][slot]
        ax.plot(
            c["premature"],
            c["added_delay_ms"],
            color=color,
            linewidth=2,
            solid_capstyle="round",
            label=name,
        )
        at = d["yield"]["test_at_theta"]
        _point(
            ax, at["premature"], at["added_delay_ms"], color, t, f"θ = {d['yield']['theta']:.2f}"
        )
        top = max(top, d["yield"]["max_wait_ms"] - d["yield"]["vad_pause_ms"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top or 1)
    ax.set_xlabel("Premature responses: share of unfinished turns answered")
    ax.set_ylabel("Mean added delay on finished turns (ms)")
    ax.set_title(
        "θ_yield tradeoff on the D1 test set (point: θ chosen on calib)", fontsize=10, loc="left"
    )
    _style(ax, t)
    _legend(ax, t, "upper right")
    fig.tight_layout()
    return fig


def reliability(d: dict[str, Any], t: dict[str, Any]) -> Figure:
    decisions = [k for k in DECISION_TITLES if k in d["reliability"]]
    cols = 2
    rows = (len(decisions) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 3.2 * rows), dpi=200, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax in axes.flat[len(decisions) :]:
        ax.set_visible(False)
    for ax, decision in zip(axes.flat, decisions, strict=False):
        r = d["reliability"][decision]
        ax.plot([0, 1], [0, 1], color=t["muted"], linewidth=1, label="perfect calibration")
        for slot, key, name in ((0, "before", "no temperature"), (1, "after", "with temperature")):
            bins = r[key]["bins"]
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                "-o",
                color=t["series"][slot],
                linewidth=2,
                markersize=5,
                markeredgecolor=t["surface"],
                markeredgewidth=1,
                label=f"{name} (ECE {r[key]['ece']:.3f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(DECISION_TITLES[decision], fontsize=10, loc="left")
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        _style(ax, t)
        _legend(ax, t, "upper left")
    fig.tight_layout()
    return fig


def render_all() -> list[Path]:
    data = _load()
    written: list[Path] = []
    FIGURES.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Any]] = []
    if any("interrupt" in d for d in data.values()):
        jobs.append(("tradeoff_interrupt", lambda t: tradeoff_interrupt(data, t)))
    if any("yield" in d for d in data.values()):
        jobs.append(("tradeoff_yield", lambda t: tradeoff_yield(data, t)))
    for model, d in data.items():
        if d.get("reliability"):
            jobs.append((f"reliability_{model}", lambda t, d=d: reliability(d, t)))
    for name, make in jobs:
        for mode, theme in THEMES.items():
            fig = make(theme)
            path = FIGURES / f"{name}.{mode}.png"
            fig.savefig(path, facecolor=theme["surface"])
            plt.close(fig)
            written.append(path)
    return written
