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

from floorcall.config import REPO_ROOT, get_settings

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


def _title_and_legend(ax: Axes, t: dict[str, Any], title: str, names: list[str]) -> None:
    """A legend for two or more series. A single series is named in the title instead: a legend
    box with one swatch only restates it (dataviz method, labels and legend)."""
    if len(names) == 1:
        title = f"{title}: {names[0]}"
    # the color goes on set_title itself: a left-aligned title is a separate artist from
    # ax.title, so _style cannot recolor it (dark mode rendered it black on black)
    ax.set_title(title, fontsize=10, loc="left", color=t["ink"])
    if len(names) >= 2:
        _legend(ax, t, "upper right")


def _no_skill(ax: Axes, t: dict[str, Any], x: list[float], y: list[float]) -> None:
    """What a scorer independent of the label traces: the reference a real model must beat."""
    ax.plot(x, y, color=t["muted"], linewidth=1, zorder=1)
    ax.annotate(
        "no skill",
        ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2),
        # above the diagonal: stock Laya's curves run on it or just under it
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color=t["muted"],
        ha="left",
    )


def tradeoff_interrupt(data: dict[str, dict[str, Any]], t: dict[str, Any]) -> Figure:
    fig, ax = _figure(t, (6.4, 4.4))
    _no_skill(ax, t, [0, 1], [1, 0])
    names: list[str] = []
    for model, d in data.items():
        if "interrupt" not in d:
            continue
        slot, name = MODELS[model]
        names.append(name)
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
    _style(ax, t)
    _title_and_legend(ax, t, "θ_interrupt on the D2 test set (point: θ chosen on calib)", names)
    fig.tight_layout()
    return fig


def tradeoff_yield(data: dict[str, dict[str, Any]], t: dict[str, Any]) -> Figure:
    fig, ax = _figure(t, (6.4, 4.4))
    top = 0.0
    names: list[str] = []
    for model, d in data.items():
        if "yield" not in d:
            continue
        slot, name = MODELS[model]
        names.append(name)
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
    _no_skill(ax, t, [0, 1], [top or 1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top or 1)
    ax.set_xlabel("Premature responses: share of unfinished turns answered")
    ax.set_ylabel("Mean added delay on finished turns (ms)")
    _style(ax, t)
    _title_and_legend(ax, t, "θ_yield on the D1 test set (point: θ chosen on calib)", names)
    fig.tight_layout()
    return fig


def reliability(d: dict[str, Any], t: dict[str, Any], *, min_count: int) -> Figure:
    """Small multiples, one panel per decision, one shared legend below them.

    Bins holding fewer than `min_count` rows are left off the plot: a bin of two rows swings
    between accuracy 0 and 1 and says nothing. ECE is still computed over every row.
    """
    decisions = [k for k in DECISION_TITLES if k in d["reliability"]]
    cols = 2
    rows = (len(decisions) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 3.1 * rows + 0.5), dpi=200, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax in axes.flat[len(decisions) :]:
        ax.set_visible(False)
    for ax, decision in zip(axes.flat, decisions, strict=False):
        r = d["reliability"][decision]
        ax.plot([0, 1], [0, 1], color=t["muted"], linewidth=1, label="perfect calibration")
        for slot, key, name in ((0, "before", "no temperature"), (1, "after", "with temperature")):
            bins = [b for b in r[key]["bins"] if b["count"] >= min_count]
            if len(bins) < 2:  # say so, rather than let a series vanish on a small test set
                ax.text(
                    0.98,
                    0.04 + 0.07 * slot,
                    f"{name}: {'only 1' if bins else 'no'} bin holds ≥ {min_count} rows",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=7.5,
                    color=t["ink2"],
                )
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                "-o",
                color=t["series"][slot],
                linewidth=2,
                markersize=5,
                markeredgecolor=t["surface"],
                markeredgewidth=1,
                label=name,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ece = f"ECE {r['before']['ece']:.3f} → {r['after']['ece']:.3f}"
        ax.set_title(f"{DECISION_TITLES[decision]}   {ece}", fontsize=9, loc="left", color=t["ink"])
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        _style(ax, t)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def render_all() -> list[Path]:
    settings = get_settings()
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
            jobs.append(
                (
                    f"reliability_{model}",
                    lambda t, d=d: reliability(
                        d, t, min_count=settings.eval.reliability_min_bin_count
                    ),
                )
            )
    for name, make in jobs:
        for mode, theme in THEMES.items():
            fig = make(theme)
            path = FIGURES / f"{name}.{mode}.png"
            fig.savefig(path, facecolor=theme["surface"])
            plt.close(fig)
            written.append(path)
    return written
