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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from floorcall.config import REPO_ROOT

RESULTS = REPO_ROOT / "results" / "table_a"
RESULTS_B = REPO_ROOT / "results" / "table_b"
RESULTS_C = REPO_ROOT / "results" / "table_c"
RESULTS_D = REPO_ROOT / "results" / "table_d"
RESULTS_CURVES = REPO_ROOT / "results" / "curves"
FIGURES = REPO_ROOT / "results" / "figures"
README = REPO_ROOT / "README.md"

DECISION_NAMES = {
    "turn_complete": "D1 turn_complete",
    "barge_in": "D2 barge_in",
    "route": "D3 route",
    "escalate": "D4 escalate",
}
# (results file model id, row label, the decisions it applies to or None for all), in the order
# CLAUDE.md §12 lists Table A's rows. The calib-threshold rows exist for binary decisions whose
# threshold is chosen on calib (D-035).
THRESHOLD_ROWS = ("escalate",)
MODELS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("majority", "majority class (train prior)", None),
    ("lexical_rule", "lexical rule: backchannel words + length", ("barge_in",)),
    ("tfidf_lr", "TF-IDF + logistic regression", None),
    ("stock_laya", "stock Laya, zero-shot", None),
    ("stock_laya_threshold", "stock Laya, calib threshold", THRESHOLD_ROWS),
    ("finetuned", "fine-tuned", None),
    ("finetuned_temp", "fine-tuned + temperature", None),
    ("finetuned_temp_threshold", "fine-tuned + temperature, calib threshold", THRESHOLD_ROWS),
    ("prompted_llm", "prompted LLM, stated probabilities", None),
)


def _load(decision: str, model: str, root: Path) -> dict[str, Any] | None:
    path = root / f"{decision}.{model}.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(m: dict[str, Any], name: str) -> str:
    """A metric with its bootstrap 95% interval when the row has one (accuracy falls back to
    Wilson's)."""
    x = m[name]
    if x is None:
        return "n/a"
    ci = (m.get("ci95") or {}).get(name)
    if ci is None and name == "accuracy":
        ci = m["accuracy_ci95"]
    return f"{x:.3f}" if ci is None else f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def table_a(root: Path = RESULTS) -> str:
    lines = [
        "| Decision | Model | Accuracy [95% CI] | Macro-F1 | ECE | Brier | Hard-subset acc. (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for decision, name in DECISION_NAMES.items():
        for model, label, only in MODELS:
            if only is not None and decision not in only:
                continue
            r = _load(decision, model, root)
            if r is None:
                lines.append(f"| {name} | {label} | TODO | TODO | TODO | TODO | TODO |")
                continue
            if "prior_split" in r:
                label = f"majority class ({r['prior_split']} prior)"
            m = r["metrics"]
            if m.get("threshold") is not None:
                label = f"{label} (θ = {m['threshold']:.3f})"
            hard = (
                "n/a"
                if m["hard_accuracy"] is None
                else f"{_ci(m, 'hard_accuracy')} ({m['hard_n']})"
            )
            lines.append(
                f"| {name} | {label} | {_ci(m, 'accuracy')} | {_ci(m, 'macro_f1')} | "
                f"{_ci(m, 'ece')} | {_ci(m, 'brier')} | {hard} |"
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
    (
        "prompted_llm",
        "prompted LLM (gpt-oss-20b via OpenRouter, pinned DeepInfra bf16), user_pause, "
        "3 questions in 1 call: network latency, request to parsed answer",
    ),
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
        parts = [f"{r[k]['p50']:.1f}" if k in r else "n/a" for k in ("forward_ms", "pack_ms")]
        lines.append(
            f"| {label} | {t['p50']:.1f} | {t['p95']:.1f} | {t['p99']:.1f} | "
            f"{parts[0]} | {parts[1]} | {verdict} (≤ {r['budget_p99_ms']:.0f} ms) |"
        )
    return "\n".join(lines)


def table_b_gpu(root: Path = RESULTS_B) -> str:
    """Stock and fine-tuned GPU rows side by side, each with the GPU's state while it was timed
    (D-037). Every value comes from the rows' telemetry; nothing is typed by hand."""
    lines = [
        "| Path | Stock p50 / p99 ms | Fine-tuned p50 / p99 ms | Fine-tuned vs stock, p50 | "
        "GPU max °C (stock / fine-tuned) | SM clock median, MHz | Power median / limit, W | "
        "Throttled while timed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stem, label in LATENCY_ROWS:
        if not stem.startswith("gpu_"):
            continue
        paths = (root / "stock" / f"{stem}.json", root / f"{stem}.json")
        if not all(p.exists() for p in paths):
            lines.append(f"| {label} | TODO | TODO | TODO | TODO | TODO | TODO | TODO |")
            continue
        s, f = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
        ts, tf = s["telemetry"]["timed"], f["telemetry"]["timed"]
        change = (f["total_ms"]["p50"] / s["total_ms"]["p50"] - 1) * 100
        throttled = (
            s["telemetry"]["throttle_reasons_while_timed"]
            + f["telemetry"]["throttle_reasons_while_timed"]
        )
        lines.append(
            f"| {label} | {s['total_ms']['p50']:.1f} / {s['total_ms']['p99']:.1f} | "
            f"{f['total_ms']['p50']:.1f} / {f['total_ms']['p99']:.1f} | {change:+.1f}% | "
            f"{ts['temperature_c']['max']:.0f} / {tf['temperature_c']['max']:.0f} | "
            f"{ts['sm_clock_mhz']['median']:.0f} / {tf['sm_clock_mhz']['median']:.0f} | "
            f"{ts['power_draw_w']['median']:.0f} / {tf['power_draw_w']['median']:.0f} of "
            f"{tf['power_limit_w']['max']:.0f} | "
            f"{', '.join(sorted(set(throttled))) or 'no'} |"
        )
    return "\n".join(lines)


PRECISIONS = ("fp32", "bf16", "fp16")
PARITY = REPO_ROOT / "results" / "precision" / "parity.json"


def table_b_precision(root: Path = RESULTS_B, parity_path: Path = PARITY) -> str:
    """The fine-tuned checkpoint at each inference precision, measured back to back (D-039), with
    each precision's parity against fp32 on calib and dev."""
    head = " | ".join(f"{p} p50 / p99 ms" for p in PRECISIONS)
    lines = [
        f"| Path | {head} | GPU max °C ({' / '.join(PRECISIONS)}) | Throttled while timed |",
        "|---|" + "---|" * (len(PRECISIONS) + 2),
    ]
    for stem, label in LATENCY_ROWS:
        if not stem.startswith("gpu_"):
            continue
        paths = [root / "precision" / p / f"{stem}.json" for p in PRECISIONS]
        rows = [json.loads(p.read_text(encoding="utf-8")) if p.exists() else None for p in paths]
        cells = [
            f"{r['total_ms']['p50']:.1f} / {r['total_ms']['p99']:.1f}" if r else "TODO"
            for r in rows
        ]
        temps = " / ".join(
            f"{r['telemetry']['timed']['temperature_c']['max']:.0f}" if r else "-" for r in rows
        )
        seen = sorted(
            {x for r in rows if r for x in r["telemetry"]["throttle_reasons_while_timed"]}
        )
        throttled = ", ".join(seen) or ("no" if any(rows) else "TODO")
        lines.append(f"| {label} | {' | '.join(cells)} | {temps} | {throttled} |")
    if parity_path.exists():
        report = json.loads(parity_path.read_text(encoding="utf-8"))
        parts = []
        for p, entry in report["precisions"].items():
            v = entry["verdict"]
            parts.append(
                f"{p}: worst argmax agreement {v['worst_argmax_agreement']:.2%} "
                f"({v['worst_set']}), largest probability difference "
                f"{v['max_abs_prob_diff']:.3f}, {'passed' if v['passes'] else 'FAILED'}"
            )
        lines.append("")
        lines.append(
            f"Parity against fp32 on calib and dev ({sum(report['sets'].values())} rows, never "
            f"test; bar {next(iter(report['precisions'].values()))['verdict']['min_agreement_required']:.1%}): "
            + "; ".join(parts)
            + "."
        )
    return "\n".join(lines)


def latency_power(root: Path = RESULTS_B) -> str:
    """How power-limited the GPU was in each latency session, from the rows' telemetry. A session
    is the rows of one run, identified by the commit it ran at."""
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(root.rglob("gpu_*.json"))]
    rows = [r for r in rows if "telemetry" in r]
    if not rows:
        return "GPU power while timed: TODO."
    sessions: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sessions.setdefault(r["code"], []).append(r)
    lines = []
    for code, rs in sorted(sessions.items(), key=lambda kv: kv[1][0].get("started_at", "")):
        timed = [r["telemetry"]["timed"] for r in rs]
        lo = min(t["power_limit_w"]["min"] for t in timed if t["power_limit_w"])
        hi = max(t["power_limit_w"]["max"] for t in timed if t["power_limit_w"])
        medians = [t["power_draw_w"]["median"] for t in timed]
        capped = sum(t["clock_events"]["sw_power_cap"] for t in timed)
        samples = sum(t["samples"] for t in timed)
        env = rs[0]["environment"]
        mode = (env.get("windows_power_mode_ac") or {}).get("name", "not recorded")
        limit = f"{lo:.0f} W" if lo == hi else f"{lo:.0f} to {hi:.0f} W"
        lines.append(
            f"- Session `{code}` ({len(rs)} rows, from {rs[0].get('started_at', '?')[:10]} UTC): "
            f"enforced power limit {limit}; each row's median draw while timed {min(medians):.0f} "
            f"to {max(medians):.0f} W; the driver's power cap active in {capped / samples:.0%} of "
            f"{samples} timed samples; Windows power mode: {mode}."
        )
    return (
        "GPU power while timed, from nvidia-smi every 500 ms. Every GPU row ran power-limited:\n\n"
        + "\n".join(lines)
    )


def _latency_env(root: Path = RESULTS_B) -> str:
    files = sorted(root.glob("*.json"))
    if not files:
        return "Measured environment: TODO."
    # The newest session describes the machine best: GPU rows (D-037) carry a start time.
    gpu = [f for f in files if f.name.startswith("gpu_")]
    env = json.loads((gpu or files)[0].read_text(encoding="utf-8"))
    e, first = env["environment"], env
    when = f" GPU rows measured {env['started_at'][:10]} (UTC)." if "started_at" in env else ""
    mode = (e.get("windows_power_mode_ac") or {}).get("name")
    power = e.get("gpu_power") or {}
    if mode:
        when += f" Windows power mode: {mode}."
    if power.get("default_w"):
        when += (
            f" GPU power limit: vendor default, {float(power['default_w']):.0f} W base plus "
            f"Dynamic Boost ({float(power['enforced_w']):.0f} W enforced at the start)."
        )
    return (
        f"Measured on {e.get('gpu')} (driver {str(e.get('gpu_driver')).split(',')[0]}) and "
        f"{e.get('cpu')} with {e.get('torch_threads')} torch threads; on AC power: "
        f"{e.get('on_ac_power')}; torch {e.get('torch')}, laya {e.get('laya')}.{when} "
        f"Batch 1, {first['warmup']} warmup and {first['iterations']} timed iterations over "
        f"{first['inputs']} fixed inputs per event; every timed call is a full `Decider.decide` "
        "(packing, tokenizing, forward, temperatures)."
    )


NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2)


def _cell(path: Path) -> str:
    if not path.exists():
        return "TODO"
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return f"{m['macro_f1']:.3f} ({m['accuracy']:.3f})"


def table_c(root: Path = RESULTS_C) -> str:
    head = " | ".join(f"noise {lvl:.2f}" for lvl in NOISE_LEVELS)
    lines = [f"| Decision | {head} |", "|---|" + "---|" * len(NOISE_LEVELS)]
    for decision, name in DECISION_NAMES.items():
        cells = " | ".join(_cell(root / f"{decision}.{lvl:.2f}.json") for lvl in NOISE_LEVELS)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


ABLATION_ROWS = (
    ("full", "full model"),
    ("no_history", "without recent_turns"),
    ("no_agent", "without agent_last_utterance"),
    ("no_normalize.written", "without normalization, scored on written text"),
    ("no_normalize.asr", "without normalization, scored on ASR-style text"),
)


def _hard(path: Path) -> str:
    if not path.exists():
        return "TODO"
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return _f(m["hard_accuracy"])


RESULTS_PAIRED_D = REPO_ROOT / "results" / "paired" / "table_d"


def _paired(path: Path, metric: str) -> str:
    if not path.exists():
        return "TODO"
    d = json.loads(path.read_text(encoding="utf-8"))["differences"].get(metric)
    if d is None:
        return "n/a"
    lo, hi = d["ci95"]
    return f"{d['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def table_d_paired(root: Path = RESULTS_PAIRED_D) -> str:
    """Each ablation minus the Kaggle full arm, with paired-bootstrap 95% intervals (D-047)."""
    lines = [
        "| Variant minus full arm | D1 macro-F1 | D1 hard acc. | D2 macro-F1 | D2 hard acc. | "
        "D3 macro-F1 | D4 macro-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, label in ABLATION_ROWS:
        if variant == "full":
            continue
        f = {d: root / f"{d}.{variant}_vs_full.json" for d in DECISION_NAMES}
        lines.append(
            f"| {label} | {_paired(f['turn_complete'], 'macro_f1')} | "
            f"{_paired(f['turn_complete'], 'hard_accuracy')} | "
            f"{_paired(f['barge_in'], 'macro_f1')} | {_paired(f['barge_in'], 'hard_accuracy')} | "
            f"{_paired(f['route'], 'macro_f1')} | {_paired(f['escalate'], 'macro_f1')} |"
        )
    return "\n".join(lines)


def table_d(root: Path = RESULTS_D) -> str:
    lines = [
        "| Variant | D1 macro-F1 (acc) | D1 hard acc. | D2 macro-F1 (acc) | D2 hard acc. | "
        "D3 macro-F1 (acc) | D4 macro-F1 (acc) |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, label in ABLATION_ROWS:
        f = {d: root / f"{variant}.{d}.json" for d in DECISION_NAMES}
        lines.append(
            f"| {label} | {_cell(f['turn_complete'])} | {_hard(f['turn_complete'])} | "
            f"{_cell(f['barge_in'])} | {_hard(f['barge_in'])} | {_cell(f['route'])} | "
            f"{_cell(f['escalate'])} |"
        )
    return "\n".join(lines)


CURVE_MODELS = (("stock_laya", "stock Laya"), ("finetuned_temp", "fine-tuned + temperature"))


def operating_points(root: Path = RESULTS_CURVES) -> str:
    lines = [
        "| Model | θ_interrupt | false stops (test) | missed interruptions (test) | θ_yield | "
        "premature responses (test) | added delay, ms (test) |",
        "|---|---|---|---|---|---|---|",
    ]
    for stem, label in CURVE_MODELS:
        path = root / f"{stem}.json"
        if not path.exists():
            lines.append(f"| {label} | TODO | TODO | TODO | TODO | TODO | TODO |")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        i, y = d.get("interrupt"), d.get("yield")
        ic = (
            f"{i['theta']:.3f}{'' if i['feasible_on_calib'] else ' (target unreachable)'} | "
            f"{i['test_at_theta']['false_stop']:.3f} | {i['test_at_theta']['missed']:.3f}"
            if i
            else "TODO | TODO | TODO"
        )
        yc = (
            f"{y['theta']:.3f}{'' if y['feasible_on_calib'] else ' (target unreachable)'} | "
            f"{y['test_at_theta']['premature']:.3f} | {y['test_at_theta']['added_delay_ms']:.0f}"
            if y
            else "TODO | TODO | TODO"
        )
        lines.append(f"| {label} | {ic} | {yc} |")
    return "\n".join(lines)


FIGURE_CAPTIONS = (
    ("tradeoff_interrupt", "θ_interrupt: false stops against missed interruptions"),
    ("tradeoff_yield", "θ_yield: premature responses against added delay"),
    ("reliability_finetuned_temp", "Reliability, fine-tuned model, before and after temperature"),
    ("reliability_stock_laya", "Reliability, stock Laya: raw logits and shipped temperatures"),
)


def figures(root: Path = FIGURES) -> str:
    blocks = []
    for stem, caption in FIGURE_CAPTIONS:
        light, dark = root / f"{stem}.light.png", root / f"{stem}.dark.png"
        if not (light.exists() and dark.exists()):
            blocks.append(f"*{caption}*: TODO")
            continue
        rel = "results/figures"
        blocks.append(
            f"<picture>\n"
            f'  <source media="(prefers-color-scheme: dark)" srcset="{rel}/{dark.name}">\n'
            f'  <img alt="{caption}" src="{rel}/{light.name}" width="640">\n'
            f"</picture>"
        )
    return "\n\n".join(blocks)


SECTIONS = {
    "table-a": table_a,
    "table-b": table_b,
    "table-b-env": _latency_env,
    "table-b-gpu": table_b_gpu,
    "table-b-precision": table_b_precision,
    "table-b-power": latency_power,
    "table-c": table_c,
    "table-d": table_d,
    "table-d-paired": table_d_paired,
    "operating-points": operating_points,
    "figures": figures,
}


def _filler(body: str) -> Callable[[re.Match[str]], str]:
    def fill(m: re.Match[str]) -> str:
        return m.group(1) + body + m.group(2)

    return fill


def render_readme(text: str, sections: Mapping[str, Callable[[], str]] = SECTIONS) -> str:
    """Every marked section, each rendered from its own results directory. The model card
    (floorcall.release) renders a subset of the same sections the same way."""
    for name, fn in sections.items():
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
