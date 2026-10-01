"""What floorcall serves, read from the committed results files and never retyped.

- **The policy's thresholds** (D-045, D-046), each chosen on calib, never on test:
  - theta_interrupt and theta_yield from results/curves/finetuned_temp.json;
  - theta_escalate from results/table_a/escalate.finetuned_temp_threshold.json;
  - theta_oos from results/thresholds/route_oos.json.
- **Each decision's budget on replay's logical clock** (D-046): the measured GPU p50 of that
  event's Table B row. The row is the one matching the served configuration (CUDA graphs on or
  off, inference precision, one batched call).

A missing file stops replay and names the command that writes it; replay never falls back to a
default. A file computed for a different checkpoint than the others also stops it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floorcall.config import REPO_ROOT, PolicySettings, Settings
from floorcall.questions import Event

RESULTS = REPO_ROOT / "results"


@dataclass(frozen=True)
class Source:
    """One served number and where it was read from."""

    value: float
    file: str  # repo-relative
    key: str


@dataclass(frozen=True)
class ServedPolicy:
    policy: PolicySettings
    sources: dict[str, Source]  # theta name -> where it came from
    checkpoint: str


@dataclass(frozen=True)
class Budget:
    event: Event
    ms: float
    source: Source
    label: str  # the Table B row's own label


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read(path: Path, made_by: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{_rel(path)} is missing: run `{made_by}`")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _ckpt(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def served_policy(settings: Settings, results: Path = RESULTS) -> ServedPolicy:
    curves_path = results / "curves" / "finetuned_temp.json"
    esc_path = results / "table_a" / "escalate.finetuned_temp_threshold.json"
    oos_path = results / "thresholds" / "route_oos.json"
    curves = _read(curves_path, "floorcall eval curves --checkpoint <run>")
    esc = _read(esc_path, "floorcall eval checkpoint --checkpoint <run>")
    oos = _read(oos_path, "floorcall eval oos-threshold --checkpoint <run>")

    checkpoints = {_ckpt(curves["checkpoint"]), _ckpt(esc["checkpoint"]), _ckpt(oos["checkpoint"])}
    if len(checkpoints) != 1:
        raise ValueError(f"the served thresholds come from different checkpoints: {checkpoints}")
    p = settings.policy
    y = curves["yield"]
    # theta_yield is a point on a curve drawn with these two timings; serving it with others
    # would serve a different tradeoff than the one reported.
    if (y["vad_pause_ms"], y["max_wait_ms"]) != (p.vad_pause_ms, p.max_wait_ms):
        raise ValueError(
            f"theta_yield was chosen with vad_pause {y['vad_pause_ms']} ms and max_wait "
            f"{y['max_wait_ms']} ms; the policy is set to {p.vad_pause_ms} and {p.max_wait_ms}"
        )
    sources = {
        "theta_interrupt": Source(
            curves["interrupt"]["theta"], _rel(curves_path), "interrupt.theta"
        ),
        "theta_yield": Source(y["theta"], _rel(curves_path), "yield.theta"),
        "theta_escalate": Source(
            esc["threshold_choice"]["theta"], _rel(esc_path), "threshold_choice.theta"
        ),
        "theta_oos": Source(oos["choice"]["theta"], _rel(oos_path), "choice.theta"),
    }
    policy = p.model_copy(update={k: s.value for k, s in sources.items()})
    return ServedPolicy(policy=policy, sources=sources, checkpoint=checkpoints.pop())


BUDGET_ROWS = {Event.USER_PAUSE: "pause", Event.USER_SPEECH_DURING_AGENT: "barge"}


def decision_budgets(
    settings: Settings, checkpoint: str, results: Path = RESULTS
) -> dict[Event, Budget]:
    """Each event's GPU p50 from Table B, for the served configuration and checkpoint."""
    prefix = "gpu_graphed" if settings.laya.cuda_graphs else "gpu_eager"
    out = {}
    for event, suffix in BUDGET_ROWS.items():
        path = results / "table_b" / f"{prefix}_{suffix}.json"
        r = _read(path, "floorcall eval latency")
        expected = {
            "device": "cuda",
            "precision": settings.laya.precision,
            "event": event.value,
            "sequential": False,
        }
        got = {k: r.get(k) for k in expected}
        if got != expected or _ckpt(r["checkpoint"]) != _ckpt(checkpoint):
            raise ValueError(
                f"{_rel(path)} is not the served configuration: {got}, checkpoint "
                f"{r['checkpoint']}; expected {expected}, checkpoint {checkpoint}"
            )
        out[event] = Budget(
            event=event,
            ms=float(r["total_ms"]["p50"]),
            source=Source(float(r["total_ms"]["p50"]), _rel(path), "total_ms.p50"),
            label=r["label"],
        )
    return out
