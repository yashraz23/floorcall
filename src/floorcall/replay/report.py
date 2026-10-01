"""The divergence report: what each agent did at each decision point, and where they differ.

The 8 scripts are illustrative demos, not an evaluation set; the evaluation is Tables A-D
(DECISIONS.md D-046). The report says so at the top of its summary, and the counts in the summary
are counts of scripted points, not accuracy figures.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from floorcall.pipeline.operating import Budget, ServedPolicy
from floorcall.questions import Event
from floorcall.replay.engine import Outcome, Run
from floorcall.replay.script import BARGE_WANTS, Script

DISCLAIMER = (
    "Illustrative demo scripts, not an evaluation set: the evaluation is Tables A-D. "
    "The counts below are scripted decision points, not accuracy."
)


def _s(ms: float) -> str:
    return f"{ms / 1000:.2f} s"


def _did(o: Outcome) -> str:
    out = o.act.action
    if o.event is Event.USER_PAUSE and o.act.action != "keep_listening":
        out += f", route {o.act.route or '- (no router)'}"
    if o.act.escalate:
        out += ", hand off to a human"
    elif o.want_escalate:
        out += ", no hand-off"
    return out


def _wanted(o: Outcome) -> str:
    out = o.want or "(no expectation)"
    if o.want_route:
        out += f", route {o.want_route}"
    if o.want_escalate:
        out += ", hand off to a human"
    return out


def _missed_want(script: Script, seg: int, event: Event) -> str:
    e = script.expectation(seg, event is Event.USER_SPEECH_DURING_AGENT)
    if e is None:
        return "(no expectation)"
    out = e.want + (f", route {e.route}" if e.route else "")
    return out + (", hand off to a human" if e.escalate else "")


def _mark(o: Outcome) -> str:
    return "  " if o.ok is None else "✓ " if o.ok else "✗ "


def _detail(o: Outcome, budgets: dict[Event, Budget]) -> str:
    a = o.act
    if a.compute_ms is None and a.budget_ms == 0:
        return a.reason
    bits = [a.reason]
    if a.heard and o.event is Event.USER_SPEECH_DURING_AGENT:
        bits.append(f'heard "{a.heard}" by {_s(a.decided_at_ms)}')
    bits.append(f"decided at {_s(a.decided_at_ms)} + {a.budget_ms:.1f} ms budget (GPU p50)")
    if a.compute_ms is not None:
        bits.append(f"compute {a.compute_ms:,.0f} ms on {a.device} (not on the timeline)")
    return " · ".join(bits)


def _key(o: Outcome) -> tuple[int, str]:
    return (o.seg, o.event.value)


def divergences(runs: dict[str, Run]) -> list[tuple[int, str]]:
    """(seg, event) where the agents acted differently, or only one of them had to decide."""
    names = list(runs)
    by = {n: {_key(o): o for o in runs[n].outcomes} for n in names}
    keys = sorted(set().union(*[set(b) for b in by.values()]))
    out = []
    for k in keys:
        acts = [
            (b[k].act.action, b[k].act.route, b[k].act.escalate) if k in b else None
            for b in by.values()
        ]
        if len(set(acts)) > 1:
            out.append(k)
    return out


def render_script(script: Script, runs: dict[str, Run], budgets: dict[Event, Budget]) -> list[str]:
    lines = [f"━━ {script.id} · {script.title}", f"   covers: {', '.join(script.covers)}"]
    if script.history:
        for h in script.history:
            lines.append(f'   (before)       {h.speaker:6s} "{h.text}"')
    by = {n: {_key(o): o for o in r.outcomes} for n, r in runs.items()}
    missed = {n: {(m.seg, m.want in BARGE_WANTS): m for m in r.missed} for n, r in runs.items()}
    split = set(divergences(runs))
    width = max(len(n) for n in runs)
    for i, seg in enumerate(script.timeline):
        lines.append(f'   {seg.at_ms / 1000:.2f}-{_s(seg.end_ms)}  {seg.who:6s} "{seg.text}"')
        if seg.who != "user":
            continue
        for event in (Event.USER_SPEECH_DURING_AGENT, Event.USER_PAUSE):
            k = (i, event.value)
            here = {n: b[k] for n, b in by.items() if k in b}
            gone = {
                n for n, m in missed.items() if (i, event is Event.USER_SPEECH_DURING_AGENT) in m
            }
            if not here and not gone:
                continue
            want_from = next(iter(here.values()), None)
            what = "speech over the agent" if event is Event.USER_SPEECH_DURING_AGENT else "pause"
            tag = "  ◆ the agents diverge" if k in split else ""
            wanted = _wanted(want_from) if want_from else _missed_want(script, i, event)
            lines.append(f"   ▸ {what} · want {wanted}{tag}")
            for n in runs:
                if n in here:
                    o = here[n]
                    lines.append(f"       {n:{width}s} {_mark(o)}{_did(o)}: {o.consequence}")
                    lines.append(f"       {'':{width}s}   {_detail(o, budgets)}")
                elif (i, event is Event.USER_SPEECH_DURING_AGENT) in missed[n]:
                    m = missed[n][(i, event is Event.USER_SPEECH_DURING_AGENT)]
                    mark = "✗ " if m.counts else "  "
                    note = "" if m.counts else " (not scored: nothing to stop)"
                    lines.append(f"       {n:{width}s} {mark}no decision: {m.why}{note}")
    return lines


def _score(runs: list[Run]) -> tuple[int, int]:
    scored = [o for r in runs for o in r.outcomes if o.ok is not None]
    missed = sum(m.counts for r in runs for m in r.missed)
    return sum(bool(o.ok) for o in scored), len(scored) + missed


def render_summary(
    scripts: list[Script],
    runs: dict[str, list[Run]],
    served: ServedPolicy,
    budgets: dict[Event, Budget],
    device: str,
) -> list[str]:
    n_points = sum(len(s.expect) for s in scripts)
    lines = [
        "━━ Summary",
        f"   {DISCLAIMER}",
        f"   {len(scripts)} scripts, {n_points} expected decision points",
    ]
    for name, rs in runs.items():
        ok, total = _score(rs)
        lines.append(f"   {name:9s} acted as wanted at {ok} of {total}")
    paired = [dict(zip(runs, r, strict=True)) for r in zip(*runs.values(), strict=True)]
    lines.append(f"   points where the agents diverge: {sum(len(divergences(p)) for p in paired)}")
    for event, b in budgets.items():
        lines.append(
            f"   timeline budget per {event.value} decision: {b.ms:.1f} ms "
            f"({b.source.file}, {b.source.key}: {b.label})"
        )
    computed = [
        o.act.compute_ms for rs in runs.values() for r in rs for o in r.outcomes if o.act.compute_ms
    ]
    if computed:
        lines.append(
            f"   measured compute in this replay, on {device}: p50 {statistics.median(computed):,.0f} ms, "
            f"max {max(computed):,.0f} ms over {len(computed)} forward passes (not on the timeline)"
        )
    for name, src in served.sources.items():
        lines.append(f"   {name} = {src.value:g}  ({src.file}, {src.key})")
    return lines


def to_json(
    scripts: list[Script],
    runs: dict[str, list[Run]],
    served: ServedPolicy,
    budgets: dict[Event, Budget],
    meta: dict[str, Any],
) -> dict[str, Any]:
    def run_json(r: Run) -> dict[str, Any]:
        d = asdict(r)
        for o in d["outcomes"]:
            o["event"] = Event(o["event"]).value
        return d

    names = list(runs)
    return {
        "note": DISCLAIMER,
        **meta,
        "thresholds": {k: asdict(v) for k, v in served.sources.items()},
        "budgets_ms": {e.value: asdict(b.source) | {"label": b.label} for e, b in budgets.items()},
        "scripts": [
            {
                "id": s.id,
                "runs": {n: run_json(runs[n][i]) for n in names},
                "divergences": [list(k) for k in divergences({n: runs[n][i] for n in names})],
            }
            for i, s in enumerate(scripts)
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
