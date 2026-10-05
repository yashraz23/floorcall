"""The Hugging Face Space's views, as HTML and Markdown strings (DECISIONS.md D-044, D-050).

The Space (space/app.py) is thin Gradio glue around these functions, so what it shows is tested
here without Gradio. Three tabs:

- **Replay** draws a committed replay run (results/replay/replay.json) over its frozen script:
  where each agent spoke, where its line was cut, and each decision point with ✓/✗. It makes no
  model call, so it is instant and exactly the committed run.
- **Try it** shows one live decision: the action, the calibrated probabilities, the packed state
  the model read, and the measured compute on the Space's CPU.
- **Results** is Tables A to D, Table C and the operating points, rendered from results/ by the
  README's renderer at staging time.

Nothing here contains real Twitter text: the replay scripts are written by hand (D-046), and the
results are numbers.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

# Colours chosen to read on both Gradio's light and dark themes.
USER = "#3b82f6"
AGENT = "#64748b"
OK = "#16a34a"
WRONG = "#dc2626"
NEUTRAL = "#94a3b8"

WANT_TEXT = {
    "keep_talking": "keep talking",
    "stop_and_listen": "stop and listen",
    "ignore": "ignore it",
    "respond": "respond",
    "keep_listening": "keep listening",
}


def _e(s: object) -> str:
    return html.escape(str(s), quote=True)


def score(replay: Mapping[str, Any], agent: str, script_id: str | None = None) -> tuple[int, int]:
    """(acted as wanted, scored points) for an agent, over one script or all of them.

    A point counts when it has an expectation, and an expected point that never arose for the agent
    counts as a miss, exactly as the replay report scores it (D-046).
    """
    ok = points = 0
    for s in replay["scripts"]:
        if script_id is not None and s["id"] != script_id:
            continue
        run = s["runs"][agent]
        for o in run["outcomes"]:
            if o["ok"] is not None:
                points += 1
                ok += bool(o["ok"])
        points += sum(1 for m in run["missed"] if m.get("counts"))
    return ok, points


def _points(script: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every decision point of a script, both agents side by side, in time order."""
    timeline = script["timeline"]
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for agent in ("naive", "floorcall"):
        run = result["runs"][agent]
        for o in run["outcomes"]:
            key = (o["seg"], o["event"])
            p = by_key.setdefault(key, {"seg": o["seg"], "event": o["event"], "want": o["want"]})
            p["want"] = p["want"] or o["want"]
            p["want_route"] = o.get("want_route") or p.get("want_route")
            p["want_escalate"] = o.get("want_escalate") or p.get("want_escalate")
            p[agent] = o
        for m in run["missed"]:
            key = (m["seg"], "user_speech_during_agent")
            p = by_key.setdefault(key, {"seg": m["seg"], "event": key[1], "want": m["want"]})
            p[f"{agent}_missed"] = m
    points = list(by_key.values())

    def when(p: Mapping[str, Any]) -> float:
        times = [float(p[a]["act"]["decided_at_ms"]) for a in ("naive", "floorcall") if a in p]
        seg = timeline[p["seg"]]
        return min(times) if times else float(seg["at_ms"] + seg["dur_ms"])

    points.sort(key=when)
    for i, p in enumerate(points, 1):
        p["n"] = i
        p["t"] = when(p)
    return points


def _verdict(o: Mapping[str, Any] | None) -> str:
    if o is None or o["ok"] is None:
        return "·"
    return "✓" if o["ok"] else "✗"


def _cell(p: Mapping[str, Any], agent: str) -> str:
    o = p.get(agent)
    missed = p.get(f"{agent}_missed")
    if o is None and missed is not None:
        return f'<span class="fc-bad">✗ no decision</span><div class="fc-why">{_e(missed["why"])}</div>'
    if o is None:
        return '<span class="fc-dim">—</span>'
    act = o["act"]
    cls = {"✓": "fc-good", "✗": "fc-bad"}.get(_verdict(o), "fc-dim")
    action = WANT_TEXT.get(act["action"], act["action"].replace("_", " "))
    extras = []
    if act.get("route"):
        extras.append(f"route {act['route']}")
    elif agent == "naive" and o["event"] == "user_pause" and act["action"].startswith("respond"):
        extras.append("no router")
    if act.get("escalate"):
        extras.append("hand off to a human")
    detail = f" · {', '.join(extras)}" if extras else ""
    out = (
        f'<span class="{cls}">{_verdict(o)} {_e(action)}{_e(detail)}</span>'
        f'<div class="fc-why">{_e(o["consequence"])}</div>'
    )
    if agent == "floorcall":
        out += f'<div class="fc-why">{_e(act["reason"])}</div>'
    return out


def _want(p: Mapping[str, Any]) -> str:
    if p["want"] is None:
        return '<span class="fc-dim">no expectation</span>'
    parts = [WANT_TEXT.get(p["want"], p["want"])]
    if p.get("want_route"):
        parts.append(f"route {p['want_route']}")
    if p.get("want_escalate"):
        parts.append("hand off")
    return _e(", ".join(parts))


def _lane_marks(points: Sequence[Mapping[str, Any]], agent: str) -> list[tuple[float, str, str]]:
    marks = []
    for p in points:
        o = p.get(agent)
        if o is None:
            continue
        act = o["act"]
        colour = {"✓": OK, "✗": WRONG}.get(_verdict(o), NEUTRAL)
        tip = f"{p['n']}. {act['action'].replace('_', ' ')}: {o['consequence']}"
        marks.append((float(act["effective_ms"]), colour, f"{p['n']}|{tip}"))
    return marks


def _cuts(result: Mapping[str, Any], agent: str) -> list[float]:
    return [
        float(o["act"]["effective_ms"])
        for o in result["runs"][agent]["outcomes"]
        if o["act"]["action"] == "stop_and_listen"
    ]


def timeline_svg(script: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Three lanes on one clock: the user's speech, then each agent's line and decisions."""
    points = _points(script, result)
    items = script["timeline"]
    end = float(max(i["at_ms"] + i["dur_ms"] for i in items) + 600)
    left, width = 118, 862

    def x(ms: float) -> float:
        return left + width * max(0.0, min(ms, end)) / end

    lanes = {"user": 34, "naive": 92, "floorcall": 150}
    parts = [
        f'<svg viewBox="0 0 1000 196" width="100%" role="img" '
        f'aria-label="Timeline of {_e(script["title"])}" style="font: 12px system-ui, sans-serif">',
        '<defs><pattern id="fc-unsaid" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="3" height="6" fill="{AGENT}" '
        'fill-opacity="0.35"/></pattern></defs>',
    ]
    for name, y in lanes.items():
        label = {"user": "user", "naive": "naive agent", "floorcall": "floorcall"}[name]
        parts.append(
            f'<text x="8" y="{y + 4}" fill="currentColor" font-weight="600">{label}</text>'
            f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" stroke="currentColor" '
            'stroke-opacity="0.15"/>'
        )
    for i in items:
        a, b = i["at_ms"], i["at_ms"] + i["dur_ms"]
        if i["who"] == "user":
            parts.append(
                f'<rect x="{x(a):.1f}" y="{lanes["user"] - 9}" width="{max(2.0, x(b) - x(a)):.1f}" '
                f'height="18" rx="4" fill="{USER}"><title>{_e(i["text"])}</title></rect>'
            )
            continue
        for agent in ("naive", "floorcall"):
            y = lanes[agent]
            cut = next((c for c in _cuts(result, agent) if a < c < b), None)
            said_to = cut if cut is not None else b
            parts.append(
                f'<rect x="{x(a):.1f}" y="{y - 8}" width="{max(2.0, x(said_to) - x(a)):.1f}" '
                f'height="16" rx="4" fill="{AGENT}"><title>{_e(i["text"])}</title></rect>'
            )
            if cut is not None:
                parts.append(
                    f'<rect x="{x(cut):.1f}" y="{y - 8}" width="{x(b) - x(cut):.1f}" height="16" '
                    f'rx="4" fill="url(#fc-unsaid)" stroke="{AGENT}" stroke-dasharray="3 3">'
                    f"<title>{(b - cut) / 1000:.1f} s of this line left unsaid</title></rect>"
                )
    for agent in ("naive", "floorcall"):
        y = lanes[agent]
        for ms, colour, tip in _lane_marks(points, agent):
            n, text = tip.split("|", 1)
            parts.append(
                f'<g><title>{_e(text)}</title><circle cx="{x(ms):.1f}" cy="{y + 21}" r="9" '
                f'fill="{colour}"/><text x="{x(ms):.1f}" y="{y + 25}" text-anchor="middle" '
                f'fill="#fff" font-size="11" font-weight="700">{_e(n)}</text></g>'
            )
    for s in range(0, int(end / 1000) + 1, 2):
        parts.append(
            f'<text x="{x(s * 1000):.1f}" y="192" text-anchor="middle" fill="currentColor" '
            f'fill-opacity="0.6" font-size="10">{s} s</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


STYLE = """
<style>
.fc-wrap { font-family: system-ui, sans-serif; line-height: 1.45; }
.fc-good { color: #16a34a; font-weight: 600; }
.fc-bad { color: #dc2626; font-weight: 600; }
.fc-dim { opacity: 0.65; }
.fc-why { font-size: 0.86em; opacity: 0.8; margin-top: 2px; }
.fc-note { font-size: 0.9em; opacity: 0.8; margin: 6px 0 12px; }
.fc-table { width: 100%; border-collapse: collapse; font-size: 0.92em; margin-top: 10px; }
.fc-table th, .fc-table td { border-top: 1px solid rgba(127,127,127,0.25); padding: 7px 8px;
  text-align: left; vertical-align: top; }
.fc-n { display: inline-block; min-width: 20px; height: 20px; border-radius: 10px;
  background: rgba(127,127,127,0.25); text-align: center; font-weight: 700; font-size: 0.8em;
  line-height: 20px; }
.fc-score { display: flex; gap: 18px; flex-wrap: wrap; margin: 8px 0; font-size: 1.05em; }
.fc-bar { height: 10px; border-radius: 5px; background: #3b82f6; }
.fc-bartrack { background: rgba(127,127,127,0.18); border-radius: 5px; width: 100%; }
.fc-action { display: inline-block; padding: 4px 12px; border-radius: 14px; font-weight: 700;
  background: rgba(59,130,246,0.15); }
</style>
"""


def replay_header(replay: Mapping[str, Any]) -> str:
    nok, npts = score(replay, "naive")
    fok, fpts = score(replay, "floorcall")
    budgets = replay["budgets_ms"]
    return (
        STYLE + '<div class="fc-wrap"><div class="fc-score">'
        f"<span>naive agent: <b>{nok} of {npts}</b> as wanted</span>"
        f"<span>floorcall: <b>{fok} of {fpts}</b> as wanted</span></div>"
        '<div class="fc-note">Eight scripted banking calls, frozen before floorcall first ran on '
        "them. They are illustrative demos, not an evaluation set: the evaluation is Tables A to D "
        "on the Results tab. The naive agent answers after "
        f"{replay['naive_silence_ms']} ms of silence and stops whenever the user speaks. Each "
        "floorcall decision costs its event's measured GPU p50 on the replay clock "
        f"({budgets['user_pause']['value']:.1f} ms for a pause, "
        f"{budgets['user_speech_during_agent']['value']:.1f} ms for speech over the agent). "
        "This is the committed replay run, shown as recorded.</div></div>"
    )


def replay_html(
    script: Mapping[str, Any], result: Mapping[str, Any], replay: Mapping[str, Any]
) -> str:
    """One script: its timeline and a table of every decision point, naive against floorcall."""
    points = _points(script, result)
    nok, npts = score(replay, "naive", script["id"])
    fok, fpts = score(replay, "floorcall", script["id"])
    history = "".join(
        f'<div class="fc-why">before: {_e(t["speaker"])} said “{_e(t["text"])}”</div>'
        for t in script.get("history", [])
    )
    rows = []
    for p in points:
        said = script["timeline"][p["seg"]]
        kind = "pause after" if p["event"] == "user_pause" else "speech over the agent"
        rows.append(
            f'<tr><td><span class="fc-n">{p["n"]}</span></td>'
            f'<td>{p["t"] / 1000:.2f} s · {kind}<div class="fc-why">“{_e(said["text"])}”</div></td>'
            f"<td>{_want(p)}</td><td>{_cell(p, 'naive')}</td><td>{_cell(p, 'floorcall')}</td></tr>"
        )
    return (
        STYLE + '<div class="fc-wrap">'
        f'<h3 style="margin:4px 0">{_e(script["title"])}</h3>'
        f'<div class="fc-note">covers: {_e(", ".join(script.get("covers", [])))} · naive '
        f"{nok}/{npts} · floorcall {fok}/{fpts}</div>{history}"
        + timeline_svg(script, result)
        + '<table class="fc-table"><thead><tr><th>#</th><th>when, and what the user said</th>'
        "<th>wanted</th><th>naive agent</th><th>floorcall</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def try_it_html(event: Mapping[str, Any], packed_state: Mapping[str, Any]) -> str:
    """One live decision: action, calibrated probabilities per question, and what the model read."""
    action = event["action"].replace("_", " ")
    extras = []
    if event.get("route"):
        extras.append(f"route: <b>{_e(event['route'])}</b>")
    if event.get("escalate"):
        extras.append("<b>hand off to a human</b>")
    blocks = []
    for qid, probs in event["probabilities"].items():
        bars = "".join(
            f'<div style="display:grid;grid-template-columns:150px 1fr 52px;gap:8px;'
            f'align-items:center;margin:3px 0"><span>{_e(label)}</span>'
            f'<div class="fc-bartrack"><div class="fc-bar" style="width:{100 * p:.1f}%"></div></div>'
            f"<span>{p:.3f}</span></div>"
            for label, p in sorted(probs.items(), key=lambda kv: -kv[1])[:6]
        )
        blocks.append(f'<div style="margin:10px 0"><b>{_e(qid)}</b>{bars}</div>')
    compute = event.get("compute_ms")
    timing = (
        f"forward pass and packing: {compute:,.0f} ms on {_e(event['device'])}"
        if compute is not None
        else "no forward pass"
    )
    return (
        STYLE + '<div class="fc-wrap">'
        f'<div><span class="fc-action">{_e(action)}</span> {" · ".join(extras)}</div>'
        f'<div class="fc-why">{_e(event["reason"])}</div>'
        + "".join(blocks)
        + f'<div class="fc-note">{timing}; the state used {event["state_tokens"]} tokens. This '
        "Space shares a small CPU, so it is far slower than the GPU path in Table B.</div>"
        "<details><summary>the state the model read (ASR-style normalized)</summary><pre "
        'style="white-space:pre-wrap">'
        + _e(json.dumps(packed_state, indent=2, ensure_ascii=False))
        + "</pre></details></div>"
    )


def results_markdown() -> str:
    """Tables A to D, Table C and the operating points, rendered from results/ (staging time)."""
    from floorcall.evaluate.report import SECTIONS

    return "\n\n".join(
        [
            "All numbers are on frozen test sets that no training, calibration or threshold "
            "choice ever saw, rendered from the committed results files by the same code as the "
            "repository's README. The fine-tuned rows are the released checkpoint, "
            "[enz23/floorcall](https://huggingface.co/enz23/floorcall).",
            "### Table A: quality per decision",
            SECTIONS["table-a"](),
            "### Table B: latency, batch 1 (budgets: p99 ≤ 50 ms GPU, ≤ 100 ms CPU)",
            SECTIONS["table-b"](),
            SECTIONS["table-b-env"](),
            "### Table C: robustness to ASR-style noise (macro-F1, accuracy in brackets)",
            SECTIONS["table-c"](),
            "### Table D: ablations, trained on Kaggle against a full arm of the same recipe",
            SECTIONS["table-d"](),
            "Each variant minus the full arm, paired-bootstrap 95% intervals:",
            SECTIONS["table-d-paired"](),
            "### Operating points, chosen on calib",
            SECTIONS["operating-points"](),
        ]
    )
