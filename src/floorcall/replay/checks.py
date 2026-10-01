"""A script's timing, checked against both agents' rules before any model runs (D-046).

An oracle agent replays the script doing exactly what it wants at every point, on floorcall's
timing. Every expectation must then arise as a decision point, and every decision point must have
an expectation. On top of that, each point's timing must make the contrast it is there to show:

- **A mid-thought pause** (want `keep_listening`) lasts longer than naive's timeout, so naive
  answers into it. It is shorter than `max_wait_ms`, so floorcall's safety net does not answer.
  It is at least `vad_pause_ms` plus the pause budget, so floorcall's decision lands before the
  user goes on. The user speaks next.
- **A finished turn** (want `respond`) is followed by at least `max_wait_ms` of silence, so even
  the safety net answers before the agent's scripted reply starts. The agent speaks next, if
  anyone does.
- **Speech over the agent** starts after the agent's first word, and floorcall's decision lands
  while the line is still playing, so stopping or not makes a difference.
"""

from __future__ import annotations

from floorcall.config import PolicySettings, ReplaySettings
from floorcall.pipeline.operating import Budget
from floorcall.questions import Event
from floorcall.replay.engine import OracleReplay, simulate
from floorcall.replay.script import Script


def check_timing(
    script: Script, policy: PolicySettings, budgets: dict[Event, Budget], cfg: ReplaySettings
) -> list[str]:
    run = simulate(
        script, OracleReplay(script, policy, budgets, cfg), vad_pause_ms=policy.vad_pause_ms
    )
    tl = script.timeline
    problems = [f"seg {m.seg}: expected {m.want}, but {m.why}" for m in run.missed]
    for o in run.outcomes:
        u = tl[o.seg]
        if o.want is None:
            problems.append(f"seg {o.seg}: a {o.event.value} arises with no expectation")
            continue
        later = [s for s in tl if s.at_ms >= u.end_ms]
        nxt = later[0] if later else None
        gap = None if nxt is None else nxt.at_ms - u.end_ms
        if o.event is Event.USER_SPEECH_DURING_AGENT:
            (line,) = [s for s in tl if s.who == "agent" and s.at_ms <= u.at_ms < s.end_ms]
            first_word = line.at_ms + line.dur_ms / len(line.words)
            if u.at_ms < first_word:
                problems.append(f"seg {o.seg}: starts before the agent has said a word")
            if o.act.effective_ms >= line.end_ms:
                problems.append(
                    f"seg {o.seg}: floorcall's decision lands at {o.act.effective_ms:.0f} ms, "
                    f"after the agent's line ends at {line.end_ms} ms"
                )
        elif o.want == "keep_listening":
            floor = policy.vad_pause_ms + budgets[Event.USER_PAUSE].ms
            if nxt is None or nxt.who != "user":
                problems.append(f"seg {o.seg}: a mid-thought pause must be followed by the user")
            elif not (cfg.naive_silence_ms < gap < policy.max_wait_ms and gap >= floor):  # type: ignore[operator]
                problems.append(
                    f"seg {o.seg}: mid-thought gap {gap} ms must be > naive's "
                    f"{cfg.naive_silence_ms} ms, < max_wait {policy.max_wait_ms} ms and "
                    f">= {floor:.0f} ms (VAD + decision budget)"
                )
        else:  # respond
            if nxt is not None and nxt.who != "agent":
                problems.append(f"seg {o.seg}: a finished turn must be followed by the agent")
            elif gap is not None and gap < policy.max_wait_ms:
                problems.append(
                    f"seg {o.seg}: {gap} ms before the agent's reply; it must be >= max_wait "
                    f"{policy.max_wait_ms} ms so every policy has answered by then"
                )
    return problems
