"""Replay a script through one agent on a logical clock (DECISIONS.md D-046).

The user's speech is fixed. The agent's lines start at their scripted times, and a line the agent
stops is cut at the moment the stop takes effect. Events fire as a pipeline would fire them:

- **Speech while the agent is talking** fires `user_speech_during_agent` at its onset. Naive
  stops right there. floorcall decides once the speech ends, or after `barge_window_ms` of it,
  whichever comes first; the words heard by then are what it reads.
- **Silence of `vad_pause_ms` or more after the user's speech**, with the agent quiet, fires
  `user_pause`. Naive responds once the silence reaches its timeout. floorcall decides at
  `vad_pause_ms`. If it keeps listening and the silence reaches `max_wait_ms`, the safety net
  responds.

**Time is logical.** Each floorcall decision takes the event's budget: the GPU p50 measured for
that event in Table B (pipeline.operating). The compute time actually measured in replay (on CPU,
by default) is reported beside it, labelled with its device, and never moves the timeline. Naive
makes no model call, so its decisions take no time. VAD's own lag applies to both agents equally,
so it is not modelled.

What replay is not: a closed-loop simulation. The user does not react to the agent. If an agent
answers while the user is mid-thought, the report says it cut the user off; the recording then
plays on as recorded.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from floorcall.config import PolicySettings, ReplaySettings
from floorcall.pipeline.naive import NaiveAgent
from floorcall.pipeline.operating import Budget
from floorcall.pipeline.processor import DecisionProcessor
from floorcall.policy import BargeAction, PauseAction
from floorcall.questions import Event
from floorcall.replay.script import Script, Segment
from floorcall.state import Snapshot, Turn

SnapshotAt = Callable[[float], Snapshot]


@dataclass(frozen=True)
class Act:
    """One agent's decision at one event."""

    action: str
    route: str | None
    escalate: bool
    reason: str
    decided_at_ms: float  # logical time the decision is made (the model's input is from then)
    effective_ms: float  # logical time the action takes effect
    budget_ms: float  # effective - decided: the logical cost of deciding
    compute_ms: float | None = None  # measured, on `device`; None: no forward pass
    device: str | None = None
    probabilities: dict[str, dict[str, float]] | None = None
    heard: str = ""  # the user's words the decision read
    timer: bool = False  # the safety net answered, not the model


@dataclass(frozen=True)
class Outcome:
    seg: int
    event: Event
    act: Act
    consequence: str
    want: str | None = None  # None: no expectation here (unscored)
    want_route: str | None = None
    want_escalate: bool | None = None
    ok: bool | None = None


@dataclass(frozen=True)
class Missed:
    """An expected decision point that never arose for this agent, and why."""

    seg: int
    want: str
    why: str
    # False when not deciding did no harm: asked to stop, the agent was already silent.
    counts: bool = True


@dataclass
class Run:
    agent: str
    outcomes: list[Outcome] = field(default_factory=list)
    missed: list[Missed] = field(default_factory=list)


class ReplayAgent(Protocol):
    name: str

    def barge(self, seg: int, at: SnapshotAt, onset_ms: float, end_ms: float) -> Act: ...

    def pause(self, seg: int, at: SnapshotAt, end_ms: float, gap_ms: float | None) -> Act: ...


# -- the agents ---------------------------------------------------------------------------------


def _silence_ok(gap_ms: float | None, silence_ms: float) -> bool:
    return gap_ms is None or gap_ms >= silence_ms


class FloorcallReplay:
    name = "floorcall"

    def __init__(
        self,
        processor: DecisionProcessor,
        policy: PolicySettings,
        budgets: dict[Event, Budget],
        cfg: ReplaySettings,
    ) -> None:
        self.processor, self.policy, self.budgets, self.cfg = processor, policy, budgets, cfg

    def barge(self, seg: int, at: SnapshotAt, onset_ms: float, end_ms: float) -> Act:
        t = min(end_ms, onset_ms + self.cfg.barge_window_ms)
        snap = at(t)
        ev = self.processor.on_user_speech_during_agent(snap)
        b = self.budgets[Event.USER_SPEECH_DURING_AGENT].ms
        return Act(
            action=ev.action,
            route=None,
            escalate=ev.escalate,
            reason=ev.reason,
            decided_at_ms=t,
            effective_ms=t + b,
            budget_ms=b,
            compute_ms=ev.compute_ms,
            device=ev.device,
            probabilities=ev.probabilities,
            heard=snap.user_partial,
        )

    def pause(self, seg: int, at: SnapshotAt, end_ms: float, gap_ms: float | None) -> Act:
        t = end_ms + self.policy.vad_pause_ms
        snap = at(t)
        ev = self.processor.on_user_pause(snap, silence_ms=self.policy.vad_pause_ms)
        b = self.budgets[Event.USER_PAUSE].ms
        common: dict[str, Any] = {
            "compute_ms": ev.compute_ms,
            "device": ev.device,
            "probabilities": ev.probabilities,
            "heard": snap.user_partial,
            "decided_at_ms": t,
            "budget_ms": b,
        }
        if ev.action == PauseAction.KEEP_LISTENING and _silence_ok(gap_ms, self.policy.max_wait_ms):
            late = self.processor.after_silence(ev, silence_ms=self.policy.max_wait_ms)
            return Act(
                action=late.action,
                route=late.route,
                escalate=late.escalate,
                reason=f"{ev.reason}; then {late.reason}",
                effective_ms=end_ms + self.policy.max_wait_ms,
                timer=True,
                **common,
            )
        return Act(
            action=ev.action,
            route=ev.route,
            escalate=ev.escalate,
            reason=ev.reason,
            effective_ms=t + b,
            **common,
        )


class NaiveReplay:
    name = "naive"

    def __init__(self, naive: NaiveAgent) -> None:
        self.naive = naive

    def barge(self, seg: int, at: SnapshotAt, onset_ms: float, end_ms: float) -> Act:
        d = self.naive.on_user_speech_during_agent()
        return Act(
            action=d.action.value,
            route=None,
            escalate=d.escalate,
            reason=d.reason,
            decided_at_ms=onset_ms,
            effective_ms=onset_ms,
            budget_ms=0.0,
            heard="",
        )

    def pause(self, seg: int, at: SnapshotAt, end_ms: float, gap_ms: float | None) -> Act:
        respond = _silence_ok(gap_ms, self.naive.silence_ms)
        silence = self.naive.silence_ms if respond else gap_ms
        assert silence is not None
        d = self.naive.on_user_pause(silence_ms=silence)
        return Act(
            action=d.action.value,
            route=d.route,
            escalate=d.escalate,
            reason=d.reason if respond else f"the user resumed after {gap_ms:.0f} ms of silence",
            decided_at_ms=end_ms + silence,
            effective_ms=end_ms + silence,
            budget_ms=0.0,
            timer=True,
        )


class OracleReplay:
    """Does exactly what the script wants, on floorcall's timing. Replay uses it only to check a
    script's timing before any model runs (check_timing)."""

    name = "oracle"

    def __init__(
        self,
        script: Script,
        policy: PolicySettings,
        budgets: dict[Event, Budget],
        cfg: ReplaySettings,
    ) -> None:
        self.script, self.policy, self.budgets, self.cfg = script, policy, budgets, cfg

    def _want(self, seg: int, over_agent: bool, fallback: str) -> tuple[str, str | None, bool]:
        e = self.script.expectation(seg, over_agent)
        if e is None:
            return fallback, None, False
        return e.want, e.route, bool(e.escalate)

    def barge(self, seg: int, at: SnapshotAt, onset_ms: float, end_ms: float) -> Act:
        want, _, esc = self._want(seg, True, BargeAction.KEEP_TALKING.value)
        t = min(end_ms, onset_ms + self.cfg.barge_window_ms)
        b = self.budgets[Event.USER_SPEECH_DURING_AGENT].ms
        return Act(want, None, esc, "oracle", t, t + b, b)

    def pause(self, seg: int, at: SnapshotAt, end_ms: float, gap_ms: float | None) -> Act:
        want, route, esc = self._want(seg, False, PauseAction.KEEP_LISTENING.value)
        t = end_ms + self.policy.vad_pause_ms
        b = self.budgets[Event.USER_PAUSE].ms
        action = PauseAction.RESPOND.value if want == "respond" else want
        return Act(action, route, esc, "oracle", t, t + b, b)


# -- the timeline -------------------------------------------------------------------------------


def _words_by(seg: Segment, t: float, until: float | None = None) -> list[str]:
    """The words of `seg` finished by `t` (words are spread evenly over its duration); an agent
    line stopped at `until` says nothing after that."""
    end = seg.end_ms if until is None else min(seg.end_ms, until)
    t = min(t, end)
    if t <= seg.at_ms:
        return []
    words = seg.words
    n = math.floor(len(words) * (t - seg.at_ms) / seg.dur_ms + 1e-9)
    return words[: min(n, len(words))]


class Timeline:
    """One agent's view of the recording: the script plus the lines this agent has cut."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self.cut: dict[int, float] = {}  # agent segment index -> when it stopped

    def spoken_end(self, i: int) -> float:
        s = self.script.timeline[i]
        return min(s.end_ms, self.cut.get(i, math.inf))

    def agent_line_at(self, t: float) -> int | None:
        for i, s in enumerate(self.script.timeline):
            if s.who == "agent" and s.at_ms <= t < self.spoken_end(i):
                return i
        return None

    def next_start(self, after: float) -> float | None:
        later = [s.at_ms for s in self.script.timeline if s.at_ms >= after]
        return min(later) if later else None

    def snapshot(self, t: float, *, min_words_from: int | None = None) -> Snapshot:
        """What the pipeline knows at `t`. `min_words_from`: the segment being decided on,
        which has been heard, so it contributes at least one word.

        Speech is put in order of when it ends: each agent word by its own end, each user
        segment as a whole by its end (or by `t`, if it is still going). So an agent line the
        user spoke over is split at the end of that user speech. A backchannel 3 s before the
        current speech stays a turn of its own, between the agent's words before and after it.
        Agent words spoken during an interruption stay before it, so the user's turn is not
        split by them.
        """
        chunks: list[tuple[float, int, str, list[str]]] = [
            (-math.inf, k, h.speaker, h.text.split()) for k, h in enumerate(self.script.history)
        ]
        for i, s in enumerate(self.script.timeline):
            if s.at_ms >= t and i != min_words_from:
                continue
            if s.who == "agent":
                spoken = _words_by(s, t, self.cut.get(i))
                per = s.dur_ms / len(s.words)
                for j, w in enumerate(spoken):  # word j ends at at + (j + 1) * per
                    chunks.append((s.at_ms + (j + 1) * per, 0, "agent", [w]))
                continue
            words = _words_by(s, t)
            if i == min_words_from and not words:
                words = s.words[:1]
            if words:
                chunks.append((min(s.end_ms, t), 1, "user", words))
        groups: list[tuple[str, list[str]]] = []
        for _, _, who, words in sorted(chunks, key=lambda c: (c[0], c[1])):
            if groups and groups[-1][0] == who:
                groups[-1][1].extend(words)
            else:
                groups.append((who, list(words)))
        user = " ".join(groups.pop()[1]) if groups and groups[-1][0] == "user" else ""
        agent = " ".join(groups.pop()[1]) if groups and groups[-1][0] == "agent" else ""
        return Snapshot(
            agent_speaking=self.agent_line_at(t) is not None,
            user_partial=user,
            agent_last_utterance=agent,
            recent_turns=tuple(Turn(w, " ".join(ws)) for w, ws in groups),  # type: ignore[arg-type]
        )


def _matches(want: str, action: str) -> bool:
    if want == "respond":
        return action in (PauseAction.RESPOND.value, PauseAction.RESPOND_TIMEOUT.value)
    return want == action


def _pause_consequence(act: Act, end_ms: float, next_user: float | None, mid_thought: bool) -> str:
    if act.action == PauseAction.KEEP_LISTENING.value:
        return "kept listening"
    after = act.effective_ms - end_ms
    if next_user is not None and act.effective_ms < next_user:
        later = next_user - act.effective_ms
        if mid_thought:
            return (
                f"answered {after:.0f} ms after the user stopped, mid-thought: it cut them off "
                f"{later:.0f} ms before they went on"
            )
        return f"answered {after:.0f} ms after the user stopped; the user spoke again {later:.0f} ms later"
    how = "safety net" if act.timer and act.budget_ms else "timeout" if act.timer else "decision"
    return f"answered {after:.0f} ms after the user stopped ({how})"


def _barge_consequence(act: Act, seg: Segment, line: Segment, line_end: float) -> str:
    if act.action == BargeAction.STOP_AND_LISTEN.value:
        return f"stopped talking at {act.effective_ms / 1000:.2f} s ({line_end - act.effective_ms:.0f} ms of its line unsaid)"
    if act.action == BargeAction.KEEP_TALKING.value:
        return "kept talking"
    return "ignored it and kept talking"


def simulate(script: Script, agent: ReplayAgent, *, vad_pause_ms: float) -> Run:
    """Replay `script` through `agent`. VAD raises a pause after `vad_pause_ms` of silence, for
    either agent; what the agent then does with the silence is its own rule."""
    tl = Timeline(script)
    run = Run(agent=agent.name)
    timeline = script.timeline
    seen: set[tuple[int, bool]] = set()  # (segment, over the agent)
    for i, u in enumerate(timeline):
        if u.who != "user":
            continue
        line = tl.agent_line_at(u.at_ms)
        if line is not None:

            def heard(t: float, i: int = i) -> Snapshot:
                return tl.snapshot(t, min_words_from=i)

            act = agent.barge(i, heard, u.at_ms, u.end_ms)
            if act.action == BargeAction.STOP_AND_LISTEN.value:
                tl.cut[line] = min(act.effective_ms, timeline[line].end_ms)
            run.outcomes.append(
                _scored(
                    script,
                    i,
                    Event.USER_SPEECH_DURING_AGENT,
                    act,
                    _barge_consequence(act, u, timeline[line], timeline[line].end_ms),
                )
            )
            seen.add((i, True))
            if tl.agent_line_at(u.end_ms) is not None:
                continue  # the agent still holds the floor: no pause for the user
        elif tl.agent_line_at(u.end_ms) is not None:
            continue
        nxt = tl.next_start(u.end_ms)
        gap = None if nxt is None else nxt - u.end_ms
        if gap is not None and gap < vad_pause_ms:
            continue
        act = agent.pause(i, lambda t: tl.snapshot(t), u.end_ms, gap)
        next_user = next((s.at_ms for s in timeline[i + 1 :] if s.who == "user"), None)
        next_any_is_user = nxt is not None and next_user == nxt
        e = script.expectation(i, False)
        mid_thought = e is not None and e.want == "keep_listening"
        consequence = _pause_consequence(
            act, u.end_ms, next_user if next_any_is_user else None, mid_thought
        )
        run.outcomes.append(_scored(script, i, Event.USER_PAUSE, act, consequence))
        seen.add((i, False))
    for e in script.expect:
        if (e.seg, e.over_agent) in seen:
            continue
        stopped = [v for v in tl.cut.values() if v <= timeline[e.seg].at_ms]
        if e.over_agent and stopped:
            why = f"the agent had already stopped talking at {max(stopped) / 1000:.2f} s"
            harmless = e.want == BargeAction.STOP_AND_LISTEN.value
            run.missed.append(Missed(e.seg, e.want, why, counts=not harmless))
        elif not e.over_agent:
            run.missed.append(Missed(e.seg, e.want, "the agent was still talking over the user"))
        else:
            run.missed.append(Missed(e.seg, e.want, "no decision point arose here"))
    run.outcomes.sort(key=lambda o: (o.seg, o.event != Event.USER_SPEECH_DURING_AGENT))
    return run


def _scored(script: Script, seg: int, event: Event, act: Act, consequence: str) -> Outcome:
    e = script.expectation(seg, event is Event.USER_SPEECH_DURING_AGENT)
    if e is None:
        return Outcome(seg, event, act, consequence)
    ok = _matches(e.want, act.action)
    if ok and e.route is not None:
        ok = act.route == e.route
    if ok and e.escalate is not None:
        ok = act.escalate == e.escalate
    if e.want == "keep_listening" and act.action == PauseAction.KEEP_LISTENING.value:
        ok = True  # nothing is routed or handed off while listening
    return Outcome(seg, event, act, consequence, e.want, e.route, e.escalate, ok)
