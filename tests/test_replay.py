"""Replay mode (D-045, D-046): scripts, freezing, both agents, the logical clock, timing checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from floorcall.config import REPO_ROOT, PolicySettings, ReplaySettings, Settings
from floorcall.decider import Decision, Timings
from floorcall.pipeline.naive import NaiveAgent
from floorcall.pipeline.operating import Budget, Source, decision_budgets, served_policy
from floorcall.pipeline.processor import DecisionProcessor
from floorcall.questions import BARGE_IN_LABELS, ROUTE_LABELS, Event
from floorcall.replay.checks import check_timing
from floorcall.replay.engine import FloorcallReplay, NaiveReplay, simulate
from floorcall.replay.script import Script, freeze, load_script, read_manifest, verify_frozen
from floorcall.state import PackedState, Snapshot

POLICY = PolicySettings(theta_yield=0.8, theta_interrupt=0.1, theta_escalate=0.5, theta_oos=0.45)
CFG = ReplaySettings()
BUDGETS = {
    e: Budget(e, ms, Source(ms, "results/table_b/x.json", "total_ms.p50"), "fake")
    for e, ms in ((Event.USER_PAUSE, 50.0), (Event.USER_SPEECH_DURING_AGENT, 40.0))
}
SCRIPTS = sorted((REPO_ROOT / "demo" / "scripts").glob("*.json"))


@dataclass
class FakeDecider:
    """Fixed calibrated probabilities per event; records every snapshot it is asked about. Its
    compute time is absurdly slow on purpose: none of it may reach the timeline."""

    p_complete: float = 0.9
    barge: str = "backchannel"
    p_escalate: float = 0.1
    route: str = "balance"
    seen: list[tuple[Event, Snapshot]] = field(default_factory=list)

    def decide(self, event: Event, snapshot: Snapshot) -> Decision:
        self.seen.append((event, snapshot))
        esc = {"false": 1 - self.p_escalate, "true": self.p_escalate}
        if event is Event.USER_PAUSE:
            rest = 0.1 / (len(ROUTE_LABELS) - 1)
            probs = {
                "turn_complete": {"false": 1 - self.p_complete, "true": self.p_complete},
                "route": {k: 0.9 if k == self.route else rest for k in ROUTE_LABELS},
                "escalate": esc,
            }
        else:
            probs = {
                "barge_in": {k: 0.9 if k == self.barge else 0.05 for k in BARGE_IN_LABELS},
                "escalate": esc,
            }
        packed = PackedState({}, 10, 100, 0, 0, 0, 0)
        return Decision(event, probs, packed, Timings(1.0, 1.0, 5000.0, 1), total_ms=5000.0)


def floorcall_agent(decider: FakeDecider) -> FloorcallReplay:
    return FloorcallReplay(DecisionProcessor(decider, POLICY, device="cpu"), POLICY, BUDGETS, CFG)


def script(**kw: Any) -> Script:
    base: dict[str, Any] = {"id": "t", "title": "t", "covers": ["x"]}
    return Script.model_validate(base | kw)


BACKCHANNEL = script(
    timeline=[
        {
            "who": "agent",
            "at_ms": 0,
            "dur_ms": 5000,
            "text": "one two three four five six seven eight",
        },
        {"who": "user", "at_ms": 1000, "dur_ms": 300, "text": "uh-huh"},
        {"who": "user", "at_ms": 3000, "dur_ms": 300, "text": "right"},
        {"who": "user", "at_ms": 5500, "dur_ms": 900, "text": "what is my balance"},
        {"who": "agent", "at_ms": 8600, "dur_ms": 1000, "text": "it is ten dollars"},
    ],
    expect=[
        {"seg": 1, "want": "keep_talking", "why": "x"},
        {"seg": 2, "want": "keep_talking", "why": "x"},
        {"seg": 3, "want": "respond", "route": "balance", "why": "x"},
    ],
)


# -- scripts --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "timeline",
    [
        [  # the user overlaps themself
            {"who": "user", "at_ms": 0, "dur_ms": 1000, "text": "a b"},
            {"who": "user", "at_ms": 500, "dur_ms": 1000, "text": "c"},
        ],
        [  # the agent starts while the user speaks
            {"who": "user", "at_ms": 0, "dur_ms": 1000, "text": "a b"},
            {"who": "agent", "at_ms": 500, "dur_ms": 1000, "text": "c"},
        ],
        [  # out of order
            {"who": "user", "at_ms": 900, "dur_ms": 100, "text": "a"},
            {"who": "agent", "at_ms": 0, "dur_ms": 100, "text": "b"},
        ],
    ],
)
def test_malformed_timelines_are_refused(timeline: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        script(timeline=timeline, expect=[{"seg": 0, "want": "respond", "why": "x"}])


def test_expectations_are_checked() -> None:
    tl = [{"who": "user", "at_ms": 0, "dur_ms": 100, "text": "a"}]
    for bad in (
        {"seg": 0, "want": "respond", "route": "no_such_intent", "why": "x"},
        {"seg": 0, "want": "keep_talking", "route": "balance", "why": "x"},  # route at a barge
        {"seg": 5, "want": "respond", "why": "x"},
    ):
        with pytest.raises(ValidationError):
            script(timeline=tl, expect=[bad])
    two = [
        {"seg": 0, "want": "respond", "why": "x"},
        {"seg": 0, "want": "keep_listening", "why": "y"},
    ]
    with pytest.raises(ValidationError):
        script(timeline=tl, expect=two)
    # one barge and one pause expectation on the same segment is fine
    script(timeline=tl, expect=[two[0], {"seg": 0, "want": "stop_and_listen", "why": "z"}])


def test_freezing_is_once(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(BACKCHANNEL.model_dump_json(), encoding="utf-8")
    assert verify_frozen([p]) == ["t.json: not frozen (run `floorcall replay --freeze`)"]
    freeze([p])
    assert verify_frozen([p]) == [] and "t.json" in read_manifest(tmp_path)
    p.write_text(p.read_text(encoding="utf-8").replace("uh-huh", "mm-hm"), encoding="utf-8")
    assert verify_frozen([p]) == ["t.json: changed since it was frozen"]
    with pytest.raises(ValueError, match="frozen"):
        freeze([p])


def test_the_demo_scripts_are_frozen_and_pass_the_timing_check() -> None:
    assert len(SCRIPTS) == 8
    assert verify_frozen(SCRIPTS) == []
    for p in SCRIPTS:
        assert check_timing(load_script(p), POLICY, BUDGETS, CFG) == [], p.name


def test_a_mid_thought_gap_naive_would_wait_out_fails_the_check() -> None:
    def mid(gap: int) -> Script:
        return script(
            timeline=[
                {"who": "user", "at_ms": 0, "dur_ms": 1000, "text": "I want to pay my"},
                {"who": "user", "at_ms": 1000 + gap, "dur_ms": 600, "text": "card bill"},
                {"who": "agent", "at_ms": 1600 + gap + 2200, "dur_ms": 900, "text": "sure"},
            ],
            expect=[
                {"seg": 0, "want": "keep_listening", "why": "x"},
                {"seg": 1, "want": "respond", "why": "x"},
            ],
        )

    assert check_timing(mid(1100), POLICY, BUDGETS, CFG) == []
    (problem,) = check_timing(mid(700), POLICY, BUDGETS, CFG)  # under naive's 800 ms
    assert "mid-thought gap 700 ms" in problem


# -- agents ---------------------------------------------------------------------------------------


def test_naive_stops_for_everything_and_answers_on_a_timer() -> None:
    n = NaiveAgent(800)
    assert n.on_user_speech_during_agent().action.value == "stop_and_listen"
    assert n.on_user_pause(silence_ms=800).action.value == "respond_after_timeout"
    assert n.on_user_pause(silence_ms=799).action.value == "keep_listening"
    assert n.on_user_pause(silence_ms=5000).route is None


def test_backchannels_naive_stops_floorcall_talks_on() -> None:
    naive = simulate(BACKCHANNEL, NaiveReplay(NaiveAgent(800)), vad_pause_ms=300)
    first = naive.outcomes[0]
    assert (first.seg, first.act.action, first.ok) == (1, "stop_and_listen", False)
    assert first.act.effective_ms == 1000  # at onset: naive makes no model call
    (missed,) = naive.missed  # "right" never reached it: it had stopped at 1.00 s
    assert missed.seg == 2 and missed.counts

    fc = simulate(BACKCHANNEL, floorcall_agent(FakeDecider()), vad_pause_ms=300)
    assert [(o.seg, o.act.action, o.ok) for o in fc.outcomes] == [
        (1, "keep_talking", True),
        (2, "keep_talking", True),
        (3, "respond", True),
    ]
    assert fc.missed == []


def test_the_timeline_uses_the_budget_never_the_compute_time() -> None:
    fc = simulate(BACKCHANNEL, floorcall_agent(FakeDecider()), vad_pause_ms=300)
    barge, _, pause = fc.outcomes
    # decided when the 300 ms "uh-huh" ended (inside the 600 ms window), plus the 40 ms budget
    assert barge.act.decided_at_ms == 1300 and barge.act.effective_ms == 1340
    assert barge.act.compute_ms == 5000.0 and barge.act.device == "cpu"  # reported, not used
    assert pause.act.decided_at_ms == 6400 + 300 and pause.act.effective_ms == 6750


def test_a_stop_cuts_the_agents_line_and_the_snapshot_holds_what_was_heard() -> None:
    d = FakeDecider(barge="interruption")
    s = script(
        timeline=[
            {"who": "agent", "at_ms": 0, "dur_ms": 4000, "text": "a1 a2 a3 a4 a5 a6 a7 a8"},
            {"who": "user", "at_ms": 1000, "dur_ms": 2000, "text": "u1 u2 u3 u4 u5 u6 u7 u8"},
        ],
        expect=[
            {"seg": 1, "want": "stop_and_listen", "why": "x"},
            {"seg": 1, "want": "respond", "why": "x"},
        ],
    )
    fc = simulate(s, floorcall_agent(d), vad_pause_ms=300)
    barge, pause = fc.outcomes
    assert barge.act.decided_at_ms == 1600  # 600 ms into the speech
    assert barge.act.heard == "u1 u2"  # 8 words over 2 s: two finished by 600 ms
    snap = d.seen[0][1]
    assert snap.agent_speaking and snap.agent_last_utterance == "a1 a2 a3"
    # the line stopped at 1640 ms, so the pause after the user's speech sees the cut line
    assert pause.event is Event.USER_PAUSE
    after = d.seen[1][1]
    assert not after.agent_speaking and after.agent_last_utterance == "a1 a2 a3"
    assert after.user_partial == "u1 u2 u3 u4 u5 u6 u7 u8"


def test_the_safety_net_answers_without_a_second_forward_pass() -> None:
    d = FakeDecider(p_complete=0.2)  # below theta_yield: keep listening
    s = script(
        timeline=[{"who": "user", "at_ms": 0, "dur_ms": 900, "text": "hello there"}],
        expect=[{"seg": 0, "want": "respond", "why": "x"}],
    )
    fc = simulate(s, floorcall_agent(d), vad_pause_ms=300)
    (o,) = fc.outcomes
    assert o.act.action == "respond_after_timeout" and o.act.timer
    assert o.act.effective_ms == 900 + POLICY.max_wait_ms
    assert len(d.seen) == 1


def test_processor_after_silence_reuses_the_probabilities() -> None:
    d = FakeDecider(p_complete=0.2)
    proc = DecisionProcessor(d, POLICY, device="cpu")
    first = proc.on_user_pause(Snapshot(False, "hi"), silence_ms=300)
    late = proc.after_silence(first, silence_ms=2000)
    assert first.action == "keep_listening" and late.action == "respond_after_timeout"
    assert late.compute_ms is None and late.probabilities == first.probabilities
    assert len(d.seen) == 1


# -- served thresholds and budgets: from the committed results files ------------------------------


def test_served_numbers_are_read_from_the_results_files() -> None:
    settings = Settings()
    served = served_policy(settings)
    curves = json.loads((REPO_ROOT / "results/curves/finetuned_temp.json").read_text())
    oos = json.loads((REPO_ROOT / "results/thresholds/route_oos.json").read_text())
    assert served.policy.theta_interrupt == curves["interrupt"]["theta"]
    assert served.policy.theta_yield == curves["yield"]["theta"]
    assert served.policy.theta_oos == oos["choice"]["theta"]
    budgets = decision_budgets(settings, served.checkpoint)
    row = json.loads((REPO_ROOT / "results/table_b/gpu_eager_pause.json").read_text())
    assert budgets[Event.USER_PAUSE].ms == row["total_ms"]["p50"]
    assert budgets[Event.USER_PAUSE].source.file == "results/table_b/gpu_eager_pause.json"


def test_a_budget_row_for_another_configuration_is_refused() -> None:
    s = Settings()
    other = s.model_copy(update={"laya": s.laya.model_copy(update={"precision": "bf16"})})
    with pytest.raises(ValueError, match="not the served configuration"):
        decision_budgets(other, "checkpoints/main-r2")


def test_an_earlier_backchannel_is_its_own_turn_not_part_of_the_current_speech() -> None:
    d = FakeDecider()
    simulate(BACKCHANNEL, floorcall_agent(d), vad_pause_ms=300)
    (_, first), (_, second) = d.seen[:2]
    assert first.user_partial == "uh-huh"
    # "right" is read alone; "uh-huh" sits between the agent's words before and after it
    assert second.user_partial == "right"
    assert [(t.speaker, t.text) for t in second.recent_turns] == [
        ("agent", "one two"),
        ("user", "uh-huh"),
    ]
    assert second.agent_last_utterance == "three four five"


def test_agent_words_during_an_interruption_do_not_split_the_users_turn() -> None:
    d = FakeDecider(barge="interruption")
    s = script(
        timeline=[
            {"who": "agent", "at_ms": 0, "dur_ms": 4000, "text": "a1 a2 a3 a4 a5 a6 a7 a8"},
            {"who": "user", "at_ms": 1000, "dur_ms": 1000, "text": "wait no"},
            {"who": "user", "at_ms": 2200, "dur_ms": 600, "text": "not that"},
        ],
        expect=[
            {"seg": 1, "want": "stop_and_listen", "why": "x"},
            {"seg": 2, "want": "respond", "why": "x"},
        ],
    )
    simulate(s, floorcall_agent(d), vad_pause_ms=300)
    pause = d.seen[-1][1]
    assert pause.user_partial == "wait no not that"
    assert pause.agent_last_utterance == "a1 a2 a3"  # cut at 1,640 ms, inside the user's speech
