"""The adapter against real Laya.

The first test needs only the installed package. The rest load the pinned checkpoint (`-m model`,
~0.9 GB download on first use) and run on a developer machine, not in CI.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.model.laya_adapter import LayaDecider, StateOverflowError, serialize
from floorcall.state import Snapshot, Turn, pack_state
from floorcall.state import serialize as packer_serialize

STATE = {
    "agent_speaking": True,
    "recent_turns": [{"speaker": "user", "text": "café au lait"}],
    "agent_last_utterance": "your card was declined because the",
    "user_partial": "yeah but why",
}


def test_packer_serializes_exactly_like_laya() -> None:
    assert packer_serialize(STATE) == serialize(STATE)


@pytest.fixture(scope="module")
def decider() -> Iterator[LayaDecider]:
    yield LayaDecider(LayaSettings(**{**get_settings().laya.model_dump(), "device": "cpu"}))


@pytest.mark.model
def test_packed_state_fits_under_the_real_tokenizer(decider: LayaDecider) -> None:
    qs = questions.questions_for(questions.Event.USER_PAUSE)
    budget = decider.state_room(qs) - get_settings().state.safety_margin_tokens
    long_turns = tuple(
        Turn("user" if i % 2 else "agent", f"this is turn number {i} about my savings account")
        for i in range(60)
    )
    packed = pack_state(
        Snapshot(
            agent_speaking=False,
            user_partial="i want to move money into my checking",
            agent_last_utterance="how can i help you today",
            recent_turns=long_turns,
        ),
        budget=budget,
        count_tokens=decider.count_tokens,
    )
    assert packed.turns_dropped > 0, "the fixture should be long enough to force packing"
    assert packed.n_tokens <= budget
    assert decider.check_fits(packed.state, qs) == packed.n_tokens


@pytest.mark.model
def test_check_fits_refuses_an_oversized_state(decider: LayaDecider) -> None:
    qs = questions.questions_for(questions.Event.USER_PAUSE)
    huge = {**STATE, "user_partial": "word " * 600}
    with pytest.raises(StateOverflowError):
        decider.check_fits(huge, qs)


@pytest.mark.model
def test_logits_with_checkpoint_temperature_reproduce_predict(decider: LayaDecider) -> None:
    # Evaluation applies temperatures to raw logits itself (Table A's calibrated and uncalibrated
    # rows). That is only valid if softmax(logits / T) with the adapter's reading of Laya's
    # temperature buckets is what Agent.predict returns.
    for event in questions.Event:
        qs = questions.questions_for(event)
        pred = decider.predict(STATE, qs)
        (logits,) = decider.logits_batch([STATE], qs)
        for qid, ql in logits.items():
            t = decider.effective_temperature(ql.qtype, len(ql.labels))
            z = ql.logits / t
            p = np.exp(z - z.max())
            p /= p.sum()
            expected = np.array([pred.answers[qid].probabilities[lab] for lab in ql.labels])
            # predict() rounds to 4 decimals
            assert np.abs(p - expected).max() < 1e-3, (event, qid, p, expected)


@pytest.mark.model
def test_inference_ignores_a_model_left_in_train_mode(decider: LayaDecider) -> None:
    qs = questions.questions_for(questions.Event.USER_SPEECH_DURING_AGENT)
    decider.model.train()
    try:
        a = decider.predict(STATE, qs).answers
        b = decider.predict(STATE, qs).answers
        (la,) = decider.logits_batch([STATE], qs)
        (lb,) = decider.logits_batch([STATE], qs)
        assert decider.model.training, "the caller's mode must be restored"
    finally:
        decider.model.eval()
    # with dropout active these would differ run to run
    assert a == b
    for qid in la:
        assert np.array_equal(la[qid].logits, lb[qid].logits)


@pytest.mark.model
def test_saved_checkpoint_reloads_through_plain_laya(decider: LayaDecider, tmp_path: Path) -> None:
    # No training: the stock weights are saved in Laya's layout and reloaded by laya.Agent. They are
    # fp16 on disk to begin with, so on CPU (fp32 compute) the logits must match exactly.
    out = tmp_path / "ckpt"
    decider.save_checkpoint(out, {"note": "untrained round trip"})
    reloaded = LayaDecider(LayaSettings(checkpoint=str(out), revision=None, device="cpu"))
    qs = questions.questions_for(questions.Event.USER_PAUSE)
    (a,) = decider.logits_batch([STATE], qs)
    (b,) = reloaded.logits_batch([STATE], qs)
    for qid in a:
        assert np.array_equal(a[qid].logits, b[qid].logits), qid
    cfg = json.loads((out / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert cfg["temperature"] == [1.0, 1.0, 1.0]  # floorcall applies its own (D-021)
    assert "temperature_by_options" not in cfg
    assert cfg["fine_tuned"] is True
    assert cfg["floorcall"] == {"note": "untrained round trip"}
