"""The D4 LLM labeller and the baseline prompts, offline (no API, no spend)."""

from pathlib import Path
from typing import Any

import pytest

from floorcall.config import REPO_ROOT
from floorcall.data import d4_llm
from floorcall.data.escalate import Candidate
from floorcall.llm import prompts
from floorcall.questions import Event, questions_for


def cand(
    i: int, split: str, text: str | None = None, thread: int | None = None, cue: bool = False
) -> Candidate:
    return Candidate(
        id=f"twcs-{i}",
        group=f"twcs:{thread if thread is not None else i}",
        split=split,  # type: ignore[arg-type]
        company="BofA_Help",
        prefiltered=cue,
        user_partial=text or f"message number {i}",
        agent_last_utterance="",
        recent_turns=[],
    )


# -- prompts ---------------------------------------------------------------------------------


# Each decision rule of docs/labelling-escalate.md, word for word.
RULES = [
    "The customer asks for a person: a human, a representative, a supervisor or manager, a phone "
    'call, "someone who can actually help".',
    'The customer is frustrated in a way support is not resolving: repeated contact ("I already '
    'DMed you", "third time"), "nobody is helping", hostility toward the support itself, or a '
    "threat to close the account, leave, or go to a regulator or the press.",
    'It is a routine question or request, even about a problem ("my card was declined, why?").',
    'The customer is supplying requested information, thanking, confirming ("ok, done"), or '
    "praising.",
    "There is annoyance at a product or situation, but no sign that support has failed them "
    '("ugh, the app is down").',
    "sarcasm you cannot read, context too thin to judge, not English, spam, or unreadable",
    "if you were the team lead watching this conversation live, would you want a person to take "
    "over now?",
]


def _flat(text: str) -> str:
    return " ".join(text.replace("**", "").replace("*", "").split())


@pytest.mark.parametrize("rule", RULES)
def test_the_labeller_gets_yashs_rules_verbatim(rule: str) -> None:
    doc = _flat((REPO_ROOT / "docs" / "labelling-escalate.md").read_text(encoding="utf-8"))
    assert rule.lower() in doc.lower(), "the rule is not in the guidelines"
    assert rule.lower() in _flat(prompts.LABELLER_SYSTEM).lower(), "the rule is not in the prompt"


def test_labeller_messages_show_what_the_tool_showed() -> None:
    msgs = prompts.labeller_messages(
        [
            {"speaker": "user", "text": "a"},
            {"speaker": "agent", "text": "b"},
            {"speaker": "user", "text": "c"},
        ],
        "",
        "why is nobody helping",
    )
    user = msgs[1]["content"]
    assert "user: a" not in user  # only the last two earlier turns, as in the labelling tool
    assert "agent: b" in user and "user: c" in user
    assert "(none: the customer opened the thread)" in user
    assert user.endswith("why is nobody helping")


def test_baseline_schema_is_strict_and_complete() -> None:
    qs = questions_for(Event.USER_PAUSE)
    schema = prompts.baseline_schema(qs)
    assert schema["required"] == list(qs) and schema["additionalProperties"] is False
    route = schema["properties"]["route"]
    assert route["required"] == list(qs["route"]["criteria"])
    assert route["additionalProperties"] is False


@pytest.mark.parametrize(
    ("answer", "probs", "ok"),
    [
        ({"false": 0.2, "true": 0.8}, [0.2, 0.8], True),
        ({"false": 2.0, "true": 2.0}, [0.5, 0.5], True),  # renormalized
        ({"false": -1.0, "true": 0.5}, [0.0, 1.0], True),  # negatives count as 0
        ({"false": 0, "true": 0}, [0.5, 0.5], False),  # no mass: uniform, flagged
        ({}, [0.5, 0.5], False),
    ],
)
def test_read_probabilities(answer: dict[str, Any], probs: list[float], ok: bool) -> None:
    got, valid = prompts.read_probabilities(answer, ["false", "true"])
    assert got == pytest.approx(probs) and valid is ok


# -- the D4 pipeline -------------------------------------------------------------------------


def test_train_pool_excludes_held_out_threads_and_texts() -> None:
    held_out = [cand(1, "test", "please get me a human"), cand(2, "calib")]
    pool = [
        cand(10, "train", "please get me a human"),  # same text as a test message, other thread
        cand(11, "train", thread=50),
        cand(12, "train", thread=50),  # same thread as 11: at most one of them
        cand(13, "train"),
        cand(14, "test"),  # not train
    ]
    chosen, report = d4_llm.train_pool(pool, held_out, n=2, seed=0)
    ids = {c.id for c in chosen}
    assert "twcs-10" not in ids and "twcs-14" not in ids
    assert len(ids & {"twcs-11", "twcs-12"}) <= 1
    assert report["dropped_text_matches_held_out"] == 1
    assert d4_llm.train_pool(pool, held_out, n=2, seed=0)[0] == chosen  # deterministic


def test_label_all_resumes_without_asking_twice(tmp_path: Path) -> None:
    asked: list[str] = []

    def ask(c: Candidate) -> str:
        asked.append(c.id)
        return "escalate" if c.id.endswith("1") else "no"

    cands = [cand(i, "train") for i in (1, 2, 3)]
    labels: dict[str, dict[str, Any]] = {}
    kw: dict[str, Any] = {
        "model": "m",
        "prompt_version": "v1",
        "split_of": lambda c: "train",
        "concurrency": 2,
        "save": lambda lab: None,
    }
    d4_llm.label_all(cands[:2], labels, ask, **kw)
    d4_llm.label_all(cands, labels, ask, **kw)
    assert sorted(asked) == ["twcs-1", "twcs-2", "twcs-3"]
    d4_llm.label_all(cands, labels, ask, **{**kw, "prompt_version": "v2"})  # new prompt: relabel
    assert len(asked) == 6


def test_train_rows_drop_unsure_and_carry_their_source() -> None:
    cands = [cand(1, "train", cue=True), cand(2, "train"), cand(3, "train")]
    labels = {
        "twcs-1": {"label": "escalate", "model": "m", "prompt": "v1"},
        "twcs-2": {"label": "unsure", "model": "m", "prompt": "v1"},
        "twcs-3": {"label": "no", "model": "m", "prompt": "v1"},
    }
    rows = d4_llm.train_rows(cands, labels)
    assert [r["id"] for r in rows] == ["twcs-1", "twcs-3"]
    assert {r["source"] for r in rows} == {"llm_labelled"}
    assert [r["label"] for r in rows] == ["true", "false"]
    assert rows[0]["hard"] is False and rows[1]["hard"] is True


def test_cohen_kappa_hand_computed() -> None:
    # 10 items: agree on 8 (6 no, 2 yes); p_o = 0.8
    # hand: 7 no / 3 yes; llm: 7 no / 3 yes -> p_e = 0.7*0.7 + 0.3*0.3 = 0.58
    # kappa = (0.8 - 0.58) / 0.42 = 0.5238
    hand = ["no"] * 6 + ["yes", "yes"] + ["no", "yes"]
    llm = ["no"] * 6 + ["yes", "yes"] + ["yes", "no"]
    assert d4_llm.cohen_kappa(hand, llm) == pytest.approx(0.22 / 0.42)


def test_agreement_counts_unsure_separately() -> None:
    hand = {"a": "true", "b": "false", "c": "false", "d": "true"}
    llm = {"a": "true", "b": "true", "c": "false", "d": "unsure"}
    out = d4_llm.agreement(
        hand, llm, {"a": "cue/reply", "b": "plain/opener", "c": "plain/opener", "d": "cue/reply"}
    )
    assert out["n"] == 3 and out["llm_unsure"] == 1
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["confusion"]["hand_false_llm_true"] == 1
    assert out["escalate_precision"] == pytest.approx(1 / 2)
    assert out["escalate_recall"] == pytest.approx(1.0)
