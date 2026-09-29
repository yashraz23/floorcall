"""Prompts for the two LLM uses: the D4 training labeller and the prompted-LLM baseline.

The labeller is given Yash's labelling guideline verbatim and the same view of a conversation the
labelling tool showed him: up to two earlier turns, the agent's last reply, the customer's
message. Its agreement with his labels then measures the labeller on his task, not on a different
definition. The prompt is versioned; changing a word is a new version, and each version's labels
are kept apart. v1 and v2 quoted guideline v1 (v2 added two clarifying sentences, D-032; both are
in git at 27c1c8c). v3 (D-033) is guideline v2, word for word, and like it has no "unsure": the
guideline says what to do when unsure. v3 is the last labeller prompt revision.

The baseline is given exactly what Laya is given: the packed, normalized state and the question's
instructions and option descriptions from floorcall.questions. It is asked for a probability per
option. Token log-probabilities are not used: provider support varies behind OpenRouter, and the
answer is a JSON object rather than a single option token.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from floorcall.data.labelling import GUIDELINE_V2

LABELLER_VERSION = "llm-labeller-v3"

# The framing is v1's, unchanged; the guideline is v2, verbatim.
LABELLER_SYSTEM = (
    "You label customer messages sent to a bank's support account on Twitter, for a dataset that "
    'trains an escalation detector. The question for every message is: "Should this conversation '
    "be handed to a human?\" The customer plays the user; the company's support account plays the "
    "agent.\n\n"
    f"{GUIDELINE_V2}\n\n"
    'Answer "y" or "n".'
)

LABELLER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["y", "n"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def labeller_messages(
    recent_turns: Sequence[Mapping[str, str]], agent_last: str, message: str
) -> list[dict[str, str]]:
    lines = [f"{t['speaker']}: {t['text']}" for t in recent_turns[-2:]]
    context = "\n".join(lines) if lines else "(none)"
    agent = agent_last or "(none: the customer opened the thread)"
    user = (
        f"Earlier turns:\n{context}\n\nAgent's last reply:\n{agent}\n\n"
        f"Customer's message:\n{message}"
    )
    return [{"role": "system", "content": LABELLER_SYSTEM}, {"role": "user", "content": user}]


BASELINE_VERSION = "llm-baseline-v1"

BASELINE_SYSTEM = """You make fast decisions for the controller of a banking voice agent. You are \
given the conversation state as JSON: whether the agent is speaking, recent turns, the agent's \
last utterance, and the user's words so far, transcribed by speech recognition (lowercase, no \
punctuation). Answer each question by giving a probability for every option. The probabilities \
of one question must sum to 1. Use the whole range: say 0.5 when you cannot tell."""


def _options(qdef: Mapping[str, Any]) -> list[tuple[str, str]]:
    crit = qdef["criteria"]
    return [(str(k), str(v)) for k, v in crit.items()]


def baseline_messages(
    state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    parts = []
    for qid, q in questions.items():
        opts = "\n".join(f"  - {label}: {desc}" for label, desc in _options(q))
        parts.append(f'Question "{qid}": {q["instructions"]}\nOptions:\n{opts}')
    user = f"State:\n{json.dumps(dict(state), ensure_ascii=False)}\n\n" + "\n\n".join(parts)
    return [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user}]


def baseline_schema(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Strict schema: one object per question, one number per option, nothing else."""
    props = {}
    for qid, q in questions.items():
        labels = [label for label, _ in _options(q)]
        props[qid] = {
            "type": "object",
            "properties": {label: {"type": "number"} for label in labels},
            "required": labels,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(questions),
        "additionalProperties": False,
    }


def read_probabilities(
    answer: Mapping[str, Any], labels: Sequence[str]
) -> tuple[list[float], bool]:
    """The stated probabilities as a distribution over `labels`, and whether they were usable.

    Negative or missing values count as 0; the rest are renormalized. A question with no mass at
    all falls back to uniform and is reported as invalid.
    """
    raw = []
    for label in labels:
        v = answer.get(label)
        raw.append(max(0.0, float(v)) if isinstance(v, int | float) else 0.0)
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(labels)] * len(labels), False
    return [x / total for x in raw], True
