"""The decision set: every typed question floorcall asks, in Laya's question format.

Wording rules, each learned from Laya's documented failure modes rather than chosen for style:

- Every `noul` carries explicit `false` / `true` criteria. On the English checkpoint a criteria-less
  `noul` answers "no" whatever the state (Laya issue #156; DECISIONS.md D-005).
- No `choice` label is a boolean word (yes / no / true / false). The checkpoint can follow such a
  label instead of its description (Laya README, Honest limits).
- Instructions and option descriptions share a 192-token head budget with the options. The adapter
  refuses any question whose options would lose a distinct token span
  (`LayaDecider.state_room`). Descriptions are therefore short, and what the budget costs the
  state is measured, not guessed.

The fine-tuned checkpoint is trained on exactly these strings. Changing a word here after training
changes the input distribution, so it means a retrain and a note in DECISIONS.md.
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Any

QuestionDef = MappingProxyType[str, Any]


class Event(StrEnum):
    """Pipeline moments that trigger a decision (CLAUDE.md §4)."""

    # VAD detects silence while the user holds the floor.
    USER_PAUSE = "user_pause"
    # User audio is detected while the agent's TTS is playing.
    USER_SPEECH_DURING_AGENT = "user_speech_during_agent"


# -- D1 --------------------------------------------------------------------------------------

TURN_COMPLETE = "turn_complete"
D1_TURN_COMPLETE: QuestionDef = MappingProxyType(
    {
        "type": "noul",
        "instructions": "Has the user finished their turn and is now waiting for a reply?",
        "criteria": {
            "false": "the user is mid-thought and will keep talking",
            "true": "the user has finished and expects the agent to respond",
        },
    }
)

# -- D2 --------------------------------------------------------------------------------------

BARGE_IN = "barge_in"
BACKCHANNEL, INTERRUPTION, NOISE = "backchannel", "interruption", "noise"
BARGE_IN_LABELS = (BACKCHANNEL, INTERRUPTION, NOISE)
D2_BARGE_IN: QuestionDef = MappingProxyType(
    {
        "type": "choice",
        "instructions": "The user spoke while the agent was talking. What was it?",
        "criteria": {
            BACKCHANNEL: "brief listener feedback that does not claim the turn, like uh-huh or right",
            INTERRUPTION: "the user wants the floor: a question, objection, correction or new request",
            NOISE: "not speech to the agent: a fragment, laughter, or talk to someone else",
        },
    }
)

# -- D3 --------------------------------------------------------------------------------------

ROUTE = "route"
OUT_OF_SCOPE = "out_of_scope"
# CLINC150's banking domain, labels exactly as in clinc/oos-eval data/domains.json, so a D3 gold
# label is the CLINC intent name with no mapping table in between.
BANKING_INTENTS: MappingProxyType[str, str] = MappingProxyType(
    {
        "transfer": "move money between accounts",
        "transactions": "details of a past transaction",
        "balance": "how much is in an account",
        "freeze_account": "the user wants an account frozen",
        "pay_bill": "pay a bill",
        "bill_balance": "how much a bill owes",
        "bill_due": "when a bill is due",
        "interest_rate": "an account's interest rate",
        "routing": "the bank's routing number",
        "min_payment": "the minimum payment due",
        "order_checks": "order new checks",
        "pin_change": "change a PIN",
        "report_fraud": "report fraud or a charge they did not make",
        "account_blocked": "why an account was blocked",
        "spending_history": "how much was spent over time",
    }
)
ROUTE_LABELS = (*BANKING_INTENTS.keys(), OUT_OF_SCOPE)
D3_ROUTE: QuestionDef = MappingProxyType(
    {
        "type": "choice",
        "instructions": "Which banking request is the user making?",
        "criteria": {
            **BANKING_INTENTS,
            OUT_OF_SCOPE: "anything a bank support agent does not handle",
        },
    }
)

# -- D4 --------------------------------------------------------------------------------------

ESCALATE = "escalate"
D4_ESCALATE: QuestionDef = MappingProxyType(
    {
        "type": "noul",
        "instructions": "Should this conversation be handed to a human?",
        "criteria": {
            "false": "the user is fine continuing with the automated agent",
            "true": "the user asks for a person, or is frustrated and not being helped",
        },
    }
)

# -- events ----------------------------------------------------------------------------------

ALL_QUESTIONS: MappingProxyType[str, QuestionDef] = MappingProxyType(
    {
        TURN_COMPLETE: D1_TURN_COMPLETE,
        BARGE_IN: D2_BARGE_IN,
        ROUTE: D3_ROUTE,
        ESCALATE: D4_ESCALATE,
    }
)

# Each event asks only the questions that matter at that moment, all in one batched call.
EVENT_QUESTIONS: MappingProxyType[Event, tuple[str, ...]] = MappingProxyType(
    {
        Event.USER_PAUSE: (TURN_COMPLETE, ROUTE, ESCALATE),
        Event.USER_SPEECH_DURING_AGENT: (BARGE_IN, ESCALATE),
    }
)


def questions_for(event: Event) -> dict[str, dict[str, Any]]:
    """The question dict for one event, in the plain-dict shape Laya accepts."""
    return {qid: _plain(ALL_QUESTIONS[qid]) for qid in EVENT_QUESTIONS[event]}


def questions_by_id(*qids: str) -> dict[str, dict[str, Any]]:
    return {qid: _plain(ALL_QUESTIONS[qid]) for qid in qids}


def _plain(q: QuestionDef) -> dict[str, Any]:
    out = dict(q)
    crit = out.get("criteria")
    if isinstance(crit, MappingProxyType | dict):
        out["criteria"] = dict(crit)
    return out
