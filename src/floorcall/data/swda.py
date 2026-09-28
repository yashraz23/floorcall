"""D1 (turn_complete) and D2 (barge_in) from the Switchboard Dialog Act corpus.

Source: https://github.com/cgpotts/swda/raw/master/swda.zip, one CSV per conversation
(DECISIONS.md D-006). Licence CC BY-NC-SA 3.0: anything derived is non-commercial.

In every example the speaker whose words are being decided about plays the "user" and the other
caller plays the "agent". Both are humans on the phone in 1990-91, not a user talking to a bot.
That domain gap is real, and the README states it (CLAUDE.md §16).

Text
----
SwDA text carries transcription markup that no ASR emits. `clean_text` strips it and keeps the
words a recogniser would hear:

  {F uh, }  {D so, }  {C and }  {E i mean }  {A aside }   fillers, markers, asides: braces go, words stay
  [ i think, + i guess ]                                    restarts: brackets and "+" go, both attempts stay
  <laughter>  <noise>  <<talking to child>>                  non-speech and comments: removed
  #  ((  ))                                                  overlap and uncertainty marks: removed
  /  -/  --                                                  unit and interruption marks: removed

Ordinary punctuation (. , ? !) is kept, so that the stored text is "written form". It is removed at
pack time by floorcall.normalize, which is what lets the Table D ablation re-run on punctuated text
from the same frozen files.

Labels: see `build_d1` and `build_d2`, and DECISIONS.md D-015 and D-016.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floorcall.data.splits import Split, assign_split
from floorcall.normalize import normalize
from floorcall.questions import BACKCHANNEL, INTERRUPTION, NOISE

SOURCE_URL = "https://github.com/cgpotts/swda/raw/master/swda.zip"
SOURCE_SHA256 = "0a08b8dd3992b446c8a920cacf7d246abe1e8a092a1ef7b5a2ed0352de9ccad2"
LICENCE = "CC BY-NC-SA 3.0"

# -- act tags ----------------------------------------------------------------------------------


def damsl_tag(act_tag: str) -> str:
    """Collapse a raw SwDA act tag to its SWBD-DAMSL class.

    A line-for-line port of `Utterance.damsl_act_tag` in cgpotts/swda's swda.py, the corpus
    author's own reading of the coders' manual. The first of several comma-separated tags wins,
    as there.
    """
    tag = re.split(r"\s*[,;]\s*", act_tag)[0]
    if tag in ("qy^d", "qw^d", "b^m"):
        return tag
    if tag == "nn^e":
        return "ng"
    if tag == "ny^e":
        return "na"
    tag = re.sub(r"(.)\^.*", r"\1", tag)
    tag = re.sub(r"[\(\)@*]", "", tag)
    return {
        "qr": "qy",
        "fe": "ba",
        "oo": "oo_co_cc",
        "co": "oo_co_cc",
        "cc": "oo_co_cc",
        "fx": "sv",
        "aap": "aap_am",
        "am": "aap_am",
        "arp": "arp_nd",
        "nd": "arp_nd",
        "fo": 'fo_o_fw_"_by_bc',
        "o": 'fo_o_fw_"_by_bc',
        "fw": 'fo_o_fw_"_by_bc',
        '"': 'fo_o_fw_"_by_bc',
        "by": 'fo_o_fw_"_by_bc',
        "bc": 'fo_o_fw_"_by_bc',
    }.get(tag, tag)


CONTINUATION = "+"
# The spec's D2 classes (CLAUDE.md §4).
BACKCHANNEL_TAGS = frozenset({"b", "bh"})
NOISE_TAGS = frozenset({"x", "%"})
# A claim on the floor: statements, questions, directives, disagreement.
INTERRUPTION_TAGS = frozenset(
    {"sd", "sv", "qy", "qw", "qo", "qh", "qrr", "qy^d", "qw^d", "ad", "ar", "arp_nd", "ng", "nn"}
)
# Everything else (aa agree, ba appreciation, bk, na/ny yes-answers, b^m repeat-phrase, closings,
# hedges ...) is left out of D2's labels. Those short responses sit between the two classes, and
# forcing them into either would put label noise into train and test.
#
# They still matter for *who holds the floor*. SwDA's coders tag listener feedback by function, and
# a short "yeah" is often aa or ny rather than b. Counting it as a turn would flip the floor: the
# other caller simply continuing would look like a barge-in. So a minimal response of a few words
# (DataSettings.minimal_response_max_words) is "minor", like a backchannel, for floor purposes.
MINIMAL_RESPONSE_TAGS = frozenset({"aa", "bk", "ba", "na", "ny", "b^m"})

# -- utterances --------------------------------------------------------------------------------

_COMMENT = re.compile(r"<<[^<>]*>>")
_NONSPEECH = re.compile(r"<[^<>]*>")
_BRACE_OPEN = re.compile(r"\{[FDCEA]\s")
_MARKS = re.compile(r"-/|--|/|#|\(\(|\)\)|[\[\]{}+]|(?<!\S)-(?!\S)")  # last: a lone "-" pause mark


def clean_text(raw: str) -> str:
    """Strip SwDA transcription markup, keeping spoken words and ordinary punctuation."""
    s = _COMMENT.sub(" ", raw)
    s = _NONSPEECH.sub(" ", s)
    s = _BRACE_OPEN.sub(" ", s)
    s = _MARKS.sub(" ", s)
    s = re.sub(r"\s+([,.?!])", r"\1", s)  # "word ," -> "word,"
    s = re.sub(r"(^|\s)[,.]+(?=\s|$)", " ", s)  # punctuation orphaned by removed markup
    return " ".join(s.split())


@dataclass(frozen=True)
class Utt:
    conversation: int
    index: int  # transcript_index
    caller: str  # "A" or "B"
    act_tag: str
    damsl: str
    raw: str
    text: str  # clean_text(raw)
    # Does not take or hold the floor: a backchannel, noise, or a short minimal response.
    minor: bool

    @property
    def ends_unit(self) -> bool:
        """The segment closes its slash unit ("/") and was not abandoned ("-/")."""
        r = self.raw.rstrip()
        return r.endswith("/") and not r.endswith("-/")

    @property
    def claims_floor(self) -> bool:
        """Said something that takes (or holds) the turn."""
        return not self.minor and bool(normalize(self.text))


def is_minor(damsl: str, text: str, minimal_max_words: int) -> bool:
    if damsl in BACKCHANNEL_TAGS or damsl in NOISE_TAGS:
        return True
    return damsl in MINIMAL_RESPONSE_TAGS and len(normalize(text).split()) <= minimal_max_words


def read_conversation(path: Path, *, minimal_max_words: int) -> list[Utt]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    utts = []
    for r in rows:
        damsl = damsl_tag(r["act_tag"])
        text = clean_text(r["text"])
        utts.append(
            Utt(
                conversation=int(r["conversation_no"]),
                index=int(r["transcript_index"]),
                caller=r["caller"],
                act_tag=r["act_tag"],
                damsl=damsl,
                raw=r["text"],
                text=text,
                minor=is_minor(damsl, text, minimal_max_words),
            )
        )
    return sorted(utts, key=lambda u: u.index)


def iter_conversations(root: Path, *, minimal_max_words: int) -> Iterator[list[Utt]]:
    for path in sorted(root.glob("sw*utt/*.utt.csv")):
        yield read_conversation(path, minimal_max_words=minimal_max_words)


# -- examples ----------------------------------------------------------------------------------


@dataclass
class Example:
    id: str
    decision: str
    event: str
    group: str
    label: str
    kind: str
    agent_speaking: bool
    user_partial: str
    agent_last_utterance: str
    recent_turns: list[dict[str, str]]
    split: Split | None = None
    hard: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision": self.decision,
            "event": self.event,
            "source": "swda",
            "group": self.group,
            "split": self.split,
            "label": self.label,
            "kind": self.kind,
            "hard": self.hard,
            "snapshot": {
                "agent_speaking": self.agent_speaking,
                "user_partial": self.user_partial,
                "agent_last_utterance": self.agent_last_utterance,
                "recent_turns": self.recent_turns,
            },
        }


def _join(utts: Iterable[Utt]) -> str:
    return " ".join(u.text for u in utts if u.text)


def _history(
    utts: Sequence[Utt], end: int, user: str, max_turns: int
) -> tuple[list[dict[str, str]], str]:
    """Merged turns before position `end`, and the agent's last turn text.

    A turn is a maximal run of one caller's floor-claiming utterances. Minor utterances
    (backchannels, noise, short minimal responses) are left out of the history: a voice agent
    does not backchannel, and the packer's budget is better spent on content. Roles: `user` is the caller being decided about, the other plays
    the agent. Returns (turns before the agent's last turn, the agent's last turn).
    """
    turns: list[tuple[str, list[Utt]]] = []
    for u in utts[:end]:
        if not u.claims_floor:
            continue
        if turns and turns[-1][0] == u.caller:
            turns[-1][1].append(u)
        else:
            turns.append((u.caller, [u]))
    as_dicts = [
        {"speaker": "user" if caller == user else "agent", "text": _join(us)}
        for caller, us in turns
    ]
    agent_last = ""
    if as_dicts and as_dicts[-1]["speaker"] == "agent":
        agent_last = as_dicts.pop()["text"]
    return as_dicts[-max_turns:], agent_last


def _current_turn_start(utts: Sequence[Utt], pos: int) -> int:
    """Index where `utts[pos]`'s caller took the floor: after the other caller's last claim."""
    speaker = utts[pos].caller
    i = pos
    while i > 0:
        prev = utts[i - 1]
        if prev.caller != speaker and prev.claims_floor:
            break
        i -= 1
    return i


@dataclass(frozen=True)
class _D1Context:
    uid: str
    group: str
    agent_last: str
    recent: list[dict[str, str]]

    def example(self, kind: str, label: str, text: str) -> Example:
        return Example(
            id=f"{self.uid}-d1-{kind}",
            decision="turn_complete",
            event="user_pause",
            group=self.group,
            label=label,
            kind=kind,
            agent_speaking=False,
            user_partial=text,
            agent_last_utterance=self.agent_last,
            recent_turns=self.recent,
        )


def build_d1(conv: Sequence[Utt], *, rng: random.Random, max_history: int) -> list[Example]:
    """turn_complete examples from one conversation.

    At every point where one caller (S) stops and the other (O) speaks:
      complete      S's segment closes a unit, and O then claims the floor before S speaks again.
      continuation  S's segment does not close its unit, O only backchannels, and S resumes with a
                    "+" continuation: a genuine mid-utterance pause. Label false.
      (excluded)    S closed a unit, O only backchannelled, S began a new unit. Whether S meant to
                    stop is not recoverable from text; either label would be noise.
      (excluded)    O claimed the floor after an unfinished or abandoned segment: S was cut off,
                    which says nothing about whether S had finished.
    Every complete example also yields a truncation: the turn cut at a random word boundary inside
    its last utterance (at least one word kept and one removed). Label false.

    `user_partial` is S's whole turn so far, every word S said since O last claimed the floor,
    short responses included: in a live pipeline the recogniser's transcript accumulates from the
    moment the user took the floor, and it transcribes a "yeah" like any other word.
    """
    out: list[Example] = []
    n = len(conv)
    for pos in range(n - 1):
        u = conv[pos]
        if u.caller == conv[pos + 1].caller or not u.claims_floor:
            continue
        j = pos + 1
        o_claimed = False
        while j < n and conv[j].caller != u.caller:
            o_claimed = o_claimed or conv[j].claims_floor
            j += 1
        s_next = conv[j] if j < n else None

        start = _current_turn_start(conv, pos)
        turn = [x for x in conv[start : pos + 1] if x.caller == u.caller]
        user_partial = _join(turn)
        if not normalize(user_partial):
            continue
        recent, agent_last = _history(conv, start, u.caller, max_history)
        ctx = _D1Context(
            uid=f"swda-{u.conversation}-{u.index:04d}",
            group=f"swda:{u.conversation}",
            agent_last=agent_last,
            recent=recent,
        )

        if o_claimed and u.ends_unit:
            out.append(ctx.example("complete", "true", user_partial))
            words = u.text.split()
            if len(words) >= 2:
                keep = rng.randint(1, len(words) - 1)
                truncated = f"{_join(turn[:-1])} {' '.join(words[:keep])}".strip()
                if normalize(truncated):
                    out.append(ctx.example("truncation", "false", truncated))
        elif (
            not o_claimed
            and not u.ends_unit
            and not u.raw.rstrip().endswith("-/")
            and s_next is not None
            and s_next.damsl == CONTINUATION
        ):
            out.append(ctx.example("continuation", "false", user_partial))
    return out


def run_label(tags: Sequence[str]) -> str | None:
    """The D2 class of one caller's run of utterances, from their act tags.

    Any floor claim makes the run an interruption: "yeah, / but that's not what i asked" is one.
    Otherwise it is a backchannel if every utterance is one, noise if every utterance is noise,
    and left out if it mixes in anything else (agreement, appreciation, yes-answers ...).
    """
    if any(t in INTERRUPTION_TAGS for t in tags):
        return INTERRUPTION
    if tags and all(t in BACKCHANNEL_TAGS for t in tags):
        return BACKCHANNEL
    if tags and all(t in NOISE_TAGS for t in tags):
        return NOISE
    return None


def build_d2(conv: Sequence[Utt], *, partial_words: int, max_history: int) -> list[Example]:
    """barge_in examples from one conversation.

    A candidate is the moment B starts speaking while A holds the floor: B plays the user, A the
    agent. B's whole run of utterances, up to A's next utterance, decides the label (`run_label`).
    SwDA segments by dialog act, so "yeah, but that's not what i asked" is two rows, "yeah," then
    "but ...". Labelling single rows would lose exactly the case that makes barge-in hard
    (DECISIONS.md D-016). The text is the run's first `partial_words` words: what a recogniser
    would have when the decision fires, not the whole turn that follows.

    Skipped: a run that opens with a "+" continuation (B resuming its own interrupted unit, not
    reacting to A), and any moment where A has no floor-claiming words in its current turn (A had
    only backchannelled or given a short "yeah", so B already held the floor).

    agent_last_utterance is A's current turn up to that moment, often mid-sentence: what the user
    is reacting to.
    """
    out: list[Example] = []
    n = len(conv)
    for pos in range(1, n):
        u, prev = conv[pos], conv[pos - 1]
        if u.caller == prev.caller or u.damsl == CONTINUATION:
            continue
        j = pos
        while j < n and conv[j].caller == u.caller:
            j += 1
        run = conv[pos:j]
        label = run_label([x.damsl for x in run])
        if label is None:
            continue
        a_start = _current_turn_start(conv, pos - 1)
        agent_turn = [x for x in conv[a_start:pos] if x.caller == prev.caller and x.claims_floor]
        agent_now = _join(agent_turn)
        if not normalize(agent_now):
            continue
        recent, agent_before = _history(conv, a_start, u.caller, max_history)
        if agent_before:
            recent = [*recent, {"speaker": "agent", "text": agent_before}][-max_history:]
        tags = list(dict.fromkeys(x.damsl for x in run))
        out.append(
            Example(
                id=f"swda-{u.conversation}-{u.index:04d}-d2",
                decision="barge_in",
                event="user_speech_during_agent",
                group=f"swda:{u.conversation}",
                label=label,
                kind="+".join(tags[:4]),
                agent_speaking=True,
                user_partial=" ".join(_join(run).split()[:partial_words]),
                agent_last_utterance=agent_now,
                recent_turns=recent,
            )
        )
    return out


# -- splits and hard subsets -------------------------------------------------------------------


def assign(examples: list[Example], *, seed: int, fractions: dict[str, float]) -> None:
    for ex in examples:
        ex.split = assign_split(ex.group, seed=seed, fractions=fractions)


def _last_word(text: str) -> str:
    words = normalize(text).split()
    return words[-1] if words else ""


def mark_hard_d1(examples: list[Example], *, min_count: int) -> dict[str, float]:
    """D1 hard subset: truncations whose last word, alone, looks like the end of a turn.

    From train only: r(w) = (complete turns ending in w) / (truncations ending in w). A truncation
    ending in a word with r(w) >= 1 is one a "last word" heuristic would call complete: the
    cut-offs that land on a complete-looking clause ("i live in texas", cut from "i live in texas
    with my wife"). Returns r for the words that qualify.
    """
    train = [e for e in examples if e.split == "train"]
    ends_complete = Counter(_last_word(e.user_partial) for e in train if e.kind == "complete")
    ends_trunc = Counter(_last_word(e.user_partial) for e in train if e.kind == "truncation")
    ratio = {w: ends_complete[w] / c for w, c in ends_trunc.items() if c >= min_count}
    turn_final = {w: r for w, r in ratio.items() if r >= 1.0}
    for e in examples:
        e.hard = e.kind == "truncation" and _last_word(e.user_partial) in turn_final
    return turn_final


def _opens_with(text: str, form: str) -> bool:
    return text == form or text.startswith(form + " ")


def mark_hard_d2(
    examples: list[Example], *, min_count: int, min_share: float
) -> dict[str, tuple[int, int]]:
    """D2 hard subset: the surface forms both classes use (CLAUDE.md §4).

    From train only: a backchannel form f ("yeah", "right", "okay" ...) qualifies when at least
    `min_count` backchannels are exactly f, at least `min_count` interruptions open with f, and
    neither class dominates: each is at least `min_share` of f's rows. Volume alone does not make a
    form ambiguous. "uh huh" opens 734 train interruptions, but those are 8% of its rows, and the
    form alone predicts the class. Hard rows are backchannels that are a qualifying form, and
    interruptions that open with one.
    "yeah" is a backchannel; "yeah but that's not what i asked" is an interruption. Aggregate
    accuracy is dominated by unambiguous "uh huh" rows; this subset is where the classes meet.
    Returns {form: (backchannels, interruptions)} for the qualifying forms.
    """
    train = [e for e in examples if e.split == "train"]
    b_forms = Counter(normalize(e.user_partial) for e in train if e.label == BACKCHANNEL)
    i_texts = [normalize(e.user_partial) for e in train if e.label == INTERRUPTION]
    shared: dict[str, tuple[int, int]] = {}
    for form, n_b in b_forms.items():
        if not form or n_b < min_count:
            continue
        n_i = sum(_opens_with(t, form) for t in i_texts)
        if n_i >= min_count and min(n_b, n_i) / (n_b + n_i) >= min_share:
            shared[form] = (n_b, n_i)
    for e in examples:
        t = normalize(e.user_partial)
        if e.label == BACKCHANNEL:
            e.hard = t in shared
        elif e.label == INTERRUPTION:
            e.hard = any(_opens_with(t, f) for f in shared)
        else:
            e.hard = False
    return shared
