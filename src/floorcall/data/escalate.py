"""D4 (escalate) candidates from Customer Support on Twitter, for hand labelling.

Source: Thought Vector's "Customer Support on Twitter" (Kaggle), twcs.csv, CC BY-NC-SA 4.0. Phone
numbers and emails are already masked by the publisher.

The test set is real customer messages labelled by Yash (CLAUDE.md §4 D4), and nothing else. This
module builds the pool he labels from:

- **Banking threads only.** Threads with one of the eight banking or payments brands in the
  corpus. The demo agent does banking support, so the D4 number should describe that domain.
- **Split by thread.** A thread's split comes from sha256 of its root tweet, exactly like a SwDA
  conversation. Test and calib candidates come from different threads, and neither can share a
  thread with future training rows.
- **One message per thread**, so a single long argument cannot dominate the set.
- **Stratified.** Half the candidates are drawn from messages matching `ESCALATION_CUES`, half
  uniformly from the rest. At the natural rate, 300 labels might hold too few positives to say
  anything. The flag is stored on every row so results can be reported per stratum, and the set's
  positive rate is not the natural one. That is stated wherever D4 numbers appear. The labelling
  tool never shows the flag.

The customer plays the user; the company account plays the agent.
"""

from __future__ import annotations

import html
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from floorcall.data.splits import Split, assign_split

LICENCE = "CC BY-NC-SA 4.0"
BANK_BRANDS = (
    "BofA_Help",
    "AskAmex",
    "AskPayPal",
    "ChaseSupport",
    "Ask_WellsFargo",
    "AskCiti",
    "askvisa",
    "KeyBank_Help",
)

# Used only to stratify the candidate pool, never as a label.
ESCALATION_CUES = re.compile(
    r"\b(human|real person|a person|representative|supervisor|manager|someone|anyone|"
    r"call me|speak to|speak with|talk to|escalat\w*|complain\w*|ridiculous|unacceptable|"
    r"terrible|worst|awful|useless|horrible|frustrat\w*|angry|upset|again|still|third|"
    r"several times|no one|nobody|never|close my account|cancel\w*|lawyer|cfpb|bbb)\b|!!",
    re.IGNORECASE,
)

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_SIGNATURE = re.compile(r"\s[\^*~]\s?[A-Za-z]{1,12}\s*$")  # "... ^MG", "... ^Clarissa"
_MASK = re.compile(r"__\w+__")  # the publisher's PII masks
# Function words, for a deliberately simple language check. Words that several languages share
# ("no", "a") are in neither list.
_ENGLISH = frozenset(
    [
        "the",
        "an",
        "i",
        "you",
        "it",
        "is",
        "to",
        "and",
        "of",
        "my",
        "me",
        "for",
        "in",
        "on",
        "not",
        "this",
        "that",
        "have",
        "was",
        "with",
        "your",
        "can",
        "why",
        "what",
        "how",
        "do",
        "does",
        "did",
        "but",
        "just",
        "we",
        "are",
        "be",
    ]
)
_OTHER = frozenset(
    [
        "de",
        "la",
        "que",
        "el",
        "en",
        "los",
        "las",
        "por",
        "para",
        "una",
        "un",
        "mi",
        "se",
        "es",
        "con",
        "del",
        "le",
        "les",
        "et",
        "je",
        "pas",
        "est",
        "il",
        "y",
        "o",
        "da",
        "em",
        "um",
        "uma",
        "voce",
        "nao",
        "che",
        "di",
        "il",
        "per",
        "non",
    ]
)


def clean_tweet(text: str, *, agent: bool = False) -> str:
    """Strip handles, URLs, PII masks and entities. Only an agent tweet loses its trailing
    signature ("^MG", "^Clarissa"): on a customer tweet a trailing "*word" may be content."""
    s = html.unescape(text)
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _MASK.sub(" ", s)
    s = " ".join(s.split())
    if agent:
        s = _SIGNATURE.sub("", s)
    return s.strip()


def looks_english(text: str) -> bool:
    """Mostly ASCII, and English function words outnumber Spanish/Portuguese/French/Italian ones.

    Simple on purpose: it only has to keep non-English messages out of a labelling pool. Anything
    it lets through by mistake can still be skipped by the labeller.
    """
    if not text:
        return False
    ascii_share = sum(ch.isascii() for ch in text) / len(text)
    words = re.findall(r"[a-z']+", text.lower())
    en = sum(w in _ENGLISH for w in words)
    other = sum(w in _OTHER for w in words)
    return ascii_share > 0.97 and en >= 1 and en > other


@dataclass(frozen=True)
class Candidate:
    id: str
    group: str
    split: Split
    company: str
    prefiltered: bool
    user_partial: str
    agent_last_utterance: str
    recent_turns: list[dict[str, str]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _roots(parent: dict[int, int]) -> dict[int, int]:
    """Thread root for every tweet, by following reply links upward (iteratively, memoised)."""
    root: dict[int, int] = {}
    for start in parent:
        path: list[int] = []
        t = start
        while t in parent and t not in root and len(path) < 10_000:
            path.append(t)
            t = parent[t]
        r = root.get(t, t)
        for p in path:
            root[p] = r
    return root


def build_candidates(
    csv_path: Path,
    *,
    seed: int,
    fractions: dict[str, float],
    max_history: int,
    min_words: int = 3,
    max_words: int = 60,
) -> list[Candidate]:
    df = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={"author_id": str},
    )
    ids = [int(t) for t in df.tweet_id]
    text: dict[int, str] = dict(zip(ids, (str(x) for x in df.text), strict=True))
    author: dict[int, str] = dict(zip(ids, (str(x) for x in df.author_id), strict=True))
    inbound: dict[int, bool] = dict(zip(ids, (bool(x) for x in df.inbound), strict=True))
    parent: dict[int, int] = {
        t: int(p)
        for t, p in zip(ids, df.in_response_to_tweet_id, strict=True)
        if pd.notna(p) and int(p) in text
    }
    root = _roots(parent)
    brand_of_root: dict[int, str] = {}
    for t, a in author.items():
        if a in BANK_BRANDS:
            brand_of_root.setdefault(root.get(t, t), a)

    out: list[Candidate] = []
    for t in sorted(text):
        if not inbound[t]:
            continue
        r = root.get(t, t)
        if r not in brand_of_root:
            continue
        msg = clean_tweet(text[t])
        n_words = len(msg.split())
        if not (min_words <= n_words <= max_words) or not looks_english(msg):
            continue
        # ancestors, oldest first, merged into turns by speaker
        chain: list[int] = []
        p = parent.get(t)
        while p is not None and len(chain) < 4 * max_history:
            chain.append(p)
            p = parent.get(p)
        turns: list[dict[str, str]] = []
        for ancestor in reversed(chain):
            speaker = "user" if inbound[ancestor] else "agent"
            words = clean_tweet(text[ancestor], agent=speaker == "agent")
            if not words:
                continue
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["text"] += " " + words
            else:
                turns.append({"speaker": speaker, "text": words})
        agent_last = turns.pop()["text"] if turns and turns[-1]["speaker"] == "agent" else ""
        group = f"twcs:{r}"
        out.append(
            Candidate(
                id=f"twcs-{t}",
                group=group,
                split=assign_split(group, seed=seed, fractions=fractions),
                company=brand_of_root[r],
                prefiltered=bool(ESCALATION_CUES.search(msg)),
                user_partial=msg,
                agent_last_utterance=agent_last,
                recent_turns=turns[-max_history:],
            )
        )
    return out


def stratum(c: Candidate) -> str:
    """Which of the four equal cells a candidate belongs to: cue x agent context."""
    return (
        f"{'cue' if c.prefiltered else 'plain'}/{'reply' if c.agent_last_utterance else 'opener'}"
    )


STRATA = ("cue/reply", "cue/opener", "plain/reply", "plain/opener")


def select_for_labelling(
    candidates: Sequence[Candidate], *, split: Split, n: int, seed: int
) -> list[Candidate]:
    """`n` candidates from `split`, one per thread, a quarter from each stratum, shuffled.

    Two axes. Cue: without preselection, 300 labels may hold too few escalations to measure. Agent
    context: 74% of messages open a thread, but "frustration the agent is not resolving" can only
    happen after the agent has replied. Equal cells cover both kinds of escalation.
    """
    if n % len(STRATA):
        raise ValueError(f"n must be a multiple of {len(STRATA)}")
    rng = random.Random(f"{seed}:{split}")
    by_thread: dict[str, list[Candidate]] = {}
    for c in candidates:
        if c.split == split:
            by_thread.setdefault(c.group, []).append(c)
    one_each = [rng.choice(cs) for _, cs in sorted(by_thread.items())]
    per = n // len(STRATA)
    chosen: list[Candidate] = []
    for s in STRATA:
        cell = [c for c in one_each if stratum(c) == s]
        if len(cell) < per:
            raise ValueError(f"{split}/{s}: need {per} threads, have {len(cell)}")
        chosen += rng.sample(cell, per)
    rng.shuffle(chosen)
    return chosen
