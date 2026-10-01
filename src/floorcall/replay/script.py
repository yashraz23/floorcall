"""Replay scripts: one recorded call, its timing, and the action wanted at each decision point.

A script is a fixed recording: the user's speech and the agent's scripted lines, each placed on a
millisecond timeline. Both agents replay the same recording. What each agent does changes only its
own speech (a line it stops is cut short), never the user's. These scripts are illustrative demos,
not an evaluation set; the evaluation is Tables A-D (DECISIONS.md D-046).

    {
      "id": "05_mid_sentence_pause",            # must equal the file name
      "title": "...", "covers": ["D1 early pause", "D3 route"],
      "history": [{"speaker": "agent", "text": "..."}],      # turns before the timeline
      "timeline": [{"who": "user", "at_ms": 0, "dur_ms": 1000, "text": "I want to pay my"}, ...],
      "expect": [{"seg": 0, "want": "keep_listening", "why": "..."}, ...]
    }

`expect[].seg` is the index of a user segment: speech over the agent is decided on that segment,
and a pause is decided after it. One segment can have both, e.g. an interruption followed by a
pause; the kind of `want` says which is meant. `want` is the right action there. A
pause also takes the right `route`, and either event takes the right `escalate`, when the script
says so.

Scripts and their expectations are frozen before any model run: demo/scripts/MANIFEST.sha256
holds each file's SHA256. Replay refuses a script that is unlisted or changed, so a script cannot
be reworded after its output has been seen (D-046).
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from floorcall.questions import ROUTE_LABELS

MANIFEST = "MANIFEST.sha256"

PAUSE_WANTS = ("respond", "keep_listening")
BARGE_WANTS = ("stop_and_listen", "keep_talking", "ignore")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HistoryTurn(_Strict):
    speaker: Literal["user", "agent"]
    text: str = Field(min_length=1)


class Segment(_Strict):
    who: Literal["user", "agent"]
    at_ms: int = Field(ge=0)
    dur_ms: int = Field(gt=0)
    text: str = Field(min_length=1)

    @property
    def end_ms(self) -> int:
        return self.at_ms + self.dur_ms

    @property
    def words(self) -> list[str]:
        return self.text.split()


class Expect(_Strict):
    seg: int = Field(ge=0)
    want: Literal["respond", "keep_listening", "stop_and_listen", "keep_talking", "ignore"]
    route: str | None = None
    escalate: bool | None = None
    why: str = Field(min_length=1)

    @property
    def over_agent(self) -> bool:
        """Wanted at speech over the agent, rather than at a pause."""
        return self.want in BARGE_WANTS

    @field_validator("route")
    @classmethod
    def _known_route(cls, v: str | None) -> str | None:
        if v is not None and v not in ROUTE_LABELS:
            raise ValueError(f"unknown route {v!r}")
        return v


class Script(_Strict):
    id: str
    title: str
    covers: list[str] = Field(min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)
    timeline: list[Segment] = Field(min_length=1)
    expect: list[Expect] = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self) -> Script:
        starts = [s.at_ms for s in self.timeline]
        if starts != sorted(starts):
            raise ValueError("timeline must be in order of at_ms")
        for who in ("user", "agent"):
            mine = [s for s in self.timeline if s.who == who]
            for a, b in pairwise(mine):
                if b.at_ms < a.end_ms:
                    raise ValueError(f"{who} segments overlap at {b.at_ms} ms")
        users = [s for s in self.timeline if s.who == "user"]
        for s in self.timeline:
            if s.who == "agent" and any(u.at_ms <= s.at_ms < u.end_ms for u in users):
                raise ValueError(f"the agent starts at {s.at_ms} ms while the user is speaking")
        keys = [(e.seg, e.over_agent) for e in self.expect]
        if len(keys) != len(set(keys)):
            raise ValueError("at most one expectation per user segment and kind of event")
        for e in self.expect:
            if e.seg >= len(self.timeline) or self.timeline[e.seg].who != "user":
                raise ValueError(f"expect.seg {e.seg} is not a user segment")
            if e.want in BARGE_WANTS and e.route is not None:
                raise ValueError(f"seg {e.seg}: a route is only decided at a pause")
        return self

    def expectation(self, seg: int, over_agent: bool) -> Expect | None:
        """The expectation for speech over the agent (over_agent) or for the pause after `seg`."""
        return next((e for e in self.expect if (e.seg, e.over_agent) == (seg, over_agent)), None)


def load_script(path: Path) -> Script:
    script = Script.model_validate_json(path.read_text(encoding="utf-8"))
    if script.id != path.stem:
        raise ValueError(f"{path.name}: id {script.id!r} must equal the file name")
    return script


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path) -> dict[str, str]:
    path = directory / MANIFEST
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    return out


def freeze(paths: list[Path]) -> dict[str, str]:
    """Add scripts to their directory's manifest. A listed script can never change its hash."""
    (directory,) = {p.parent.resolve() for p in paths}
    manifest = read_manifest(directory)
    for p in paths:
        digest = sha256(p)
        if manifest.get(p.name, digest) != digest:
            raise ValueError(f"{p.name} is frozen with another hash: a frozen script never changes")
        manifest[p.name] = digest
    lines = "".join(f"{d}  {n}\n" for n, d in sorted(manifest.items()))
    with (directory / MANIFEST).open("w", encoding="utf-8", newline="\n") as f:
        f.write(lines)
    return manifest


def verify_frozen(paths: list[Path]) -> list[str]:
    """Problems: scripts not frozen, or changed since."""
    problems = []
    for p in paths:
        listed = read_manifest(p.parent).get(p.name)
        if listed is None:
            problems.append(f"{p.name}: not frozen (run `floorcall replay --freeze`)")
        elif listed != sha256(p):
            problems.append(f"{p.name}: changed since it was frozen")
    return problems
