"""Hand labels for D4: storage, and turning them into the frozen test set and the calib set.

`data/labels/escalate.candidates.v1.jsonl` is the pool, in the order it is shown. It is committed,
so what was offered for labelling is on the record. `data/labels/escalate.labels.jsonl` holds one
record per labelled candidate: `{id, label, guidelines, labeller, at}`, where label is `true`,
`false` or `skip`. The file is rewritten atomically after every keypress, so an interrupted session
loses nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floorcall.data.escalate import Candidate, stratum

GUIDELINES = "v1"
LABELS = ("true", "false", "skip")


def load_candidates(path: Path) -> list[Candidate]:
    with path.open(encoding="utf-8") as f:
        return [Candidate(**json.loads(line)) for line in f if line.strip()]


def save_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    if path.exists():
        raise FileExistsError(
            f"{path} exists. The pool is fixed once labelling starts; a new pool is a new version."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in candidates:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in records}


def save_labels(path: Path, labels: Mapping[str, Mapping[str, Any]], order: Sequence[str]) -> None:
    """Write labels in pool order, via a temp file and an atomic rename."""
    rank = {cid: i for i, cid in enumerate(order)}
    rows = sorted(labels.values(), key=lambda r: rank.get(r["id"], len(rank)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def label_record(cid: str, label: str, labeller: str) -> dict[str, Any]:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    return {
        "id": cid,
        "label": label,
        "guidelines": GUIDELINES,
        "labeller": labeller,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def to_rows(
    candidates: Sequence[Candidate], labels: Mapping[str, Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    """Labelled (true/false) candidates of one split, in the dataset row format. Skips drop out."""
    out = []
    for c in candidates:
        rec = labels.get(c.id)
        if c.split != split or rec is None or rec["label"] == "skip":
            continue
        if rec["guidelines"] != GUIDELINES:
            raise ValueError(f"{c.id} was labelled under guidelines {rec['guidelines']}")
        out.append(
            {
                "id": c.id,
                "decision": "escalate",
                "event": "user_pause",
                "source": "twcs",
                "group": c.group,
                "split": c.split,
                "label": rec["label"],
                "kind": stratum(c),
                # hard: no escalation keyword in the message, so a keyword match cannot find it
                "hard": not c.prefiltered,
                "company": c.company,
                "labeller": rec["labeller"],
                "guidelines": rec["guidelines"],
                "snapshot": {
                    "agent_speaking": False,
                    "user_partial": c.user_partial,
                    "agent_last_utterance": c.agent_last_utterance,
                    "recent_turns": c.recent_turns,
                },
            }
        )
    return out


def progress(
    candidates: Sequence[Candidate], labels: Mapping[str, Mapping[str, Any]]
) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for c in candidates:
        rec = labels.get(c.id)
        out.setdefault(c.split, Counter())["pending" if rec is None else rec["label"]] += 1
    return out
