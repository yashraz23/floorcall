"""Hand labels for D4: storage, and turning them into the frozen test set and the calib set.

`data/labels/escalate.candidates.v1.jsonl` is the pool, in the order it is shown. It is committed,
so what was offered for labelling is on the record. `data/labels/escalate.labels.jsonl` holds one
record per labelled candidate: `{id, label, guidelines, labeller, at}`, where label is `true`,
`false` or `skip`. The file is rewritten atomically after every keypress, so an interrupted session
loses nothing.

Guideline v2 (DECISIONS.md D-033) is applied to a seeded, stratified sample of the v1 eval sets,
relabelled blind into `escalate.labels.v2.jsonl`. v2 has no skip: its own text says what to do
when unsure. The v1 file is never written again, and v1 and v2 labels never share an eval set.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from floorcall.data.escalate import STRATA, Candidate, stratum

GUIDELINES = "v1"  # the version escalate.labels.jsonl was made under
LABELS = ("true", "false", "skip")
LABELS_BY_GUIDELINES = {"v1": LABELS, "v2": ("true", "false")}

# Guideline v2, Yash's text, verbatim (D-033). docs/labelling-escalate.md carries the same three
# lines (a test checks), and the relabel tool prints them above every message.
GUIDELINE_V2 = (
    "Escalate (y) if any of: (1) Asks for a human, a call, or a manager, or says they can't reach "
    "or get help from support. (2) Says they already raised it (called, DM'd, visited, told a rep) "
    "and it's still unresolved or past a promised time. (3) Says they'll leave, close the account, "
    "switch banks, or take it public/viral.\n"
    "Not escalate (n): venting or insults with no prior contact and no threat to leave; a first "
    'report of a problem, even an angry one; questions, thanks, "sent you a DM"; fraud or '
    "phishing reports; general sarcasm about the company.\n"
    "If unsure: apply the three rules literally. If none clearly fits, n."
)
RELABEL_SAMPLE_FILE = "escalate.relabel_v2.sample.json"
LABELS_V2_FILE = "escalate.labels.v2.jsonl"
# The primary D4 model's training messages (D-034), in display order, and Yash's labels for them.
TRAIN_SAMPLE_FILE = "escalate.train_sample.v2.jsonl"
TRAIN_LABELS_FILE = "escalate.train_labels.v2.jsonl"


def load_candidates(path: Path) -> list[Candidate]:
    from floorcall.data import private

    if not path.exists() and path.name in (private.CANDIDATES, private.TRAIN_SAMPLE):
        raise private.PrivateDataMissingError(path)  # real Twitter text, not public (D-049)
    with path.open(encoding="utf-8") as f:
        return [Candidate(**json.loads(line)) for line in f if line.strip()]


def save_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    if path.exists():
        raise FileExistsError(
            f"{path} exists. The pool is fixed once labelling starts; a new pool is a new version."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(candidates_bytes(candidates))


def candidates_bytes(candidates: Sequence[Candidate]) -> bytes:
    """A candidates file's exact bytes: one JSON object per line, UTF-8, LF line ends."""
    lines = (json.dumps(c.to_json(), ensure_ascii=False) + "\n" for c in candidates)
    return "".join(lines).encode("utf-8")


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


def label_record(
    cid: str, label: str, labeller: str, guidelines: str = GUIDELINES
) -> dict[str, Any]:
    allowed = LABELS_BY_GUIDELINES[guidelines]
    if label not in allowed:
        raise ValueError(
            f"label must be one of {allowed} under guidelines {guidelines}, not {label!r}"
        )
    return {
        "id": cid,
        "label": label,
        "guidelines": guidelines,
        "labeller": labeller,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def to_rows(
    candidates: Sequence[Candidate],
    labels: Mapping[str, Mapping[str, Any]],
    split: str,
    *,
    guidelines: str = GUIDELINES,
) -> list[dict[str, Any]]:
    """Labelled (true/false) candidates of one split, in the dataset row format. Skips drop out.

    Every label must have been made under `guidelines`: versions are never mixed in one set.
    """
    out = []
    for c in candidates:
        rec = labels.get(c.id)
        if c.split != split or rec is None or rec["label"] == "skip":
            continue
        if rec["guidelines"] != guidelines:
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


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Split `n` across strata in proportion to their sizes: largest remainder, ties by name."""
    total = sum(sizes.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot draw {n} from {total}")
    quota = {s: Fraction(n * k, total) for s, k in sizes.items()}
    out = {s: int(q) for s, q in quota.items()}
    by_remainder = sorted(quota, key=lambda s: (-(quota[s] - out[s]), s))
    for s in by_remainder[: n - sum(out.values())]:
        out[s] += 1
    return out


def draw_relabel_sample(
    candidates: Sequence[Candidate],
    v1_labels: Mapping[str, Mapping[str, Any]],
    sizes: Mapping[str, int],
    seed: int,
) -> dict[str, Any]:
    """The messages relabelled under guideline v2: a seeded, stratified sample of each v1 eval set.

    A split's population is its v1 eval set: its candidates that have a v1 label of true or false
    (a skipped message never entered an eval set). Each of the four sampling strata gets a share
    of the split's sample in proportion to its size, and its messages are drawn uniformly by a
    generator seeded from `seed`, the split and the stratum. Only ids, splits, strata and whether
    a v1 label exists are read: never a label's value, an LLM label or a disagreement. The display
    order shuffles both splits together, so the tool cannot reveal which split a message is from.
    """
    population: dict[str, dict[str, list[str]]] = {}
    for c in candidates:
        rec = v1_labels.get(c.id)
        if c.split in sizes and rec is not None and rec["label"] != "skip":
            population.setdefault(c.split, {s: [] for s in STRATA})[stratum(c)].append(c.id)
    allocation: dict[str, dict[str, int]] = {}
    ids: dict[str, list[str]] = {}
    for split, n in sizes.items():
        cells = population.get(split, {})
        allocation[split] = allocate({s: len(v) for s, v in cells.items()}, n)
        ids[split] = sorted(
            cid
            for s, k in allocation[split].items()
            for cid in random.Random(f"{seed}:relabel-v2:{split}:{s}").sample(sorted(cells[s]), k)
        )
    order = sorted(cid for split_ids in ids.values() for cid in split_ids)
    random.Random(f"{seed}:relabel-v2:order").shuffle(order)
    return {
        "guidelines": "v2",
        "seed": seed,
        "sizes": dict(sizes),
        "method": (
            "per split, the v1 eval set (true/false v1 labels); proportional allocation across the "
            "four sampling strata (largest remainder); uniform draw per stratum with "
            "random.Random(f'{seed}:relabel-v2:{split}:{stratum}'); display order shuffled with "
            "random.Random(f'{seed}:relabel-v2:order')"
        ),
        "population": {sp: {s: len(v) for s, v in c.items()} for sp, c in population.items()},
        "allocation": allocation,
        "ids": ids,
        "order": order,
    }


def draw_train_sample(
    pool: Sequence[Candidate], eval_groups: set[str], *, n: int, seed: int
) -> list[Candidate]:
    """The primary D4 model's training messages (D-034): `n` of the train pool, uniformly at random.

    They come back in the order the labelling tool shows them (random.sample's own order, which
    is random). Refuses a pool message that is not train-split, or whose thread is in an eval set:
    the training data must be conversation-disjoint from test and calib.
    """
    if not_train := [c.id for c in pool if c.split != "train"]:
        raise ValueError(f"{len(not_train)} pool messages are not train-split, e.g. {not_train[0]}")
    if shared := {c.group for c in pool} & eval_groups:
        raise ValueError(f"{len(shared)} pool threads are also in an eval set")
    return random.Random(f"{seed}:train-hand-v2").sample(sorted(pool, key=lambda c: c.id), n)


def save_relabel_sample(path: Path, sample: Mapping[str, Any]) -> bool:
    """Write the sample once. Returns False if the same sample is already there; refuses a change."""
    text = json.dumps(sample, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise FileExistsError(f"{path} holds a different sample; a drawn sample is never redrawn")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def progress(
    candidates: Sequence[Candidate], labels: Mapping[str, Mapping[str, Any]]
) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for c in candidates:
        rec = labels.get(c.id)
        out.setdefault(c.split, Counter())["pending" if rec is None else rec["label"]] += 1
    return out
