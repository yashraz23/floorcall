"""D3 (route) from CLINC150: the banking domain's 15 intents plus out-of-scope.

Source: clinc/oos-eval, data/data_oos_plus.json (the "plus" variant: 250 out-of-scope training
queries rather than 100). Its counts match the Hugging Face `clinc/clinc_oos` "plus" parquet row
for row. Licence CC BY 3.0. Domain membership comes from the same repo's data/domains.json.

Splits are CLINC's own: train -> train, val -> calib, test -> test. A utterance is its own group,
so there is no conversation to leak. What can leak is the same query text on both sides, so any
calib or test row whose normalized text also appears in train is dropped and counted.

Only CLINC's `oos` queries become out_of_scope. The other nine domains' in-scope queries (travel,
kitchen ...) are not used. That keeps the label definition CLINC's own. It also means the hard
out-of-scope cases are CLINC's near-domain ones ("how much is an overdraft fee for bank").

CLINC has no dialogue, so each query is packed as a `user_pause` state with no history. The agent's
last utterance is one of a few fixed openers, picked from a hash of the row id so that it is
deterministic, and so the model does not learn one constant string as part of the input.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from floorcall.normalize import normalize
from floorcall.questions import BANKING_INTENTS, OUT_OF_SCOPE

SOURCE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
SOURCE_SHA256 = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"
DOMAINS_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json"
DOMAINS_SHA256 = "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"
LICENCE = "CC BY 3.0"

CLINC_SPLIT = {"train": "train", "val": "calib", "test": "test"}
AGENT_OPENERS = (
    "how can i help you today",
    "what can i do for you",
    "is there anything else i can help you with",
    "sure what do you need",
    "",
)


@dataclass
class BuildReport:
    rows: dict[str, Counter[str]] = field(default_factory=dict)
    dropped_duplicates: dict[str, int] = field(default_factory=dict)


def _opener(row_id: str) -> str:
    h = int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:4], "big")
    return AGENT_OPENERS[h % len(AGENT_OPENERS)]


def build_d3(raw_json: Path, domains_json: Path) -> tuple[list[dict[str, Any]], BuildReport]:
    data = json.loads(raw_json.read_text(encoding="utf-8"))
    banking = json.loads(domains_json.read_text(encoding="utf-8"))["banking"]
    if set(banking) != set(BANKING_INTENTS):
        raise ValueError(f"CLINC banking intents {sorted(banking)} differ from questions.py")

    report = BuildReport()
    rows: list[dict[str, Any]] = []
    seen_train: set[str] = set()
    for clinc_split in ("train", "val", "test"):  # train first: dedup is against train
        split = CLINC_SPLIT[clinc_split]
        pairs = [(t, i) for t, i in data[clinc_split] if i in banking]
        pairs += [(t, OUT_OF_SCOPE) for t, _ in data[f"oos_{clinc_split}"]]
        counts: Counter[str] = Counter()
        dropped = 0
        for n, (text, intent) in enumerate(pairs):
            key = normalize(text)
            if split == "train":
                seen_train.add(key)
            elif key in seen_train:
                dropped += 1
                continue
            row_id = f"clinc-{clinc_split}-{n:05d}"
            rows.append(
                {
                    "id": row_id,
                    "decision": "route",
                    "event": "user_pause",
                    "source": "clinc150",
                    "group": row_id,
                    "split": split,
                    "label": intent,
                    "kind": "oos" if intent == OUT_OF_SCOPE else "in_scope",
                    "hard": False,
                    "snapshot": {
                        "agent_speaking": False,
                        "user_partial": text,
                        "agent_last_utterance": _opener(row_id),
                        "recent_turns": [],
                    },
                }
            )
            counts[intent] += 1
        report.rows[split] = counts
        report.dropped_duplicates[split] = dropped
    return rows, report
