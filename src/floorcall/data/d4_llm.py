"""D4 training rows: real customer messages from train-split threads, labelled by an LLM.

Yash's decision (DECISIONS.md D-030): real messages from the Twitter support corpus, labelled by a
LLM (via OpenRouter, D-031), never synthetic ones, every row tagged `source=llm_labelled`. The test and calib
sets stay his hand labels.

- **Pool.** Banking threads in the *train* split (thread-level hash, as for test and calib), one
  message per thread, sampled with a fixed seed. A message whose normalized text equals any test
  or calib message is also dropped: the same complaint can be pasted into more than one thread,
  and the thread split alone would miss it.
- **Labels.** One LLM call per message with Yash's guidelines (floorcall.llm.prompts). "unsure"
  drops the message, as his skip did. Every label is written to
  `data/labels/escalate.llm_labels.v1.jsonl` with its model and prompt version, and that file is
  committed, so the train set can be rebuilt without calling the API again.
- **Agreement.** The labeller also labels the test and calib candidates. Its agreement with Yash's
  labels (Cohen's kappa, accuracy, confusion) is reported, and is **measurement only**: the prompt
  is fixed before it sees a test message, and the test agreement changes nothing.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floorcall.data.escalate import Candidate, stratum
from floorcall.normalize import normalize

LLM_LABELS_FILE = "escalate.llm_labels.v1.jsonl"
LABEL_TO_BOOL = {"escalate": "true", "no": "false"}


def train_pool(
    candidates: Sequence[Candidate], held_out: Sequence[Candidate], *, n: int, seed: int
) -> tuple[list[Candidate], dict[str, int]]:
    """`n` train-split messages, one per thread, none matching a held-out message's text."""
    banned = {normalize(c.user_partial) for c in held_out}
    train = [c for c in candidates if c.split == "train"]
    kept = [c for c in train if normalize(c.user_partial) not in banned]
    rng = random.Random(f"{seed}:d4-train")
    by_thread: dict[str, list[Candidate]] = {}
    for c in kept:
        by_thread.setdefault(c.group, []).append(c)
    one_each = [rng.choice(cs) for _, cs in sorted(by_thread.items())]
    if len(one_each) < n:
        raise ValueError(f"only {len(one_each)} train threads available, {n} requested")
    chosen = sorted(rng.sample(one_each, n), key=lambda c: c.id)
    report = {
        "train_candidates": len(train),
        "dropped_text_matches_held_out": len(train) - len(kept),
        "train_threads": len(one_each),
        "chosen": n,
    }
    return chosen, report


def load_llm_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def save_llm_labels(path: Path, labels: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for key in sorted(labels):
            f.write(json.dumps(labels[key], ensure_ascii=False) + "\n")
    tmp.replace(path)


def label_all(
    candidates: Sequence[Candidate],
    labels: dict[str, dict[str, Any]],
    ask: Callable[[Candidate], str],
    *,
    model: str,
    prompt_version: str,
    split_of: Callable[[Candidate], str],
    concurrency: int,
    save: Callable[[dict[str, dict[str, Any]]], None],
    save_every: int = 100,
) -> dict[str, dict[str, Any]]:
    """Label every candidate not yet labelled under this model and prompt version.

    `ask` returns "escalate", "no" or "unsure". Progress is saved every `save_every` labels, so an
    interrupted run (or one stopped by the budget) resumes where it left off.
    """
    todo = [
        c
        for c in candidates
        if not (
            c.id in labels
            and labels[c.id]["model"] == model
            and labels[c.id]["prompt"] == prompt_version
        )
    ]

    def one(c: Candidate) -> tuple[Candidate, str]:
        return c, ask(c)

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for c, label in pool.map(one, todo):
            labels[c.id] = {
                "id": c.id,
                "split": split_of(c),
                "label": label,
                "model": model,
                "prompt": prompt_version,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            done += 1
            if done % save_every == 0:
                save(labels)
    save(labels)
    return labels


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, non-empty label sequences")
    cats = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(
    hand: Mapping[str, str], llm: Mapping[str, str], strata: Mapping[str, str]
) -> dict[str, Any]:
    """LLM labels against Yash's, on messages both labelled (his skips and its unsures aside).

    Labels are "true"/"false"; `strata` maps an id to its sampling stratum.
    """
    both = sorted(i for i in hand if i in llm and llm[i] in ("true", "false"))
    unsure = sum(1 for i in hand if llm.get(i) == "unsure")
    h = [hand[i] for i in both]
    m = [llm[i] for i in both]
    confusion = {
        f"hand_{x}_llm_{y}": sum(1 for a, b in zip(h, m, strict=True) if a == x and b == y)
        for x in ("true", "false")
        for y in ("true", "false")
    }
    tp = confusion["hand_true_llm_true"]
    by_stratum = {}
    for s in sorted(set(strata.values())):
        ids = [i for i in both if strata.get(i) == s]
        if ids:
            by_stratum[s] = {
                "n": len(ids),
                "accuracy": sum(hand[i] == llm[i] for i in ids) / len(ids),
            }
    return {
        "n": len(both),
        "llm_unsure": unsure,
        "accuracy": sum(a == b for a, b in zip(h, m, strict=True)) / len(both) if both else None,
        "cohen_kappa": cohen_kappa(h, m) if both else None,
        # with the hand label as reference
        "escalate_precision": tp / max(1, tp + confusion["hand_false_llm_true"]),
        "escalate_recall": tp / max(1, tp + confusion["hand_true_llm_false"]),
        "confusion": confusion,
        "by_stratum": by_stratum,
    }


def train_rows(
    candidates: Iterable[Candidate], labels: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Processed train rows for every labelled, not-unsure candidate."""
    out = []
    for c in candidates:
        rec = labels.get(c.id)
        if rec is None or rec["label"] not in LABEL_TO_BOOL:
            continue
        out.append(
            {
                "id": c.id,
                "decision": "escalate",
                "event": "user_pause",
                "source": "llm_labelled",
                "group": c.group,
                "split": "train",
                "label": LABEL_TO_BOOL[rec["label"]],
                "kind": stratum(c),
                "hard": not c.prefiltered,
                "company": c.company,
                "labeller": rec["model"],
                "prompt": rec["prompt"],
                "snapshot": {
                    "agent_speaking": False,
                    "user_partial": c.user_partial,
                    "agent_last_utterance": c.agent_last_utterance,
                    "recent_turns": c.recent_turns,
                },
            }
        )
    return out


def run(settings: Any, split: str) -> dict[str, Any]:
    """Label one split with the LLM: "calib" or "test" (agreement with Yash), or "train"."""
    from floorcall.config import REPO_ROOT, Settings
    from floorcall.data.build import balance, processed_file
    from floorcall.data.download import fetch
    from floorcall.data.escalate import build_candidates
    from floorcall.data.freeze import write_jsonl_gz
    from floorcall.data.labelling import load_candidates, load_labels
    from floorcall.llm.client import Ledger, LLMClient
    from floorcall.llm.prompts import LABELLER_SCHEMA, LABELLER_VERSION, labeller_messages

    s: Settings = settings
    if split not in ("calib", "test", "train"):
        raise ValueError(f"split must be calib, test or train, not {split!r}")
    if s.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env, which git ignores)")
    labels_dir = s.paths.labels
    held_out = load_candidates(labels_dir / "escalate.candidates.v1.jsonl")
    pool_report: dict[str, int] = {}
    if split == "train":
        pool = build_candidates(
            fetch("twcs", s.paths.data_raw),
            seed=s.splits.seed,
            fractions=s.splits.model_dump(include={"train", "calib", "test"}),
            max_history=s.data.max_history_turns,
        )
        cands, pool_report = train_pool(pool, held_out, n=s.llm.d4_train_rows, seed=s.splits.seed)
    else:
        cands = [c for c in held_out if c.split == split]

    ledger = Ledger(REPO_ROOT / "runs" / "llm" / "ledger.sqlite", s.llm)
    client = LLMClient(s.llm, s.openrouter_api_key.get_secret_value(), ledger)
    model = s.llm.labeller_model

    def ask(c: Candidate) -> str:
        out = client.chat_json(
            model=model,
            messages=labeller_messages(c.recent_turns, c.agent_last_utterance, c.user_partial),
            schema=LABELLER_SCHEMA,
            purpose=f"d4-label-{split}",
        )
        return str(out.data["label"])

    path = labels_dir / LLM_LABELS_FILE
    labels = label_all(
        cands,
        load_llm_labels(path),
        ask,
        model=model,
        prompt_version=LABELLER_VERSION,
        split_of=lambda _c: split,
        concurrency=s.llm.concurrency,
        save=lambda lab: save_llm_labels(path, lab),
    )
    client.close()
    summary: dict[str, Any] = {
        "split": split,
        "model": model,
        "prompt": LABELLER_VERSION,
        "labels": dict(Counter(labels[c.id]["label"] for c in cands)),
        "spent_usd_total": ledger.spent(),
        **pool_report,
    }
    if split == "train":
        rows = train_rows(cands, labels)
        write_jsonl_gz(s.paths.data_processed / processed_file("escalate", "train"), rows)
        summary["train_rows"] = len(rows)
        summary["train_balance"] = balance(rows)["train"]["labels"]
    else:
        hand = {
            k: v["label"]
            for k, v in load_labels(labels_dir / "escalate.labels.jsonl").items()
            if v["label"] in ("true", "false") and k in {c.id for c in cands}
        }
        llm = {c.id: LABEL_TO_BOOL.get(labels[c.id]["label"], "unsure") for c in cands}
        summary["agreement"] = agreement(hand, llm, {c.id: stratum(c) for c in cands})
    out = REPO_ROOT / "results" / "d4_labeller" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary
