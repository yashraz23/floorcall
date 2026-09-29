"""D4 training rows: real customer messages from train-split threads, labelled by an LLM.

Yash's decision (DECISIONS.md D-030): real messages from the Twitter support corpus, labelled by an
LLM (via OpenRouter, D-031), never synthetic ones, every row tagged `source=llm_labelled`. The test
and calib sets stay his hand labels: since D-033, the v2 eval sets (his blind relabel under
guideline v2), which are what every agreement is measured against.

- **Pool.** Banking threads in the *train* split (thread-level hash, as for test and calib), one
  message per thread, sampled with a fixed seed. A message whose normalized text equals any test
  or calib message is also dropped: the same complaint can be pasted into more than one thread,
  and the thread split alone would miss it.
- **Labels.** One LLM call per message with Yash's guideline (floorcall.llm.prompts). Prompt v3
  answers y or n, as guideline v2 does; v1 and v2 could say "unsure", which dropped the message. Every label is written to one file per prompt version,
  `data/labels/escalate.llm_labels.v<N>.jsonl`, with its model, prompt version and the provider
  that served it, and those files are committed, so the train set can be rebuilt without calling
  the API again and an earlier version's labels are never overwritten.
- **Agreement and the gate** (D-032). A prompt is judged on calib only: its agreement with Yash's
  labels there must reach the configured kappa and escalate precision before it may label test or
  train, and `run` refuses otherwise. The test agreement is **measurement only**: it is taken once,
  with a prompt already accepted, and changes nothing.
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

# "escalate"/"no" are prompts v1 and v2; "y"/"n" are v3, in guideline v2's own terms.
LABEL_TO_BOOL = {"escalate": "true", "no": "false", "y": "true", "n": "false"}
# D4's training rows (D-035): the train pool labelled by prompt v3, tagged with this source and
# written to build.TRAIN_FILES["escalate"].
TRAIN_PROMPT = "llm-labeller-v3"
TRAIN_SOURCE = "llm_v3"


def llm_labels_file(prompt_version: str) -> str:
    """One labels file per prompt version: "llm-labeller-v2" -> "escalate.llm_labels.v2.jsonl"."""
    return f"escalate.llm_labels.{prompt_version.removeprefix('llm-labeller-')}.jsonl"


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
    ask: Callable[[Candidate], tuple[str, str | None]],
    *,
    model: str,
    prompt_version: str,
    split_of: Callable[[Candidate], str],
    concurrency: int,
    save: Callable[[dict[str, dict[str, Any]]], None],
    save_every: int = 100,
) -> dict[str, dict[str, Any]]:
    """Label every candidate not yet labelled under this model and prompt version.

    `ask` returns the label ("escalate", "no" or "unsure") and the provider that served it.
    Progress is saved every `save_every` labels, so an interrupted run (or one stopped by the
    budget) resumes where it left off.
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

    def one(c: Candidate) -> tuple[Candidate, tuple[str, str | None]]:
        return c, ask(c)

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for c, (label, provider) in pool.map(one, todo):
            labels[c.id] = {
                "id": c.id,
                "split": split_of(c),
                "label": label,
                "model": model,
                "prompt": prompt_version,
                "provider": provider,
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


def gate_failures(
    calib: Mapping[str, Any] | None,
    *,
    model: str,
    prompt_version: str,
    endpoint: str | None,
    hand_labels: str,
    min_kappa: float,
    min_escalate_precision: float,
) -> list[str]:
    """Why this labeller may not label test or train yet; empty once calib has accepted it.

    `calib` is the calib summary `run` wrote for this prompt version, or None if there is none.
    `hand_labels` names the hand labels it must have been judged against ("guideline v2").
    """
    if calib is None:
        return [f"calib has not been labelled with {prompt_version}"]
    want = (model, prompt_version, endpoint, hand_labels)
    served = tuple(calib.get(k) for k in ("model", "prompt", "provider_pin", "hand_labels"))
    if served != want:
        return [f"calib was labelled and judged as {served}, not {want}"]
    a = calib["agreement"]
    out = []
    if a["cohen_kappa"] is None or a["cohen_kappa"] < min_kappa:
        out.append(f"kappa {a['cohen_kappa']} is below {min_kappa}")
    if a["escalate_precision"] < min_escalate_precision:
        out.append(
            f"escalate precision {a['escalate_precision']} is below {min_escalate_precision}"
        )
    return out


def disagreements(
    candidates: Iterable[Candidate], hand: Mapping[str, str], llm: Mapping[str, str]
) -> list[dict[str, Any]]:
    """The messages where the LLM's label differs from Yash's, with what each saw."""
    return [
        {
            "id": c.id,
            "stratum": stratum(c),
            "hand": hand[c.id],
            "llm": llm[c.id],
            "agent_last": c.agent_last_utterance,
            "message": c.user_partial,
        }
        for c in candidates
        if c.id in hand and llm.get(c.id) in ("true", "false") and hand[c.id] != llm[c.id]
    ]


def train_rows(
    candidates: Iterable[Candidate],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    source: str = "llm_labelled",
) -> list[dict[str, Any]]:
    """Processed train rows for every labelled, not-unsure candidate, tagged with `source`."""
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
                "source": source,
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
    """Label one split with the LLM: "calib" or "test" (agreement with Yash), or "train".

    "train" labels the D-030 pool with prompt v3, D4's training data since D-035, tagged
    `source=llm_v3`. The gate did not accept v3; D-035 chose it anyway, so train is not gated.
    """
    from floorcall.config import REPO_ROOT, Settings
    from floorcall.data.build import TEST_VERSIONS, balance, escalate_eval_rows, processed_file
    from floorcall.data.download import fetch
    from floorcall.data.escalate import build_candidates
    from floorcall.data.freeze import write_jsonl_gz
    from floorcall.data.labelling import load_candidates
    from floorcall.llm.client import Ledger, LLMClient
    from floorcall.llm.prompts import LABELLER_SCHEMA, LABELLER_VERSION, labeller_messages

    s: Settings = settings
    if split not in ("calib", "test", "train"):
        raise ValueError(f"split must be calib, test or train, not {split!r}")
    if split == "train" and LABELLER_VERSION != TRAIN_PROMPT:
        raise RuntimeError(
            f"D4's training labels are {TRAIN_PROMPT} (D-035), not {LABELLER_VERSION}"
        )
    model = s.llm.labeller_model
    pin = s.llm.provider_pins.get(model)
    guidelines = TEST_VERSIONS["escalate"]  # the hand labels of the eval sets D4 is scored on
    gate_kw: dict[str, Any] = {
        "model": model,
        "prompt_version": LABELLER_VERSION,
        "endpoint": pin.endpoint if pin else None,
        "hand_labels": f"guideline {guidelines}",
        "min_kappa": s.llm.labeller_min_kappa,
        "min_escalate_precision": s.llm.labeller_min_escalate_precision,
    }
    results_dir = REPO_ROOT / "results" / "d4_labeller" / LABELLER_VERSION
    if split == "test":  # measurement against Yash's test labels needs an accepted prompt
        calib_file = results_dir / "calib.json"
        calib = json.loads(calib_file.read_text(encoding="utf-8")) if calib_file.exists() else None
        if failures := gate_failures(calib, **gate_kw):
            raise RuntimeError(
                f"{LABELLER_VERSION} has not been accepted on calib, so it may not label {split}: "
                + "; ".join(failures)
            )
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
        test_rows, calib_rows = escalate_eval_rows(s, guidelines)
        eval_rows = test_rows if split == "test" else calib_rows
        by_id = {c.id: c for c in held_out}
        cands = [by_id[r["id"]] for r in eval_rows]
        hand = {r["id"]: r["label"] for r in eval_rows}

    ledger = Ledger(REPO_ROOT / "runs" / "llm" / "ledger.sqlite", s.llm)
    client = LLMClient(s.llm, s.openrouter_api_key.get_secret_value(), ledger)

    def ask(c: Candidate) -> tuple[str, str | None]:
        out = client.chat_json(
            model=model,
            messages=labeller_messages(c.recent_turns, c.agent_last_utterance, c.user_partial),
            schema=LABELLER_SCHEMA,
            purpose=f"d4-label-{split}",
        )
        return str(out.data["label"]), out.provider

    path = labels_dir / llm_labels_file(LABELLER_VERSION)
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
        "provider_pin": gate_kw["endpoint"],
        "hand_labels": gate_kw["hand_labels"],
        "served_by": dict(Counter(str(labels[c.id].get("provider")) for c in cands)),
        "labels": dict(Counter(labels[c.id]["label"] for c in cands)),
        "spent_usd_total": ledger.spent(),
        **pool_report,
    }
    if split == "train":
        rows = train_rows(cands, labels, source=TRAIN_SOURCE)
        write_jsonl_gz(s.paths.data_processed / processed_file("escalate", "train"), rows)
        summary["train_file"] = processed_file("escalate", "train")
        summary["train_rows"] = len(rows)
        summary["train_balance"] = balance(rows)["train"]["labels"]
    else:
        summary["eval_set"] = f"{split} {guidelines}, {len(cands)} messages"
        llm = {c.id: LABEL_TO_BOOL.get(labels[c.id]["label"], "unsure") for c in cands}
        summary["agreement"] = agreement(hand, llm, {c.id: stratum(c) for c in cands})
        if split == "calib":
            failures = gate_failures(summary, **gate_kw)
            summary["gate"] = {
                "min_kappa": gate_kw["min_kappa"],
                "min_escalate_precision": gate_kw["min_escalate_precision"],
                "accepted": not failures,
                "failures": failures,
            }
        summary["disagreements"] = disagreements(cands, hand, llm)
    out = results_dir / (f"train.{TRAIN_SOURCE}.json" if split == "train" else f"{split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary
