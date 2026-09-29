"""Build every dataset from its raw source: train and calib to data/processed/, test frozen.

    uv run floorcall data build

Order of operations, the same for every decision:
  1. build examples from the raw source
  2. assign splits (by conversation for SwDA; CLINC's own splits for CLINC)
  3. mark hard subsets, from train statistics only
  4. assert no group appears in two splits
  5. write train and calib (overwritten freely); freeze test (never edited; a rebuild must match)
  6. write class balance per split to data/processed/cards/<decision>.json, which is committed
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from floorcall.config import Settings
from floorcall.data import clinc, swda
from floorcall.data.download import SOURCES, fetch
from floorcall.data.freeze import freeze, sha256_file, write_jsonl_gz
from floorcall.data.splits import SPLITS, assert_disjoint
from floorcall.provenance import git_head

# The frozen test set each decision is evaluated on. D4 moved to v2 with its guideline v2
# (DECISIONS.md D-033); its v1 file stays frozen beside it, never edited, and is no longer evaluated.
TEST_VERSIONS = {"turn_complete": "v1", "barge_in": "v1", "route": "v1", "escalate": "v2"}
# D4's hand-label file per guideline version. For D4 the test set version is the guideline version.
D4_LABEL_FILES = {"v1": "escalate.labels.jsonl", "v2": "escalate.labels.v2.jsonl"}


def test_file(decision: str, version: str | None = None) -> str:
    return f"{decision}.test.{version or TEST_VERSIONS[decision]}.jsonl.gz"


def processed_file(decision: str, split: str) -> str:
    return f"{decision}.{split}.jsonl.gz"


def balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        by_split[split] = {
            "rows": len(part),
            "labels": dict(sorted(Counter(r["label"] for r in part).items())),
            "hard": sum(bool(r["hard"]) for r in part),
            "kinds": dict(Counter(r["kind"] for r in part).most_common()),
        }
    return by_split


def _emit(
    settings: Settings,
    decision: str,
    rows: list[dict[str, Any]],
    source: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["id"])
    assert_disjoint(rows, group_key="group")
    processed = settings.paths.data_processed
    for split in ("train", "calib"):
        write_jsonl_gz(
            processed / processed_file(decision, split), [r for r in rows if r["split"] == split]
        )
    src = SOURCES[source]
    digest, new = freeze(
        settings.paths.test_frozen,
        test_file(decision),
        [r for r in rows if r["split"] == "test"],
        {
            "decision": decision,
            "source_url": src.url,
            "source_sha256": src.sha256,
            "licence": src.licence,
            "built_at": git_head(),
            "labels": balance(rows)["test"]["labels"],
        },
    )
    card: dict[str, Any] = {
        "decision": decision,
        "source": source,
        "licence": src.licence,
        "test_file": test_file(decision),
        "test_sha256": digest,
        "balance": balance(rows),
        **extra,
    }
    cards = processed / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    with (cards / f"{decision}.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
        f.write("\n")
    card["test_newly_frozen"] = new
    return card


def build_swda(settings: Settings) -> list[dict[str, Any]]:
    fetch("swda", settings.paths.data_raw)
    root = settings.paths.data_raw / "swda" / "swda"
    d = settings.data
    fractions = settings.splits.model_dump(include={"train", "calib", "test"})
    rng = random.Random(settings.splits.seed)
    d1: list[swda.Example] = []
    d2: list[swda.Example] = []
    for conv in swda.iter_conversations(root, minimal_max_words=d.minimal_response_max_words):
        d1 += swda.build_d1(conv, rng=rng, max_history=d.max_history_turns)
        d2 += swda.build_d2(conv, partial_words=d.d2_partial_words, max_history=d.max_history_turns)
    swda.assign(d1, seed=settings.splits.seed, fractions=fractions)
    swda.assign(d2, seed=settings.splits.seed, fractions=fractions)
    turn_final = swda.mark_hard_d1(d1, min_count=d.d1_hard_min_count)
    shared = swda.mark_hard_d2(d2, min_count=d.d2_hard_min_count, min_share=d.d2_hard_min_share)
    return [
        _emit(
            settings,
            "turn_complete",
            [e.to_json() for e in d1],
            "swda",
            {
                "hard_subset": {
                    "definition": "truncations ending in a turn-final word (train r >= 1)",
                    "turn_final_words": dict(sorted(turn_final.items())),
                }
            },
        ),
        _emit(
            settings,
            "barge_in",
            [e.to_json() for e in d2],
            "swda",
            {
                "hard_subset": {
                    "definition": "backchannel forms shared with interruptions (train)",
                    "forms": {
                        f: {"backchannels": b, "interruptions": i}
                        for f, (b, i) in sorted(shared.items())
                    },
                }
            },
        ),
    ]


def build_clinc(settings: Settings) -> list[dict[str, Any]]:
    raw = fetch("clinc", settings.paths.data_raw)
    domains = fetch("clinc_domains", settings.paths.data_raw)
    rows, report = clinc.build_d3(raw, domains)
    return [
        _emit(
            settings,
            "route",
            rows,
            "clinc",
            {"dropped_duplicates_of_train": report.dropped_duplicates},
        )
    ]


def escalate_eval_rows(
    settings: Settings, guidelines: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """D4's (test, calib) rows under one guideline version, from the committed label files."""
    from floorcall.data import labelling

    labels_dir = settings.paths.labels
    cands = labelling.load_candidates(labels_dir / "escalate.candidates.v1.jsonl")
    labels = labelling.load_labels(labels_dir / D4_LABEL_FILES[guidelines])
    return (
        labelling.to_rows(cands, labels, "test", guidelines=guidelines),
        labelling.to_rows(cands, labels, "calib", guidelines=guidelines),
    )


def freeze_escalate(settings: Settings, guidelines: str | None = None) -> dict[str, Any]:
    """D4: Yash's labels to a frozen test set and a calib file. Refuses below the minimum.

    Builds the current version (TEST_VERSIONS) by default. An older version can be rebuilt to
    check it still matches its frozen bytes; only the current version writes the calib file and
    the card.
    """
    from floorcall.data import labelling

    current = TEST_VERSIONS["escalate"]
    version = guidelines or current
    test, calib = escalate_eval_rows(settings, version)
    need = settings.data.d4_min_test_labels
    if len(test) < need:
        raise ValueError(f"D4 test has {len(test)} true/false labels; at least {need} are required")
    rows = sorted(test + calib, key=lambda r: r["id"])
    assert_disjoint(rows, group_key="group")
    src = SOURCES["twcs"]
    meta: dict[str, Any] = {
        "decision": "escalate",
        "source_url": src.url,
        "source_sha256": src.sha256,
        "licence": src.licence,
        "built_at": git_head(),
        "labels": balance(rows)["test"]["labels"],
        "labeller": "yash",
        "guidelines": f"docs/labelling-escalate.md {version}",
    }
    if version == "v2":
        sample = json.loads(
            (settings.paths.labels / labelling.RELABEL_SAMPLE_FILE).read_text(encoding="utf-8")
        )
        meta["sample"] = f"data/labels/{labelling.RELABEL_SAMPLE_FILE}, seed {sample['seed']}"
    digest, new = freeze(
        settings.paths.test_frozen,
        test_file("escalate", version),
        sorted(test, key=lambda r: r["id"]),
        meta,
    )
    card: dict[str, Any] = {
        "decision": "escalate",
        "source": "twcs",
        "licence": src.licence,
        "guidelines": version,
        "test_file": test_file("escalate", version),
        "test_sha256": digest,
        "balance": balance(rows),
        "sampling": (
            "four equal strata (escalation-word cue x agent context); "
            "the positive rate is not the natural rate"
        ),
    }
    if version == current:
        processed = settings.paths.data_processed
        calib_path = processed / processed_file("escalate", "calib")
        write_jsonl_gz(calib_path, calib)
        card["calib_file"] = calib_path.name
        card["calib_sha256"] = sha256_file(calib_path)
        if "sample" in meta:
            card["sampling"] += (
                f"; v2 is a seeded sample of the v1 eval sets, stratified by the same strata "
                f"({meta['sample']}), relabelled blind under guideline v2"
            )
        cards = processed / "cards"
        cards.mkdir(parents=True, exist_ok=True)
        with (cards / "escalate.json").open("w", encoding="utf-8", newline="\n") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
            f.write("\n")
    card["test_newly_frozen"] = new
    return card


def build_all(settings: Settings) -> list[dict[str, Any]]:
    return build_swda(settings) + build_clinc(settings)


def summarize(cards: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for c in cards:
        lines.append(
            f"{c['decision']}  (test {c['test_file']}, sha256 {c['test_sha256'][:12]}..., newly frozen: {c.get('test_newly_frozen')})"
        )
        for split, b in c["balance"].items():
            lines.append(f"  {split:5s} {b['rows']:6d} rows  hard {b['hard']:5d}  {b['labels']}")
    return "\n".join(lines)


def cards_dir(settings: Settings) -> Path:
    return settings.paths.data_processed / "cards"
