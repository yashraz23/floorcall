"""D4's real Twitter text is private (DECISIONS.md D-049). Only ids, labels and hashes are public.

    uv run floorcall data restore-escalate     # rebuild the four files from twcs.csv, verified

The public repository holds no customer message text: Customer Support on Twitter is real people's
writing, and some of it carries their phone numbers, names and reference numbers. Everything else
D4 needs is public: the hand and LLM labels by id, the relabel sample, and the sha256 of every
private file. The four files rebuild byte for byte from the corpus (twcs.csv, pinned by its content
hash) with the project's seeds, so anyone who downloads the corpus can restore them and check them
against those hashes. The private copies live in the floorcall-archive repository and the private
Kaggle dataset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from floorcall.config import Settings
from floorcall.data.freeze import _read_manifest, jsonl_gz_bytes, sha256_file

CANDIDATES = "escalate.candidates.v1.jsonl"
TRAIN_SAMPLE = "escalate.train_sample.v2.jsonl"
TEST_FILES = ("escalate.test.v1.jsonl.gz", "escalate.test.v2.jsonl.gz")
# The private label files' hashes (the frozen test files' are in MANIFEST.sha256).
LABELS_MANIFEST = "PRIVATE.sha256"


class PrivateDataMissingError(FileNotFoundError):
    """A D4 file with real Twitter text is not here: restore it from the corpus."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} holds real Twitter text and is not in the public repository "
            "(DECISIONS.md D-049). Rebuild it from the corpus: `uv run floorcall data "
            "restore-escalate`."
        )


def private_paths(settings: Settings) -> list[Path]:
    labels, frozen = settings.paths.labels, settings.paths.test_frozen
    return [labels / CANDIDATES, labels / TRAIN_SAMPLE, *(frozen / t for t in TEST_FILES)]


def is_private(path: Path, settings: Settings) -> bool:
    return path.resolve() in {p.resolve() for p in private_paths(settings)}


def require(path: Path, settings: Settings) -> Path:
    """`path`, or a clear error saying how to restore it if it is a missing private file."""
    if not path.exists() and is_private(path, settings):
        raise PrivateDataMissingError(path)
    return path


def missing(settings: Settings) -> list[Path]:
    return [p for p in private_paths(settings) if not p.exists()]


def expected_hashes(settings: Settings) -> dict[Path, str]:
    """sha256 of every private file, from the committed manifests."""
    labels = settings.paths.labels
    out: dict[Path, str] = {}
    for line in (labels / LABELS_MANIFEST).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[labels / name.strip()] = digest
    frozen = _read_manifest(settings.paths.test_frozen)
    for name in TEST_FILES:
        out[settings.paths.test_frozen / name] = frozen[name]
    return out


def _put(path: Path, data: bytes, digest: str) -> str:
    """Write `data` to `path` if it matches `digest`. Never overwrites a file that differs."""
    got = hashlib.sha256(data).hexdigest()
    if got != digest:
        raise ValueError(f"{path.name}: the rebuild gives {got}, the manifest says {digest}")
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError(f"{path} exists and differs from the manifest; it is left untouched")
        return "already present, verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "restored, verified"


def restore(settings: Settings) -> dict[str, str]:
    """Rebuild every private D4 file from twcs.csv and the committed id/label files.

    The same functions that first drew the pool, the training sample and the frozen test sets
    rebuild them, so a restored file is byte for byte the original, and each is checked against its
    committed sha256 before it is written.
    """
    from floorcall.data import escalate, labelling
    from floorcall.data.build import TEST_VERSIONS, escalate_eval_rows
    from floorcall.data.d4_llm import train_pool
    from floorcall.data.download import fetch

    want = expected_hashes(settings)
    labels = settings.paths.labels
    status: dict[str, str] = {}

    pool = escalate.build_candidates(
        fetch("twcs", settings.paths.data_raw),  # refuses a corpus whose content hash differs
        seed=settings.splits.seed,
        fractions=settings.splits.model_dump(include={"train", "calib", "test"}),
        max_history=settings.data.max_history_turns,
    )
    chosen = escalate.select_for_labelling(
        pool, split="test", n=settings.data.d4_candidates_test, seed=settings.splits.seed
    ) + escalate.select_for_labelling(
        pool, split="calib", n=settings.data.d4_candidates_calib, seed=settings.splits.seed
    )
    status[CANDIDATES] = _put(
        labels / CANDIDATES, labelling.candidates_bytes(chosen), want[labels / CANDIDATES]
    )

    held_out = labelling.load_candidates(labels / CANDIDATES)
    train, _ = train_pool(pool, held_out, n=settings.llm.d4_train_rows, seed=settings.splits.seed)
    test, calib = escalate_eval_rows(settings, TEST_VERSIONS["escalate"])
    eval_groups = {r["group"] for r in test + calib} | {c.group for c in held_out}
    sample = labelling.draw_train_sample(
        train,
        eval_groups,
        n=settings.data.d4_train_hand_rows,
        seed=settings.data.d4_train_hand_seed,
    )
    status[TRAIN_SAMPLE] = _put(
        labels / TRAIN_SAMPLE, labelling.candidates_bytes(sample), want[labels / TRAIN_SAMPLE]
    )

    for name in TEST_FILES:
        version = name.split(".")[2]  # escalate.test.<version>.jsonl.gz
        rows, _ = escalate_eval_rows(settings, version)
        path = settings.paths.test_frozen / name
        status[name] = _put(path, jsonl_gz_bytes(sorted(rows, key=lambda r: r["id"])), want[path])
    return status
