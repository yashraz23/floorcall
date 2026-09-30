"""Paired bootstrap of the difference between two models on the same test rows (DECISIONS.md D-036).

Both models' saved per-row predictions (runs/eval) are resampled with the *same* row indices, so
each resample compares them on identical rows. That is what the unpaired intervals in Table A
cannot do: two overlapping intervals can still hide a consistent difference. For each metric:
the point difference (a - b), its 95% percentile interval, and the share of resamples in which a
is not better than b. Nothing is re-scored; every model is checked first against its committed
Table A row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall.config import REPO_ROOT, Settings
from floorcall.data.build import test_file
from floorcall.data.freeze import sha256_file
from floorcall.evaluate.dataset import option_labels
from floorcall.evaluate.metrics import accuracy, brier, ece, macro_f1
from floorcall.evaluate.scoring import score, softmax

PREDICTIONS = REPO_ROOT / "runs" / "eval"
TABLE_A = REPO_ROOT / "results" / "table_a"
RESULTS = REPO_ROOT / "results" / "paired"
# Lower is better for these; the difference is still reported as a - b.
LOWER_IS_BETTER = {"ece", "brier"}


@dataclass(frozen=True)
class Scored:
    """One model's test predictions, as a Table A row scores them."""

    name: str  # the Table A model id it reproduces
    probs: npt.NDArray[np.float64]
    pred: npt.NDArray[np.int64]
    ids: npt.NDArray[Any]
    y: npt.NDArray[np.int64]
    hard: npt.NDArray[np.bool_]


def load_scored(settings: Settings, decision: str, npz: str, model: str) -> Scored:
    """Rebuild a committed Table A row from saved logits, and refuse if it does not reproduce."""
    z = np.load(PREDICTIONS / f"{npz}.npz")
    row = json.loads((TABLE_A / f"{decision}.{model}.json").read_text(encoding="utf-8"))
    if row["test_sha256"] != sha256_file(settings.paths.test_frozen / test_file(decision)):
        raise ValueError(f"{decision}.{model}: its test file has changed since it was scored")
    labels = option_labels(decision)
    theta = row["metrics"].get("threshold")
    probs = softmax(np.asarray(z["logits"], dtype=np.float64), row["temperature"])
    m = score(probs, z["y"], z["hard"], labels, n_bins=settings.eval.ece_bins, threshold=theta)
    for key in ("accuracy", "macro_f1", "ece", "brier", "confusion"):
        if m[key] != row["metrics"][key]:
            raise ValueError(f"{decision}.{model}: saved predictions do not reproduce its {key}")
    pred = (
        probs.argmax(1) if theta is None else (probs[:, labels.index("true")] >= theta).astype(int)
    )
    return Scored(model, probs, pred.astype(np.int64), z["ids"], z["y"], z["hard"])


def _metrics(s: Scored, idx: npt.NDArray[np.int64], k: int, n_bins: int) -> dict[str, float]:
    y, pred, probs = s.y[idx], s.pred[idx], s.probs[idx]
    argmax = probs.argmax(1)
    return {
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred, labels=list(range(k))),
        # ECE and Brier describe the probabilities, as in Table A: argmax answer and confidence
        "ece": ece(probs.max(1), (argmax == y).astype(int), n_bins=n_bins),
        "brier": brier(probs, y),
    }


def paired_differences(
    a: Scored, b: Scored, *, k: int, n_bins: int, samples: int, seed: int
) -> dict[str, dict[str, Any]]:
    if not (np.array_equal(a.ids, b.ids) and np.array_equal(a.y, b.y)):
        raise ValueError(f"{a.name} and {b.name} are not scored on the same rows in the same order")
    n = len(a.y)
    everything = np.arange(n)
    point_a, point_b = _metrics(a, everything, k, n_bins), _metrics(b, everything, k, n_bins)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {m: [] for m in point_a}
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        ma, mb = _metrics(a, idx, k, n_bins), _metrics(b, idx, k, n_bins)
        for m in diffs:
            diffs[m].append(ma[m] - mb[m])
    out = {}
    for m, values in diffs.items():
        v = np.asarray(values)
        lo, hi = np.quantile(v, [0.025, 0.975])
        not_better = v >= 0 if m in LOWER_IS_BETTER else v <= 0
        out[m] = {
            "a": point_a[m],
            "b": point_b[m],
            "difference": point_a[m] - point_b[m],
            "ci95": [float(lo), float(hi)],
            "share_a_not_better": float(not_better.mean()),
        }
    return out


# (decision, a: (npz, Table A model), b: (npz, Table A model))
COMPARISONS = (
    ("escalate", ("escalate.finetuned", "finetuned_temp_threshold"), ("escalate.stock_laya", "stock_laya")),
    ("escalate", ("escalate.finetuned", "finetuned_temp_threshold"), ("escalate.stock_laya", "stock_laya_threshold")),
    ("turn_complete", ("turn_complete.finetuned", "finetuned_temp"), ("turn_complete.stock_laya", "stock_laya")),
    ("barge_in", ("barge_in.finetuned", "finetuned_temp"), ("barge_in.stock_laya", "stock_laya")),
    ("route", ("route.finetuned", "finetuned_temp"), ("route.stock_laya", "stock_laya")),
)  # fmt: skip


# Fine-tuned against the best cheap baseline of each decision (D-036): the lexical rule for D2, where
# it beats TF-IDF + LR; TF-IDF + LR elsewhere, D4's at its own calib threshold.
BASELINE_COMPARISONS = (
    ("escalate", ("escalate.finetuned", "finetuned_temp_threshold"), ("escalate.tfidf_lr", "tfidf_lr")),
    ("turn_complete", ("turn_complete.finetuned", "finetuned_temp"), ("turn_complete.tfidf_lr", "tfidf_lr")),
    ("barge_in", ("barge_in.finetuned", "finetuned_temp"), ("barge_in.lexical_rule", "lexical_rule")),
    ("route", ("route.finetuned", "finetuned_temp"), ("route.tfidf_lr", "tfidf_lr")),
)  # fmt: skip
SETS = {"stock": COMPARISONS, "baselines": BASELINE_COMPARISONS}


def run_paired(settings: Settings, against: str = "stock") -> list[dict[str, Any]]:
    from floorcall.provenance import git_head

    code = git_head()
    out = []
    for decision, (npz_a, model_a), (npz_b, model_b) in SETS[against]:
        a = load_scored(settings, decision, npz_a, model_a)
        b = load_scored(settings, decision, npz_b, model_b)
        payload = {
            "decision": decision,
            "test_file": test_file(decision),
            "test_sha256": sha256_file(settings.paths.test_frozen / test_file(decision)),
            "a": model_a,
            "b": model_b,
            "n": len(a.y),
            "samples": settings.eval.bootstrap_samples,
            "seed": settings.eval.bootstrap_seed,
            "method": "paired percentile bootstrap: both models on the same resampled rows",
            "code": code,
            "differences": paired_differences(
                a,
                b,
                k=len(option_labels(decision)),
                n_bins=settings.eval.ece_bins,
                samples=settings.eval.bootstrap_samples,
                seed=settings.eval.bootstrap_seed,
            ),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{decision}.{model_a}_vs_{model_b}.json"
        with Path(path).open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        out.append(payload)
    return out
