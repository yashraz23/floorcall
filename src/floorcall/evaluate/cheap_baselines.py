"""Sanity baselines added after the main test pass (DECISIONS.md D-036): nothing is tuned on test.

- **TF-IDF + logistic regression**, all four decisions. TF-IDF over the user's word 1- and
  2-grams and the context's words (the agent's last utterance and the recent turns), all through
  the same ASR-style normalizer as everything else. It is trained on the same rows the fine-tuned
  model trained on: each task's train rows minus the same dev split (`carve_dev`, same fraction
  and seed). C is chosen on that dev split by cross-entropy, the criterion that chose the
  fine-tuned checkpoint. D4 uses balanced class weights, the counterpart of D-035's class-balanced
  sampling, and its threshold is chosen on calib by the D-035 rule.
- **A lexical rule for D2**, fixed before it was scored: no words or fillers only -> noise; up to
  three words, all from a fixed list of backchannel words -> backchannel; anything else ->
  interruption. Its probabilities are the train class frequencies of rows taking the same branch
  (add-one smoothed), so its ECE and Brier score mean something.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from floorcall.config import REPO_ROOT, Settings
from floorcall.data.build import test_file
from floorcall.data.freeze import sha256_file
from floorcall.evaluate.dataset import EvalSet, _make, load_processed, load_test
from floorcall.evaluate.scoring import score
from floorcall.evaluate.thresholds import choose_threshold
from floorcall.normalize import normalize
from floorcall.train.data import carve_dev

TABLE_A = REPO_ROOT / "results" / "table_a"
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FILLERS = frozenset({"um", "uh", "er", "ah"})
BACKCHANNEL_WORDS = frozenset(
    {"uh", "huh", "mhm", "hm", "hmm", "mm", "yeah", "yep", "yes", "right", "okay", "ok",
     "oh", "sure", "really", "wow", "i", "see", "true", "exactly"}
)  # fmt: skip
MAX_BACKCHANNEL_WORDS = 3


def _user(row: dict[str, Any]) -> str:
    return normalize(row["snapshot"]["user_partial"])


def _context(row: dict[str, Any]) -> str:
    s = row["snapshot"]
    turns = " ".join(t["text"] for t in s.get("recent_turns", []))
    return normalize(f"{turns} {s.get('agent_last_utterance', '')}")


def features(row: dict[str, Any]) -> list[str]:
    """The user's word 1- and 2-grams, and the context's words tagged "ctx:"."""
    u = _user(row).split()
    return u + [f"{a} {b}" for a, b in pairwise(u)] + [f"ctx:{w}" for w in _context(row).split()]


class TfidfLR:
    def __init__(self, c: float, balanced: bool) -> None:
        self.vec = TfidfVectorizer(analyzer=features, min_df=2, sublinear_tf=True)
        self.lr = LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced" if balanced else None
        )

    def fit(self, data: EvalSet) -> TfidfLR:
        self.lr.fit(self.vec.fit_transform(data.rows), data.y)
        return self

    def proba(self, data: EvalSet, k: int) -> npt.NDArray[np.float64]:
        p = self.lr.predict_proba(self.vec.transform(data.rows))
        out = np.zeros((len(data.rows), k))
        out[:, self.lr.classes_] = p  # a class absent from train keeps probability 0
        return out


def _ce(probs: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _split(settings: Settings, decision: str) -> tuple[EvalSet, EvalSet]:
    rows = load_processed(settings, decision, "train").rows
    train, dev = carve_dev(
        rows, fraction=settings.train.dev_fraction, seed=settings.train.seed, task=decision
    )
    return _make(decision, "train", train), _make(decision, "dev", dev)


def _payload(
    settings: Settings, test: EvalSet, model: str, code: str, **extra: Any
) -> dict[str, Any]:
    return {
        "decision": test.decision,
        "test_file": test_file(test.decision),
        "test_sha256": sha256_file(settings.paths.test_frozen / test_file(test.decision)),
        "code": code,
        "labels": list(test.labels),
        "model": model,
        "added": "after the main test pass, as a sanity baseline; nothing tuned on test (D-036)",
        **extra,
    }


def _write(decision: str, model: str, payload: dict[str, Any]) -> None:
    TABLE_A.mkdir(parents=True, exist_ok=True)
    with (TABLE_A / f"{decision}.{model}.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run_tfidf_lr(settings: Settings, decision: str, code: str) -> dict[str, Any]:
    from floorcall.evaluate.checkpoint import save_predictions

    train, dev = _split(settings, decision)
    k = len(train.labels)
    balanced = decision in settings.train.balance_classes
    dev_ce = {c: _ce(TfidfLR(c, balanced).fit(train).proba(dev, k), dev.y) for c in C_GRID}
    best_c = min(C_GRID, key=lambda c: (dev_ce[c], c))
    model = TfidfLR(best_c, balanced).fit(train)
    choice = None
    if decision in settings.eval.threshold_decisions:  # the D-035 rule, on calib
        calib = load_processed(settings, decision, "calib")
        p = model.proba(calib, k)[:, calib.labels.index("true")]
        choice = choose_threshold(p, calib.y).to_json()
    test = load_test(settings, decision)
    probs = model.proba(test, k)
    save_predictions(f"{decision}.tfidf_lr", np.log(np.clip(probs, 1e-12, 1.0)), test)
    payload = _payload(
        settings,
        test,
        "tfidf_lr",
        code,
        train_rows=len(train.rows),
        dev_rows=len(dev.rows),
        c_grid=list(C_GRID),
        dev_cross_entropy={str(c): v for c, v in dev_ce.items()},
        chosen_c=best_c,
        class_weight="balanced" if balanced else None,
        temperature=1.0,
        **({"threshold_choice": choice} if choice else {}),
        metrics=score(
            probs,
            test.y,
            test.hard,
            test.labels,
            n_bins=settings.eval.ece_bins,
            bootstrap=(settings.eval.bootstrap_samples, settings.eval.bootstrap_seed),
            threshold=None if choice is None else choice["theta"],
        ),
    )
    _write(decision, "tfidf_lr", payload)
    return payload


def lexical_branch(text: str) -> str:
    words = text.split()
    if not words or all(w in FILLERS for w in words):
        return "empty_or_filler"
    if len(words) <= MAX_BACKCHANNEL_WORDS and all(w in BACKCHANNEL_WORDS for w in words):
        return "backchannel_words"
    return "other"


def run_lexical_rule(settings: Settings, code: str) -> dict[str, Any]:
    from floorcall.evaluate.checkpoint import save_predictions

    decision = "barge_in"
    train, _ = _split(settings, decision)
    k = len(train.labels)
    counts: dict[str, npt.NDArray[np.float64]] = {}
    for row, y in zip(train.rows, train.y, strict=True):
        counts.setdefault(lexical_branch(_user(row)), np.ones(k))[y] += 1  # add-one
    branch_probs = {b: c / c.sum() for b, c in counts.items()}
    test = load_test(settings, decision)
    probs = np.stack([branch_probs[lexical_branch(_user(r))] for r in test.rows])
    save_predictions(f"{decision}.lexical_rule", np.log(probs), test)
    payload = _payload(
        settings,
        test,
        "lexical_rule",
        code,
        rule={
            "fillers": sorted(FILLERS),
            "backchannel_words": sorted(BACKCHANNEL_WORDS),
            "max_backchannel_words": MAX_BACKCHANNEL_WORDS,
        },
        branch_probabilities={
            b: dict(zip(train.labels, map(float, p), strict=True)) for b, p in branch_probs.items()
        },
        temperature=1.0,
        metrics=score(
            probs,
            test.y,
            test.hard,
            test.labels,
            n_bins=settings.eval.ece_bins,
            bootstrap=(settings.eval.bootstrap_samples, settings.eval.bootstrap_seed),
        ),
    )
    _write(decision, "lexical_rule", payload)
    return payload
