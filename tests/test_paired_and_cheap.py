"""The paired bootstrap and the cheap sanity baselines (D-036), on synthetic data."""

from typing import Any

import numpy as np
import pytest

from floorcall.evaluate.cheap_baselines import TfidfLR, features, lexical_branch
from floorcall.evaluate.dataset import _make
from floorcall.evaluate.paired import Scored, paired_differences


def scored(name: str, probs: np.ndarray, y: np.ndarray, ids: np.ndarray | None = None) -> Scored:
    n = len(y)
    return Scored(
        name,
        probs,
        probs.argmax(1),
        np.arange(n) if ids is None else ids,
        y,
        np.zeros(n, bool),
    )


Y = np.array([0, 1] * 50)


def test_a_model_against_itself_differs_by_nothing() -> None:
    p = np.stack([np.full(100, 0.4), np.full(100, 0.6)], 1)
    out = paired_differences(
        scored("a", p, Y), scored("b", p, Y), k=2, n_bins=15, samples=200, seed=0
    )
    for m in ("accuracy", "macro_f1", "ece", "brier"):
        assert out[m]["difference"] == 0 and out[m]["ci95"] == [0.0, 0.0]
        assert out[m]["share_a_not_better"] == 1.0  # a tie is "not better"


def test_a_uniformly_better_model_wins_every_resample() -> None:
    good = np.stack([np.where(Y == 0, 0.9, 0.1), np.where(Y == 1, 0.9, 0.1)], 1)
    bad = good[:, ::-1]  # always wrong
    out = paired_differences(
        scored("a", good, Y), scored("b", bad, Y), k=2, n_bins=15, samples=200, seed=0
    )
    assert out["accuracy"]["difference"] == 1.0 and out["accuracy"]["share_a_not_better"] == 0.0
    assert (
        out["brier"]["difference"] < 0 and out["brier"]["share_a_not_better"] == 0.0
    )  # lower is better


def test_the_pairing_is_checked() -> None:
    p = np.stack([np.full(100, 0.5), np.full(100, 0.5)], 1)
    with pytest.raises(ValueError, match="same rows"):
        paired_differences(
            scored("a", p, Y), scored("b", p, Y, ids=np.arange(100)[::-1]), k=2, n_bins=15,
            samples=10, seed=0,
        )  # fmt: skip


@pytest.mark.parametrize(
    ("text", "branch"),
    [
        ("", "empty_or_filler"),
        ("um uh", "empty_or_filler"),
        ("uh huh", "backchannel_words"),
        ("oh yeah really", "backchannel_words"),
        ("yeah yeah yeah yeah", "other"),  # longer than three words
        ("yeah but why", "other"),
        ("no", "other"),
    ],
)
def test_the_lexical_rule(text: str, branch: str) -> None:
    assert lexical_branch(text) == branch


def row(user: str, agent: str = "", label: str = "false") -> dict[str, Any]:
    return {
        "id": user,
        "group": user,
        "label": label,
        "hard": False,
        "event": "user_pause",
        "snapshot": {
            "agent_speaking": False,
            "user_partial": user,
            "agent_last_utterance": agent,
            "recent_turns": [],
        },
    }


def test_features_are_user_ngrams_and_tagged_context() -> None:
    f = features(row("Yeah, but WHY?", agent="Your card was declined."))
    assert f[:4] == ["yeah", "but", "why", "yeah but"]
    assert "ctx:declined" in f and "declined" not in f


def test_tfidf_lr_learns_a_separable_toy_task() -> None:
    rows = [row(f"please get me a person {i}", label="true") for i in range(20)] + [
        row(f"what is my balance {i}", label="false") for i in range(20)
    ]
    data = _make("escalate", "train", rows)
    p = TfidfLR(1.0, balanced=True).fit(data).proba(data, 2)
    assert (p.argmax(1) == data.y).all()
