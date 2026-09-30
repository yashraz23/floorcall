"""Evaluate a trained checkpoint on the frozen test sets: Table A's fine-tuned rows, Tables C and D.

    uv run floorcall eval checkpoint  --checkpoint checkpoints/<run>
    uv run floorcall eval robustness  --checkpoint checkpoints/<run>
    uv run floorcall eval ablation    --checkpoint checkpoints/<run-trained-with-an-ablation>

All three score the same way: states packed with the event's budget and the StateSettings the
checkpoint was trained with, raw logits from one forward pass, then either no temperature
("fine-tuned") or the checkpoint's per-decision temperatures fitted on calib ("+ temperature").
The checkpoint's run.json names its ablation; scoring it under any other StateSettings is refused,
so a Table D row cannot be computed with the wrong flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from floorcall import questions
from floorcall.config import ABLATIONS, REPO_ROOT, LayaSettings, Settings, StateSettings
from floorcall.data.build import processed_file, test_file
from floorcall.data.freeze import read_jsonl_gz, sha256_file
from floorcall.evaluate.calibration import Calibration, load_calibration
from floorcall.evaluate.dataset import EvalSet, load_test
from floorcall.evaluate.robustness import noisy_row, train_vocab
from floorcall.evaluate.scoring import score, softmax
from floorcall.provenance import git_head
from floorcall.train.data import event_budget

RESULTS = REPO_ROOT / "results"
DECISIONS = ("turn_complete", "barge_in", "route", "escalate")
NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2)
VOCAB_SIZE = 5000


def run_info(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"{checkpoint} has no run.json: not a floorcall training run")
    info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return info


ABLATION_FLAGS = ("normalize", "include_history", "include_agent")


def trained_ablation(checkpoint: Path) -> str:
    """The ablation a checkpoint was trained under, from the StateSettings in its run.json."""
    trained = StateSettings(**run_info(checkpoint)["settings"]["state"])
    flags = {k: getattr(trained, k) for k in ABLATION_FLAGS}
    for name, overrides in ABLATIONS.items():
        if flags == {**dict.fromkeys(ABLATION_FLAGS, True), **overrides}:
            return name
    raise ValueError(f"{checkpoint}: trained with {flags}, which is no named ablation")


def _load(checkpoint: Path, device: str) -> Any:
    from floorcall.model.laya_adapter import LayaDecider

    return LayaDecider(
        LayaSettings(checkpoint=str(checkpoint), revision=None, device=device, cuda_graphs=False)
    )


def _calibration(checkpoint: Path) -> Calibration:
    cal = load_calibration(checkpoint)
    if cal is None:
        raise FileNotFoundError(f"{checkpoint} is not calibrated: run `floorcall train calibrate`")
    return cal


def logits_for(decider: Any, test: EvalSet, state: StateSettings) -> npt.NDArray[np.float64]:
    budget = event_budget(decider, test.event, state)
    states = test.packed_states(state=state, budget=budget, count_tokens=decider.count_tokens)
    rows = decider.logits_batch(states, questions.questions_by_id(test.decision))
    return np.stack([r[test.decision].logits for r in rows])


def _frozen(settings: Settings) -> list[str]:
    return [d for d in DECISIONS if (settings.paths.test_frozen / test_file(d)).exists()]


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _base(settings: Settings, checkpoint: Path, test: EvalSet, code: str) -> dict[str, Any]:
    return {
        "decision": test.decision,
        "test_file": test_file(test.decision),
        "test_sha256": sha256_file(settings.paths.test_frozen / test_file(test.decision)),
        "checkpoint": str(checkpoint),
        "trained_at": run_info(checkpoint).get("code"),
        "code": code,
        "labels": list(test.labels),
    }


PREDICTIONS = REPO_ROOT / "runs" / "eval"


def save_predictions(
    name: str, logits: npt.NDArray[np.floating], test: EvalSet, **extra: Any
) -> Path:
    """Per-row logits of one scoring pass, so any later comparison reuses them (D-036): never a
    second pass over a test set. `runs/` is gitignored; the rows are rebuilt by rerunning."""
    path = PREDICTIONS / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "logits": np.asarray(logits, dtype=np.float64),
        "y": test.y,
        "hard": test.hard,
        "ids": np.array([r["id"] for r in test.rows]),
        **{k: np.asarray(v) for k, v in extra.items()},
    }
    np.savez_compressed(path, **arrays)
    return path


def table_a_rows(cal: Calibration, decision: str) -> list[tuple[str, float, dict[str, Any] | None]]:
    """(model id, temperature, threshold choice) for each fine-tuned Table A row of a decision."""
    t_cal = cal.temperatures[decision]
    rows: list[tuple[str, float, dict[str, Any] | None]] = [
        ("finetuned", 1.0, None),
        ("finetuned_temp", t_cal, None),
    ]
    if decision in cal.thresholds:  # chosen on calib after temperature (D-035)
        rows.append(("finetuned_temp_threshold", t_cal, cal.thresholds[decision]))
    return rows


def score_row(
    settings: Settings,
    logits: npt.NDArray[np.floating],
    test: EvalSet,
    temperature: float,
    choice: dict[str, Any] | None,
) -> dict[str, Any]:
    return score(
        softmax(np.asarray(logits, dtype=np.float64), temperature),
        test.y,
        test.hard,
        test.labels,
        n_bins=settings.eval.ece_bins,
        bootstrap=(settings.eval.bootstrap_samples, settings.eval.bootstrap_seed),
        threshold=None if choice is None else choice["theta"],
    )


def evaluate_table_a(
    settings: Settings, checkpoint: Path, *, device: str = "cuda"
) -> list[dict[str, Any]]:
    code = git_head()
    if trained_ablation(checkpoint) != "full":
        raise ValueError("Table A rows come from the full model; this checkpoint is an ablation")
    cal = _calibration(checkpoint)
    decider = _load(checkpoint, device)
    out = []
    for decision in _frozen(settings):
        test = load_test(settings, decision)
        z = logits_for(decider, test, settings.state)
        save_predictions(f"{decision}.finetuned", z, test, temperature=cal.temperatures[decision])
        for model, t, choice in table_a_rows(cal, decision):
            payload = {
                **_base(settings, checkpoint, test, code),
                "model": model,
                "precision": decider.precision,
                "temperature": t,
                "metrics": score_row(settings, z, test, t, choice),
            }
            if choice is not None:
                payload["threshold_choice"] = choice
            _write(RESULTS / "table_a" / f"{decision}.{model}.json", payload)
            out.append(payload)
    return out


def recover_predictions(
    settings: Settings, checkpoint: Path, *, device: str = "cuda"
) -> dict[str, list[str]]:
    """Recompute a checkpoint's test logits and check them against its committed Table A rows.

    Inference only (D-036): it writes the per-row logits to runs/eval and nothing to results/.
    Every committed metric of every fine-tuned row, bootstrap intervals and confusion included,
    must come out identical from the recovered logits. Returns the differences per decision,
    which must all be empty before the logits are used.
    """
    cal = _calibration(checkpoint)
    decider = _load(checkpoint, device)
    diffs: dict[str, list[str]] = {}
    for decision in _frozen(settings):
        test = load_test(settings, decision)
        z = logits_for(decider, test, settings.state)
        found: list[str] = []
        for model, t, choice in table_a_rows(cal, decision):
            committed = json.loads(
                (RESULTS / "table_a" / f"{decision}.{model}.json").read_text(encoding="utf-8")
            )
            if committed["test_sha256"] != sha256_file(
                settings.paths.test_frozen / test_file(decision)
            ):
                found.append(f"{model}: the test file changed")
            if committed["checkpoint"] != str(checkpoint) or committed["temperature"] != t:
                found.append(f"{model}: checkpoint or temperature differs")
            again = score_row(settings, z, test, t, choice)
            for key in sorted(set(again) | set(committed["metrics"])):
                if again.get(key) != committed["metrics"].get(key):
                    found.append(
                        f"{model}.{key}: committed {committed['metrics'].get(key)!r}, "
                        f"recovered {again.get(key)!r}"
                    )
        diffs[decision] = found
        if not found:
            save_predictions(
                f"{decision}.finetuned", z, test, temperature=cal.temperatures[decision]
            )
    return diffs


def evaluate_table_c(
    settings: Settings, checkpoint: Path, *, device: str = "cuda"
) -> list[dict[str, Any]]:
    code = git_head()
    cal = _calibration(checkpoint)
    decider = _load(checkpoint, device)
    texts = []
    for d in DECISIONS:
        path = settings.paths.data_processed / processed_file(d, "train")
        if path.exists():
            texts += [r["snapshot"]["user_partial"] for r in read_jsonl_gz(path)]
    vocab = train_vocab(texts, VOCAB_SIZE)
    rng_seed = settings.splits.seed
    out = []
    for decision in _frozen(settings):
        clean = load_test(settings, decision)
        for level in NOISE_LEVELS:
            test = load_test(settings, decision)
            test.rows = [noisy_row(r, level, seed=rng_seed, vocab=vocab) for r in clean.rows]
            z = logits_for(decider, test, settings.state)
            t = cal.temperatures[decision]
            save_predictions(f"table_c/{decision}.{level:.2f}", z, test, temperature=t)
            payload = {
                **_base(settings, checkpoint, test, code),
                "model": "finetuned_temp",
                "precision": decider.precision,
                "noise_level": level,
                "temperature": t,
                "metrics": score(
                    softmax(z, t), test.y, test.hard, test.labels, n_bins=settings.eval.ece_bins
                ),
            }
            _write(RESULTS / "table_c" / f"{decision}.{level:.2f}.json", payload)
            out.append(payload)
    return out


def evaluate_table_d(
    settings: Settings, checkpoint: Path, *, device: str = "cuda"
) -> list[dict[str, Any]]:
    """Score an ablation checkpoint under its own StateSettings.

    The normalization ablation is scored twice: on written text (what it trained on; the leak
    makes it look good) and on ASR-style normalized text (what a live pipeline gives it).
    """
    code = git_head()
    name = trained_ablation(checkpoint)
    cal = _calibration(checkpoint)
    decider = _load(checkpoint, device)
    trained_state = settings.state.model_copy(update=ABLATIONS[name])
    variants = {name: trained_state}
    if name == "no_normalize":
        variants = {
            "no_normalize.written": trained_state,
            "no_normalize.asr": trained_state.model_copy(update={"normalize": True}),
        }
    out = []
    for decision in _frozen(settings):
        test = load_test(settings, decision)
        for variant, state in variants.items():
            z = logits_for(decider, test, state)
            t = cal.temperatures[decision]
            save_predictions(f"table_d/{decision}.{variant}", z, test, temperature=t)
            payload = {
                **_base(settings, checkpoint, test, code),
                "model": "finetuned_temp",
                "precision": decider.precision,
                "ablation": variant,
                "state": state.model_dump(),
                "temperature": t,
                "metrics": score(
                    softmax(z, t), test.y, test.hard, test.labels, n_bins=settings.eval.ece_bins
                ),
            }
            _write(RESULTS / "table_d" / f"{variant}.{decision}.json", payload)
            out.append(payload)
    return out
