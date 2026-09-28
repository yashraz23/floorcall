"""The only module in floorcall that imports laya.

Laya's API is young and changes between patch releases, so everything floorcall needs from it
goes through here, and a version bump touches one file. Written against laya 0.3.21's source
(the version pinned in pyproject.toml, asserted by tests/test_environment.py). A few calls reach
into underscore-prefixed internals: `Agent._check_question`, `Agent._to_internal`,
`Agent._encode_state`, `Agent._forward`. They are the only way to get the pre-temperature logits
and the exact per-question token room, and the version pin is what makes relying on them safe.
Each use is marked `# laya-internal`.

What goes in and out of this module is plain Python and numpy. No laya object escapes it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import laya
import numpy as np
import torch
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    encode_text,
    proper_reward,
    render_options,
    serialize_state,
    temp_bucket,
)

from floorcall.config import LayaSettings

QuestionType = Literal["choice", "score", "noul"]
Questions = Mapping[str, Mapping[str, Any]]
State = Mapping[str, Any]


class StateOverflowError(ValueError):
    """A state that does not fit the room its questions leave. Laya would truncate it silently.

    Raised instead, because a truncated state is a packer bug (CLAUDE.md §5; DECISIONS.md D-004).
    """


class QuestionBudgetError(ValueError):
    """A question whose options do not survive the head budget intact."""


@dataclass(frozen=True)
class Answer:
    qid: str
    qtype: QuestionType
    # Option label -> probability, in the question's option order. A noul answer is reported
    # as {"false": 1 - p, "true": p}, so every question type has the same shape.
    probabilities: dict[str, float]
    label: str
    # max(p): the quantity temperature scaling fits and ECE is computed on.
    answer_confidence: float


@dataclass(frozen=True)
class Prediction:
    answers: dict[str, Answer]
    latency_ms: float
    input_tokens: int


@dataclass(frozen=True)
class QuestionLogits:
    """One question's raw decision-head scores for one state, before any temperature."""

    qid: str
    qtype: QuestionType
    labels: tuple[str, ...]
    logits: np.ndarray  # shape (n_options,), float64


def option_labels(qdef: Mapping[str, Any]) -> tuple[str, ...]:
    """Answer labels in the order the model scores them."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        keys = crit if isinstance(crit, list) else list(cast(Mapping[str, Any], crit).keys())
        return tuple(str(k) for k in keys)
    if t == "score":
        return tuple(str(i) for i in range(len(cast(list[Any], crit))))
    return ("false", "true")  # noul: fixed semantic order, laya.common.render_options


class LayaDecider:
    """A loaded Laya checkpoint that answers typed questions about one state at a time."""

    def __init__(self, settings: LayaSettings) -> None:
        self._agent = laya.Agent(
            settings.checkpoint, device=settings.device, revision=settings.revision
        )
        self.checkpoint = settings.checkpoint
        self.revision: str | None = self._agent.revision
        self.max_len: int = int(self._agent.cfg.get("max_len", 512))
        self.head_max_len: int = int(self._agent.cfg.get("head_max_len", 192))
        self._tok = self._agent.tok

    # -- introspection ---------------------------------------------------------------------

    @property
    def device(self) -> str:
        """The device the weights are actually on.

        Not the device that was asked for: Laya falls back to CPU with only a printed warning when
        it cannot place the model on the GPU (DECISIONS.md D-010).
        """
        return str(next(self._agent.model.parameters()).device.type)

    @property
    def autocast_dtype(self) -> str | None:
        return str(self._agent.dtype).removeprefix("torch.") if self._agent.amp_enabled else None

    def effective_temperature(self, qtype: QuestionType, n_options: int) -> float:
        """The temperature Agent.predict applies to a question of this type and size.

        A per-(type, option-count) bucket wins over the per-type value, as in
        `Agent._decode_answers`. Both were clamped to [0.5, 5.0] when the checkpoint loaded.
        """
        qt = QTYPES[qtype]
        return float(
            self._agent.temperature_by_options.get(
                temp_bucket(qt, n_options), self._agent.temperature[qt]
            )
        )

    # -- token accounting ------------------------------------------------------------------

    def serialize(self, state: State) -> str:
        """The exact text the model reads for a state (laya.common.serialize_state)."""
        return str(serialize_state(dict(state)))

    def count_tokens(self, text: str) -> int:
        """Token count of `text` exactly as `Agent._encode_state` tokenizes a serialized state."""
        mask = self._tok.mask_token
        ids = encode_text(self._tok, text.replace(mask, " "), add_special_tokens=False)
        return len(ids["input_ids"])

    def state_room(self, questions: Questions) -> int:
        """Tokens left for the state: the minimum over `questions` (DECISIONS.md D-004).

        Each question is its own row, `[CLS] head [SEP] options [SEP] state [SEP]`, and the state
        gets `max_len - len(prefix) - 1`. Building the row with an empty state gives
        `len(prefix) + 1` tokens, so the room is `max_len` minus that length.
        """
        if not questions:
            raise ValueError("an event must ask at least one question")
        rooms = []
        for qid, qdef in questions.items():
            internal = self._validated(qid, qdef)
            ids, markers, stats = build_sequence(
                self._tok,
                "",
                internal,
                self.max_len,
                self.head_max_len,
                state_ids=[],
                return_stats=True,
            )
            n_options = len(render_options(internal))
            if len(markers) != n_options or stats["options_distinct"] < n_options:
                raise QuestionBudgetError(
                    f"question {qid!r}: {stats['options_distinct']} of {n_options} options keep "
                    f"a distinct token span within head_max_len={self.head_max_len}; shorten the "
                    "option descriptions"
                )
            rooms.append(self.max_len - len(ids))
        return min(rooms)

    def check_fits(self, state: State, questions: Questions) -> int:
        """Return the state's token count, or raise if any question's row would truncate it."""
        n = self.count_tokens(self.serialize(state))
        room = self.state_room(questions)
        if n > room:
            raise StateOverflowError(
                f"state is {n} tokens but the questions leave room for {room}; the packer must "
                "keep states inside the budget"
            )
        return n

    # -- inference -------------------------------------------------------------------------

    def predict(self, state: State, questions: Questions, *, check: bool = True) -> Prediction:
        """Answer every question about one state in one batched forward call.

        This is the live path: the checkpoint's own temperatures, timed end to end (tokenize,
        forward, decode). The forward copies its outputs to host memory, so on CUDA the timer
        covers the kernels and needs no explicit synchronize.
        """
        if check:
            self.check_fits(state, questions)
        t0 = time.perf_counter()
        raw = self._agent.predict(dict(state), {k: dict(v) for k, v in questions.items()})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        answers = {
            qid: self._to_answer(qid, questions[qid], a) for qid, a in raw["answers"].items()
        }
        return Prediction(
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=int(raw["usage"]["input_tokens"]),
        )

    def predict_sequential(self, state: State, questions: Questions) -> Prediction:
        """The same answers from one forward call per question. The Table B comparison row."""
        self.check_fits(state, questions)
        t0 = time.perf_counter()
        answers: dict[str, Answer] = {}
        tokens = 0
        for qid, qdef in questions.items():
            raw = self._agent.predict(dict(state), {qid: dict(qdef)})
            answers[qid] = self._to_answer(qid, qdef, raw["answers"][qid])
            tokens += int(raw["usage"]["input_tokens"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(answers=answers, latency_ms=latency_ms, input_tokens=tokens)

    def logits_batch(
        self,
        states: Sequence[State],
        questions: Questions,
        *,
        batch_size: int = 32,
        check: bool = True,
    ) -> list[dict[str, QuestionLogits]]:
        """Raw decision-head scores for many states, before temperature. The evaluation path.

        Uncalibrated and calibrated rows of Table A both come from these logits, so they are
        computed from the same forward pass, and temperature fitting needs exactly this.
        """
        if check:
            for st in states:
                self.check_fits(st, questions)
        ids = list(questions.keys())
        internal = {qid: self._validated(qid, questions[qid]) for qid in ids}
        types = {qid: cast(QuestionType, questions[qid]["type"]) for qid in ids}
        labels = {qid: option_labels(questions[qid]) for qid in ids}
        out: list[dict[str, QuestionLogits]] = []
        for start in range(0, len(states), batch_size):
            chunk = states[start : start + batch_size]
            encoded = [
                self._agent._encode_state(dict(st), ids, internal) for st in chunk
            ]  # laya-internal
            batch = collate_items(encoded, self._tok.pad_token_id)
            logits, _act = self._agent._forward(batch)  # laya-internal
            row = 0
            for items in encoded:
                per_q: dict[str, QuestionLogits] = {}
                for j, qid in enumerate(ids):
                    k = len(items[j]["markers"])
                    per_q[qid] = QuestionLogits(
                        qid=qid,
                        qtype=types[qid],
                        labels=labels[qid],
                        logits=np.asarray(logits[row + j, :k], dtype=np.float64),
                    )
                out.append(per_q)
                row += len(ids)
        return out

    # -- training ----------------------------------------------------------------------------
    #
    # The item format, the collate, and the reward are Laya's own, reproduced from
    # notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (CLAUDE.md §11). The training loop
    # itself lives in floorcall.train and sees only these functions and plain tensors.

    def training_item(
        self, state: State, qid: str, qdef: Mapping[str, Any], target: Sequence[float]
    ) -> dict[str, Any]:
        """One (state, question, gold distribution) training row, as the notebook builds it.

        `target` is a probability per option in `option_labels(qdef)` order. The notebook drops a
        row whose markers do not survive; here that is an error, as is a truncated state, since
        both mean the row is not the input the model will see at inference.
        """
        internal = self._validated(qid, qdef)
        n_options = len(render_options(internal))
        if len(target) != n_options:
            raise ValueError(f"question {qid!r}: {len(target)} targets for {n_options} options")
        total = float(sum(target))
        if total <= 0:
            raise ValueError(f"question {qid!r}: target has no mass")
        dist = [float(t) / total for t in target]
        n_state = self.count_tokens(self.serialize(state))
        ids, markers = build_sequence(
            self._tok, dict(state), internal, self.max_len, self.head_max_len
        )
        room = self.state_room({qid: qdef})
        if n_state > room:
            raise StateOverflowError(f"training state is {n_state} tokens, room is {room}")
        if len(markers) != n_options:
            raise QuestionBudgetError(f"question {qid!r}: options lost to the head budget")
        return {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[qdef["type"]],
            "target": dist,
            "label": dist.index(max(dist)),
        }

    def collate(self, items: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad a list of training rows into one batch (laya.common.collate_items)."""
        batch = collate_items([list(items)], self._tok.pad_token_id)
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    @property
    def model(self) -> torch.nn.Module:
        """The underlying DecisionModel, for the training loop. Forward signature:
        `(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)`."""
        return cast(torch.nn.Module, self._agent.model)

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint both the encoder and the decision head, as the notebook does."""
        self._agent.model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self._agent.model.head_checkpointing = True

    @staticmethod
    def proper_reward(
        q: torch.Tensor,
        target: torch.Tensor,
        qtype: torch.Tensor,
        mask: torch.Tensor,
        *,
        w_sph: float,
        w_rps: float,
    ) -> torch.Tensor:
        """Laya's strictly proper scoring-rule reward: log + spherical score, minus RPS on score
        questions (laya.common.proper_reward)."""
        return cast(torch.Tensor, proper_reward(q, target, qtype, mask, w_sph=w_sph, w_rps=w_rps))

    # -- helpers ---------------------------------------------------------------------------

    def _validated(self, qid: str, qdef: Mapping[str, Any]) -> dict[str, Any]:
        laya.Agent._check_question(qid, dict(qdef))  # laya-internal
        return cast(dict[str, Any], laya.Agent._to_internal(dict(qdef)))  # laya-internal

    @staticmethod
    def _to_answer(qid: str, qdef: Mapping[str, Any], raw: Mapping[str, Any]) -> Answer:
        qtype = cast(QuestionType, raw["type"])
        if qtype == "noul":
            p_true = float(raw["noul"])
            probs = {"false": 1.0 - p_true, "true": p_true}
        else:
            probs = {str(k): float(v) for k, v in raw["probabilities"].items()}
        order = option_labels(qdef)
        probs = {label: probs[label] for label in order}
        label = max(probs, key=probs.__getitem__)
        return Answer(
            qid=qid,
            qtype=qtype,
            probabilities=probs,
            label=label,
            answer_confidence=float(raw["answer_confidence"]),
        )
