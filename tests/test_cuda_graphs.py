"""CUDA-graphed forward against the eager one (DECISIONS.md D-012).

Measured on 2026-09-28 (RTX 5070 Ti Laptop, bf16, stock checkpoint, 60 to 90 states from the
frozen test sets, 160 to 240 decisions): replay bit-identical; max |dp| 0.031 at T=1; max |dlogit|
0.14; argmax flips on 1 of 240 and 4 of 160 decisions. **Every flip had an eager top-2 margin of
exactly 2^-8**, one bf16 step: the decision head scores under bf16 autocast, so its logits are
quantized, and those decisions were exact ties that the two kernels broke differently. The stock
checkpoint makes such ties common (44% of its decisions have a top-2 margin under 0.1).

So the test does not ask for a fixed agreement rate. It asks that nothing flips unless it was a tie
at bf16 resolution, and bounds the logit difference. A broken capture (wrong mask, stale buffer,
wrong padding) would flip decisions with real margins and move logits far more than this.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.evaluate.dataset import load_test
from floorcall.evaluate.scoring import softmax
from floorcall.model.laya_adapter import LayaDecider, bucket_length


@pytest.mark.parametrize(
    ("length", "bucket", "max_len", "expected"),
    [
        (1, 32, 512, 32),
        (32, 32, 512, 32),
        (33, 32, 512, 64),
        (500, 32, 512, 512),
        (512, 32, 512, 512),
        (223, 32, 512, 224),
        (510, 64, 512, 512),
        (40, 64, 50, 50),
    ],
)
def test_bucket_length(length: int, bucket: int, max_len: int, expected: int) -> None:
    assert bucket_length(length, bucket, max_len) == expected


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# A top-2 margin at or below this is a tie at bf16 resolution: a few steps of 2^-8 near |z| ~ 1-2.
BF16_TIE = 0.02


@pytest.fixture(scope="module")
def decider() -> Iterator[LayaDecider]:
    s = get_settings()
    d = LayaDecider(LayaSettings(**{**s.laya.model_dump(), "device": "cuda"}))
    assert d.precision == s.laya.precision == "fp16"  # D-040: applied at load, on CUDA
    yield d
    d.disable_cuda_graphs()


def cases(d: LayaDecider, per_decision: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    s = get_settings()
    out = []
    for decision in ("turn_complete", "barge_in", "route"):
        ev = load_test(s, decision)
        qs = questions.questions_for(ev.event)
        budget = d.state_room(qs) - s.state.safety_margin_tokens
        step = len(ev.rows) // per_decision
        ev.rows = ev.rows[::step][:per_decision]
        packed = ev.packed_states(state=s.state, budget=budget, count_tokens=d.count_tokens)
        out += [(st, qs) for st in packed]
    return out


@pytest.mark.model
@pytest.mark.gpu
@needs_gpu
@pytest.mark.parametrize("precision", ["bf16", "fp32", "fp16"])
def test_graphed_matches_eager(decider: LayaDecider, precision: str) -> None:
    # every inference precision (D-039) captures and replays the same function as its eager forward
    decider.set_precision(precision)
    assert decider.precision == precision
    assert (
        decider.autocast_dtype == {"fp32": None, "bf16": "bfloat16", "fp16": "float16"}[precision]
    )
    sample = cases(decider, per_decision=20)
    decider.disable_cuda_graphs()
    eager = [decider.logits_one(st, qs) for st, qs in sample]
    decider.enable_cuda_graphs(get_settings().laya.graph_bucket_tokens)
    graphed = [decider.logits_one(st, qs) for st, qs in sample]
    again = [decider.logits_one(st, qs) for st, qs in sample]

    total = 0
    max_dp = max_dz = 0.0
    real_flips = []
    for e, g, g2 in zip(eager, graphed, again, strict=True):
        assert g.graphed and not e.graphed
        for qid, el in e.logits.items():
            zg = g.logits[qid].logits
            assert np.array_equal(zg, g2.logits[qid].logits), "replay must be deterministic"
            max_dp = max(max_dp, float(np.abs(softmax(el.logits[None]) - softmax(zg[None])).max()))
            max_dz = max(max_dz, float(np.abs(el.logits - zg).max()))
            top2 = np.sort(el.logits)[-2:]
            if el.logits.argmax() != zg.argmax() and top2[1] - top2[0] > BF16_TIE:
                real_flips.append((qid, float(top2[1] - top2[0])))
            total += 1
    assert total > 100
    assert not real_flips, f"decisions with a real margin flipped: {real_flips}"
    assert max_dz <= 0.3, f"max |dlogit| {max_dz:.4f}"
    assert max_dp <= 0.06, f"max |dp| {max_dp:.4f}"
    decider.set_precision(get_settings().laya.precision)  # the default, for later tests


@pytest.mark.model
@pytest.mark.gpu
@needs_gpu
def test_shapes_are_bucketed_and_reused(decider: LayaDecider) -> None:
    decider.enable_cuda_graphs(32)
    sample = cases(decider, per_decision=10)
    for st, qs in sample:
        decider.logits_one(st, qs)
    shapes = decider.captured_shapes
    assert all(length % 32 == 0 for _, length, _ in shapes)
    assert len(shapes) < len(sample), "one graph per bucket, not one per state"
    for st, qs in sample:
        decider.logits_one(st, qs)
    assert decider.captured_shapes == shapes, "a second pass must not capture anything new"
