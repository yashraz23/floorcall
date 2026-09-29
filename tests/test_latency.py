"""The latency harness's bookkeeping, with a fake model (no GPU, no checkpoint)."""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from floorcall import questions
from floorcall.config import StateSettings, get_settings
from floorcall.decider import Decider
from floorcall.evaluate import latency
from floorcall.model.laya_adapter import QuestionLogits, TimedLogits, option_labels


@dataclass
class FakeModel:
    device: str = "cuda"
    fall_back_after: int | None = None  # simulate Laya's silent CPU fallback
    calls: int = 0
    seen: list[str] = field(default_factory=list)

    def state_room(self, qs: Any) -> int:
        return 400

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def effective_temperature(self, qtype: Any, n: int) -> float:
        return 1.0

    def logits_one(self, state: Any, qs: Any, *, check: bool = True) -> TimedLogits:
        self.calls += 1
        if self.fall_back_after is not None and self.calls > self.fall_back_after:
            self.device = "cpu"
        out = {
            qid: QuestionLogits(qid, q["type"], option_labels(q), np.zeros(len(option_labels(q))))
            for qid, q in qs.items()
        }
        return TimedLogits(out, encode_ms=0.5, forward_ms=1.5, input_tokens=5, graphed=False)


ROW = latency.Row("t", "test", "cuda", False, questions.Event.USER_PAUSE, False)


def test_percentiles() -> None:
    p = latency.percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert p["p99"] == pytest.approx(99.01)
    assert p["max"] == 100


def test_inputs_are_fixed_and_mix_decisions() -> None:
    s = get_settings()
    a = latency.inputs(s, questions.Event.USER_PAUSE)
    b = latency.inputs(s, questions.Event.USER_PAUSE)
    assert a == b
    assert len(a) == latency.N_INPUTS
    assert len(latency.inputs(s, questions.Event.USER_SPEECH_DURING_AGENT)) == latency.N_INPUTS


def test_measure_reports_every_component() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    out = latency.measure(Decider(FakeModel(), StateSettings()), ROW, snaps, warmup=3, iters=20)
    assert set(out) == {"total_ms", "pack_ms", "encode_ms", "forward_ms", "state_tokens_mean"}
    assert out["forward_ms"]["p50"] == pytest.approx(1.5)
    # The fake reports a forward time without spending it, so the wall-clock total is only
    # bounded below by the parts that are really timed here: packing.
    assert out["total_ms"]["p50"] >= out["pack_ms"]["p50"] > 0


def test_a_gpu_row_refuses_a_cpu_model() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    with pytest.raises(RuntimeError, match="D-010"):
        latency.measure(Decider(FakeModel(device="cpu"), StateSettings()), ROW, snaps, 1, 2)


def test_a_silent_fallback_during_timing_is_caught() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    model = FakeModel(fall_back_after=5)
    with pytest.raises(RuntimeError, match="left cuda"):
        latency.measure(Decider(model, StateSettings()), ROW, snaps, warmup=2, iters=10)


# -- thermal rules (D-037), with fake telemetry: no GPU, no nvidia-smi --------------------------

LINE = (
    "2026/09/30 10:00:00.000, 61, 2400, 12000, 3090, 88.50, [N/A], P0, 0x0000000000000004, "
    "Active, Not Active, Not Active, Not Active, Not Active"
)


def sample(phase: str, temp: float = 60.0, **reasons: bool) -> dict[str, Any]:
    s = latency.parse_sample(LINE)
    assert s is not None
    s.update({"phase": phase, "temperature.gpu": temp})
    for r in latency.THROTTLE_REASONS:
        s[r] = reasons.get(r.split(".", 1)[1], False)
    return s


def test_a_telemetry_line_parses_including_na_fields() -> None:
    s = latency.parse_sample(LINE)
    assert s is not None
    assert s["temperature.gpu"] == 61 and s["clocks.sm"] == 2400 and s["power.draw"] == 88.5
    assert s["enforced.power.limit"] is None  # "[N/A]"
    assert s["clocks_event_reasons.sw_power_cap"] is True
    assert s["clocks_event_reasons.hw_thermal_slowdown"] is False
    assert latency.parse_sample("garbage") is None


GPU_ROW = latency.Row("g", "gpu", "cuda", False, questions.Event.USER_PAUSE, False)
CPU_ROW = latency.Row("c", "cpu", "cpu", False, questions.Event.USER_PAUSE, False)


@pytest.mark.parametrize(
    ("samples", "row", "discard"),
    [
        ([sample("timed")], GPU_ROW, None),  # the power cap alone is not throttling
        ([sample("timed", sw_thermal_slowdown=True)], GPU_ROW, "sw_thermal_slowdown"),
        ([sample("timed", hw_slowdown=True)], GPU_ROW, "hw_slowdown"),
        ([sample("warmup", hw_thermal_slowdown=True), sample("timed")], GPU_ROW, None),
        ([sample("timed", sw_thermal_slowdown=True)], CPU_ROW, None),  # GPU clocks: not a CPU row's
        ([sample("warmup")], GPU_ROW, "no telemetry"),
    ],
)
def test_the_verdict(samples: list[dict[str, Any]], row: latency.Row, discard: str | None) -> None:
    reason = latency.verdict(row, latency.summarise_telemetry(samples))
    assert (reason is None) if discard is None else (discard in (reason or ""))


def test_a_row_waits_until_the_gpu_is_cool_and_never_runs_blind() -> None:
    temps = iter([70.0, 60.0, 50.0])
    slept: list[float] = []
    gate = latency.wait_until_cool(
        55,
        poll_s=15,
        max_wait_s=600,
        read=lambda: next(temps),
        sleep=slept.append,
        log=lambda _: None,
    )
    assert gate == {"start_temp_c": 50.0, "waited_s": 30.0} and slept == [15, 15]
    with pytest.raises(RuntimeError, match="blind"):
        latency.wait_until_cool(55, poll_s=1, max_wait_s=10, read=lambda: None, log=lambda _: None)
    with pytest.raises(TimeoutError):
        latency.wait_until_cool(
            55, poll_s=5, max_wait_s=10, read=lambda: 80.0, sleep=lambda _: None, log=lambda _: None
        )


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0  # every read advances a second
        return self.t


def test_cpu_rows_run_in_chunks_with_cooldowns() -> None:
    snaps = latency.inputs(get_settings(), questions.Event.USER_PAUSE)
    model = FakeModel(device="cpu")
    cooled: list[int] = []
    out = latency.measure(
        Decider(model, StateSettings()), CPU_ROW, snaps, warmup=4, iters=30,
        chunk_s=10, chunk_warmup=2, between_chunks=cooled.append, clock=Clock(),
    )  # fmt: skip
    chunks = out["chunks"]
    assert sum(c["timed_calls"] for c in chunks) == 30 and len(chunks) > 1
    assert all(c["seconds"] <= 12 for c in chunks)  # at most the limit, plus the last call
    assert cooled == list(range(1, len(chunks)))  # a cooldown before every chunk but the first
    # untimed calls: the first chunk's warmup, then each later chunk's short one
    assert model.calls == 30 + 4 + 2 * (len(chunks) - 1)


class FakeProbe(latency.GpuProbe):
    """Telemetry without nvidia-smi: each phase change records one sample."""

    def __init__(self, timed_sample: dict[str, Any], abort_after: int | None = None) -> None:
        super().__init__(interval_ms=500, abort_temp_c=85)
        self.timed_sample, self.abort_after, self.checks = timed_sample, abort_after, 0

    def __enter__(self) -> "FakeProbe":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def phase(self, name: str) -> None:
        super().phase(name)
        if name in ("warmup", "timed"):
            self.samples.append({**self.timed_sample, "phase": name})

    def check(self) -> None:
        self.checks += 1
        if self.abort_after is not None and self.checks >= self.abort_after:
            raise latency.RowAbortedError("GPU at 90 C, abort at 85 C")


@dataclass
class FakeGpuModel(FakeModel):
    checkpoint: str = "checkpoints/x"
    revision: str | None = None
    cuda_graphs: bool = False
    autocast_dtype: str = "bf16"

    def enable_cuda_graphs(self, bucket: int) -> None:
        self.cuda_graphs = True

    def disable_cuda_graphs(self) -> None:
        self.cuda_graphs = False


def test_a_throttled_row_is_discarded_and_retried(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latency, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(latency, "DISCARDED", tmp_path / "discarded")
    probes = iter(
        [FakeProbe(sample("timed", sw_thermal_slowdown=True)), FakeProbe(sample("timed"))]
    )
    s = get_settings()
    s = s.model_copy(
        update={"eval": s.eval.model_copy(update={"latency_warmup": 2, "latency_iters": 10})}
    )
    kept = latency.measure_row(
        s, GPU_ROW, FakeGpuModel(), "configured", "abc", {}, lambda _: None,
        probe_factory=lambda: next(probes), wait=lambda *a, **k: {"start_temp_c": 50.0},
    )  # fmt: skip
    assert kept is not None and kept["attempt"] == 2 and "discarded" not in kept
    assert (tmp_path / "results" / "g.json").exists()
    (discarded,) = list((tmp_path / "discarded" / "configured").iterdir())
    assert "sw_thermal_slowdown" in json.loads(discarded.read_text())["discarded"]


def test_a_row_that_gets_too_hot_stops_and_is_not_kept(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latency, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(latency, "DISCARDED", tmp_path / "discarded")
    s = get_settings()
    s = s.model_copy(
        update={
            "eval": s.eval.model_copy(
                update={"latency_warmup": 2, "latency_iters": 50, "latency_max_attempts": 1}
            )
        }
    )
    kept = latency.measure_row(
        s, GPU_ROW, FakeGpuModel(), "configured", "abc", {}, lambda _: None,
        probe_factory=lambda: FakeProbe(sample("timed"), abort_after=1),
        wait=lambda *a, **k: {"start_temp_c": 50.0},
    )  # fmt: skip
    assert kept is None and not (tmp_path / "results").exists()
    assert len(list((tmp_path / "discarded" / "configured").iterdir())) == 1
