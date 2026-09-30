"""Table B: batch-1 decision latency (CLAUDE.md §12).

    uv run floorcall eval latency [--rows gpu|cpu|all]

Protocol: batch 1 (real time is batch 1), `EvalSettings.latency_warmup` untimed iterations, then
`EvalSettings.latency_iters` timed ones, over a fixed input set of 50 real snapshots per event,
drawn at even spacing from the frozen test sets. The warmup cycles through every input, so every
CUDA-graph bucket is captured before timing starts. What is timed is the whole `Decider.decide`
call (packing, tokenizing, the forward pass, temperatures), because that is what sits on the
pipeline's critical path, and each component is reported beside it.

The device the weights are on is checked before and after timing. Laya falls back to CPU silently,
and a "GPU" row must never be a CPU number (DECISIONS.md D-010).

Thermal rules (DECISIONS.md D-037: the laptop crashed from heat, overclocked). Every row starts
only once the GPU is cool enough, and is sampled by nvidia-smi throughout: temperature, SM and
memory clocks, power draw and limit, and the clock-event (throttle) reasons. The samples are
written beside the row. A GPU row that throttled while timed (thermal or hardware slowdown, power
brake) is discarded to runs/latency/discarded/ and retried after a cooldown; the software power cap
is recorded but is not a discard reason. A row that reaches the abort temperature stops at once.
CPU rows run in chunks of at most `latency_cpu_chunk_s`, each after a cooldown.

`--compare-stock` measures the stock checkpoint beside the configured one, row by row, alternating
which goes first. The first fine-tuned measurement came in about a third faster than the stock one,
so weight-independence is measured, not assumed. Stock rows go to results/table_b/stock/.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.decider import Decider
from floorcall.evaluate.dataset import load_test, snapshot_of
from floorcall.provenance import git_head
from floorcall.state import Snapshot

# Windows 11 power mode overlays (the Settings > Power "Power mode" slider), on AC.
POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}
RESULTS = REPO_ROOT / "results" / "table_b"
DISCARDED = REPO_ROOT / "runs" / "latency" / "discarded"
N_INPUTS = 50


@dataclass(frozen=True)
class Row:
    name: str
    label: str
    device: str
    graphs: bool
    event: questions.Event
    sequential: bool


PAUSE, BARGE = questions.Event.USER_PAUSE, questions.Event.USER_SPEECH_DURING_AGENT
ROWS: tuple[Row, ...] = (
    Row(
        "gpu_graphed_pause",
        "GPU, CUDA graphs: user_pause (3 questions, 1 call)",
        "cuda",
        True,
        PAUSE,
        False,
    ),
    Row(
        "gpu_graphed_pause_seq",
        "GPU, CUDA graphs: user_pause (3 questions, 3 calls)",
        "cuda",
        True,
        PAUSE,
        True,
    ),
    Row(
        "gpu_graphed_barge",
        "GPU, CUDA graphs: user_speech_during_agent (2 questions, 1 call)",
        "cuda",
        True,
        BARGE,
        False,
    ),
    Row(
        "gpu_eager_pause",
        "GPU, eager: user_pause (3 questions, 1 call)",
        "cuda",
        False,
        PAUSE,
        False,
    ),
    Row(
        "gpu_eager_pause_seq",
        "GPU, eager: user_pause (3 questions, 3 calls)",
        "cuda",
        False,
        PAUSE,
        True,
    ),
    Row(
        "gpu_eager_barge",
        "GPU, eager: user_speech_during_agent (2 questions, 1 call)",
        "cuda",
        False,
        BARGE,
        False,
    ),
    Row("cpu_pause", "CPU: user_pause (3 questions, 1 call)", "cpu", False, PAUSE, False),
    Row("cpu_pause_seq", "CPU: user_pause (3 questions, 3 calls)", "cpu", False, PAUSE, True),
    Row(
        "cpu_barge",
        "CPU: user_speech_during_agent (2 questions, 1 call)",
        "cpu",
        False,
        BARGE,
        False,
    ),
)


def inputs(settings: Settings, event: questions.Event, n: int = N_INPUTS) -> list[Snapshot]:
    """A fixed input set: evenly spaced test rows. user_pause mixes D1 and D3 states."""
    decisions = ("turn_complete", "route") if event is PAUSE else ("barge_in",)
    per = n // len(decisions)
    out: list[Snapshot] = []
    for d in decisions:
        rows = load_test(settings, d).rows
        step = max(1, len(rows) // per)
        out += [snapshot_of(r) for r in rows[::step][:per]]
    return out


def percentiles(xs: Sequence[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict[str, Any]:
    import torch

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "laya": version("laya"),
        "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_driver": _run(
            ["nvidia-smi", "--query-gpu=driver_version,power.limit", "--format=csv,noheader"]
        ),
    }
    if sys.platform == "win32":
        env["on_ac_power"] = (
            _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Battery).BatteryStatus",
                ]
            )
            == "2"
        )
        env["power_plan"] = _run(["powercfg", "/getactivescheme"])
        overlay = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User"
                "\\PowerSchemes').ActiveOverlayAcPowerScheme",
            ]
        )
        env["windows_power_mode_ac"] = {
            "guid": overlay,
            "name": POWER_MODES.get(overlay, "unknown"),
        }
    limits = _run(
        [
            "nvidia-smi",
            "--query-gpu=enforced.power.limit,power.default_limit,power.min_limit,power.max_limit,"
            "clocks.max.sm,clocks.max.mem",
            "--format=csv,noheader,nounits",
        ]
    ).split(",")
    keys = ("enforced_w", "default_w", "min_w", "max_w", "max_sm_clock_mhz", "max_mem_clock_mhz")
    env["gpu_power"] = dict(zip(keys, (x.strip() for x in limits), strict=False))
    return env


class RowAbortedError(RuntimeError):
    """The GPU reached the abort temperature during a row."""


class Probe:
    """What `measure` tells the telemetry: the phase it is in, and when to check the abort."""

    def phase(self, name: str) -> None:
        return None

    def check(self) -> None:
        return None


def measure(
    decider: Decider,
    row: Row,
    snaps: Sequence[Snapshot],
    warmup: int,
    iters: int,
    *,
    probe: Probe | None = None,
    chunk_s: float | None = None,
    chunk_warmup: int = 0,
    between_chunks: Callable[[int], None] = lambda _: None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time `iters` calls after `warmup` untimed ones.

    With `chunk_s`, the timed calls are split into chunks of at most that long (warmup included):
    `between_chunks(k)` runs before chunk k > 0 (the cooldown), and each later chunk starts with
    `chunk_warmup` untimed calls. Percentiles are over every timed call.
    """
    probe = probe or Probe()
    device = getattr(decider.model, "device", None)
    if device != row.device:
        raise RuntimeError(f"{row.name}: model is on {device}, row needs {row.device} (D-010)")
    total, pack, encode, forward = [], [], [], []
    tokens = []
    chunks: list[dict[str, Any]] = []
    done = 0
    while done < iters:
        k = len(chunks)
        if k:
            between_chunks(k)
        t0 = clock()
        probe.phase("warmup")
        for i in range(warmup if k == 0 else chunk_warmup):
            decider.decide(row.event, snaps[i % len(snaps)], sequential=row.sequential)
        probe.phase("timed")
        n = 0
        while done < iters and (chunk_s is None or n == 0 or clock() - t0 < chunk_s):
            d = decider.decide(row.event, snaps[done % len(snaps)], sequential=row.sequential)
            total.append(d.total_ms)
            pack.append(d.timings.pack_ms)
            encode.append(d.timings.encode_ms)
            forward.append(d.timings.forward_ms)
            tokens.append(d.packed.n_tokens)
            done += 1
            n += 1
            if done % 10 == 0:
                probe.check()
        probe.phase("idle")
        chunks.append({"timed_calls": n, "seconds": clock() - t0})
    if getattr(decider.model, "device", None) != row.device:
        raise RuntimeError(f"{row.name}: model left {row.device} during timing (D-010)")
    out: dict[str, Any] = {
        "total_ms": percentiles(total),
        "pack_ms": percentiles(pack),
        "encode_ms": percentiles(encode),
        "forward_ms": percentiles(forward),
        "state_tokens_mean": float(np.mean(tokens)),
    }
    if chunk_s is not None:
        out["chunks"] = chunks
    return out


# -- GPU telemetry -------------------------------------------------------------------------------

TELEMETRY_FIELDS = (
    "timestamp",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "power.draw",
    "enforced.power.limit",
    "pstate",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)
NUMERIC = frozenset(
    {
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "clocks.max.sm",
        "power.draw",
        "enforced.power.limit",
    }
)
# Clock-event reasons that make a GPU row invalid. The software power cap is not one: a laptop
# GPU runs against its power limit whenever it is busy, so it is recorded, not a discard reason.
THROTTLE_REASONS = (
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def parse_sample(line: str) -> dict[str, Any] | None:
    """One `nvidia-smi --format=csv,noheader,nounits` line, or None if it is not one."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(TELEMETRY_FIELDS):
        return None
    out: dict[str, Any] = {}
    for key, value in zip(TELEMETRY_FIELDS, parts, strict=True):
        if key in NUMERIC:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = None  # "[N/A]" on fields this GPU does not report
        elif key.startswith("clocks_event_reasons.") and key != "clocks_event_reasons.active":
            out[key] = value == "Active"
        else:
            out[key] = value
    return out


def _stats(xs: Sequence[float | None]) -> dict[str, float] | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}


def summarise_telemetry(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per phase: temperature, clocks, power and clock-event counts; throttling while timed."""
    phases: dict[str, Any] = {}
    for phase in ("warmup", "timed"):
        ss = [x for x in samples if x.get("phase") == phase]
        phases[phase] = {
            "samples": len(ss),
            "temperature_c": _stats([x["temperature.gpu"] for x in ss]),
            "sm_clock_mhz": _stats([x["clocks.sm"] for x in ss]),
            "mem_clock_mhz": _stats([x["clocks.mem"] for x in ss]),
            "max_sm_clock_mhz": _stats([x["clocks.max.sm"] for x in ss]),
            "power_draw_w": _stats([x["power.draw"] for x in ss]),
            "power_limit_w": _stats([x["enforced.power.limit"] for x in ss]),
            "pstates": sorted({x["pstate"] for x in ss}),
            "clock_events": {
                r.split(".", 1)[1]: sum(bool(x[r]) for x in ss)
                for r in (*THROTTLE_REASONS, "clocks_event_reasons.sw_power_cap")
            },
        }
    timed = [x for x in samples if x.get("phase") == "timed"]
    seen = sorted({r.split(".", 1)[1] for x in timed for r in THROTTLE_REASONS if x[r]})
    return {**phases, "throttle_reasons_while_timed": seen}


def verdict(row: Row, telemetry: dict[str, Any]) -> str | None:
    """Why a measured row must be discarded, or None to keep it."""
    if telemetry["timed"]["samples"] == 0:
        return "no telemetry while timed: throttling cannot be ruled out"
    if row.device == "cuda" and telemetry["throttle_reasons_while_timed"]:
        return "throttled while timed: " + ", ".join(telemetry["throttle_reasons_while_timed"])
    return None


class GpuProbe(Probe):
    """nvidia-smi sampling in the background, tagged with the phase `measure` reports."""

    def __init__(self, interval_ms: int, abort_temp_c: float) -> None:
        self.interval_ms = interval_ms
        self.abort_temp_c = abort_temp_c
        self.samples: list[dict[str, Any]] = []
        self._phase = "idle"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuProbe:
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            self.add(line)

    def add(self, line: str) -> None:
        sample = parse_sample(line)
        if sample is not None:
            with self._lock:
                sample["phase"] = self._phase
                self.samples.append(sample)

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def phase(self, name: str) -> None:
        with self._lock:
            self._phase = name

    def check(self) -> None:
        with self._lock:
            temps = [x["temperature.gpu"] for x in self.samples[-3:] if x["temperature.gpu"]]
        if temps and max(temps) >= self.abort_temp_c:
            raise RowAbortedError(f"GPU at {max(temps):.0f} C, abort at {self.abort_temp_c:.0f} C")


def read_gpu_temp() -> float | None:
    out = _run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
    try:
        return float(out.splitlines()[0])
    except (ValueError, IndexError):
        return None


def wait_until_cool(
    max_temp_c: float,
    *,
    poll_s: float,
    max_wait_s: float,
    read: Callable[[], float | None] = read_gpu_temp,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Block until the GPU is at or below `max_temp_c`. Refuses to run blind (no reading)."""
    waited = 0.0
    while True:
        t = read()
        if t is None:
            raise RuntimeError("cannot read the GPU temperature; no latency row runs blind")
        if t <= max_temp_c:
            return {"start_temp_c": t, "waited_s": waited}
        if waited >= max_wait_s:
            raise TimeoutError(f"GPU still at {t:.0f} C after {waited:.0f} s")
        log(f"cooling: GPU at {t:.0f} C, waiting for <= {max_temp_c:.0f} C")
        sleep(poll_s)
        waited += poll_s


def weight_dtypes(model: Any) -> dict[str, int]:
    """Parameter count per dtype of the loaded network (D-037: weights stored differently could
    run at different speeds)."""
    net = getattr(model, "model", None)
    if net is None or not hasattr(net, "parameters"):
        return {}
    counts: dict[str, int] = {}
    for p in net.parameters():
        key = str(p.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run(
    settings: Settings,
    which: str = "all",
    *,
    compare_stock: bool = False,
    precisions: Sequence[str] = (),
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Measure the selected rows under the thermal rules; return the rows kept.

    `precisions` (D-039) measures the configured checkpoint at each precision instead, interleaved
    row by row like the stock comparison; each goes to results/table_b/precision/<precision>/, and a
    precision that failed the parity check is refused.
    """
    from floorcall.evaluate.precision import passing
    from floorcall.model.laya_adapter import LayaDecider

    selected = [r for r in ROWS if which == "all" or r.name.startswith(which)]
    # arms: (name, which loaded model, precision to set or None, where kept rows go)
    arms: list[tuple[str, str, str | None, Path]] = [("configured", "configured", None, RESULTS)]
    lays = {"configured": settings.laya}
    if compare_stock:
        if settings.laya.checkpoint == LayaSettings().checkpoint:
            raise ValueError("--compare-stock needs a non-stock checkpoint configured")
        lays["stock"] = LayaSettings()
        arms.append(("stock", "stock", None, RESULTS / "stock"))
    if precisions:
        if refused := [p for p in precisions if not passing(p)]:
            raise ValueError(f"no latency rows for {refused}: they did not pass the parity check")
        arms = [(p, "configured", p, RESULTS / "precision" / p) for p in precisions]
    # Provenance is read when the run starts: the code executing is what was loaded then, and
    # edits to files during a long run do not change it.
    code = git_head()
    env = environment()
    out = []
    for device in ("cuda", "cpu"):
        rows = [r for r in selected if r.device == device]
        if not rows:
            continue
        loaded = {
            name: LayaDecider(LayaSettings(**{**lay.model_dump(), "device": device}))
            for name, lay in lays.items()
            if any(a[1] == name for a in arms)
        }
        for i, row in enumerate(rows):
            # rotate which arm goes first, so drift over the session hits every arm alike
            k = i % len(arms)
            for name, model_key, precision, target in arms[k:] + arms[:k]:
                kept = measure_row(
                    settings,
                    row,
                    loaded[model_key],
                    name,
                    code,
                    env,
                    log,
                    target=target,
                    precision=precision,
                )
                if kept is not None:
                    out.append(kept)
        for model in loaded.values():
            model.disable_cuda_graphs()
        del loaded
    return out


def measure_row(
    settings: Settings,
    row: Row,
    model: Any,
    name: str,
    code: str,
    env: dict[str, Any],
    log: Callable[[str], None],
    *,
    probe_factory: Callable[[], GpuProbe] | None = None,
    wait: Callable[..., dict[str, Any]] = wait_until_cool,
    sleep: Callable[[float], None] = time.sleep,
    target: Path | None = None,
    precision: str | None = None,
) -> dict[str, Any] | None:
    """One row for one model: gate, measure with telemetry, then keep or discard and retry."""
    ev = settings.eval
    if target is None:
        target = RESULTS / "stock" if name == "stock" else RESULTS
    decider = Decider(model, settings.state)
    snaps = inputs(settings, row.event)
    gate_kw = {
        "poll_s": ev.latency_cooldown_poll_s,
        "max_wait_s": ev.latency_cooldown_max_s,
        "log": log,
    }

    def cooldown(k: int) -> None:
        log(f"{row.name}: cooldown before chunk {k + 1} ({ev.latency_cpu_cooldown_s:.0f} s)")
        sleep(ev.latency_cpu_cooldown_s)
        wait(ev.latency_start_max_temp_c, **gate_kw)

    for attempt in range(1, ev.latency_max_attempts + 1):
        gate = wait(ev.latency_start_max_temp_c, **gate_kw)
        if precision is not None:
            model.set_precision(precision)  # drops any captured graphs; recaptured below
        if row.graphs:
            model.enable_cuda_graphs(settings.laya.graph_bucket_tokens)
        else:
            model.disable_cuda_graphs()
        started = datetime.now(UTC).isoformat(timespec="seconds")
        aborted: str | None = None
        summary: dict[str, Any] = {}
        probe = (
            probe_factory()
            if probe_factory
            else GpuProbe(ev.latency_telemetry_ms, ev.latency_abort_temp_c)
        )
        with probe:
            try:
                summary = measure(
                    decider,
                    row,
                    snaps,
                    ev.latency_warmup,
                    ev.latency_iters,
                    probe=probe,
                    chunk_s=ev.latency_cpu_chunk_s if row.device == "cpu" else None,
                    chunk_warmup=ev.latency_cpu_chunk_warmup,
                    between_chunks=cooldown,
                )
            except RowAbortedError as exc:
                aborted = str(exc)
        telemetry = summarise_telemetry(probe.samples)
        reason = aborted or verdict(row, telemetry)
        budget = ev.gpu_p99_budget_ms if row.device == "cuda" else ev.cpu_p99_budget_ms
        payload: dict[str, Any] = {
            "row": row.name,
            "label": row.label,
            "device": model.device,
            "cuda_graphs": model.cuda_graphs,
            "autocast": model.autocast_dtype,
            "precision": getattr(model, "precision", None),
            "event": row.event.value,
            "questions": list(questions.EVENT_QUESTIONS[row.event]),
            "sequential": row.sequential,
            "warmup": ev.latency_warmup,
            "iterations": ev.latency_iters,
            "inputs": len(snaps),
            "checkpoint": model.checkpoint,
            "revision": model.revision,
            "weight_dtypes": weight_dtypes(model),
            "model": name,
            "code": code,
            "environment": env,
            "started_at": started,
            "attempt": attempt,
            "thermal_gate": gate,
            "telemetry": telemetry,
            "telemetry_samples": probe.samples,
            "budget_p99_ms": budget,
            **summary,
        }
        if reason is None:
            payload["fits_budget"] = summary["total_ms"]["p99"] <= budget
            _write(target / f"{row.name}.json", payload)
            t = summary["total_ms"]
            log(f"{name:10s} {row.name:22s} kept: p50 {t['p50']:.1f}  p99 {t['p99']:.1f} ms")
            return payload
        payload["discarded"] = reason
        stamp = started.replace(":", "")
        _write(DISCARDED / name / f"{row.name}.{stamp}.json", payload)
        log(f"{name:10s} {row.name:22s} discarded (attempt {attempt}): {reason}")
    log(f"{name:10s} {row.name:22s} no valid measurement in {ev.latency_max_attempts} attempts")
    return None
