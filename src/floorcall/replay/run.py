"""`floorcall replay`: check, freeze or run the replay scripts (DECISIONS.md D-046).

    uv run floorcall replay demo/scripts/*.json --compare naive    # run; writes results/replay/
    uv run floorcall replay --check                                # timing only, no model
    uv run floorcall replay --freeze                               # check, then freeze hashes

A run refuses scripts that are not frozen or have changed since. It reads the served thresholds
and the timeline budgets from the committed results files (pipeline.operating), and it computes
on CPU by default, so replay needs neither a GPU nor a key.
"""

from __future__ import annotations

from pathlib import Path

from floorcall.config import REPO_ROOT, LayaSettings, Settings
from floorcall.pipeline.naive import NaiveAgent
from floorcall.pipeline.operating import decision_budgets, served_policy
from floorcall.provenance import git_head
from floorcall.replay.checks import check_timing
from floorcall.replay.engine import FloorcallReplay, NaiveReplay, Run, simulate
from floorcall.replay.report import render_script, render_summary, to_json, write_json
from floorcall.replay.script import MANIFEST, load_script, read_manifest, verify_frozen
from floorcall.replay.script import freeze as freeze_scripts

RESULTS = REPO_ROOT / "results" / "replay"


def run_replay(
    settings: Settings,
    paths: list[Path],
    *,
    check_only: bool = False,
    freeze: bool = False,
    checkpoint: Path | None = None,
    device: str | None = None,
    results: Path = RESULTS,
) -> tuple[list[str], bool]:
    """(report lines, success)."""
    if not paths:
        return ["no scripts given"], False
    served = served_policy(settings)
    budgets = decision_budgets(settings, served.checkpoint)
    scripts = [load_script(p) for p in paths]
    lines: list[str] = []
    problems = {s.id: check_timing(s, served.policy, budgets, settings.replay) for s in scripts}
    for sid, ps in problems.items():
        lines += [f"{sid}: {p}" for p in ps]
    # naive needs no model: its run is part of the check, so the intended contrasts can be read
    naive = NaiveReplay(NaiveAgent(settings.replay.naive_silence_ms))
    vad = served.policy.vad_pause_ms
    if any(problems.values()):
        return [*lines, "timing check FAILED: fix the scripts before freezing or running"], False
    if check_only:
        for s in scripts:
            run = simulate(s, naive, vad_pause_ms=vad)
            lines += render_script(s, {"naive": run}, budgets)
        lines.append(f"timing check passed for {len(scripts)} scripts")
        if freeze:
            manifest = freeze_scripts(paths)
            lines.append(f"frozen in {paths[0].parent / MANIFEST}: {len(manifest)} scripts")
        return lines, True

    unfrozen = verify_frozen(paths)
    if unfrozen:
        return [*unfrozen, "refusing to replay unfrozen scripts"], False

    from floorcall.decider import Decider
    from floorcall.pipeline.processor import DecisionProcessor

    ckpt = checkpoint or settings.replay.checkpoint
    dev = device or settings.replay.device
    laya = LayaSettings(
        checkpoint=str(ckpt),
        revision=None,
        device=dev,
        cuda_graphs=False,
        precision=settings.laya.precision,
    )
    decider = Decider.load(settings.model_copy(update={"laya": laya}))
    label = f"{dev} ({'fp32' if dev == 'cpu' else settings.laya.precision})"
    processor = DecisionProcessor(decider, served.policy, device=label)
    floorcall = FloorcallReplay(processor, served.policy, budgets, settings.replay)

    runs: dict[str, list[Run]] = {"naive": [], "floorcall": []}
    for s in scripts:
        pair = {
            "naive": simulate(s, naive, vad_pause_ms=vad),
            "floorcall": simulate(s, floorcall, vad_pause_ms=vad),
        }
        for n, r in pair.items():
            runs[n].append(r)
        lines += render_script(s, pair, budgets)
        lines.append("")
    lines += render_summary(scripts, runs, served, budgets, label)
    meta = {
        "code": git_head(),
        "checkpoint": Path(ckpt).resolve().relative_to(REPO_ROOT).as_posix()
        if Path(ckpt).resolve().is_relative_to(REPO_ROOT)
        else str(ckpt),
        "device": label,
        "naive_silence_ms": settings.replay.naive_silence_ms,
        "barge_window_ms": settings.replay.barge_window_ms,
        "vad_pause_ms": served.policy.vad_pause_ms,
        "max_wait_ms": served.policy.max_wait_ms,
        "scripts_manifest": read_manifest(paths[0].parent),
    }
    write_json(results / "replay.json", to_json(scripts, runs, served, budgets, meta))
    text = results / "replay.txt"
    with text.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    lines.append(f"wrote {results / 'replay.json'} and {text}")
    return lines, True
