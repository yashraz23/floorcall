"""Command-line entry point: `uv run floorcall --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from floorcall.config import get_settings

app = typer.Typer(
    help="floorcall: turn-taking, barge-in, routing and escalation decisions for voice agents.",
    no_args_is_help=True,
)
data_app = typer.Typer(
    help="Download, build, freeze and verify the datasets.", no_args_is_help=True
)
label_app = typer.Typer(help="Hand labelling.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate on the frozen test sets.", no_args_is_help=True)
train_app = typer.Typer(help="Fine-tune and calibrate.", no_args_is_help=True)
llm_app = typer.Typer(help="LLM usage (OpenRouter): spend against the cap.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(llm_app, name="llm")
app.add_typer(train_app, name="train")
app.add_typer(label_app, name="label")
app.add_typer(eval_app, name="eval")
console = Console()

ESCALATE_CANDIDATES = "escalate.candidates.v1.jsonl"
ESCALATE_LABELS = "escalate.labels.jsonl"


def _read_key() -> str:
    """One keypress on a terminal; one character of stdin otherwise (tests, piping).

    On Windows click.getchar reads the console through msvcrt and never sees redirected stdin,
    so without the fallback a scripted session blocks forever. End of input quits.
    """
    import sys

    import click

    if sys.stdin.isatty():
        return click.getchar().lower()
    ch = sys.stdin.read(1)
    return ch.lower() if ch else "q"


@data_app.command("download")
def data_download() -> None:
    """Fetch every raw source and check it against its pinned SHA256."""
    from floorcall.data.download import SOURCES, fetch

    raw = get_settings().paths.data_raw
    for name in SOURCES:
        path = fetch(name, raw)
        console.print(f"[green]ok[/green] {name}: {path}")


@data_app.command("build")
def data_build(
    only: str = typer.Option("all", help="all | swda | clinc"),
) -> None:
    """Build train/calib into data/processed and freeze (or re-verify) the test sets."""
    from floorcall.data import build

    settings = get_settings()
    runners = {"all": build.build_all, "swda": build.build_swda, "clinc": build.build_clinc}
    if only not in runners:
        raise typer.BadParameter(f"--only must be one of {sorted(runners)}")
    cards = runners[only](settings)
    console.print(build.summarize(cards))


@data_app.command("verify")
def data_verify() -> None:
    """Check every frozen test set against data/test_frozen/MANIFEST.sha256."""
    from floorcall.data.freeze import verify

    manifest = verify(get_settings().paths.test_frozen)
    for name, digest in manifest.items():
        console.print(f"[green]ok[/green] {digest}  {name}")


@data_app.command("escalate-candidates")
def data_escalate_candidates() -> None:
    """Draw the fixed D4 labelling pool from Twitter banking threads (test and calib splits)."""
    from floorcall.data import escalate
    from floorcall.data.download import fetch
    from floorcall.data.labelling import save_candidates

    s = get_settings()
    csv_path = fetch("twcs", s.paths.data_raw)
    fractions = s.splits.model_dump(include={"train", "calib", "test"})
    pool = escalate.build_candidates(
        csv_path, seed=s.splits.seed, fractions=fractions, max_history=s.data.max_history_turns
    )
    chosen = escalate.select_for_labelling(
        pool, split="test", n=s.data.d4_candidates_test, seed=s.splits.seed
    ) + escalate.select_for_labelling(
        pool, split="calib", n=s.data.d4_candidates_calib, seed=s.splits.seed
    )
    save_candidates(s.paths.labels / ESCALATE_CANDIDATES, chosen)
    console.print(f"wrote {len(chosen)} candidates to {s.paths.labels / ESCALATE_CANDIDATES}")


@label_app.command("escalate")
def label_escalate(
    split: str = typer.Option("test", help="test | calib"),
    labeller: str = typer.Option("yash", help="recorded on every label"),
) -> None:
    """Label D4 candidates: y escalate, n no, s skip, u undo, ? guidelines, q quit.

    Guidelines: docs/labelling-escalate.md. Every keypress is saved.
    """
    import click
    from rich.panel import Panel

    from floorcall.config import REPO_ROOT
    from floorcall.data import labelling

    s = get_settings()
    cands = labelling.load_candidates(s.paths.labels / ESCALATE_CANDIDATES)
    labels_path = s.paths.labels / ESCALATE_LABELS
    labels = labelling.load_labels(labels_path)
    order = [c.id for c in cands]
    queue = [c for c in cands if c.split == split]
    history: list[str] = []
    guidelines = (REPO_ROOT / "docs" / "labelling-escalate.md").read_text(encoding="utf-8")
    keys = {"y": "true", "n": "false", "s": "skip"}

    while True:
        pending = [c for c in queue if c.id not in labels]
        if not pending:
            console.print(f"[green]All {len(queue)} {split} candidates are labelled.[/green]")
            break
        c = pending[0]
        counts: dict[str, int] = dict(labelling.progress(cands, labels).get(split) or {})
        done = sum(v for k, v in counts.items() if k != "pending")
        console.clear()
        console.print(
            f"[bold]{split}[/bold]  {done}/{len(queue)} done  "
            f"(escalate {counts.get('true', 0)}, no {counts.get('false', 0)}, "
            f"skip {counts.get('skip', 0)})   company: {c.company}"
        )
        for t in c.recent_turns[-2:]:
            console.print(f"[dim]{t['speaker']:>5}: {t['text']}[/dim]")
        if c.agent_last_utterance:
            console.print(
                Panel(c.agent_last_utterance, title="agent (last reply)", border_style="blue")
            )
        else:
            console.print("[dim](no agent reply yet: the customer opened the thread)[/dim]")
        console.print(Panel(c.user_partial, title="customer", border_style="yellow"))
        console.print(
            "Hand to a human?  [bold]y[/bold] yes   [bold]n[/bold] no   [bold]s[/bold] skip   "
            "[bold]u[/bold] undo   [bold]?[/bold] guidelines   [bold]q[/bold] quit"
        )
        key = _read_key()
        if key == "q":
            break
        if key == "?":
            console.clear()
            console.print(guidelines)
            click.pause()
            continue
        if key == "u":
            if history:
                labels.pop(history.pop(), None)
                labelling.save_labels(labels_path, labels, order)
            continue
        if key not in keys:
            continue
        labels[c.id] = labelling.label_record(c.id, keys[key], labeller)
        labelling.save_labels(labels_path, labels, order)
        history.append(c.id)
    console.print(f"labels saved to {labels_path}")


@data_app.command("escalate-relabel-sample")
def data_escalate_relabel_sample() -> None:
    """Draw the guideline-v2 relabel sample from the v1 eval sets (seeded, stratified; D-033)."""
    from floorcall.data import labelling

    s = get_settings()
    cands = labelling.load_candidates(s.paths.labels / ESCALATE_CANDIDATES)
    v1 = labelling.load_labels(s.paths.labels / ESCALATE_LABELS)
    sample = labelling.draw_relabel_sample(
        cands, v1, s.data.d4_relabel_sizes, s.data.d4_relabel_seed
    )
    path = s.paths.labels / labelling.RELABEL_SAMPLE_FILE
    wrote = labelling.save_relabel_sample(path, sample)
    console.print(f"seed {sample['seed']}")
    for split, cells in sample["allocation"].items():
        console.print(f"{split}: {cells} of {sample['population'][split]}")
    console.print(f"{'wrote' if wrote else 'unchanged:'} {path}")


def _blind_label(order: list[Any], labels_path: Path, labeller: str) -> None:
    """The blind labelling screen under guideline v2 (D-033, D-034): y, n, u undo, q quit.

    Shows `order` (candidates) one at a time with guideline v2 above each and the same context as
    the first pass. It reads no other label, and shows no split or stratum. Every keypress is saved
    to `labels_path`, and a rerun resumes at the first unlabelled message.
    """
    from rich.markup import escape
    from rich.panel import Panel
    from rich.text import Text

    from floorcall.data import labelling

    ids = [c.id for c in order]
    labels = labelling.load_labels(labels_path)
    history: list[str] = []
    keys = {"y": "true", "n": "false"}

    while True:
        pending = [c for c in order if c.id not in labels]
        if not pending:
            console.print(f"[green]All {len(order)} messages are labelled.[/green]")
            break
        c = pending[0]
        console.clear()
        console.print(
            Panel(Text(labelling.GUIDELINE_V2), title="guideline v2", border_style="green")
        )
        console.print(
            f"[bold]label[/bold]  {len(order) - len(pending)}/{len(order)} done   "
            f"company: {escape(c.company)}"
        )
        for t in c.recent_turns[-2:]:
            console.print(Text(f"{t['speaker']:>5}: {t['text']}", style="dim"))
        if c.agent_last_utterance:
            console.print(
                Panel(Text(c.agent_last_utterance), title="agent (last reply)", border_style="blue")
            )
        else:
            console.print("[dim](no agent reply yet: the customer opened the thread)[/dim]")
        console.print(Panel(Text(c.user_partial), title="customer", border_style="yellow"))
        console.print(
            "Hand to a human?  [bold]y[/bold] escalate   [bold]n[/bold] not escalate   "
            "[bold]u[/bold] undo   [bold]q[/bold] quit"
        )
        key = _read_key()
        if key == "q":
            break
        if key == "u":
            if history:
                labels.pop(history.pop(), None)
                labelling.save_labels(labels_path, labels, ids)
            continue
        if key not in keys:
            continue
        labels[c.id] = labelling.label_record(c.id, keys[key], labeller, guidelines="v2")
        labelling.save_labels(labels_path, labels, ids)
        history.append(c.id)
    console.print(f"labels saved to {labels_path}")


@label_app.command("escalate-relabel")
def label_escalate_relabel(
    labeller: str = typer.Option("yash", help="recorded on every label"),
) -> None:
    """Blind relabel of the D4 eval sample under guideline v2: y, n, u undo, q quit (D-033).

    The sample in its shuffled order, test and calib together. Earlier labels (Yash's v1 or any
    LLM's) are never read. Every keypress is saved to escalate.labels.v2.jsonl.
    """
    import json

    from floorcall.data import labelling

    s = get_settings()
    sample_path = s.paths.labels / labelling.RELABEL_SAMPLE_FILE
    if not sample_path.exists():
        raise typer.BadParameter(f"{sample_path} is missing: run data escalate-relabel-sample")
    order_ids: list[str] = json.loads(sample_path.read_text(encoding="utf-8"))["order"]
    by_id = {c.id: c for c in labelling.load_candidates(s.paths.labels / ESCALATE_CANDIDATES)}
    _blind_label([by_id[i] for i in order_ids], s.paths.labels / labelling.LABELS_V2_FILE, labeller)


@data_app.command("escalate-train-sample")
def data_escalate_train_sample() -> None:
    """Draw the train-pool messages Yash labels for the primary D4 model (seeded; D-034)."""
    from collections import Counter

    from floorcall.data import labelling
    from floorcall.data.build import TEST_VERSIONS, escalate_eval_rows
    from floorcall.data.d4_llm import train_pool
    from floorcall.data.download import fetch
    from floorcall.data.escalate import build_candidates, stratum

    s = get_settings()
    held_out = labelling.load_candidates(s.paths.labels / ESCALATE_CANDIDATES)
    candidates = build_candidates(
        fetch("twcs", s.paths.data_raw),
        seed=s.splits.seed,
        fractions=s.splits.model_dump(include={"train", "calib", "test"}),
        max_history=s.data.max_history_turns,
    )
    # the D-030 pool: train threads only, one message each, none sharing text with a held-out one
    pool, _ = train_pool(candidates, held_out, n=s.llm.d4_train_rows, seed=s.splits.seed)
    test, calib = escalate_eval_rows(s, TEST_VERSIONS["escalate"])
    eval_groups = {r["group"] for r in test + calib} | {c.group for c in held_out}
    sample = labelling.draw_train_sample(
        pool, eval_groups, n=s.data.d4_train_hand_rows, seed=s.data.d4_train_hand_seed
    )
    path = s.paths.labels / labelling.TRAIN_SAMPLE_FILE
    if path.exists():
        if [c.id for c in labelling.load_candidates(path)] != [c.id for c in sample]:
            raise typer.BadParameter(f"{path} holds a different sample; it is never redrawn")
        console.print(f"unchanged: {path}")
    else:
        labelling.save_candidates(path, sample)
        console.print(f"wrote {path}")
    console.print(
        f"seed {s.data.d4_train_hand_seed}: {len(sample)} of {len(pool)} pool messages, "
        f"{len({c.group for c in sample})} threads, none in test or calib; "
        f"strata {dict(Counter(stratum(c) for c in sample).most_common())}"
    )


@label_app.command("escalate-train")
def label_escalate_train(
    labeller: str = typer.Option("yash", help="recorded on every label"),
) -> None:
    """Label D4 training messages by hand under guideline v2: y, n, u, q (D-034). Unused.

    D-035 dropped hand-labelled training data before any was labelled; D4 trains on the llm_v3
    labels. The tool stays: same blind screen as the relabel, every keypress saved to
    escalate.train_labels.v2.jsonl, resumable.
    """
    from floorcall.data import labelling

    s = get_settings()
    sample_path = s.paths.labels / labelling.TRAIN_SAMPLE_FILE
    if not sample_path.exists():
        raise typer.BadParameter(f"{sample_path} is missing: run data escalate-train-sample")
    _blind_label(
        labelling.load_candidates(sample_path),
        s.paths.labels / labelling.TRAIN_LABELS_FILE,
        labeller,
    )


@data_app.command("freeze-escalate")
def data_freeze_escalate(
    guidelines: Annotated[
        str | None, typer.Option(help="v1 | v2; default: the version D4 is evaluated on")
    ] = None,
) -> None:
    """Freeze the hand-labelled D4 test set and write the D4 calib set.

    An older version only rebuilds its test set, which must match its frozen bytes.
    """
    from floorcall.data import build

    card = build.freeze_escalate(get_settings(), guidelines)
    console.print(build.summarize([card]))


@eval_app.command("stock-threshold")
def eval_stock_threshold(
    decision: str = typer.Option("escalate", help="a decision in threshold_decisions"),
    device: str = typer.Option("cuda", help="device for stock Laya"),
) -> None:
    """Table A: stock Laya with its threshold chosen on calib by macro-F1 (D-035)."""
    from floorcall.evaluate.baselines import run_stock_laya_threshold

    r = run_stock_laya_threshold(get_settings(), decision, device=device)
    m, c = r["metrics"], r["threshold_choice"]
    console.print(
        f"{decision}  stock_laya_threshold  theta {c['theta']:.3f} (calib macro-F1 "
        f"{c['macro_f1']:.3f}, {c['tied']} tied)  test acc {m['accuracy']:.3f}  "
        f"macroF1 {m['macro_f1']:.3f}  (n={m['n']})"
    )


@eval_app.command("baselines")
def eval_baselines(
    decision: str = typer.Option("all", help="turn_complete | barge_in | route | escalate | all"),
    device: str = typer.Option("cuda", help="device for stock Laya"),
    skip_laya: bool = typer.Option(False, help="majority class only"),
) -> None:
    """Majority-class and stock-Laya rows of Table A -> results/table_a/."""
    from floorcall.data.build import test_file
    from floorcall.evaluate import baselines

    s = get_settings()
    wanted = baselines.DECISIONS if decision == "all" else (decision,)
    for d in wanted:
        if not (s.paths.test_frozen / test_file(d)).exists():
            console.print(f"[yellow]skip {d}: no frozen test set yet[/yellow]")
            continue
        runs = [baselines.run_majority(s, d)]
        if not skip_laya:
            runs.append(baselines.run_stock_laya(s, d, device=device))
        for r in runs:
            m = r["metrics"]
            hard = "n/a" if m["hard_accuracy"] is None else f"{m['hard_accuracy']:.3f}"
            console.print(
                f"{d:14s} {r['model']:11s} acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
                f"ECE {m['ece']:.3f}  Brier {m['brier']:.3f}  hard {hard}  (n={m['n']})"
            )


@eval_app.command("latency")
def eval_latency(
    rows: str = typer.Option("gpu", help="gpu | cpu | all | a row name prefix"),
    compare_stock: bool = typer.Option(
        False, help="also measure the stock checkpoint, row by row (results/table_b/stock/)"
    ),
    precisions: str = typer.Option(
        "", help="e.g. fp32,bf16,fp16: the configured checkpoint at each precision (D-039)"
    ),
) -> None:
    """Table B: batch-1 decision latency under the thermal rules (D-037) -> results/table_b/.

    Each row waits for the GPU to cool, is sampled by nvidia-smi throughout, and is discarded and
    retried if the GPU throttled while it was timed. CPU rows run in chunks with cooldowns.
    """
    from floorcall.evaluate import latency

    chosen = tuple(p.strip() for p in precisions.split(",") if p.strip())
    for r in latency.run(
        get_settings(), rows, compare_stock=compare_stock, precisions=chosen, log=console.print
    ):
        t, tel = r["total_ms"], r["telemetry"]["timed"]
        temp, sm = tel["temperature_c"] or {}, tel["sm_clock_mhz"] or {}
        console.print(
            f"{r['model']:10s} {r['row']:22s} p50 {t['p50']:7.1f}  p95 {t['p95']:7.1f}  "
            f"p99 {t['p99']:7.1f} ms  fits {r['budget_p99_ms']:.0f} ms: "
            f"{'yes' if r['fits_budget'] else 'no'}  (GPU max {temp.get('max', float('nan')):.0f} C, "
            f"SM median {sm.get('median', float('nan')):.0f} MHz)"
        )


@train_app.command("run")
def train_run(
    out: Annotated[Path, typer.Option(help="new, empty directory for the checkpoint")],
    ablation: str = typer.Option("full", help="full | no_history | no_agent | no_normalize"),
    tasks: str = typer.Option("", help="comma list of decisions; default: every task with a quota"),
) -> None:
    """Fine-tune one multi-task checkpoint, then calibrate it. Refuses until all test sets are frozen."""
    from floorcall.config import with_ablation
    from floorcall.train.run import run_training

    settings = with_ablation(get_settings(), ablation)
    chosen = [t.strip() for t in tasks.split(",") if t.strip()] or None
    path = run_training(settings, out, tasks=chosen)
    console.print(f"checkpoint and calibration written to {path}")


@train_app.command("calibrate")
def train_calibrate(
    checkpoint: Annotated[Path, typer.Option(help="a floorcall training run directory")],
) -> None:
    """Refit per-decision temperatures on calib, under the state settings the run trained with."""
    from floorcall.config import with_ablation
    from floorcall.evaluate.checkpoint import trained_ablation
    from floorcall.train.run import calibrate

    settings = with_ablation(get_settings(), trained_ablation(checkpoint))
    cal = calibrate(settings, checkpoint)
    for d, t in cal.temperatures.items():
        flag = "  (at a bound: check it)" if cal.fits[d]["at_bound"] else ""
        console.print(f"{d:14s} T = {t:.3f}{flag}")


def _print_rows(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        m = r["metrics"]
        tag = r.get("ablation") or r.get("noise_level", r["model"])
        console.print(
            f"{r['decision']:14s} {tag!s:22s} acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
            f"ECE {m['ece']:.3f}  Brier {m['brier']:.3f}"
        )


@eval_app.command("checkpoint")
def eval_checkpoint(
    checkpoint: Annotated[Path, typer.Option(help="a floorcall training run directory")],
) -> None:
    """Table A's fine-tuned and fine-tuned + temperature rows."""
    from floorcall.evaluate.checkpoint import evaluate_table_a

    _print_rows(evaluate_table_a(get_settings(), checkpoint))


@eval_app.command("precision-parity")
def eval_precision_parity(
    checkpoint: Annotated[Path, typer.Option(help="a calibrated training run directory")],
) -> None:
    """D-039: do bf16 and fp16 forwards agree with fp32 on calib and dev (never test)?"""
    from floorcall.evaluate.precision import run_parity

    report = run_parity(get_settings(), checkpoint)
    for p, entry in report["precisions"].items():
        v = entry["verdict"]
        console.print(
            f"{p}: {'PASS' if v['passes'] else 'FAIL'}  worst argmax agreement "
            f"{v['worst_argmax_agreement']:.4%} ({v['worst_set']}), max |dp| "
            f"{v['max_abs_prob_diff']:.2e}"
        )
        for key, st in entry["sets"].items():
            extra = (
                f"  threshold agreement {st['threshold_agreement']:.4%}"
                if "threshold" in st
                else ""
            )
            console.print(
                f"   {key:22s} n {st['n']:6d}  argmax {st['argmax_agreement']:.4%} "
                f"({st['argmax_disagreements']} differ)  max |dp| {st['max_abs_prob_diff']:.2e}  "
                f"mean |dp| {st['mean_abs_prob_diff']:.2e}{extra}"
            )


@eval_app.command("recover-predictions")
def eval_recover_predictions(
    checkpoint: Annotated[Path, typer.Option(help="a floorcall training run directory")],
) -> None:
    """Inference only (D-036): recompute a checkpoint's test logits, check every committed Table A
    metric reproduces from them, and only then save them to runs/eval. Writes no results."""
    from floorcall.evaluate.checkpoint import recover_predictions

    diffs = recover_predictions(get_settings(), checkpoint)
    for decision, found in diffs.items():
        console.print(f"{decision}: {'identical' if not found else f'{len(found)} differences'}")
        for line in found:
            console.print(f"  {line}")
    if any(diffs.values()):
        raise typer.Exit(1)


@eval_app.command("paired")
def eval_paired(
    against: str = typer.Option(
        "stock", help="stock | baselines (the best cheap one per decision)"
    ),
) -> None:
    """Paired bootstrap of the fine-tuned model against stock Laya or the best cheap baseline, on
    the same test rows, from saved predictions (no re-scoring)."""
    from floorcall.evaluate.paired import run_paired

    for r in run_paired(get_settings(), against):
        d = r["differences"]
        console.print(
            f"{r['decision']:14s} {r['a']} - {r['b']}: "
            + "  ".join(
                f"{m} {v['difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                for m, v in d.items()
            )
        )


@eval_app.command("cheap-baselines")
def eval_cheap_baselines() -> None:
    """Sanity baselines (D-036): TF-IDF + logistic regression for every decision, and a lexical
    rule for D2. Trained on train minus the dev split, C chosen on dev, scored once on test."""
    from floorcall.evaluate.cheap_baselines import run_lexical_rule, run_tfidf_lr
    from floorcall.provenance import git_head

    s, code = get_settings(), git_head()
    rows = [run_tfidf_lr(s, d, code) for d in ("turn_complete", "barge_in", "route", "escalate")]
    _print_rows([*rows, run_lexical_rule(s, code)])


@eval_app.command("robustness")
def eval_robustness(
    checkpoint: Annotated[Path, typer.Option(help="a floorcall training run directory")],
) -> None:
    """Table C: the calibrated model on ASR-degraded test text."""
    from floorcall.evaluate.checkpoint import evaluate_table_c

    _print_rows(evaluate_table_c(get_settings(), checkpoint))


@eval_app.command("ablation")
def eval_ablation(
    checkpoint: Annotated[Path, typer.Option(help="a floorcall training run directory")],
) -> None:
    """Table D: an ablation checkpoint, scored under the state settings it trained with."""
    from floorcall.evaluate.checkpoint import evaluate_table_d

    _print_rows(evaluate_table_d(get_settings(), checkpoint))


@eval_app.command("curves")
def eval_curves(
    checkpoint: Annotated[
        Path | None, typer.Option(help="a calibrated training run; omit for stock Laya")
    ] = None,
) -> None:
    """Operating points chosen on calib, tradeoff curves and reliability on test -> results/curves/."""
    from floorcall.evaluate.operating_points import run_curves

    out = run_curves(get_settings(), checkpoint=checkpoint)
    for key in ("interrupt", "yield"):
        if key in out:
            console.print(f"{key}: theta {out[key]['theta']:.3f}  test {out[key]['test_at_theta']}")


@eval_app.command("oos-threshold")
def eval_oos_threshold(
    checkpoint: Annotated[Path, typer.Option(help="the calibrated, served training run")],
    device: str = typer.Option("cpu", help="cpu | cuda"),
) -> None:
    """D-045: choose theta_oos on D3 calib (never test) -> results/thresholds/route_oos.json."""
    from floorcall.evaluate.oos_threshold import run_oos_threshold

    out = run_oos_threshold(get_settings(), checkpoint, device=device)
    c = out["choice"]
    console.print(
        f"theta_oos {c['theta']:.3f}  calib macro-F1 {c['macro_f1']:.3f} (n={c['n']}, "
        f"{c['tied']} tied; at 0.5: {out['calib_macro_f1_at_0.5']:.3f})"
    )


@eval_app.command("figures")
def eval_figures() -> None:
    """Render results/curves/*.json to light and dark PNGs in results/figures/."""
    from floorcall.evaluate.figures import render_all

    for path in render_all():
        console.print(f"wrote {path}")


@data_app.command("escalate-llm-label")
def data_escalate_llm_label(
    split: Annotated[str, typer.Option(help="calib | test | train")],
) -> None:
    """Label D4 messages with the LLM labeller. calib: agreement with Yash, and the gate the
    prompt must pass. test (measurement only) runs only with a prompt calib accepted. train: the
    D-030 pool labelled by prompt v3, D4's training rows (D-035), source=llm_v3."""
    from floorcall.data.d4_llm import run

    out = run(get_settings(), split)
    console.print_json(
        data={k: v for k, v in out.items() if k not in ("agreement", "disagreements")}
    )
    if "agreement" in out:
        a = out["agreement"]
        console.print(
            f"agreement with Yash on {split}: n={a['n']}  accuracy {a['accuracy']:.3f}  "
            f"kappa {a['cohen_kappa']:.3f}  escalate precision {a['escalate_precision']:.3f}  "
            f"recall {a['escalate_recall']:.3f}  (LLM unsure on {a['llm_unsure']}; "
            f"{len(out['disagreements'])} disagreements in the results file)"
        )


@eval_app.command("llm-estimate")
def eval_llm_estimate() -> None:
    """D-041: project the prompted-LLM baseline's spend from a pilot on calib rows (not test)."""
    from floorcall.evaluate.llm_baseline import estimate_spend

    console.print_json(data=estimate_spend(get_settings()))


@eval_app.command("llm-baseline")
def eval_llm_baseline() -> None:
    """Table A: the prompted-LLM baseline on every decision (D1/D2 subsampled)."""
    from floorcall.evaluate.llm_baseline import run_table_a

    for r in run_table_a(get_settings()):
        m = r["metrics"]
        console.print(
            f"{r['decision']:14s} n={m['n']:5d}  acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  "
            f"ECE {m['ece']:.3f}  invalid {r['invalid_answers']}  ${r['cost_usd']:.4f}"
        )


@eval_app.command("llm-recover")
def eval_llm_recover() -> None:
    """D-041: the prompted-LLM rows' per-row answers, from the response cache only (no API)."""
    from floorcall.evaluate.llm_baseline import recover_table_a

    for decision, found in recover_table_a(get_settings()).items():
        console.print(f"{decision}: {found if found else 'identical'}")


@eval_app.command("llm-latency")
def eval_llm_latency() -> None:
    """Table B: the prompted LLM, end to end with the network, user_pause event."""
    from floorcall.evaluate.llm_baseline import run_latency

    r = run_latency(get_settings())
    t = r["total_ms"]
    console.print(
        f"prompted_llm p50 {t['p50']:.1f}  p95 {t['p95']:.1f}  p99 {t['p99']:.1f} ms  "
        f"failed {r['failed_calls']}  ${r['cost_usd']:.4f}"
    )


@llm_app.command("spend")
def llm_spend() -> None:
    """LLM spend so far, by purpose and provider, against the $4.75 stop. Needs no key."""
    from floorcall.config import REPO_ROOT
    from floorcall.llm.client import Ledger

    console.print_json(
        data=Ledger(REPO_ROOT / "runs" / "llm" / "ledger.sqlite", get_settings().llm).summary()
    )


def _script_paths(given: list[Path] | None, default: Path) -> list[Path]:
    """Script paths; a glob the shell did not expand (PowerShell) is expanded here."""
    raw = [str(p) for p in given] if given else [str(default / "*.json")]
    out: list[Path] = []
    for item in raw:
        out += (
            sorted(Path(item).parent.glob(Path(item).name))
            if any(c in item for c in "*?[")
            else [Path(item)]
        )
    return out


@app.command("replay")
def replay(
    scripts: Annotated[
        list[Path] | None, typer.Argument(help="script files (default: demo/scripts/*.json)")
    ] = None,
    compare: str = typer.Option("naive", help="the baseline agent to compare with: naive"),
    check: bool = typer.Option(False, "--check", help="check the scripts' timing only; no model"),
    freeze: bool = typer.Option(False, "--freeze", help="check, then freeze the scripts' hashes"),
    checkpoint: Annotated[
        Path | None, typer.Option(help="default: ReplaySettings.checkpoint")
    ] = None,
    device: Annotated[str | None, typer.Option(help="default: ReplaySettings.device (cpu)")] = None,
) -> None:
    """Replay scripted calls through the naive agent and floorcall; print where they diverge.

    Needs no API key. The scripts are illustrative demos, not an evaluation set (D-046)."""
    from floorcall.replay.run import run_replay

    if compare != "naive":
        raise typer.BadParameter("the only baseline agent is naive")
    settings = get_settings()
    paths = _script_paths(scripts, settings.replay.scripts)
    lines, ok = run_replay(
        settings,
        paths,
        check_only=check or freeze,
        freeze=freeze,
        checkpoint=checkpoint,
        device=device,
    )
    for line in lines:
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if not ok:
        raise typer.Exit(1)


@eval_app.command("readme")
def eval_readme() -> None:
    """Rewrite README.md's result tables from results/ (TODO where no file exists)."""
    from floorcall.evaluate.report import write_readme

    changed = write_readme()
    console.print("README.md updated" if changed else "README.md already up to date")


if __name__ == "__main__":
    app()
