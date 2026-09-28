"""Command-line entry point: `uv run floorcall --help`."""

from __future__ import annotations

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
app.add_typer(data_app, name="data")
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


@data_app.command("freeze-escalate")
def data_freeze_escalate() -> None:
    """Freeze the hand-labelled D4 test set and write the D4 calib set."""
    from floorcall.data import build

    card = build.freeze_escalate(get_settings())
    console.print(build.summarize([card]))


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


@eval_app.command("readme")
def eval_readme() -> None:
    """Rewrite README.md's result tables from results/ (TODO where no file exists)."""
    from floorcall.evaluate.report import write_readme

    changed = write_readme()
    console.print("README.md updated" if changed else "README.md already up to date")


if __name__ == "__main__":
    app()
