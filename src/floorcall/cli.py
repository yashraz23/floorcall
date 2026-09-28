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
app.add_typer(data_app, name="data")
console = Console()


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


if __name__ == "__main__":
    app()
