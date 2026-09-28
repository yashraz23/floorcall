"""Command-line entry point: `uv run floorcall --help`."""

import typer

app = typer.Typer(
    help="floorcall: turn-taking, barge-in, routing and escalation decisions for voice agents."
)


@app.callback()
def main() -> None:
    """floorcall command line."""


if __name__ == "__main__":
    app()
