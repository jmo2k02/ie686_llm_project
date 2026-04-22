"""Evaluation commands for the TravelPlanner CLI."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="Run and inspect evaluation jobs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def run(
    name: str = typer.Option("baseline", "--name", "-n", help="Evaluation run name."),
    dataset: str = typer.Option("sample", "--dataset", "-d", help="Dataset label."),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, help="Number of items to evaluate."
    ),
) -> None:
    """Run a simple evaluation command."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("[bold]Run[/bold]", name)
    table.add_row("[bold]Dataset[/bold]", dataset)
    table.add_row("[bold]Limit[/bold]", str(limit))
    table.add_row("[bold]Status[/bold]", "configured")

    console.print(
        Panel.fit(
            table,
            border_style="cyan",
            title="Evaluation",
        )
    )
