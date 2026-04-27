"""Commands to run workflows"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="Run and inspect Agent Workflows.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()