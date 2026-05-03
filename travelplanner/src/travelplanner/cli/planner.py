"""Commands to run travel-planning workflows."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .interactive import run_planner_dashboard

app = typer.Typer(
    help="Run the travel planner app.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"


def _discover_workflows() -> list[dict[str, str]]:
    workflows: list[dict[str, str]] = []
    for workflow_file in sorted(WORKFLOWS_DIR.glob("*.py")):
        if workflow_file.stem == "__init__":
            continue

        label = workflow_file.stem.replace("_", "-")
        workflows.append(
            {
                "name": label,
                "module": f"travelplanner.workflows.{workflow_file.stem}",
            }
        )

    return workflows


def _render_workflow_table(workflows: list[dict[str, str]]) -> None:
    table = Table(title="Available Workflows")
    table.add_column("Workflow", style="cyan", no_wrap=True)
    table.add_column("Module", style="dim")

    for workflow in workflows:
        table.add_row(workflow["name"], workflow["module"])

    console.print(table)


@app.command()
def run() -> None:
    """Start an interactive TravelPlanner run."""
    workflows = _discover_workflows()
    if not workflows:
        raise typer.BadParameter(f"No workflows were found in {WORKFLOWS_DIR}.")

    console.print(
        Panel.fit(
            "[bold cyan]TravelPlanner[/bold cyan]\n"
            "The dashboard will ask for the workflow and trip details.",
            border_style="blue",
        )
    )

    run_planner_dashboard(workflows)


    print("Finished")
