"""Commands to run travel-planning workflows."""

from __future__ import annotations

from pathlib import Path

import typer
from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
import json

from travelplanner.utils.imports import load_callable

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
            "Choose a workflow, then describe the trip you want help planning.",
            border_style="blue",
        )
    )
    _render_workflow_table(workflows)

    selected_workflow = inquirer.select(
        message="Choose a workflow",
        choices=[
            {
                "name": f"{workflow['name']} ({workflow['module']})",
                "value": workflow,
            }
            for workflow in workflows
        ],
    ).execute()
    workflow_builder = load_callable(f"{selected_workflow["module"]}:make_graph")
    compiled_workflow = workflow_builder()
    
    console.print("[bold green]Successfully loaded workflow[/bold green]")
    console.print(
      Panel(
          JSON(json.dumps(selected_workflow, indent=2)),
          title="Workflow",
          border_style="cyan",
      )
    )

    query = inquirer.text(
        message="Please tell me what you want help with:",
        validate=lambda value: bool(value.strip()),
        invalid_message="Please enter a travel query.",
    ).execute()

    

    console.print(
        Panel.fit
        (
            f"[bold green]Workflow selected:[/bold green] {selected_workflow['name']}\n"
            f"[bold green]Travel query:[/bold green] {query.strip()}\n\n"
            "[yellow]Execution is not wired yet. Stopping after input collection.[/yellow]",
            title="Run Summary",
            border_style="green",
        )
    )
