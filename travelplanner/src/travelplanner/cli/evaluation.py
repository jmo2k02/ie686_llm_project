"""Evaluation commands for the TravelPlanner CLI."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langgraph.graph.state import CompiledStateGraph

from travelplanner.evaluation import AVAILABLE_DATASETS, AVAILABLE_WORKFLOWS
from travelplanner.utils.imports import load_callable, load_object

app = typer.Typer(
    help="Run and inspect evaluation jobs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_dataset_path(dataset: str) -> Path:
    dataset_path = AVAILABLE_DATASETS.get(dataset, dataset)
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / dataset_path
    if not path.exists():
        raise typer.BadParameter(f"Dataset path does not exist: {path}")
    return path


def _load_dataset_records(dataset: str, limit: int) -> list[dict[str, Any]]:
    dataset_path = _resolve_dataset_path(dataset)
    payload = dataset_path.read_text(encoding="utf-8").strip()
    if dataset_path.suffix == ".jsonl":
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    else:
        records = json.loads(payload)
    if not isinstance(records, list):
        raise typer.BadParameter("Datasets must deserialize into a list of records.")
    return records[:limit]


def _call_with_supported_kwargs(target: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(target)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if accepts_kwargs or name in signature.parameters
    }
    return target(**supported_kwargs)


def _resolve_graph_reference(
    workflow: str | None, graph: str | None
) -> tuple[str, str]:
    if graph:
        return graph, graph

    workflow_label = workflow or "minimal"
    if workflow_label not in AVAILABLE_WORKFLOWS:
        raise typer.BadParameter(
            f"Unknown workflow '{workflow_label}'. Choose from: {', '.join(sorted(AVAILABLE_WORKFLOWS))}"
        )
    return workflow_label, AVAILABLE_WORKFLOWS[workflow_label]["graph"]


def _compile_graph(graph_ref: str) -> CompiledStateGraph:
    graph_target = load_callable(graph_ref)
    if hasattr(graph_target, "invoke"):
        return graph_target # type: ignore

    compiled_graph = _call_with_supported_kwargs(
        graph_target
    )
    if hasattr(compiled_graph, "compile"):
        compiled_graph = compiled_graph.compile()
    if not hasattr(compiled_graph, "invoke"):
        raise TypeError(
            f"{graph_ref} did not return a compiled graph with an invoke(...) method"
        )
    return compiled_graph


@app.command()
def list_workflows():
    """List available workflows for evaluation"""
    console.print(AVAILABLE_WORKFLOWS.keys())
    return


@app.command()
def list_datasets():
    """List available datasets for evaluation"""
    console.print(AVAILABLE_DATASETS.keys())
    return


@app.command()
def run(
    name: str = typer.Option("baseline", "--name", "-n", help="Evaluation run name."),
    dataset: str = typer.Option(
        "travel_queries", "--dataset", "-d", help="Dataset label or JSON/JSONL path."
    ),
    workflow: str | None = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Workflow label from the registry.",
    ),
    graph: str | None = typer.Option(
        None,
        "--graph",
        "-g",
        help="Import path to a compiled graph or graph builder, for example pkg.module:make_graph.",
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", min=1, help="Number of items to evaluate."
    ),
) -> None:
    """Run a simple evaluation command."""
    workflow_label, graph_ref = _resolve_graph_reference(workflow, graph)
    dataset_records = _load_dataset_records(dataset, limit)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("[bold]Run[/bold]", name)
    table.add_row("[bold]Dataset[/bold]", dataset)
    table.add_row("[bold]Workflow[/bold]", workflow_label)
    table.add_row("[bold]Graph[/bold]", graph_ref)
    table.add_row("[bold]Limit[/bold]", str(limit))
    table.add_row("[bold]Records[/bold]", str(len(dataset_records)))
    table.add_row("[bold]Status[/bold]", "configured")

    console.print(
        Panel.fit(
            table,
            border_style="cyan",
            title="[bold cyan]Travel Planner[/bold cyan] Evaluation Settings",
        )
    )
    if not inquirer.confirm(
        message="Do you want to start evaluation with these settings?"
    ).execute():
        console.print("[bold red]Stopped evaluation[/bold red]")
        return

    compiled_graph = _compile_graph(graph_ref)
    console.print("[green]Successfully compiled graph[/green]")

    for record in tqdm(dataset_records, desc="Processing Dataset Records", total=len(dataset_records)):
        graph_query = record["query"]
        result = compiled_graph.invoke({
          "query":graph_query
        })
        console.print(f"Processed [bold]{record.get('id', 'unknown')}[/bold]")
        print(result)
        break


    console.print(f"Finished evaluation run for {len(dataset_records)} record(s)")
