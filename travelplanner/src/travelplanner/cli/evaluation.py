"""Evaluation commands for the TravelPlanner CLI."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

load_dotenv()
from InquirerPy import inquirer
from rich.console import Console
from langgraph.graph.state import CompiledStateGraph

from travelplanner.cli.evaluation_rendering import (
    normalize_for_display,
    print_record_start,
    print_state_update,
    render_evaluation_settings,
    state_summary,
)
from travelplanner.schema.system_state import StateContractModel

from travelplanner.evaluation import AVAILABLE_DATASETS, AVAILABLE_WORKFLOWS
from travelplanner.utils.imports import load_callable

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
        return graph_target  # type: ignore

    compiled_graph = _call_with_supported_kwargs(graph_target)
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

    render_evaluation_settings(
        console,
        name=name,
        dataset=dataset,
        workflow_label=workflow_label,
        graph_ref=graph_ref,
        limit=limit,
        record_count=len(dataset_records),
    )
    if not inquirer.confirm(
        message="Do you want to start evaluation with these settings?"
    ).execute():
        console.print("[bold red]Stopped evaluation[/bold red]")
        return

    compiled_graph = _compile_graph(graph_ref)
    console.print(
        f"[green]Successfully compiled graph[/green] [cyan]{graph_ref}[/cyan]"
    )
    console.print(f"[green]Starting evaluation on dataset[/green] [cyan]{dataset}[/cyan]")
    for index, record in enumerate(dataset_records, start=1):
        record_id = str(record.get("id", f"record-{index}"))
        graph_query = record["query"]

        print_record_start(
            console,
            index=index,
            record_id=record_id,
            graph_query=graph_query,
        )

        state_snapshot = StateContractModel(query=graph_query)
        step = 0
        for streamed_update in compiled_graph.stream(
            {"query": graph_query},
            stream_mode="updates",
        ):
            for node_name, node_update in streamed_update.items():
                step += 1
                normalized_update = normalize_for_display(node_update)
                merged_state = state_snapshot.model_dump(mode="json")
                merged_state.update(normalized_update)
                next_state = StateContractModel.model_validate(merged_state)
                print_state_update(
                    console,
                    record_id=record_id,
                    step=step,
                    node_name=node_name,
                    update_payload=normalized_update,
                    previous_state=state_snapshot,
                    current_state=next_state,
                )
                state_snapshot = next_state

        console.print(
            f"Processed [bold]{record_id}[/bold] with final state: [dim]{state_summary(state_snapshot)}[/dim]"
        )

    console.print(f"Finished evaluation run for {len(dataset_records)} record(s)")
