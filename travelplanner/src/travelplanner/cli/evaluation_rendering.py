"""Rich rendering helpers for evaluation CLI output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from rich.console import Console, Group
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from travelplanner.schema.system_state import StateContractModel


def normalize_for_display(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [normalize_for_display(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_for_display(item) for key, item in value.items()}
    return value


def render_evaluation_settings(
    console: Console,
    *,
    name: str,
    dataset: str,
    workflow_label: str,
    graph_ref: str,
    limit: int,
    record_count: int,
) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("[bold]Run[/bold]", name)
    table.add_row("[bold]Dataset[/bold]", dataset)
    table.add_row("[bold]Workflow[/bold]", workflow_label)
    table.add_row("[bold]Graph[/bold]", graph_ref)
    table.add_row("[bold]Limit[/bold]", str(limit))
    table.add_row("[bold]Records[/bold]", str(record_count))
    table.add_row("[bold]Status[/bold]", "configured")

    console.print(
        Panel.fit(
            table,
            border_style="cyan",
            title="[bold cyan]Travel Planner[/bold cyan] Evaluation Settings",
        )
    )


def print_record_start(
    console: Console,
    *,
    index: int,
    record_id: str,
    graph_query: str,
) -> None:
    console.print(
        Panel.fit(
            f"[bold]Record[/bold] {record_id}\n[bold]Query[/bold] {graph_query}",
            border_style="magenta",
            title=f"[bold magenta]Evaluation Item {index}[/bold magenta]",
        )
    )


def state_summary(state: StateContractModel) -> str:
    timetable_status = "set" if state.timetable is not None else "empty"
    return (
        f"constraints={len(state.constraint_list)} | "
        f"tasks={len(state.task_list)} | "
        f"histories={len(state.message_histories)} | "
        f"artifacts={len(state.agent_artifacts)} | "
        f"timetable={timetable_status}"
    )


def print_state_update(
    console: Console,
    *,
    record_id: str,
    step: int,
    node_name: str,
    update_payload: dict[str, Any],
    previous_state: StateContractModel,
    current_state: StateContractModel,
) -> None:
    detail_grid = Table.grid(expand=True, padding=(0, 1))
    detail_grid.add_column(style="bold cyan", width=26)
    detail_grid.add_column(ratio=1)

    for field_name, current_value in update_payload.items():
        previous_value = getattr(previous_state, field_name, None)
        detail_grid.add_row(
            _field_change_label(field_name, previous_value, current_value),
            _render_state_field(field_name, current_value, previous_value),
        )

    console.print(
        Panel(
            Group(detail_grid),
            title=f"[bold cyan]{record_id}[/bold cyan] Step {step}: {node_name}",
            subtitle=state_summary(current_state),
            border_style="blue",
        )
    )


def _truncate_text(value: str, limit: int = 100) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _render_constraints(constraints: list[dict[str, Any]]) -> Table | str:
    if not constraints:
        return "[dim]No constraints[/dim]"

    table = Table(box=None, pad_edge=False)
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Skipped", no_wrap=True)
    table.add_column("Text", overflow="fold")
    for constraint in constraints:
        table.add_row(
            str(constraint.get("type", "-")),
            "yes" if constraint.get("user_skipped") else "no",
            str(constraint.get("text", "-")),
        )
    return table


def _render_tasks(tasks: list[dict[str, Any]]) -> Table | str:
    if not tasks:
        return "[dim]No tasks[/dim]"

    table = Table(box=None, pad_edge=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Valid", no_wrap=True)
    table.add_column("Text", overflow="fold")
    table.add_column("Comment", overflow="fold")
    for task in tasks:
        table.add_row(
            str(task.get("name", "-")),
            str(task.get("type", "-")),
            "yes" if task.get("is_valid") else "no",
            str(task.get("text", "-")),
            str(task.get("validation_comment") or "-"),
        )
    return table


def _render_message_histories(
    current_histories: dict[str, dict[str, Any]],
    previous_histories: dict[str, dict[str, Any]],
) -> Table | str:
    changed_keys = [
        key
        for key, value in current_histories.items()
        if previous_histories.get(key) != value
    ]
    if not changed_keys:
        return "[dim]No message history changes[/dim]"

    table = Table(box=None, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Agent", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Messages", no_wrap=True)
    table.add_column("Last Message", overflow="fold")
    for key in changed_keys:
        history = current_histories[key]
        messages = history.get("messages") or []
        last_message = messages[-1].get("content", "") if messages else ""
        table.add_row(
            key,
            str(history.get("user_agent") or "-"),
            str(history.get("model") or "-"),
            str(len(messages)),
            _truncate_text(str(last_message), limit=120) if last_message else "-",
        )
    return table


def _render_agent_artifacts(
    current_artifacts: dict[str, list[dict[str, Any]]],
    previous_artifacts: dict[str, list[dict[str, Any]]],
) -> Table | str:
    changed_keys = [
        key
        for key, value in current_artifacts.items()
        if previous_artifacts.get(key) != value
    ]
    if not changed_keys:
        return "[dim]No artifact changes[/dim]"

    table = Table(box=None, pad_edge=False)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Count", no_wrap=True)
    table.add_column("Latest", overflow="fold")
    for key in changed_keys:
        artifacts = current_artifacts.get(key) or []
        latest_name = artifacts[-1].get("name", "-") if artifacts else "-"
        table.add_row(key, str(len(artifacts)), str(latest_name))
    return table


def _render_state_field(
    field_name: str,
    current_value: Any,
    previous_value: Any,
) -> Table | JSON | str:
    if field_name == "constraint_list":
        return _render_constraints(current_value)
    if field_name == "task_list":
        return _render_tasks(current_value)
    if field_name == "message_histories":
        return _render_message_histories(current_value, previous_value or {})
    if field_name == "agent_artifacts":
        return _render_agent_artifacts(current_value, previous_value or {})
    if field_name == "query":
        return str(current_value)
    return JSON.from_data(current_value)


def _field_change_label(
    field_name: str, previous_value: Any, current_value: Any
) -> str:
    if isinstance(current_value, list):
        previous_count = len(previous_value or [])
        return f"{field_name} ({previous_count} -> {len(current_value)} items)"
    if isinstance(current_value, dict):
        previous_count = len(previous_value or {})
        return f"{field_name} ({previous_count} -> {len(current_value)} entries)"
    if previous_value is None and current_value is not None:
        return f"{field_name} (set)"
    return field_name
