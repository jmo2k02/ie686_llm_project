"""Interactive dashboard for TravelPlanner workflows."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import typer
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from travelplanner.schema.system_state import ConstraintModel
from travelplanner.utils.checkpoint import make_memory_checkpointer
from travelplanner.utils.imports import load_callable
from travelplanner.utils.runtime_monitor import reset_run_monitor, set_run_monitor

from .evaluation_rendering import normalize_for_display

app = typer.Typer(
    help="Run an interactive dashboard for workflow execution.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

_AGENT_LABELS: dict[str, tuple[str, str]] = {
    "constraint_agent": ("Constraints", "Constraint Agent"),
    "planner_agent": ("Planning", "Planner Agent"),
    "reviewer_agent": ("Review", "Reviewer Agent"),
    "general_web_search_agent": ("Search", "General Web Search"),
}


@dataclass(frozen=True)
class _AgentRow:
    key: str
    team: str
    label: str


class DashboardState:
    def __init__(
        self,
        travel_query: str = "",
        workflow_name: str = "Not selected",
        agents: list[_AgentRow] | None = None,
    ) -> None:
        self.travel_query = travel_query
        self.workflow_name = workflow_name
        self.agents = list(agents or [])
        self.agent_status = {agent.key: "pending" for agent in self.agents}
        self.messages: deque[tuple[str, str, str]] = deque(maxlen=200)
        self.tool_calls: deque[tuple[str, str, str]] = deque(maxlen=200)
        self.status_text = "Waiting to start workflow execution."
        self.prompt_title = "Input"
        self.prompt_text = "Questions will appear here before each prompt."
        self.last_input = ""
        self.hard_constraints: list[ConstraintModel | dict[str, Any]] = []
        self.llm_calls = 0
        self.tool_call_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.start_time = time.time()

    def add_message(self, message_type: str, content: str) -> None:
        text = str(content).strip()
        if not text:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, text))

    def add_tool_call(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, _format_tool_args(args or {})))

    def set_status(self, text: str) -> None:
        self.status_text = str(text).strip() or self.status_text

    def set_prompt(self, title: str, text: str) -> None:
        self.prompt_title = title
        self.prompt_text = text.strip() or self.prompt_text

    def clear_prompt(self) -> None:
        self.prompt_title = "Input"
        self.prompt_text = "Questions will appear here before each prompt."

    def set_last_input(self, value: str) -> None:
        self.last_input = value.strip()

    def set_workflow_name(self, workflow_name: str) -> None:
        self.workflow_name = workflow_name

    def set_travel_query(self, travel_query: str) -> None:
        self.travel_query = travel_query

    def set_agents(self, agents: list[_AgentRow]) -> None:
        self.agents = agents
        self.agent_status = {agent.key: "pending" for agent in agents}

    def set_hard_constraints(
        self,
        constraints: list[ConstraintModel | dict[str, Any]],
    ) -> None:
        self.hard_constraints = constraints

    def record_llm_call(
        self,
        *,
        model_name: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        self.llm_calls += 1
        self.tokens_in += max(tokens_in, 0)
        self.tokens_out += max(tokens_out, 0)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> None:
        self.tool_call_count += 1
        self.add_tool_call(tool_name, args)

    def mark_started(self, agent_key: str) -> None:
        if agent_key in self.agent_status and self.agent_status[agent_key] == "pending":
            self.agent_status[agent_key] = "in_progress"

    def mark_completed(self, agent_key: str) -> None:
        if agent_key not in self.agent_status:
            return
        self.agent_status[agent_key] = "completed"
        next_agent = self._next_pending_agent()
        if next_agent is not None:
            self.agent_status[next_agent] = "in_progress"

    def summarize_update(self, agent_key: str, update: dict[str, Any]) -> None:
        if agent_key == "constraint_agent":
            constraint_count = len(update.get("constraint_list") or [])
            self.set_status(
                f"Constraint agent finalized {constraint_count} constraints."
            )
            self.add_message(
                "Agent",
                f"Constraint agent finalized {constraint_count} constraints.",
            )
            return

        if agent_key == "planner_agent":
            task_count = len(update.get("task_list") or [])
            self.set_status(f"Planner agent proposed {task_count} tasks.")
            self.add_message("Agent", f"Planner agent proposed {task_count} tasks.")
            return

        if agent_key == "reviewer_agent":
            task_count = len(update.get("task_list") or [])
            self.set_status(f"Reviewer agent approved {task_count} tasks.")
            self.add_message("Agent", f"Reviewer agent approved {task_count} tasks.")
            return

        if agent_key == "general_web_search_agent":
            artifact_groups = update.get("agent_artifacts") or {}
            artifact_count = sum(len(items) for items in artifact_groups.values())
            self.set_status(
                f"General web search completed with {artifact_count} artifact(s)."
            )
            self.add_message(
                "Agent",
                f"General web search completed with {artifact_count} artifact(s).",
            )
            return

        self.set_status(f"Completed {agent_key}.")
        self.add_message("Agent", f"Completed {agent_key}.")

    def _next_pending_agent(self) -> str | None:
        for agent in self.agents:
            if self.agent_status.get(agent.key) == "pending":
                return agent.key
        return None


def _get_agent_rows(compiled_workflow: CompiledStateGraph) -> list[_AgentRow]:
    rows: list[_AgentRow] = []
    for node_name in compiled_workflow.nodes:
        if node_name.startswith("__"):
            continue
        team, label = _AGENT_LABELS.get(
            node_name,
            ("Workflow", node_name.replace("_", " ").title()),
        )
        rows.append(_AgentRow(node_name, team, label))
    return rows


def _format_tool_args(args: dict[str, Any], max_length: int = 120) -> str:
    result = ", ".join(f"{key}={value}" for key, value in args.items()) or "-"
    if len(result) > max_length:
        return result[: max_length - 3] + "..."
    return result


def _format_tokens(token_count: int) -> str:
    if token_count >= 1000:
        return f"{token_count / 1000:.1f}k"
    return str(token_count)


def _constraint_text(constraint: ConstraintModel | dict[str, Any]) -> str:
    if isinstance(constraint, ConstraintModel):
        return constraint.text
    if isinstance(constraint, dict):
        return str(constraint.get("text", constraint)).strip()
    return str(constraint).strip()


def _snapshot_has_hard_constraints(snapshot: Any) -> bool:
    return any(
        getattr(task, "state", None) and task.state.values.get("hard_constraints")
        for task in getattr(snapshot, "tasks", [])
        if getattr(task, "name", None) == "constraint_agent"
    )


def _get_snapshot_hard_constraints(
    snapshot: Any,
) -> list[ConstraintModel | dict[str, Any]]:
    return next(
        (
            task.state.values.get("hard_constraints")
            for task in getattr(snapshot, "tasks", [])
            if getattr(task, "name", None) == "constraint_agent"
            and getattr(task, "state", None)
        ),
        [],
    )



def _build_layout(dashboard: DashboardState) -> Layout:
    terminal_height = console.size.height
    reserved_prompt_space = 6
    dashboard_height = min(
        max(20, int(terminal_height * 0.8)),
        max(12, terminal_height - reserved_prompt_space),
    )
    layout = Layout(size=dashboard_height)
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3),
        Layout(name="status", ratio=1),
        Layout(name="input", ratio=3),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="messages", ratio=3),
    )

    layout["header"].update(
        Panel(
            "[bold green]TravelPlanner CLI[/bold green]\n"
            f"[dim]Workflow: {dashboard.workflow_name}[/dim]",
            title="TravelPlanner",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAD,
        padding=(0, 2),
        expand=True,
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=18)
    progress_table.add_column("Agent", style="green", justify="center", width=24)
    progress_table.add_column("Status", style="yellow", justify="center", width=16)
    for agent in dashboard.agents:
        status = dashboard.agent_status.get(agent.key, "pending")
        if status == "in_progress":
            status_cell: Spinner | str = Spinner(
                "dots",
                text="[blue]in_progress[/blue]",
                style="bold cyan",
            )
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(agent.team, agent.label, status_cell)
    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        expand=True,
        box=box.MINIMAL,
        show_lines=True,
        padding=(0, 1),
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column("Content", style="white", ratio=1)
    combined: list[tuple[str, str, str]] = []
    combined.extend(
        (timestamp, "Tool", f"{name}: {args}")
        for timestamp, name, args in dashboard.tool_calls
    )
    combined.extend(dashboard.messages)
    combined.sort(key=lambda item: item[0], reverse=True)
    for timestamp, message_type, content in combined[:12]:
        content_text = content if len(content) <= 220 else content[:217] + "..."
        messages_table.add_row(
            timestamp,
            message_type,
            Text(content_text, overflow="fold"),
        )
    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    status_lines = [
        # dashboard.status_text,
        f"[bold cyan]Selected Workflow:[/bold cyan] {f'[green]{dashboard.workflow_name}[/green]' or '-'}",
        f"[bold cyan]Original Query:[/bold cyan] {f'[green]{dashboard.travel_query}[/green]' or '-'}",
    ]
    if dashboard.hard_constraints:
        status_lines.extend(
            [
                "",
                f"[bold green]Hard constraints:[/bold green] {len(dashboard.hard_constraints)}",
                "".join([
                    f"[green]{idx+1}[/green]:<{_constraint_text(constraint)}/>, "
                    for idx, constraint in enumerate(dashboard.hard_constraints)
                ]),
            ]
        )
    layout["status"].update(
        Panel(
            Text.from_markup("\n".join(status_lines), overflow="fold"),
            title="Status",
            border_style="green",
            padding=(1, 2),
        )
    )

    prompt_lines = [
        dashboard.prompt_text,
        "",
        "[cyan]↓↓ The terminal prompt below is where you type. ↓↓[/cyan]",
    ]
    if dashboard.last_input:
        prompt_lines.extend(["", f"[bold cyan]Last input:[/bold cyan] {dashboard.last_input}"])
    layout["input"].update(
        Panel(
            Text.from_markup("\n".join(prompt_lines), overflow="fold"),
            title=dashboard.prompt_title,
            border_style="yellow",
            padding=(1, 2),
        )
    )

    completed_agents = sum(
        1 for status in dashboard.agent_status.values() if status == "completed"
    )
    total_agents = len(dashboard.agent_status)
    elapsed_seconds = int(time.time() - dashboard.start_time)
    elapsed = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
    stats_parts = [
        f"Agents: {completed_agents}/{total_agents}",
        f"LLM: {dashboard.llm_calls}",
        f"Tools: {dashboard.tool_call_count}",
        f"Tokens: {_format_tokens(dashboard.tokens_in)} {_format_tokens(dashboard.tokens_out)}",
        f"Time elapsed: {elapsed}",
    ]
    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))
    layout["footer"].update(Panel(stats_table, border_style="grey50"))
    return layout


def _draw_dashboard(dashboard: DashboardState) -> None:
    console.clear()
    console.print(_build_layout(dashboard))


def _prompt_user(
    dashboard: DashboardState,
    *,
    title: str,
    prompt_text: str,
    input_label: str,
    default: str | None = None,
) -> str:
    dashboard.set_prompt(title, prompt_text)
    dashboard.set_status(prompt_text)
    _draw_dashboard(dashboard)
    prompt_label = f"[bold cyan]{input_label}[/bold cyan]"
    if default is not None:
        prompt_label += f" [dim](blank = {default})[/dim]"
    response = console.input(f"{prompt_label}> ").strip()
    if not response and default is not None:
        response = default
    dashboard.set_last_input(response)
    dashboard.clear_prompt()
    return response


def _resolve_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    if "workflow_builder" in workflow:
        return workflow
    resolved = dict(workflow)
    resolved["workflow_builder"] = load_callable(f"{resolved['module']}:make_graph")
    return resolved


def _select_workflow(
    dashboard: DashboardState,
    workflows: list[dict[str, str]],
) -> dict[str, Any]:
    while True:
        workflow_lines = ["Choose a workflow by number:", ""]
        workflow_lines.extend(
            f"{index}. {workflow['name']} ({workflow['module']})"
            for index, workflow in enumerate(workflows, start=1)
        )
        selection = _prompt_user(
            dashboard,
            title="Workflow Selection",
            prompt_text="\n".join(workflow_lines),
            input_label="Workflow",
        )
        if selection.isdigit():
            selected_index = int(selection)
            if 1 <= selected_index <= len(workflows):
                selected_workflow = _resolve_workflow(workflows[selected_index - 1])
                dashboard.set_workflow_name(selected_workflow["name"])
                dashboard.add_message(
                    "System",
                    f"Workflow selected: {selected_workflow['name']} ({selected_workflow['module']})",
                )
                dashboard.set_status("Workflow selected. Enter your travel query.")
                _draw_dashboard(dashboard)
                return selected_workflow

        dashboard.set_status("Please enter a valid workflow number.")
        dashboard.add_message(
            "System",
            f"Invalid workflow selection: {selection or '<empty>'}",
        )


def _ask_travel_query(dashboard: DashboardState) -> str:
    while True:
        query = _prompt_user(
            dashboard,
            title="Travel Query",
            prompt_text="Describe the trip or planning task you want help with.",
            input_label="Query",
        )
        if query.strip():
            dashboard.set_travel_query(query)
            dashboard.add_message("User", query)
            dashboard.set_status("Starting workflow execution.")
            _draw_dashboard(dashboard)
            return query
        dashboard.set_status("Please enter a travel query.")


async def _execute_workflow(
    dashboard: DashboardState,
    *,
    travel_query: str,
    workflow: dict[str, Any],
) -> None:
    compiled_workflow: CompiledStateGraph = workflow["workflow_builder"]().compile(
        checkpointer=make_memory_checkpointer()
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    agents = _get_agent_rows(compiled_workflow)

    dashboard.set_workflow_name(str(workflow["name"]))
    dashboard.set_travel_query(travel_query)
    dashboard.set_agents(agents)
    if agents:
        dashboard.mark_started(agents[0].key)
    dashboard.set_status("Starting workflow execution.")
    dashboard.add_message("System", f"Running workflow: {workflow['name']}")
    _draw_dashboard(dashboard)

    token = set_run_monitor(dashboard)
    try:
        current_input: dict[str, Any] | Command = {"query": travel_query}
        while True:
            interrupt_message: str | None = None
            async for mode, data in compiled_workflow.astream(
                current_input,
                config=config,
                stream_mode=["updates", "values"],
            ):
                if mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    for node_name, node_update in data.items():
                        normalized_update = normalize_for_display(node_update)
                        if node_name in dashboard.agent_status:
                            dashboard.mark_completed(node_name)
                            if isinstance(normalized_update, dict):
                                dashboard.summarize_update(node_name, normalized_update)
                        _draw_dashboard(dashboard)
                elif mode == "values":
                    if isinstance(data, dict) and data.get("__interrupt__"):
                        interrupt_value = data["__interrupt__"][0]
                        interrupt_message = str(
                            getattr(interrupt_value, "value", interrupt_value)
                        )
                        dashboard.set_status(interrupt_message)
                        dashboard.add_message("Agent", interrupt_message)
                        snapshot = await compiled_workflow.aget_state(
                            config,
                            subgraphs=True,
                        )
                        if _snapshot_has_hard_constraints(snapshot):
                            dashboard.set_hard_constraints(
                                _get_snapshot_hard_constraints(snapshot)
                            )
                        _draw_dashboard(dashboard)

            if interrupt_message is None:
                dashboard.set_status("Travel planning finished.")
                dashboard.set_prompt(
                    "Finished",
                    f"Workflow finished. Review the summary above or press Ctrl-C to exit. {dashboard.agent_status}",
                )
                dashboard.add_message("System", "Travel planning finished.")
                for agent in agents:
                    if dashboard.agent_status.get(agent.key) != "completed":
                        dashboard.agent_status[agent.key] = "completed"
                _draw_dashboard(dashboard)
                return

            response = _prompt_user(
                dashboard,
                title="Agent Question",
                prompt_text=interrupt_message,
                input_label="Reply",
                default="ok",
            )
            dashboard.add_message("User", response)
            dashboard.set_status("Resuming workflow execution.")
            _draw_dashboard(dashboard)
            current_input = Command(resume=response)
    finally:
        reset_run_monitor(token)


def _run_dashboard(
    *,
    travel_query: str | None,
    workflow: dict[str, Any] | None,
    workflows: list[dict[str, str]] | None,
) -> None:
    dashboard = DashboardState(travel_query=travel_query or "")
    _draw_dashboard(dashboard)

    selected_workflow = _resolve_workflow(workflow) if workflow is not None else None
    if selected_workflow is None:
        if not workflows:
            raise typer.BadParameter("No workflows are available.")
        dashboard.set_status("Choose a workflow to begin.")
        selected_workflow = _select_workflow(dashboard, workflows)
    else:
        dashboard.set_workflow_name(selected_workflow["name"])

    resolved_query = travel_query
    if not resolved_query:
        resolved_query = _ask_travel_query(dashboard)
    else:
        dashboard.set_travel_query(resolved_query)
        dashboard.add_message("User", resolved_query)

    asyncio.run(
        _execute_workflow(
            dashboard,
            travel_query=resolved_query,
            workflow=selected_workflow,
        )
    )


def run_interactive_shell(travel_query: str, workflow: dict[str, Any]) -> None:
    _run_dashboard(travel_query=travel_query, workflow=workflow, workflows=None)


def run_planner_dashboard(workflows: list[dict[str, str]]) -> None:
    _run_dashboard(travel_query=None, workflow=None, workflows=workflows)


@app.command()
def run(
    query: str = typer.Argument(..., help="Travel planning query to run."),
    graph: str = typer.Option(
        "travelplanner.workflows.task_planning:make_graph",
        "--graph",
        help="Workflow graph factory import string.",
    ),
) -> None:
    """Run a workflow directly in the dashboard."""
    workflow_builder = load_callable(graph)
    run_interactive_shell(query, {"name": graph, "workflow_builder": workflow_builder})


if __name__ == "__main__":
    app()
