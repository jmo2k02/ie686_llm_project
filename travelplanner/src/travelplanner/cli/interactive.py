"""Simple interactive shell mockup."""

from __future__ import annotations

import asyncio
import uuid
from itertools import cycle

import typer
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

from travelplanner.utils.checkpoint import make_memory_checkpointer

app = typer.Typer(
    help="Run a simple interactive shell mockup.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

STATES = ["Workflow Selection", "1", "2"]
MOCK_QUESTIONS = ["What is 1+1", "How is the weather"]


def _progress_bar(current_step: int, total_steps: int, width: int = 24) -> str:
    filled = int(width * current_step / total_steps)
    return f"[{'#' * filled}{'-' * (width - filled)}]"

def _snapshot_has_hard_constraints(snapshot):
  return any(
    getattr(t, "state", None) and t.state.values.get("hard_constraints")
    for t in snapshot.tasks
    if t.name == "constraint_agent"
  )

def _get_snapshot_hard_constraints(snapshot):
    return next(
        (
            t.state.values.get("hard_constraints")
            for t in snapshot.tasks
            if t.name == "constraint_agent" and getattr(t, "state", None)
        )
    )

def run_interactive_shell(travel_query: str, workflow: dict) -> None:
    compiled_workflow: CompiledStateGraph = workflow["workflow_builder"]().compile(
        checkpointer=make_memory_checkpointer()
    )
    STATES = list(compiled_workflow.nodes)[1:]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = {
        "current_state_index": 0,
        "current_query": "No message yet",
        "status": f"Workflow selected: {workflow['name']}\nPlease answer the model questions. Ctrl-C exits.",
        "loading_message": None,
        "is_loading": False,
        "started_workflow": False,
        "is_interrupted": False,
        "hard_constraints": []
    }

    application: Application | None = None

    def invalidate() -> None:
        if application is not None:
            application.invalidate()

    def render_progress() -> str:
        current_step = state["current_state_index"] + 1
        current_label = STATES[state["current_state_index"]]
        output = (
            f"Current: {current_label}\n"
            f"{_progress_bar(current_step, len(STATES))} {current_step}/{len(STATES)}\n"
            f"{' -> '.join(STATES)}\n\n"
            f"Original Query: '{travel_query}'\n"
        )
        if len(state["hard_constraints"]) > 0:
            output += f"Hardconstraint count: {len(state["hard_constraints"])}\n"
            output += str([constraint.text for constraint in state["hard_constraints"]])
        return output

    def render_status() -> str:
        if state["loading_message"] is not None:
            return state["loading_message"]
        return str(state["status"])

    def render_query_label() -> str:
        if state["started_workflow"]:
            return f"Last Message: {state['current_query']}"
        return "Please type (y) to start"

    input_field = TextArea(
        height=1,
        prompt="> ",
        multiline=False,
        wrap_lines=False,
    )

    async def process_query(query: str) -> None:
        state["is_loading"] = True
        state["current_state_index"] = 1
        state["loading_message"] = "Our Agents are crunching your request"
        invalidate()
        result = {}
        if query in {"y", "Y", "yes", "start"}:
            state["started_workflow"] = True
            invalidate()
        if state["is_interrupted"]:
            result = await compiled_workflow.ainvoke(
                Command(resume=query), config=config
            )
            state["is_interrupted"] = False
        else:
            result = await compiled_workflow.ainvoke(
                {"query": travel_query}, config=config
            )
        if "__interrupt__" in result:
            snapshot = await compiled_workflow.aget_state(config, subgraphs=True)
            agent_message = result["__interrupt__"][0].value
            state["is_interrupted"] = True
            state["current_state_index"] = 2
            state["loading_message"] = None
            state["status"] = f"The Agent has a question.\n\n{agent_message}"
            state["is_loading"] = False
            invalidate()
            if _snapshot_has_hard_constraints(snapshot):
                state["hard_constraints"] = _get_snapshot_hard_constraints(snapshot)
            return
        else:
            state["status"] = f"TravelPlanning is finished :)"

    def accept_input(buffer) -> bool:
        query = buffer.text.strip()
        buffer.text = ""

        if not query:
            state["status"] = "Please enter a query."
            invalidate()
            return False

        if query in {"/exit", "exit", "quit"}:
            if application is not None:
                application.exit()
            return False

        if state["is_loading"]:
            state["status"] = "Still loading. Wait a moment."
            invalidate()
            return False

        state["current_query"] = query
        if application is not None:
            application.create_background_task(process_query(query))
        invalidate()
        return False

    input_field.buffer.accept_handler = accept_input

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _exit_app(event) -> None:
        event.app.exit()

    root_container = HSplit(
        [
            Frame(
                Window(FormattedTextControl(render_progress), always_hide_cursor=True),
                title="Progress",
            ),
            Frame(
                Window(FormattedTextControl(render_status), always_hide_cursor=True),
                title="Model",
            ),
            Frame(
                HSplit(
                    [
                        Window(
                            FormattedTextControl(render_query_label),
                            height=1,
                            always_hide_cursor=True,
                        ),
                        input_field,
                    ]
                ),
                title="Human",
            ),
        ]
    )

    application = Application(
        layout=Layout(root_container, focused_element=input_field),
        key_bindings=bindings,
        full_screen=True,
    )
    application.run()


@app.command()
def run() -> None:
    """Start the simple interactive shell."""
    run_interactive_shell()


if __name__ == "__main__":
    app()
