from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.agents.llm_utils import invoke_structured_model
from travelplanner.schema.system_state import (
    ConstraintModel,
    MessageHistoryModel,
    TaskModel,
)


SYSTEM_PROMPT = """You are the TravelPlanner planner agent.

Turn the extracted constraints into an initial task list for downstream search agents.

Rules:
- Return JSON only.
- Use only the allowed task types: flight, hotel, restaurant, attraction, opening_times, routing-check, general-web-search.
- Each task must be grounded in the user request and extracted constraints.
- Keep the task list small and useful.
- Do not mark tasks as valid yet. The reviewer does that.
"""


class PlannerAgentState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    task_list: list[TaskModel] = Field(default_factory=list)
    message_history: MessageHistoryModel | None = None


class PlanningResponse(BaseModel):
    tasks: list[TaskModel] = Field(default_factory=list)


def _build_user_prompt(query: str, constraints: list[ConstraintModel]) -> str:
    return "\n".join(
        [
            "Build a concise first-pass task list for the travel request below.",
            "",
            f"User request: {query.strip()}",
            "",
            "Constraints JSON:",
            json.dumps(
                [constraint.model_dump() for constraint in constraints],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Return strictly valid JSON with this shape:",
            '{"tasks": [{"name": "...", "type": "flight|hotel|restaurant|attraction|opening_times|routing-check|general-web-search", "text": "...", "is_valid": false, "validation_comment": null}]}',
        ]
    )


def _build_message_history(
    query: str,
    user_prompt: str,
    raw_response: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="planner_agent",
        model="llm",
        agent_ref="travelplanner.agents.planner_agent",
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )


def make_graph():
    def planner_node(state: PlannerAgentState) -> dict[str, Any]:
        structured_output, user_prompt, raw_response = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(state.query, state.constraint_list),
            response_model=PlanningResponse,
        )
        history = _build_message_history(state.query, user_prompt, raw_response)
        return {
            "task_list": structured_output.tasks,
            "message_history": history,
        }

    graph = StateGraph(PlannerAgentState)
    graph.add_node("planner_agent", planner_node)
    graph.set_entry_point("planner_agent")
    graph.add_edge("planner_agent", END)
    return graph.compile()
