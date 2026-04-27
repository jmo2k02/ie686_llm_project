from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from travelplanner.utils.llm import make_chat_model
from travelplanner.schema.system_state import MessageHistoryModel, StateContractModel


SYSTEM_PROMPT = (
    "You are TravelPlanner, an assistant that writes day-by-day itineraries "
    "covering transportation, meals, attractions, and lodging while obeying "
    "budget and commonsense constraints. Be concrete, reference the provided "
    "facts when possible, and keep each day easy to parse."
)


def _format_state_context(state: StateContractModel) -> str:
    context: dict[str, Any] = {
        "constraints": [
            constraint.model_dump() for constraint in state.constraint_list
        ],
        "tasks": [task.model_dump() for task in state.task_list],
        "timetable": (
            state.timetable.model_dump() if state.timetable is not None else None
        ),
    }
    return json.dumps(context, indent=2, ensure_ascii=True)


def _build_user_prompt(query: str, context_blob: str) -> str:
    sections = [
        "Request:",
        query.strip(),
        "",
        "Current State:",
        context_blob,
        "",
        "Return a structured multi-day itinerary.",
    ]
    return "\n".join(section for section in sections if section is not None)


def make_graph(
    model_name: str,
    temperature: float = 0.6,
    history_key: str = "minimal_agent",
) -> Pregel:
    client = make_chat_model(model_name=model_name, temperature=temperature)

    def planner_node(state: StateContractModel) -> dict[str, Any]:
        prompt = _build_user_prompt(state.query, _format_state_context(state))
        response = client.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        plan_text = (response.content or "").strip()

        message_histories = dict(state.message_histories)
        message_histories[history_key] = MessageHistoryModel(
            user_agent="planner",
            model=model_name,
            agent_ref="travelplanner.agents.minimal_agent",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan_text},
            ],
        )
        return {"message_histories": message_histories}

    graph = StateGraph(StateContractModel)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()
