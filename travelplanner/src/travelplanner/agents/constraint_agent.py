from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.agents.llm_utils import invoke_structured_model
from travelplanner.schema.system_state import ConstraintModel, MessageHistoryModel


SYSTEM_PROMPT = """You are the TravelPlanner constraint agent.

Extract planning constraints directly from the user request.

Rules:
- Return JSON only.
- Use the schema exactly.
- Extract hard constraints only when the user states them or strongly implies them.
- Add 1 or 2 commonsense constraints only if they help downstream planning.
- Do not invent destinations, dates, budgets, or preferences that are not present.
"""


class ConstraintAgentState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    message_history: MessageHistoryModel | None = None


class ConstraintExtractionResponse(BaseModel):
    constraints: list[ConstraintModel] = Field(default_factory=list)


def _build_user_prompt(query: str) -> str:
    return "\n".join(
        [
            "Extract constraints from the following user travel-planning request.",
            "",
            f"User request: {query.strip()}",
            "",
            "Return strictly valid JSON with this shape:",
            '{"constraints": [{"type": "hard" | "commonsense", "text": "...", "user_skipped": false}]}',
        ]
    )


def _build_message_history(
    query: str,
    user_prompt: str,
    raw_response: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="constraint_agent",
        model="llm",
        agent_ref="travelplanner.agents.constraint_agent",
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )


def make_graph():
    def constraint_node(state: ConstraintAgentState) -> dict[str, Any]:
        structured_output, user_prompt, raw_response = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(state.query),
            response_model=ConstraintExtractionResponse,
        )
        history = _build_message_history(state.query, user_prompt, raw_response)
        return {
            "constraint_list": structured_output.constraints,
            "message_history": history,
        }

    graph = StateGraph(ConstraintAgentState)
    graph.add_node("constraint_agent", constraint_node)
    graph.set_entry_point("constraint_agent")
    graph.add_edge("constraint_agent", END)
    return graph.compile()
