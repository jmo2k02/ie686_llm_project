from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from travelplanner.schema.commonsense_constraints import COMMONSENSE_CONSTRAINTS
from travelplanner.schema.system_state import ConstraintModel, MessageHistoryModel
from travelplanner.utils.llm import invoke_structured_model


# ── Constants ────────────────────────────────────────────────────────────────

HARD_CONSTRAINT_CATEGORIES: list[str] = [
    "destination",
    "travel_dates",
    "budget",
    "accommodation",
    "transport",
    "purpose",
]

CATEGORY_QUESTIONS: dict[str, str] = {
    "destination": (
        "You didn't mention a destination. Where would you like to travel? "
        "(or type 'skip' if not yet decided)"
    ),
    "travel_dates": (
        "No travel dates were specified. When do you plan to travel? "
        "Please provide a start and end date. (or 'skip')"
    ),
    "budget": (
        "No budget was mentioned. Do you have a maximum budget in mind? "
        "(or 'skip' if you don't)"
    ),
    "accommodation": (
        "No accommodation preference was found. Any preference for accommodation type "
        "(hotel, hostel, Airbnb, etc.)? (or 'skip')"
    ),
    "transport": (
        "No transport preference was found. How would you prefer to travel "
        "(flight, train, car, etc.)? (or 'skip')"
    ),
    "purpose": (
        "What is the purpose of this trip (leisure, business, honeymoon, family, etc.)? "
        "(or 'skip')"
    ),
}

EXTRACTION_SYSTEM_PROMPT = """You are the TravelPlanner constraint extraction agent.

Extract hard constraints directly from the user's travel request.

Rules:
- Return JSON only.
- Use the schema exactly.
- Extract only what the user explicitly states or strongly implies.
- For missing_categories, list only those from the provided category list that you could NOT find in the request.
- Do not invent destinations, dates, budgets, or preferences not present in the request.
"""

_SKIP_TOKENS: frozenset[str] = frozenset({
    "", "skip", "s", "no", "nein", "n", "nope", "egal",
    "doesn't matter", "doesnt matter", "don't care", "dont care",
    "not applicable", "na", "n/a", "none", "nothing",
})

_CONFIRM_TOKENS: frozenset[str] = frozenset({
    "ok", "yes", "ja", "y", "correct", "fine", "sure",
    "sounds good", "confirmed", "confirm", "good", "great",
    "looks good", "looks correct", "that's right", "thats right",
})


# ── State ────────────────────────────────────────────────────────────────────

class ConstraintIterationState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0

    query_context: str = ""
    phase: Literal["extract", "missing", "commonsense"] = "extract"

    hard_constraints: list[ConstraintModel] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    category_index: int = 0

    commonsense_constraints: list[ConstraintModel] = Field(
        default_factory=lambda: [c.model_copy() for c in COMMONSENSE_CONSTRAINTS]
    )
    constraint_index: int = 0

    messages: list[dict] = Field(default_factory=list)


# ── LLM response model ───────────────────────────────────────────────────────

class HardConstraintExtractionResponse(BaseModel):
    constraints: list[ConstraintModel] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)


# ── Prompt helpers ───────────────────────────────────────────────────────────

def _build_extraction_prompt(query: str, query_context: str) -> str:
    full_query = (
        f"{query}\n\nAdditional context / corrections: {query_context}"
        if query_context
        else query
    )
    return "\n".join([
        "Extract hard constraints from the travel request below.",
        "",
        f"User request: {full_query.strip()}",
        "",
        f"Category list to check: {json.dumps(HARD_CONSTRAINT_CATEGORIES)}",
        "",
        "Return strictly valid JSON with this shape:",
        '{"constraints": [{"type": "hard", "text": "...", "user_skipped": false}], '
        '"missing_categories": ["destination", "travel_dates"]}',
    ])


def _format_hard_constraints_message(constraints: list[ConstraintModel]) -> str:
    if not constraints:
        return (
            "I couldn't identify any hard constraints from your request. "
            "Please provide more details, or type 'ok' to continue."
        )
    lines = ["I identified the following constraints from your request:", ""]
    for i, c in enumerate(constraints, 1):
        lines.append(f"  {i}. {c.text}")
    lines += ["", "Is this correct? Type 'ok' to confirm, or describe any corrections."]
    return "\n".join(lines)


def _format_commonsense_question(index: int, total: int, constraint: ConstraintModel) -> str:
    return (
        f"[Commonsense check {index}/{total}]\n"
        f"{constraint.text}\n"
        "Does this apply to your trip? Type 'ok' to confirm or 'skip' if not applicable."
    )


def _build_message_history(messages: list[dict]) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="constraint_iteration_agent",
        model="llm",
        agent_ref="travelplanner.agents.constraint_iteration_agent",
        messages=messages,
    )


# ── Response helpers ─────────────────────────────────────────────────────────

def _is_skip(text: str) -> bool:
    return text.strip().lower() in _SKIP_TOKENS


def _is_confirm(text: str) -> bool:
    return text.strip().lower() in _CONFIRM_TOKENS


# ── Graph ────────────────────────────────────────────────────────────────────

def make_graph():
    def extract_hard_constraints(state: ConstraintIterationState) -> dict[str, Any]:
        structured_output, _, _ = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=_build_extraction_prompt(state.query, state.query_context),
            response_model=HardConstraintExtractionResponse,
        )
        return {
            "hard_constraints": structured_output.constraints,
            "missing_categories": [
                c for c in structured_output.missing_categories
                if c in HARD_CONSTRAINT_CATEGORIES
            ],
            "category_index": 0,
            "phase": "extract",
        }

    def present_hard_constraints(state: ConstraintIterationState) -> dict[str, Any]:
        agent_msg = _format_hard_constraints_message(state.hard_constraints)
        messages = [*state.messages, {"role": "assistant", "content": agent_msg}]

        user_input: str = interrupt(agent_msg)
        messages = [*messages, {"role": "user", "content": user_input}]

        if _is_confirm(user_input):
            next_phase: Literal["missing", "commonsense"] = (
                "missing" if state.missing_categories else "commonsense"
            )
            return {"messages": messages, "phase": next_phase}

        context = (
            f"{state.query_context}\n{user_input}".strip()
            if state.query_context
            else user_input
        )
        return {"messages": messages, "phase": "extract", "query_context": context}

    def ask_missing_category(state: ConstraintIterationState) -> dict[str, Any]:
        category = state.missing_categories[state.category_index]
        question = CATEGORY_QUESTIONS.get(
            category,
            f"No '{category}' was specified. Please provide details, or type 'skip'.",
        )
        messages = [*state.messages, {"role": "assistant", "content": question}]

        user_input: str = interrupt(question)
        messages = [*messages, {"role": "user", "content": user_input}]

        new_hard_constraints = list(state.hard_constraints)
        if not _is_skip(user_input):
            new_hard_constraints.append(
                ConstraintModel(
                    type="hard",
                    text=user_input.strip(),
                    user_skipped=False,
                )
            )

        return {
            "messages": messages,
            "hard_constraints": new_hard_constraints,
            "category_index": state.category_index + 1,
        }

    def present_commonsense_constraint(state: ConstraintIterationState) -> dict[str, Any]:
        idx = state.constraint_index
        constraint = state.commonsense_constraints[idx]
        total = len(state.commonsense_constraints)
        question = _format_commonsense_question(idx + 1, total, constraint)
        messages = [*state.messages, {"role": "assistant", "content": question}]

        user_input: str = interrupt(question)
        messages = [*messages, {"role": "user", "content": user_input}]

        updated = list(state.commonsense_constraints)
        updated[idx] = constraint.model_copy(
            update={"user_skipped": _is_skip(user_input)}
        )

        return {
            "messages": messages,
            "commonsense_constraints": updated,
            "constraint_index": idx + 1,
        }

    def _route_after_present_hard(state: ConstraintIterationState) -> str:
        if state.phase == "extract":
            return "extract_hard_constraints"
        if state.phase == "missing" and state.category_index < len(state.missing_categories):
            return "ask_missing_category"
        return "present_commonsense_constraint"

    def _route_after_missing(state: ConstraintIterationState) -> str:
        if state.category_index < len(state.missing_categories):
            return "ask_missing_category"
        return "present_commonsense_constraint"

    def _route_after_commonsense(state: ConstraintIterationState) -> str:
        if state.constraint_index < len(state.commonsense_constraints):
            return "present_commonsense_constraint"
        return END

    graph = StateGraph(ConstraintIterationState)
    graph.add_node("extract_hard_constraints", extract_hard_constraints)
    graph.add_node("present_hard_constraints", present_hard_constraints)
    graph.add_node("ask_missing_category", ask_missing_category)
    graph.add_node("present_commonsense_constraint", present_commonsense_constraint)

    graph.set_entry_point("extract_hard_constraints")
    graph.add_edge("extract_hard_constraints", "present_hard_constraints")
    graph.add_conditional_edges("present_hard_constraints", _route_after_present_hard)
    graph.add_conditional_edges("ask_missing_category", _route_after_missing)
    graph.add_conditional_edges("present_commonsense_constraint", _route_after_commonsense)

    return graph.compile(checkpointer=MemorySaver())


def get_constraint_list(state: dict[str, Any]) -> list[ConstraintModel]:
    """Returns the final merged constraint list from a completed graph state."""
    hard: list[ConstraintModel] = state.get("hard_constraints", [])
    commonsense: list[ConstraintModel] = state.get("commonsense_constraints", [])
    return hard + commonsense


def get_message_history(state: dict[str, Any]) -> MessageHistoryModel:
    """Returns the MessageHistoryModel from a completed graph state."""
    return _build_message_history(state.get("messages", []))
