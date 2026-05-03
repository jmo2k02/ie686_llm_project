from __future__ import annotations

import json
import re
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from travelplanner.config import get_setting
from travelplanner.schema.system_state import (
    ConstraintModel,
    MessageHistoryModel,
    TaskModel,
)
from travelplanner.utils.llm import invoke_structured_model


PLANNER_HISTORY_KEY = "planner_agent"
REVIEWER_HISTORY_KEY = "reviewer_agent"
MAX_TASKS = 6
MAX_REVIEW_ATTEMPTS = 3
ALLOWED_TASK_TYPES: tuple[str, ...] = (
    "flight",
    "hotel",
    "restaurant",
    "attraction",
    "opening_times",
    "routing-check",
    "general-web-search",
)

TASK_TYPE_GUIDANCE: dict[str, str] = {
    "flight": "Use for flight discovery or comparison when air travel is explicitly needed.",
    "hotel": "Use for lodging or accommodation selection for overnight stays.",
    "restaurant": "Use for dining, cuisine, or meal-specific recommendations.",
    "attraction": "Use for museums, landmarks, activities, sightseeing, or events to visit.",
    "opening_times": "Use to verify operating hours or reservation feasibility for time-sensitive places.",
    "routing-check": "Use to verify travel time, transit feasibility, or distance between planned stops.",
    "general-web-search": "Use for edge-case research that does not cleanly fit a specialized task type.",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "best",
    "between",
    "by",
    "for",
    "from",
    "help",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "near",
    "need",
    "of",
    "on",
    "or",
    "our",
    "plan",
    "please",
    "recommend",
    "show",
    "that",
    "the",
    "their",
    "this",
    "to",
    "trip",
    "travel",
    "us",
    "we",
    "with",
}

_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "flight": ("flight", "fly", "airport", "airline", "airfare"),
    "hotel": (
        "hotel",
        "stay",
        "stays",
        "accommodation",
        "lodging",
        "hostel",
        "check-in",
    ),
    "restaurant": (
        "restaurant",
        "food",
        "eat",
        "dinner",
        "lunch",
        "breakfast",
        "brunch",
        "cafe",
        "cuisine",
    ),
    "attraction": (
        "attraction",
        "museum",
        "landmark",
        "sightseeing",
        "activity",
        "activities",
        "visit",
        "tour",
        "explore",
        "gallery",
    ),
    "opening_times": (
        "hours",
        "opening",
        "open",
        "closing",
        "closed",
        "reservation",
        "book",
        "availability",
    ),
    "routing-check": (
        "route",
        "routing",
        "travel time",
        "commute",
        "transfer",
        "distance",
        "walk",
        "walking",
        "train",
        "metro",
        "bus",
    ),
    "general-web-search": (
        "research",
        "verify",
        "check",
        "guide",
        "option",
        "options",
        "recommendation",
        "information",
    ),
}

_DAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

PLANNER_SYSTEM_PROMPT = """You are the TravelPlanner planner agent.

Turn the extracted constraints into an initial task list for downstream search agents.

Rules:
- Return JSON only.
- Use only the allowed task types: flight, hotel, restaurant, attraction, opening_times, routing-check, general-web-search.
- Each task must be grounded in the user request and extracted constraints.
- Keep the task list small and useful.
- Do not mark tasks as valid yet. The reviewer does that.
"""

REVIEWER_SYSTEM_PROMPT = """You are the TravelPlanner reviewer agent.

Review the planner's task list and return the reviewed task list.

Rules:
- Return JSON only.
- Validate whether each task is useful, specific, and grounded in the user request.
- Remove duplicates or merge obviously redundant tasks.
- Set is_valid to true only for tasks that are ready for downstream execution.
- If a task is invalid, explain why in validation_comment.
- If the planner missed an essential task, you may add it.
"""


class PlannerAgentState(BaseModel):
    query: str
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    task_list: list[TaskModel] = Field(default_factory=list)
    message_histories: dict[str, MessageHistoryModel] = Field(default_factory=dict)
    planner_review_feedback: str | None = None
    planner_review_attempts: int = 0
    planner_approved: bool = False
    review_summary: str | None = None


class ReviewerAgentState(BaseModel):
    query: str
    model_name: str = get_setting("models.workflows.task_planning.model_name")
    temperature: float = 0.0
    constraint_list: list[ConstraintModel] = Field(default_factory=list)
    proposed_task_list: list[TaskModel] = Field(default_factory=list)
    approved_task_list: list[TaskModel] = Field(default_factory=list)
    planner_approved: bool = False
    review_summary: str | None = None
    message_history: MessageHistoryModel | None = None


class PlanningResponse(BaseModel):
    tasks: list[TaskModel] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    approved_task_list: list[TaskModel] = Field(default_factory=list)
    review_summary: str | None = None


def _normalize_whitespace(text: str | None) -> str:
    return " ".join((text or "").split())


def _task_slug(task_type: str, index: int) -> str:
    return f"{task_type.replace('_', '-').replace(' ', '-')}-{index}"


def _task_signature(task: TaskModel) -> tuple[str, str]:
    return (task.type, _normalize_whitespace(task.text).lower())


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in _STOPWORDS}


def _full_request_text(query: str, constraints: list[ConstraintModel]) -> str:
    parts = [
        query.strip(),
        *[constraint.text for constraint in constraints if not constraint.user_skipped],
    ]
    return "\n".join(part for part in parts if part)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _trip_likely_has_overnight_stay(text: str) -> bool:
    return bool(
        re.search(
            r"\b(\d+\s*(day|days|night|nights|week|weeks)|overnight)\b",
            text.lower(),
        )
    )


def _mentions_specific_time(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b", lowered)) or any(
        day in lowered for day in _DAY_NAMES
    )


def infer_required_task_types(
    query: str,
    constraints: list[ConstraintModel],
) -> list[str]:
    request_text = _full_request_text(query, constraints).lower()
    required: list[str] = []

    if _contains_any(request_text, _TYPE_KEYWORDS["flight"]):
        required.append("flight")

    if _contains_any(
        request_text, _TYPE_KEYWORDS["hotel"]
    ) or _trip_likely_has_overnight_stay(request_text):
        required.append("hotel")

    if _contains_any(request_text, _TYPE_KEYWORDS["restaurant"]):
        required.append("restaurant")

    if _contains_any(request_text, _TYPE_KEYWORDS["attraction"]):
        required.append("attraction")

    if _contains_any(
        request_text, _TYPE_KEYWORDS["opening_times"]
    ) or _mentions_specific_time(request_text):
        required.append("opening_times")

    if _contains_any(request_text, _TYPE_KEYWORDS["routing-check"]) or (
        " from " in request_text and " to " in request_text
    ):
        required.append("routing-check")

    if _contains_any(request_text, _TYPE_KEYWORDS["general-web-search"]):
        required.append("general-web-search")

    if not required:
        required.append("general-web-search")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in required:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def normalize_tasks(
    tasks: list[TaskModel],
    *,
    default_is_valid: bool,
    max_tasks: int = MAX_TASKS,
) -> list[TaskModel]:
    normalized: list[TaskModel] = []
    seen_signatures: set[tuple[str, str]] = set()

    for index, task in enumerate(tasks, 1):
        text = _normalize_whitespace(task.text) or _normalize_whitespace(task.name)
        if not text:
            continue

        normalized_task = TaskModel(
            name=_normalize_whitespace(task.name) or _task_slug(task.type, index),
            type=task.type,
            text=text,
            is_valid=default_is_valid,
            validation_comment=_normalize_whitespace(task.validation_comment) or None,
        )
        signature = _task_signature(normalized_task)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        normalized.append(normalized_task)
        if len(normalized) >= max_tasks:
            break

    return normalized


def coverage_check(
    tasks: list[TaskModel],
    constraints: list[ConstraintModel],
) -> dict[str, list[str]]:
    task_keywords: set[str] = set()
    task_types = {task.type for task in tasks}
    for task in tasks:
        task_keywords.update(_extract_keywords(task.name))
        task_keywords.update(_extract_keywords(task.text))

    uncovered_constraints: list[str] = []
    for constraint in constraints:
        if constraint.user_skipped:
            continue
        constraint_keywords = _extract_keywords(constraint.text)
        if constraint_keywords and not constraint_keywords.intersection(task_keywords):
            uncovered_constraints.append(constraint.text)

    return {
        "covered_task_types": sorted(task_types),
        "uncovered_constraints": uncovered_constraints,
    }


def _task_type_matches_text(task: TaskModel) -> bool:
    if task.type == "general-web-search":
        return True
    keywords = _TYPE_KEYWORDS.get(task.type, ())
    haystack = f"{task.name} {task.text}".lower()
    return any(keyword in haystack for keyword in keywords)


def _task_issues(task: TaskModel, *, request_keywords: set[str]) -> list[str]:
    issues: list[str] = []
    if len(task.text) < 18:
        issues.append("task text is too short")

    lowered = task.text.lower()
    if any(token in lowered for token in ("...", "todo", "tbd", "something", "stuff")):
        issues.append("task text looks like a placeholder")

    if not _task_type_matches_text(task):
        issues.append("task text does not clearly match its task type")

    task_keywords = _extract_keywords(task.name) | _extract_keywords(task.text)
    if request_keywords and not task_keywords.intersection(request_keywords):
        issues.append("task is not clearly grounded in the request or constraints")

    return issues


def validate_task_list(
    tasks: list[TaskModel],
    query: str,
    constraints: list[ConstraintModel],
) -> tuple[list[TaskModel], list[str]]:
    normalized = normalize_tasks(tasks, default_is_valid=False)
    request_keywords = _extract_keywords(_full_request_text(query, constraints))
    required_types = infer_required_task_types(query, constraints)
    present_types = {task.type for task in normalized}
    reviewed: list[TaskModel] = []
    summary: list[str] = []

    missing_types = [
        task_type for task_type in required_types if task_type not in present_types
    ]
    if missing_types:
        summary.append("Missing likely task types: " + ", ".join(missing_types) + ".")

    coverage = coverage_check(normalized, constraints)
    if coverage["uncovered_constraints"]:
        summary.append(
            "Potentially uncovered constraints: "
            + "; ".join(coverage["uncovered_constraints"][:3])
            + "."
        )

    for task in normalized:
        issues = _task_issues(task, request_keywords=request_keywords)
        reviewed.append(
            task.model_copy(
                update={
                    "is_valid": not issues,
                    "validation_comment": "; ".join(issues) if issues else None,
                }
            )
        )

    return reviewed, summary


def _format_task_type_guide() -> str:
    return "\n".join(
        f"- {task_type}: {description}"
        for task_type, description in TASK_TYPE_GUIDANCE.items()
    )


def _format_summary_lines(summary: list[str]) -> str:
    if not summary:
        return "- No deterministic issues detected."
    return "\n".join(f"- {item}" for item in summary)


def _build_planner_prompt(
    query: str,
    constraints: list[ConstraintModel],
    review_feedback: str | None,
) -> str:
    required_types = infer_required_task_types(query, constraints)
    sections = [
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
        "Task type guide:",
        _format_task_type_guide(),
        "",
        f"Likely required task types: {', '.join(required_types)}",
        f"Keep the list to at most {MAX_TASKS} tasks.",
    ]
    if review_feedback:
        sections.extend(
            [
                "",
                "Reviewer feedback from the previous attempt:",
                review_feedback.strip(),
            ]
        )
    sections.extend(
        [
            "",
            "Return strictly valid JSON with this shape:",
            '{"tasks": [{"name": "...", "type": "flight|hotel|restaurant|attraction|opening_times|routing-check|general-web-search", "text": "...", "is_valid": false, "validation_comment": null}]}',
        ]
    )
    return "\n".join(sections)


def _build_reviewer_prompt(
    query: str,
    constraints: list[ConstraintModel],
    proposed_task_list: list[TaskModel],
) -> str:
    deterministic_review, summary = validate_task_list(
        proposed_task_list,
        query,
        constraints,
    )
    return "\n".join(
        [
            "Review the planner task list for the travel request below.",
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
            "Task type guide:",
            _format_task_type_guide(),
            "",
            "Proposed tasks JSON:",
            json.dumps(
                [task.model_dump() for task in proposed_task_list],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Deterministic validation preview:",
            json.dumps(
                [task.model_dump() for task in deterministic_review],
                indent=2,
                ensure_ascii=True,
            ),
            "",
            "Deterministic summary:",
            _format_summary_lines(summary),
            "",
            "Return strictly valid JSON with this shape:",
            '{"approved_task_list": [{"name": "...", "type": "flight|hotel|restaurant|attraction|opening_times|routing-check|general-web-search", "text": "...", "is_valid": true, "validation_comment": null}], "review_summary": "..."}',
        ]
    )


def _build_message_history(
    *,
    user_agent: str,
    agent_ref: str,
    query: str,
    user_prompt: str,
    raw_response: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent=user_agent,
        model="llm",
        agent_ref=agent_ref,
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )


def _review_feedback_text(tasks: list[TaskModel], summary: list[str]) -> str | None:
    lines = list(summary)
    for task in tasks:
        if task.validation_comment:
            lines.append(f"Task '{task.name}': {task.validation_comment}.")
    if not lines:
        return None
    return "\n".join(f"- {line}" for line in lines)


def _planner_node(
    state: PlannerAgentState,
    *,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    structured_output, user_prompt, raw_response = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=_build_planner_prompt(
            state.query,
            state.constraint_list,
            state.planner_review_feedback,
        ),
        response_model=PlanningResponse,
    )
    history = _build_message_history(
        user_agent=PLANNER_HISTORY_KEY,
        agent_ref="travelplanner.agents.planner_agent",
        query=state.query,
        user_prompt=user_prompt,
        raw_response=raw_response,
    )
    return {
        "task_list": normalize_tasks(structured_output.tasks, default_is_valid=False),
        "planner_approved": False,
        "review_summary": None,
        "message_histories": {
            **state.message_histories,
            PLANNER_HISTORY_KEY: history,
        },
    }


def _reviewer_node(
    state: PlannerAgentState,
    *,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    structured_output, user_prompt, raw_response = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_prompt=_build_reviewer_prompt(
            state.query,
            state.constraint_list,
            state.task_list,
        ),
        response_model=ReviewResponse,
    )
    reviewed_tasks, summary = validate_task_list(
        structured_output.approved_task_list,
        state.query,
        state.constraint_list,
    )
    if structured_output.review_summary:
        summary = [*summary, structured_output.review_summary]

    planner_approved = (
        bool(reviewed_tasks)
        and all(task.is_valid for task in reviewed_tasks)
        and not any(
            line.startswith("Missing likely task types:")
            or line.startswith("Potentially uncovered constraints:")
            for line in summary
        )
    )
    feedback = (
        None if planner_approved else _review_feedback_text(reviewed_tasks, summary)
    )

    history = _build_message_history(
        user_agent=REVIEWER_HISTORY_KEY,
        agent_ref="travelplanner.agents.planner_agent",
        query=state.query,
        user_prompt=user_prompt,
        raw_response=raw_response,
    )
    if summary:
        history.messages.append(
            {
                "role": "assistant",
                "content": "Deterministic review summary: " + " | ".join(summary),
            }
        )
    
    return {
        "task_list": reviewed_tasks,
        "planner_review_attempts": state.planner_review_attempts + 1,
        "planner_review_feedback": feedback,
        "planner_approved": planner_approved,
        "review_summary": feedback,
        "message_histories": {
            **state.message_histories,
            REVIEWER_HISTORY_KEY: history,
        },
    }


def _route_after_review(state: PlannerAgentState) -> str:
    if state.planner_approved or state.planner_review_attempts >= MAX_REVIEW_ATTEMPTS:
        return "finalize_planner_output"
    return "planner_draft"


def _finalize_output(state: PlannerAgentState) -> dict[str, Any]:
    interrupt("Decided for final task list")
    return {
        "task_list": state.task_list,
        "message_histories": state.message_histories,
    }


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    effective_model_name = model_name or str(
        get_setting(
            "models.workflows.task_planning.model_name",
            "gpt-5.4-nano-2026-03-17",
        )
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )

    graph = StateGraph(PlannerAgentState)
    graph.add_node(
        "planner_draft",
        lambda state: _planner_node(
            state,
            model_name=effective_model_name,
            temperature=effective_temperature,
        ),
    )
    graph.add_node(
        "reviewer_agent",
        lambda state: _reviewer_node(
            state,
            model_name=effective_model_name,
            temperature=effective_temperature,
        ),
    )
    graph.add_node("finalize_planner_output", _finalize_output)

    graph.set_entry_point("planner_draft")
    graph.add_edge("planner_draft", "reviewer_agent")
    graph.add_conditional_edges("reviewer_agent", _route_after_review)
    graph.add_edge("finalize_planner_output", END)
    return graph


def make_reviewer_graph(
    model_name: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    effective_model_name = model_name or str(
        get_setting(
            "models.workflows.task_planning.model_name",
            "gpt-5.4-nano-2026-03-17",
        )
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )

    def reviewer_only(state: ReviewerAgentState) -> dict[str, Any]:
        structured_output, user_prompt, raw_response = invoke_structured_model(
            model_name=effective_model_name,
            temperature=effective_temperature,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=_build_reviewer_prompt(
                state.query,
                state.constraint_list,
                state.proposed_task_list,
            ),
            response_model=ReviewResponse,
        )
        reviewed_tasks, summary = validate_task_list(
            structured_output.approved_task_list,
            state.query,
            state.constraint_list,
        )
        if structured_output.review_summary:
            summary = [*summary, structured_output.review_summary]
        approved = bool(reviewed_tasks) and all(
            task.is_valid for task in reviewed_tasks
        )
        return {
            "approved_task_list": reviewed_tasks,
            "planner_approved": approved,
            "review_summary": _review_feedback_text(reviewed_tasks, summary),
            "message_history": _build_message_history(
                user_agent=REVIEWER_HISTORY_KEY,
                agent_ref="travelplanner.agents.planner_agent",
                query=state.query,
                user_prompt=user_prompt,
                raw_response=raw_response,
            ),
        }

    graph = StateGraph(ReviewerAgentState)
    graph.add_node("reviewer_agent", reviewer_only)
    graph.set_entry_point("reviewer_agent")
    graph.add_edge("reviewer_agent", END)
    return graph
