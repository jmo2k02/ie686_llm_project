from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from travelplanner.config import get_setting
from travelplanner.schema.commonsense_constraints import COMMONSENSE_CONSTRAINTS
from travelplanner.schema.constraint_artifact import ConstraintArtifactContentModel
from travelplanner.schema.system_state import AgentArtifactModel, ConstraintModel, MessageHistoryModel
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
}

STRUCTURED_OPTIONS: dict[str, dict[str, str]] = {
    "accommodation": {
        "a": "Hotel",
        "b": "Airbnb / Vacation rental",
        "c": "Hostel",
        "d": "Apartment",
        "e": "Resort",
        "f": "Other / No preference",
    },
    "transport": {
        "a": "Flight",
        "b": "Train",
        "c": "Car / Road trip",
        "d": "Bus / Coach",
        "e": "Other / No preference",
    },
    "purpose": {
        "a": "Leisure / Vacation",
        "b": "Business",
        "c": "Honeymoon / Romance",
        "d": "Family trip",
        "e": "Adventure / Backpacking",
        "f": "Other",
    },
}

EXTRACTION_SYSTEM_PROMPT = """You are the TravelPlanner constraint extraction agent.

Extract hard constraints directly from the user's travel request.

Rules:
- Return JSON only.
- Use the schema exactly.
- Extract only what the user explicitly states or strongly implies.
- For missing_categories, list only those from the provided category list that you could NOT find in the request.
- Do not invent destinations, dates, budgets, or preferences not present in the request.
- If the user has provided corrections, the corrections are the authoritative source and OVERRIDE any conflicting information from the original request.
- Always use the current date provided in the prompt to assess temporal constraints.
"""

VIOLATION_CHECK_SYSTEM_PROMPT = """You are the TravelPlanner constraint validation agent.

Your job is to assess every commonsense constraint against the extracted hard constraints and set is_violated to true ONLY when a constraint is definitively and clearly broken by the current data.

Rules:
- Return JSON only.
- You MUST include every constraint from the provided list in your response.
- For each constraint set is_violated: true if and only if it is definitively broken right now.
- Set is_violated: false if the constraint is satisfied, cannot yet be assessed, or is only potentially relevant.
- For violated constraints, provide a short explanation of WHY it is broken and exactly two concrete suggestions to fix it.
- For non-violated constraints, explanation and suggestions may be empty.
- Always use the current date provided in the prompt to assess whether dates are in the past or future.
- Base your assessment solely on the constraints as they are currently listed.

Examples of is_violated: false (NOT broken):
- A trip date that is in the future → is_violated: false for "trip must be in the future"
- An end date that is after the start date → is_violated: false for "end must be after start"
- Accommodation not yet booked → is_violated: false if the user hasn't been asked yet
"""

_DEFAULT_MODEL_NAME = "gpt-5.4-nano-2026-03-17"
_DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class ConstraintAgentConfig:
    model_name: str = _DEFAULT_MODEL_NAME
    temperature: float = _DEFAULT_TEMPERATURE


def load_config_from_env() -> ConstraintAgentConfig:
    cfg_prefix = "agents.constraint"
    return ConstraintAgentConfig(
        model_name=str(
            os.getenv(
                "TRAVELPLANNER_CONSTRAINT_MODEL_NAME",
                get_setting(f"{cfg_prefix}.model_name", _DEFAULT_MODEL_NAME),
            )
        ),
        temperature=float(
            os.getenv(
                "TRAVELPLANNER_CONSTRAINT_TEMPERATURE",
                str(get_setting(f"{cfg_prefix}.temperature", _DEFAULT_TEMPERATURE)),
            )
        ),
    )


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

class ViolationModel(BaseModel):
    violated_constraint: str
    is_violated: bool
    explanation: str
    suggestions: list[str] = Field(default_factory=list)


class ConstraintIterationState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0

    query_context: str = ""
    phase: Literal["extract", "missing"] = "extract"

    hard_constraints: list[ConstraintModel] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    categories_skipped: list[str] = Field(default_factory=list)
    category_index: int = 0

    commonsense_constraints: list[ConstraintModel] = Field(
        default_factory=lambda: [c.model_copy() for c in COMMONSENSE_CONSTRAINTS]
    )
    constraint_index: int = 0

    violations: list[ViolationModel] = Field(default_factory=list)

    messages: list[dict] = Field(default_factory=list)
    agent_artifacts: dict[str, list[AgentArtifactModel]] = Field(default_factory=dict)


SPELL_CHECK_SYSTEM_PROMPT = """You are a spell-check assistant for a travel planning application.

Fix ONLY clear typos and misspellings in the user's input. Nothing else.

Rules:
- Fix only typos (transposed, missing, or wrong letters)
- Do NOT rephrase, restructure, or change meaning in any way
- Do NOT alter numbers, dates, amounts, currencies, or punctuation
- Use travel planning context to resolve ambiguous cases
  (e.g. "barelona" → "Barcelona", "apirl" → "April", "flgiht" → "flight")
- If the text has no typos, return it unchanged with an empty corrections list
- Return JSON only
"""


# ── LLM response models ──────────────────────────────────────────────────────

class SpellCorrectionItem(BaseModel):
    original: str
    corrected: str


class SpellCheckResponse(BaseModel):
    corrections: list[SpellCorrectionItem] = Field(default_factory=list)
    corrected_text: str


class HardConstraintExtractionResponse(BaseModel):
    constraints: list[ConstraintModel] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)


class ConstraintViolationCheckResponse(BaseModel):
    violations: list[ViolationModel] = Field(default_factory=list)


# ── Prompt helpers ───────────────────────────────────────────────────────────

def _build_extraction_prompt(query: str, query_context: str) -> str:
    lines = [
        "Extract hard constraints from the travel request below.",
        "",
        f"Today's date: {date.today().isoformat()}",
        "",
        f"Original user request: {query.strip()}",
    ]
    if query_context:
        lines += [
            "",
            "User corrections (these are the authoritative, up-to-date values — "
            "they override any conflicting information in the original request):",
            query_context.strip(),
        ]
    lines += [
        "",
        f"Category list to check: {json.dumps(HARD_CONSTRAINT_CATEGORIES)}",
        "",
        "Return strictly valid JSON with this shape:",
        '{"constraints": [{"type": "hard", "text": "...", "user_skipped": false}], '
        '"missing_categories": ["destination", "travel_dates"]}',
    ]
    return "\n".join(lines)


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


def _build_violation_check_prompt(
    query: str,
    query_context: str,
    hard_constraints: list[ConstraintModel],
    commonsense_constraints: list[ConstraintModel],
) -> str:
    lines = [
        "Check the extracted hard constraints for commonsense violations.",
        "",
        f"Today's date: {date.today().isoformat()}",
        "",
        f"Original user request: {query.strip()}",
    ]
    if query_context:
        lines += [
            "",
            "User corrections (authoritative — override the original request):",
            query_context.strip(),
        ]
    lines += [
        "",
        "Currently extracted hard constraints (already reflect the latest corrections):",
        json.dumps([c.model_dump() for c in hard_constraints], indent=2, ensure_ascii=True),
        "",
        "Commonsense constraints to check against:",
        json.dumps([c.text for c in commonsense_constraints], indent=2, ensure_ascii=True),
        "",
        "Return strictly valid JSON with this shape:",
        '{"violations": [{"violated_constraint": "...", "is_violated": true, '
        '"explanation": "...", "suggestions": ["suggestion 1", "suggestion 2"]}]}',
    ]
    return "\n".join(lines)


def _format_violation_message(violations: list[ViolationModel]) -> str:
    lines = [
        "⚠ I found the following conflict(s) in your travel request:",
        "",
    ]
    for i, v in enumerate(violations, 1):
        lines.append(f"Conflict {i}: {v.violated_constraint}")
        lines.append(f"  → {v.explanation}")
        lines.append("")
        lines.append("  To resolve this, you could:")
        for j, suggestion in enumerate(v.suggestions, 1):
            lines.append(f"    {j}. {suggestion}")
        lines.append("")
    lines.append("Please describe how you'd like to resolve this:")
    return "\n".join(lines)


def _format_options_question(category: str, options: dict[str, str], error: str = "") -> str:
    label = category.replace("_", " ").capitalize()
    lines = []
    if error:
        lines += [f"⚠ {error}", ""]
    lines.append(f"{label} — please choose an option:")
    lines.append("")
    for letter, text in options.items():
        lines.append(f"  {letter}) {text}")
    lines.append("")
    lines.append("Enter a letter, or type 'skip' to leave this open.")
    return "\n".join(lines)


def _should_spell_check(text: str) -> bool:
    """Return True for substantive free-text worth checking (not a control token)."""
    stripped = text.strip()
    return len(stripped) >= 4 and not _is_confirm(stripped) and not _is_skip(stripped)


def _spell_check_with_context(
    text: str,
    query: str,
    recent_messages: list[dict],
    model_name: str,
    temperature: float = 0.0,
) -> tuple[str, str | None]:
    """LLM-based context-aware spell check.

    Returns (corrected_text, prompt_message) if typos were found,
    or (original_text, None) if the text is clean or the check fails.
    """
    context_lines = [f"Original trip query: {query.strip()}"]
    relevant = [m for m in recent_messages if m.get("role") in ("user", "assistant")][-6:]
    if relevant:
        context_lines.append("Recent conversation:")
        for m in relevant:
            label = "Agent" if m["role"] == "assistant" else "User"
            context_lines.append(f"  {label}: {str(m.get('content', ''))[:120]}")

    user_prompt = "\n".join([
        *context_lines,
        "",
        f'Text to spell-check: "{text}"',
        "",
        "Return strictly valid JSON:",
        '{"corrections": [{"original": "...", "corrected": "..."}], "corrected_text": "..."}',
        'If no typos: {"corrections": [], "corrected_text": "<original text unchanged>"}',
    ])

    try:
        result, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=SPELL_CHECK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=SpellCheckResponse,
        )
    except Exception:
        return text, None

    if not result.corrections:
        return text, None

    corrected = result.corrected_text.strip() or text
    summary = ", ".join(f"'{c.original}' → '{c.corrected}'" for c in result.corrections)
    msg = (
        f"Possible typo(s) detected: {summary}\n"
        f"Corrected text: \"{corrected}\"\n"
        "Accept correction? (ok / skip)"
    )
    return corrected, msg


def _build_message_history(messages: list[dict]) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="constraint_iteration_agent",
        model="llm",
        agent_ref="travelplanner.agents.constraint_iteration_agent",
        messages=messages,
    )


def _is_skip(text: str) -> bool:
    return text.strip().lower() in _SKIP_TOKENS


def _is_confirm(text: str) -> bool:
    return text.strip().lower() in _CONFIRM_TOKENS


# ── Artifact ─────────────────────────────────────────────────────────────────

def _build_artifact_node(state: ConstraintIterationState) -> dict[str, Any]:
    hard = state.hard_constraints
    content = ConstraintArtifactContentModel(
        query=state.query,
        status="success" if hard else "partial",
        hard_constraints=[c.model_dump() for c in hard],
        commonsense_constraints=[c.model_dump() for c in state.commonsense_constraints],
        categories_missing=state.categories_skipped,
        categories_skipped_by_user=state.categories_skipped,
        interaction_turns=len([m for m in state.messages if m["role"] == "user"]),
        model=state.model_name,
    )
    artifact = AgentArtifactModel(
        name="constraint_agent",
        type="constraint-extraction-result",
        content=content.model_dump(mode="json"),
        description=f"Constraint extraction for: {state.query[:80]}",
    )
    artifacts = dict(state.agent_artifacts)
    artifacts["constraint_agent"] = [artifact]
    return {"agent_artifacts": artifacts}


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

        if _should_spell_check(user_input):
            corrected, spell_msg = _spell_check_with_context(
                user_input, state.query, messages, state.model_name, state.temperature
            )
            if spell_msg:
                messages = [*messages, {"role": "assistant", "content": spell_msg}]
                accept: str = interrupt(spell_msg)
                messages = [*messages, {"role": "user", "content": accept}]
                if _is_confirm(accept):
                    user_input = corrected

        messages = [*messages, {"role": "user", "content": user_input}]

        if _is_confirm(user_input):
            next_phase: Literal["missing", "extract"] = (
                "missing" if state.missing_categories else "extract"
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
        new_hard_constraints = list(state.hard_constraints)

        # ── Structured options (accommodation / transport / purpose) ──────────
        if category in STRUCTURED_OPTIONS:
            options = STRUCTURED_OPTIONS[category]
            valid_letters = set(options.keys())

            # Detect retry: last assistant message was an options question for this category
            last_assistant = next(
                (m["content"] for m in reversed(state.messages) if m["role"] == "assistant"),
                "",
            )
            error_prefix = ""
            if category.replace("_", " ").capitalize() in last_assistant and "please choose" in last_assistant:
                error_prefix = "Invalid input. Please enter one of: " + ", ".join(valid_letters) + "."

            question = _format_options_question(category, options, error=error_prefix)
            messages = [*state.messages, {"role": "assistant", "content": question}]

            letter_input: str = interrupt(question)
            messages = [*messages, {"role": "user", "content": letter_input}]

            if _is_skip(letter_input):
                new_hard_constraints.append(
                    ConstraintModel(type="hard", text=f"{category}: not specified", user_skipped=True)
                )
                return {
                    "messages": messages,
                    "hard_constraints": new_hard_constraints,
                    "category_index": state.category_index + 1,
                    "categories_skipped": [*state.categories_skipped, category],
                }

            letter = letter_input.strip().lower().rstrip(")")
            if letter not in valid_letters:
                # Invalid — return without incrementing so routing re-asks
                return {"messages": messages}

            selected_text = options[letter]

            # ── Optional free-text follow-up ──────────────────────────────────
            followup_q = (
                f"Selected: {selected_text}.\n"
                "Any additional notes or preferences? (press Enter to skip)"
            )
            messages = [*messages, {"role": "assistant", "content": followup_q}]
            free_text: str = interrupt(followup_q)

            if free_text.strip() and not _is_skip(free_text):
                if _should_spell_check(free_text):
                    corrected, spell_msg = _spell_check_with_context(
                        free_text, state.query, messages, state.model_name, state.temperature
                    )
                    if spell_msg:
                        messages = [*messages, {"role": "assistant", "content": spell_msg}]
                        accept: str = interrupt(spell_msg)
                        messages = [*messages, {"role": "user", "content": accept}]
                        if _is_confirm(accept):
                            free_text = corrected

                messages = [*messages, {"role": "user", "content": free_text}]
                selected_text = f"{selected_text} ({free_text.strip()})"
            else:
                messages = [*messages, {"role": "user", "content": free_text}]

            new_hard_constraints.append(
                ConstraintModel(type="hard", text=f"{category}: {selected_text}", user_skipped=False)
            )
            return {
                "messages": messages,
                "hard_constraints": new_hard_constraints,
                "category_index": state.category_index + 1,
            }

        # ── Free-text categories (destination / travel_dates / budget) ────────
        question = CATEGORY_QUESTIONS.get(
            category,
            f"No '{category}' was specified. Please provide details, or type 'skip'.",
        )
        messages = [*state.messages, {"role": "assistant", "content": question}]
        user_input: str = interrupt(question)

        if not _is_skip(user_input) and _should_spell_check(user_input):
            corrected, spell_msg = _spell_check_with_context(
                user_input, state.query, messages, state.model_name, state.temperature
            )
            if spell_msg:
                messages = [*messages, {"role": "assistant", "content": spell_msg}]
                accept: str = interrupt(spell_msg)
                messages = [*messages, {"role": "user", "content": accept}]
                if _is_confirm(accept):
                    user_input = corrected

        messages = [*messages, {"role": "user", "content": user_input}]

        if not _is_skip(user_input):
            new_hard_constraints.append(
                ConstraintModel(type="hard", text=f"{category}: {user_input.strip()}", user_skipped=False)
            )
            return {
                "messages": messages,
                "hard_constraints": new_hard_constraints,
                "category_index": state.category_index + 1,
            }

        new_hard_constraints.append(
            ConstraintModel(type="hard", text=f"{category}: not specified", user_skipped=True)
        )
        return {
            "messages": messages,
            "hard_constraints": new_hard_constraints,
            "category_index": state.category_index + 1,
            "categories_skipped": [*state.categories_skipped, category],
        }

    def check_commonsense_violations(state: ConstraintIterationState) -> dict[str, Any]:
        structured_output, _, _ = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=VIOLATION_CHECK_SYSTEM_PROMPT,
            user_prompt=_build_violation_check_prompt(
                state.query,
                state.query_context,
                state.hard_constraints,
                state.commonsense_constraints,
            ),
            response_model=ConstraintViolationCheckResponse,
        )
        real_violations = [
            v for v in structured_output.violations
            if v.is_violated
        ]
        return {"violations": real_violations}

    def present_violations(state: ConstraintIterationState) -> dict[str, Any]:
        agent_msg = _format_violation_message(state.violations)
        messages = [*state.messages, {"role": "assistant", "content": agent_msg}]

        user_input: str = interrupt(agent_msg)

        if _should_spell_check(user_input):
            corrected, spell_msg = _spell_check_with_context(
                user_input, state.query, messages, state.model_name, state.temperature
            )
            if spell_msg:
                messages = [*messages, {"role": "assistant", "content": spell_msg}]
                accept: str = interrupt(spell_msg)
                messages = [*messages, {"role": "user", "content": accept}]
                if _is_confirm(accept):
                    user_input = corrected

        messages = [*messages, {"role": "user", "content": user_input}]

        context = (
            f"{state.query_context}\n{user_input}".strip()
            if state.query_context
            else user_input
        )
        return {
            "messages": messages,
            "query_context": context,
            "violations": [],
        }

    def _route_after_check(state: ConstraintIterationState) -> str:
        if state.violations:
            return "present_violations"
        return "present_hard_constraints"

    def _route_after_present_hard(state: ConstraintIterationState) -> str:
        if state.phase == "extract":
            return "extract_hard_constraints"
        if state.missing_categories and state.category_index < len(state.missing_categories):
            return "ask_missing_category"
        return "build_artifact"

    def _route_after_missing(state: ConstraintIterationState) -> str:
        if state.category_index < len(state.missing_categories):
            return "ask_missing_category"
        return "build_artifact"

    graph = StateGraph(ConstraintIterationState)
    graph.add_node("extract_hard_constraints", extract_hard_constraints)
    graph.add_node("check_commonsense_violations", check_commonsense_violations)
    graph.add_node("present_violations", present_violations)
    graph.add_node("present_hard_constraints", present_hard_constraints)
    graph.add_node("ask_missing_category", ask_missing_category)
    graph.add_node("build_artifact", _build_artifact_node)

    graph.set_entry_point("extract_hard_constraints")
    graph.add_edge("extract_hard_constraints", "check_commonsense_violations")
    graph.add_conditional_edges("check_commonsense_violations", _route_after_check)
    graph.add_edge("present_violations", "extract_hard_constraints")
    graph.add_conditional_edges("present_hard_constraints", _route_after_present_hard)
    graph.add_conditional_edges("ask_missing_category", _route_after_missing)
    graph.add_edge("build_artifact", END)

    return graph.compile(checkpointer=MemorySaver())


def make_pipeline_graph():
    """Non-interactive graph for automated pipelines.

    Runs extraction and violation check without any LangGraph interrupts.
    Use get_constraint_list() and get_message_history() to read results.
    """
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
        }

    def check_commonsense_violations(state: ConstraintIterationState) -> dict[str, Any]:
        structured_output, _, _ = invoke_structured_model(
            model_name=state.model_name,
            temperature=state.temperature,
            system_prompt=VIOLATION_CHECK_SYSTEM_PROMPT,
            user_prompt=_build_violation_check_prompt(
                state.query,
                state.query_context,
                state.hard_constraints,
                state.commonsense_constraints,
            ),
            response_model=ConstraintViolationCheckResponse,
        )
        real_violations = [
            v for v in structured_output.violations
            if v.is_violated
        ]
        return {"violations": real_violations}

    graph = StateGraph(ConstraintIterationState)
    graph.add_node("extract_hard_constraints", extract_hard_constraints)
    graph.add_node("check_commonsense_violations", check_commonsense_violations)
    graph.add_node("build_artifact", _build_artifact_node)
    graph.set_entry_point("extract_hard_constraints")
    graph.add_edge("extract_hard_constraints", "check_commonsense_violations")
    graph.add_edge("check_commonsense_violations", "build_artifact")
    graph.add_edge("build_artifact", END)
    return graph.compile()


def get_constraint_list(state: dict[str, Any]) -> list[ConstraintModel]:
    """Returns the final merged constraint list from a completed graph state."""
    hard: list[ConstraintModel] = state.get("hard_constraints", [])
    commonsense: list[ConstraintModel] = state.get("commonsense_constraints", [])
    return hard + commonsense


def get_message_history(state: dict[str, Any]) -> MessageHistoryModel:
    """Returns the MessageHistoryModel from a completed graph state."""
    return _build_message_history(state.get("messages", []))
