from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from spellchecker import SpellChecker

from travelplanner.schema.commonsense_constraints import COMMONSENSE_CONSTRAINTS
from travelplanner.schema.system_state import ConstraintModel, MessageHistoryModel
from travelplanner.utils.checkpoint import make_memory_checkpointer
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

_SPELL_CHECKER = SpellChecker()

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

Your job is to find constraints that are ACTUALLY BROKEN — not constraints that are satisfied, not constraints that are merely relevant.

Rules:
- Return JSON only.
- ONLY include a constraint in the violations list if it is definitively and clearly broken.
- If a constraint is satisfied or cannot be assessed from the given information, do NOT include it.
- Before adding any item to violations, verify: "Is this constraint actually broken right now?" If the answer is no or maybe, leave it out.
- For each real violation, provide a short explanation of WHY it is broken and exactly two concrete suggestions to fix it.
- If nothing is broken, return {"violations": []}.
- Do not include constraints that are met, assumed to be met, or only potentially relevant.
- Always use the current date provided in the prompt to assess whether dates are in the past or future.
- Base your assessment solely on the constraints as they are currently listed — ignore any previously invalid values that the user has already corrected.

Examples of what NOT to include:
- A trip date that is in the future → NOT a violation of "trip must be in the future"
- An end date that is after the start date → NOT a violation of "end must be after start"
- Accommodation not yet booked → NOT a violation if the user hasn't been asked yet
"""

_SKIP_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "skip",
        "s",
        "no",
        "nein",
        "n",
        "nope",
        "egal",
        "doesn't matter",
        "doesnt matter",
        "don't care",
        "dont care",
        "not applicable",
        "na",
        "n/a",
        "none",
        "nothing",
    }
)

_CONFIRM_TOKENS: frozenset[str] = frozenset(
    {
        "ok",
        "yes",
        "ja",
        "y",
        "correct",
        "fine",
        "sure",
        "sounds good",
        "confirmed",
        "confirm",
        "good",
        "great",
        "looks good",
        "looks correct",
        "that's right",
        "thats right",
    }
)


# ── State ────────────────────────────────────────────────────────────────────


class ViolationModel(BaseModel):
    violated_constraint: str
    explanation: str
    suggestions: list[str] = Field(default_factory=list)


class ConstraintIterationState(BaseModel):
    query: str
    message_histories: dict[str, dict]
    model_name: str
    temperature: float = 0.0

    query_context: str = ""
    phase: Literal["extract", "missing"] = "extract"

    hard_constraints: list[ConstraintModel] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    category_index: int = 0

    commonsense_constraints: list[ConstraintModel] = Field(
        default_factory=lambda: [c.model_copy() for c in COMMONSENSE_CONSTRAINTS]
    )
    constraint_index: int = 0

    violations: list[ViolationModel] = Field(default_factory=list)

    messages: list[dict] = Field(default_factory=list)


# ── LLM response models ──────────────────────────────────────────────────────


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
        json.dumps(
            [c.model_dump() for c in hard_constraints], indent=2, ensure_ascii=True
        ),
        "",
        "Commonsense constraints to check against:",
        json.dumps(
            [c.text for c in commonsense_constraints], indent=2, ensure_ascii=True
        ),
        "",
        "Return strictly valid JSON with this shape:",
        '{"violations": [{"violated_constraint": "...", "explanation": "...", '
        '"suggestions": ["suggestion 1", "suggestion 2"]}]}',
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


def _format_options_question(
    category: str, options: dict[str, str], error: str = ""
) -> str:
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


def _spell_check_text(text: str) -> tuple[str, str | None]:
    """Returns (corrected_text, message_or_None). Only flags clear single-word corrections."""
    words = text.split()
    misspelled = _SPELL_CHECKER.unknown(words)
    corrections: dict[str, str] = {}
    for word in misspelled:
        suggestion = _SPELL_CHECKER.correction(word)
        if suggestion and suggestion.lower() != word.lower() and len(word) > 3:
            corrections[word] = suggestion

    if not corrections:
        return text, None

    corrected = text
    for original, fix in corrections.items():
        corrected = corrected.replace(original, fix)

    summary = ", ".join(f"'{k}' → '{v}'" for k, v in corrections.items())
    msg = (
        f"Possible typo(s) detected: {summary}\n"
        f'Corrected text: "{corrected}"\n'
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


# ── Response helpers ─────────────────────────────────────────────────────────

_FALSE_POSITIVE_PHRASES: frozenset[str] = frozenset(
    {
        "there are no violations",
        "no violation",
        "is valid",
        "is correct",
        "is in the future",
        "which is valid",
        "which is correct",
        "are no violations",
        "not a violation",
        "is not violated",
        "does not violate",
        "is satisfied",
        "are satisfied",
    }
)


def _is_false_positive(violation: ViolationModel) -> bool:
    """Returns True if the LLM explanation itself says the constraint is actually fine."""
    explanation_lower = violation.explanation.lower()
    return any(phrase in explanation_lower for phrase in _FALSE_POSITIVE_PHRASES)


def _is_skip(text: str) -> bool:
    return text.strip().lower() in _SKIP_TOKENS


def _is_confirm(text: str) -> bool:
    return text.strip().lower() in _CONFIRM_TOKENS


# ── Graph ────────────────────────────────────────────────────────────────────


def make_graph() -> StateGraph:
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
                c
                for c in structured_output.missing_categories
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
                (
                    m["content"]
                    for m in reversed(state.messages)
                    if m["role"] == "assistant"
                ),
                "",
            )
            error_prefix = ""
            if (
                category.replace("_", " ").capitalize() in last_assistant
                and "please choose" in last_assistant
            ):
                error_prefix = (
                    "Invalid input. Please enter one of: "
                    + ", ".join(valid_letters)
                    + "."
                )

            question = _format_options_question(category, options, error=error_prefix)
            messages = [*state.messages, {"role": "assistant", "content": question}]

            letter_input: str = interrupt(question)
            messages = [*messages, {"role": "user", "content": letter_input}]

            if _is_skip(letter_input):
                return {
                    "messages": messages,
                    "category_index": state.category_index + 1,
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
            messages = [*messages, {"role": "user", "content": free_text}]

            if free_text.strip() and not _is_skip(free_text):
                # ── Spell check ───────────────────────────────────────────────
                corrected, spell_msg = _spell_check_text(free_text)
                if spell_msg:
                    messages = [*messages, {"role": "assistant", "content": spell_msg}]
                    accept: str = interrupt(spell_msg)
                    messages = [*messages, {"role": "user", "content": accept}]
                    if _is_confirm(accept):
                        free_text = corrected

                selected_text = f"{selected_text} ({free_text.strip()})"

            new_hard_constraints.append(
                ConstraintModel(type="hard", text=selected_text, user_skipped=False)
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
        messages = [*messages, {"role": "user", "content": user_input}]

        if not _is_skip(user_input):
            new_hard_constraints.append(
                ConstraintModel(
                    type="hard", text=user_input.strip(), user_skipped=False
                )
            )

        return {
            "messages": messages,
            "hard_constraints": new_hard_constraints,
            "category_index": state.category_index + 1,
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
            v for v in structured_output.violations if not _is_false_positive(v)
        ]
        return {"violations": real_violations}

    def present_violations(state: ConstraintIterationState) -> dict[str, Any]:
        agent_msg = _format_violation_message(state.violations)
        messages = [*state.messages, {"role": "assistant", "content": agent_msg}]

        user_input: str = interrupt(agent_msg)
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
        if state.missing_categories and state.category_index < len(
            state.missing_categories
        ):
            return "ask_missing_category"
        return END

    def _route_after_missing(state: ConstraintIterationState) -> str:
        if state.category_index < len(state.missing_categories):
            return "ask_missing_category"
        return END
    
    def finalize_constraint_output(state: ConstraintIterationState) -> dict[str, Any]:
        return {
            "constraint_list": [
                *state.hard_constraints,
                *state.commonsense_constraints
            ],
            "message_histories": {
                **state.message_histories,

            }
        }

    graph = StateGraph(ConstraintIterationState)
    graph.add_node("extract_hard_constraints", extract_hard_constraints)
    graph.add_node("check_commonsense_violations", check_commonsense_violations)
    graph.add_node("present_violations", present_violations)
    graph.add_node("present_hard_constraints", present_hard_constraints)
    graph.add_node("ask_missing_category", ask_missing_category)

    graph.set_entry_point("extract_hard_constraints")
    graph.add_edge("extract_hard_constraints", "check_commonsense_violations")
    graph.add_conditional_edges("check_commonsense_violations", _route_after_check)
    graph.add_edge("present_violations", "extract_hard_constraints")
    graph.add_conditional_edges("present_hard_constraints", _route_after_present_hard)
    graph.add_conditional_edges("ask_missing_category", _route_after_missing)

    return graph


def get_constraint_list(state: dict[str, Any]) -> list[ConstraintModel]:
    """Returns the final merged constraint list from a completed graph state."""
    hard: list[ConstraintModel] = state.get("hard_constraints", [])
    commonsense: list[ConstraintModel] = state.get("commonsense_constraints", [])
    return hard + commonsense


def get_message_history(state: dict[str, Any]) -> MessageHistoryModel:
    """Returns the MessageHistoryModel from a completed graph state."""
    return _build_message_history(state.get("messages", []))
