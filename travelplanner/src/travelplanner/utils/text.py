"""General text helpers shared across TravelPlanner modules.

This module contains pure string utilities and lightweight text heuristics.
Functions here should not know about planner task types, LangGraph state, LLM
prompts, or agent-specific behavior.
"""

import re
from collections.abc import Collection


_DAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def extract_keywords(text: str, stopwords: Collection[str] = frozenset()) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in stopwords}


def normalize_whitespace(text: str | None) -> str:
    return " ".join((text or "").split())


def trip_likely_has_overnight_stay(text: str) -> bool:
    return bool(
        re.search(
            r"\b(\d+\s*(day|days|night|nights|week|weeks)|overnight)\b",
            text.lower(),
        )
    )


def mentions_specific_time(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b", lowered)) or any(
        day in lowered for day in _DAY_NAMES
    )
