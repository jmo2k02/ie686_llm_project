"""Planner-specific constants and task type guidance.

The canonical task type values come from TaskModel. This module only defines
planner-facing labels, descriptions, keywords, and history keys used to turn
constraints into downstream search tasks.
"""

from travelplanner.schema.system_state import get_allowed_task_types

PLANNER_HISTORY_KEY = "planner_agent"
PLANNER_REVIEWER_HISTORY_KEY = "planner_reviewer_agent"

ALLOWED_TASK_TYPES: tuple[str, ...] = tuple(get_allowed_task_types())
TASK_TYPE_GUIDANCE: dict[str, str] = {
    "flight": "Use for flight discovery or comparison when air travel is explicitly needed.",
    "hotel": "Use for lodging or accommodation selection for overnight stays.",
    "restaurant": "Use for dining, cuisine, or meal-specific recommendations.",
    "attraction": "Use for museums, landmarks, activities, sightseeing, or events to visit.",
    "opening_times": "Use to verify operating hours or reservation feasibility for time-sensitive places.",
    "routing-check": "Use to verify travel time, transit feasibility, or distance between planned stops.",
    "general-web-search": "Use for edge-case research that does not cleanly fit a specialized task type.",
}

TASK_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
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
"""keywords that might allow to infer that a certain task type is needed"""

STOPWORDS = {
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
