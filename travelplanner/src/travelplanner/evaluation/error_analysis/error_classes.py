from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorClass(str, Enum):
    REPEATED_TOOL_CALL = "REPEATED_TOOL_CALL"
    REPEATED_QUERY = "REPEATED_QUERY"
    DEAD_LOOP = "DEAD_LOOP"
    CASCADING_ERROR = "CASCADING_ERROR"
    HANDOFF_FAILURE = "HANDOFF_FAILURE"


@dataclass
class ToolCallRecord:
    tool_name: str
    inputs_json: str        # json.dumps(args, sort_keys=True)
    position: int           # index in the flattened call sequence (no timestamps in traces)
    error: str | None = None
    agent_name: str | None = None


@dataclass
class RunClassification:
    run_id: str
    source: str             # "baseline" | "travel_agent"
    error_classes: list[ErrorClass] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)

    def is_clean(self) -> bool:
        return len(self.error_classes) == 0


@dataclass
class ErrorReport:
    classifications: list[RunClassification]
    per_class: dict[str, int]
    per_class_pct: dict[str, float]
    total_runs: int
    clean_runs: int


@dataclass
class AnalysisConfig:
    repeated_query_threshold: float = 0.85
    dead_loop_min_repetitions: int = 3
    # agents in travel_agent message_histories that must be present
    expected_travel_agents: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "key",                   # constraint_iteration_agent (serialized under "key")
            "planner_agent",
            "planner_reviewer_agent",
            "itinerary_validator",
        })
    )
