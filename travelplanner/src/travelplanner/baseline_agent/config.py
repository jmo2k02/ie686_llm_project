from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from travelplanner.config import get_setting


SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]
_VALID_SEARCH_DEPTHS: set[str] = {"basic", "advanced", "fast", "ultra-fast"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_rooted_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


@dataclass(frozen=True)
class BaselineAgentConfig:
    model_name: str
    temperature: float
    recursion_limit: int
    min_tool_calls: int
    max_tool_calls: int
    output_dir: Path
    tavily_max_results: int
    tavily_search_depth: SearchDepth
    tavily_include_answer: bool


def _coerce_search_depth(value: str) -> SearchDepth:
    if value not in _VALID_SEARCH_DEPTHS:
        raise ValueError(
            f"agents.baseline_agent.tavily.search_depth must be one of "
            f"{sorted(_VALID_SEARCH_DEPTHS)}; got {value!r}."
        )
    return cast(SearchDepth, value)


def load_config_from_env() -> BaselineAgentConfig:
    model_name = _env("TRAVELPLANNER_BASELINE_AGENT_MODEL") or str(
        get_setting(
            "agents.baseline_agent.model_name",
            "openrouter:minimax/minimax-m2.5",
        )
    )
    temperature = _env_float(
        "TRAVELPLANNER_BASELINE_AGENT_TEMPERATURE",
        float(get_setting("agents.baseline_agent.temperature", 0.2)),
    )
    recursion_limit = _env_int(
        "TRAVELPLANNER_BASELINE_AGENT_RECURSION_LIMIT",
        int(get_setting("agents.baseline_agent.recursion_limit", 20)),
    )
    max_tool_calls = _env_int(
        "TRAVELPLANNER_BASELINE_AGENT_MAX_TOOL_CALLS",
        int(get_setting("agents.baseline_agent.max_tool_calls", 4)),
    )
    min_tool_calls = _env_int(
        "TRAVELPLANNER_BASELINE_AGENT_MIN_TOOL_CALLS",
        int(get_setting("agents.baseline_agent.min_tool_calls", 1)),
    )
    if max_tool_calls < 1:
        raise ValueError(
            "agents.baseline_agent.max_tool_calls must be at least 1; "
            f"got {max_tool_calls}."
        )
    if min_tool_calls < 0:
        raise ValueError(
            "agents.baseline_agent.min_tool_calls must be non-negative; "
            f"got {min_tool_calls}."
        )
    if min_tool_calls > max_tool_calls:
        min_tool_calls = max_tool_calls
    output_dir = _resolve_rooted_path(
        _env("TRAVELPLANNER_BASELINE_AGENT_OUTPUT_DIR")
        or str(get_setting("agents.baseline_agent.output_dir", ".output/baseline_agent"))
    )
    tavily_max_results = _env_int(
        "TRAVELPLANNER_BASELINE_AGENT_TAVILY_MAX_RESULTS",
        int(get_setting("agents.baseline_agent.tavily.max_results", 5)),
    )
    tavily_search_depth = _coerce_search_depth(
        _env("TRAVELPLANNER_BASELINE_AGENT_TAVILY_SEARCH_DEPTH")
        or str(get_setting("agents.baseline_agent.tavily.search_depth", "basic"))
    )
    tavily_include_answer = _env_bool(
        "TRAVELPLANNER_BASELINE_AGENT_TAVILY_INCLUDE_ANSWER",
        bool(get_setting("agents.baseline_agent.tavily.include_answer", True)),
    )

    return BaselineAgentConfig(
        model_name=model_name,
        temperature=temperature,
        recursion_limit=recursion_limit,
        min_tool_calls=min_tool_calls,
        max_tool_calls=max_tool_calls,
        output_dir=output_dir,
        tavily_max_results=tavily_max_results,
        tavily_search_depth=tavily_search_depth,
        tavily_include_answer=tavily_include_answer,
    )
