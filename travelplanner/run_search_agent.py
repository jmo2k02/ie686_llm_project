from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from travelplanner.agents.general_web_search_agent import (
    make_graph as make_general_web_search_graph,
)
from travelplanner.schema.general_web_search_artifact import (
    GeneralWebSearchArtifactContentModel,
)
from travelplanner.schema.system_state import TaskModel


@dataclass(frozen=True)
class SearchAgentSpec:
    task_type: str
    artifact_key: str
    artifact_model: type[BaseModel]
    make_graph: Callable[[], Any]
    required_env: tuple[str, ...]
    default_task: str


SEARCH_AGENT_SPECS: dict[str, SearchAgentSpec] = {
    "general_web_search": SearchAgentSpec(
        task_type="general-web-search",
        artifact_key="general_web_search_agent",
        artifact_model=GeneralWebSearchArtifactContentModel,
        make_graph=make_general_web_search_graph,
        required_env=("TAVILY_API_KEY",),
        default_task=(
            "Best city areas with practical transit access, useful nearby food options, "
            "and opening-hour caveats relevant for planning"
        ),
    )
}


def _ensure_required_env(required_env: tuple[str, ...]) -> None:
    missing = [name for name in required_env if not os.getenv(name, "").strip()]
    if missing:
        raise SystemExit(
            "Missing required env vars for this run: " + ", ".join(missing)
        )


def _print_report(content: BaseModel) -> None:
    payload = content.model_dump(mode="json")
    print("\n=== Search Agent Output Report ===")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def run_once(*, agent_name: str, query: str, task_text: str) -> int:
    spec = SEARCH_AGENT_SPECS[agent_name]
    _ensure_required_env(spec.required_env)
    graph = spec.make_graph()
    result: dict[str, Any] = graph.invoke(
        {
            "query": query,
            "task_list": [
                TaskModel(
                    name=f"{agent_name}-demo-1",
                    type=spec.task_type,  # type: ignore[arg-type]
                    text=task_text,
                    is_valid=True,
                    validation_comment=None,
                )
            ],
            "agent_artifacts": {},
        }
    )
    artifacts = result.get("agent_artifacts", {}).get(spec.artifact_key, [])
    if not artifacts:
        print("No artifact returned.")
        return 1

    artifact = artifacts[-1]
    raw_content = (
        artifact["content"] if isinstance(artifact, dict) else artifact.content
    )
    parsed = spec.artifact_model.model_validate(raw_content)
    _print_report(parsed)
    if hasattr(parsed, "final_answer"):
        return 0 if str(getattr(parsed, "final_answer", "")).strip() else 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a search agent LangGraph and validate structured artifact output."
    )
    parser.add_argument(
        "--agent",
        choices=sorted(SEARCH_AGENT_SPECS.keys()),
        default="general_web_search",
        help="Which search agent to run from the registry.",
    )
    parser.add_argument(
        "--query",
        default="Help me plan a practical 3-day city trip.",
        help="Top-level user query context.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task text to execute. If omitted, uses agent default task template.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the test suite instead of the agent.",
    )
    args = parser.parse_args()
    if args.run_tests:
        raise SystemExit(
            subprocess.call(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_*.py",
                    "-v",
                ]
            )
        )
    spec = SEARCH_AGENT_SPECS[args.agent]
    task_text = args.task or spec.default_task
    raise SystemExit(
        run_once(agent_name=args.agent, query=args.query, task_text=task_text)
    )


if __name__ == "__main__":
    main()
