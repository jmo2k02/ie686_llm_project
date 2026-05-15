from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from travelplanner.baseline_agent.agent import run_baseline
from travelplanner.baseline_agent.config import load_config_from_env


class BaselineCase(BaseModel):
    name: str | None = Field(default=None, description="Stable output filename stem.")
    id: str | None = None
    type: str | None = None
    description: str | None = None
    hard_constraints: Any = Field(default_factory=dict)
    query: str = Field(description="User travel-planning request.")
    constraints: Any = Field(
        default_factory=dict,
        description="Hard and soft constraints as a string, list, or mapping.",
    )

    def planning_constraints(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if self.hard_constraints not in ({}, [], None, ""):
            merged["hard_constraints"] = self.hard_constraints
        if self.constraints not in ({}, [], None, ""):
            merged["constraints"] = self.constraints
        return merged


def _load_cases(path: Path) -> list[BaselineCase]:
    if not path.exists():
        raise ValueError(f"Input JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, dict) and "cases" in payload:
        raw_cases = payload["cases"]
    elif isinstance(payload, dict):
        raw_cases = [payload]
    else:
        raise ValueError("Input JSON must be one case, a case list, or {'cases': [...]}.")

    if not isinstance(raw_cases, list):
        raise ValueError("Input JSON field 'cases' must be a list.")
    return [BaselineCase.model_validate(item) for item in raw_cases]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "baseline-case"


def _write_markdown(output_dir: Path, case: BaselineCase, index: int, markdown: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = case.name or f"{case.id}"
    path = output_dir / f"{index:02d}-{_slugify(stem)}.md"
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path


def run_cases(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    config = load_config_from_env()
    effective_output_dir = output_dir or config.output_dir
    written_paths: list[Path] = []
    for index, case in enumerate(_load_cases(input_path), start=1):
        result = run_baseline(
            query=case.query,
            constraints=case.planning_constraints(),
            config=config,
        )
        path = _write_markdown(effective_output_dir, case, index, result.markdown)
        _ = print(
            f"wrote {path} "
            f"(model={result.model_name}, "
            f"tavily_executed={result.executed_tool_calls}, "
            f"tool_slots_requested={result.requested_tool_calls})"
        )
        written_paths.append(path)
    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the simple Tavily-only baseline agent over JSON cases."
    )
    _ = parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON case file. Accepts one case, a list, or {'cases': [...]}.",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for markdown outputs. Defaults to agents.baseline_agent.output_dir.",
    )
    args = parser.parse_args()
    try:
        _ = run_cases(args.input, args.output_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, ValidationError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
