#!/usr/bin/env python3
"""Extract the structured TravelPlan JSON from a local trace export.

Usage:
    uv run python scripts/extract_travelplan.py 019e31a3
    uv run python scripts/extract_travelplan.py run-019e31a3-434b-7083-8b8c-e909e9119ac5
    uv run python scripts/extract_travelplan.py travel_agent/run-019e31a3-434b-7083-8b8c-e909e9119ac5.json > travelplan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIRS = [ROOT / "travel_agent", ROOT / "baseline"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_trace(run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path)
    if candidate.exists():
        return candidate.resolve()

    candidate = ROOT / run_id_or_path
    if candidate.exists():
        return candidate.resolve()

    needle = run_id_or_path.removesuffix(".json")
    if needle.startswith("run-"):
        needle = needle[4:]

    matches = []
    for trace_dir in TRACE_DIRS:
        matches.extend(path for path in trace_dir.glob("*.json") if needle in path.stem)

    if not matches:
        raise SystemExit(f"No trace JSON found for run id/path: {run_id_or_path}")
    if len(matches) > 1:
        formatted = "\n".join(str(path.relative_to(ROOT)) for path in matches)
        raise SystemExit(f"Run id is ambiguous. Matches:\n{formatted}")
    return matches[0].resolve()


def extract_travelplan(path: Path) -> dict:
    obj = load_json(path)
    outputs = obj.get("outputs") or {}
    travelplan = outputs.get("travelplan")

    if not isinstance(travelplan, dict):
        raise SystemExit(f"No structured outputs.travelplan found in {path.relative_to(ROOT)}")
    return travelplan


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract outputs.travelplan from a local trace JSON export.")
    parser.add_argument("run_id", help="Full/partial run id, filename, or path to a JSON export")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON instead of pretty JSON")
    args = parser.parse_args()

    path = find_trace(args.run_id)
    travelplan = extract_travelplan(path)
    indent = None if args.compact else 2
    print(json.dumps(travelplan, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()
