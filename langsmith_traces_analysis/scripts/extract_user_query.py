#!/usr/bin/env python3
"""Extract the original user query from a local LangSmith JSON export.

Usage:
    uv run python scripts/extract_user_query.py 019e31d9
    uv run python scripts/extract_user_query.py run-019e31d9-cb29-70f1-87d9-a7166645dacc
    uv run python scripts/extract_user_query.py baseline/run-019e31d9-cb29-70f1-87d9-a7166645dacc.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIRS = [ROOT / "baseline", ROOT / "travel_agent"]


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


def extract_travel_agent_query(obj: dict) -> str:
    outputs = obj.get("outputs") or {}
    inputs = obj.get("inputs") or {}

    if isinstance(outputs.get("query"), str) and outputs["query"].strip():
        return outputs["query"].strip()

    resume = ((inputs.get("input") or {}).get("resume")) if isinstance(inputs.get("input"), dict) else None
    if isinstance(resume, str) and resume.strip():
        return resume.strip()

    return ""


def extract_baseline_query(obj: dict) -> str:
    messages = (obj.get("outputs") or {}).get("messages") or []
    for msg in messages:
        content = msg.get("content") or ""
        match = re.search(r"USER QUERY\s*(.*?)\n\nCONSTRAINTS", content, re.S)
        if match:
            return match.group(1).strip()

    for msg in messages:
        content = msg.get("content") or ""
        if msg.get("type") in {"human", "user"} or msg.get("role") in {"human", "user"}:
            if content.strip():
                return content.strip()

    return ""


def extract_user_query(path: Path) -> str:
    obj = load_json(path)
    if "travel_agent" in path.parts:
        query = extract_travel_agent_query(obj)
    else:
        query = extract_baseline_query(obj)

    if not query:
        raise SystemExit(f"No user query found in {path.relative_to(ROOT)}")
    return query


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract original user query from a baseline/ or travel_agent/ trace JSON.")
    parser.add_argument("run_id", help="Full/partial run id, filename, or path to a JSON export")
    parser.add_argument("--json", action="store_true", help="Print metadata and query as JSON")
    args = parser.parse_args()

    path = find_trace(args.run_id)
    query = extract_user_query(path)

    if args.json:
        print(json.dumps({"file": str(path.relative_to(ROOT)), "run_id": path.stem.removeprefix("run-"), "query": query}, ensure_ascii=False, indent=2))
    else:
        print(query)


if __name__ == "__main__":
    main()
