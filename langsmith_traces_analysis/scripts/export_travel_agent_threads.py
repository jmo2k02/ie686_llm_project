#!/usr/bin/env python3
"""Export notebook-friendly LangSmith thread data for local travel_agent traces.

Offline mode always works and writes a manifest from local JSON exports.
Online mode uses LANGSMITH_API_KEY to fetch LangSmith runs grouped by thread_id.

Usage:
    uv run python scripts/export_travel_agent_threads.py --offline
    LANGSMITH_API_KEY=... uv run python scripts/export_travel_agent_threads.py --fetch-langsmith
    LANGSMITH_API_KEY=... uv run python scripts/export_travel_agent_threads.py --run-id 019e31a3 --fetch-langsmith
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
TRAVEL_DIR = ROOT / "travel_agent"
OUT_DIR = ROOT / "thread_analysis" / "travel_agent"


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("run-")


def local_manifest(
    travel_dir: Path = TRAVEL_DIR,
    root: Path = ROOT,
    run_id_filter: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(travel_dir.glob("run-*.json")):
        root_run_id = run_id_from_path(path)
        if run_id_filter and run_id_filter not in root_run_id and run_id_filter not in path.name:
            continue

        obj = load_json(path)
        outputs = obj.get("outputs") or {}
        metadata = obj.get("metadata") or {}
        langsmith = obj.get("langsmith") or {}
        project = (langsmith.get("tracing_project") or {}).get("name") or metadata.get("LANGSMITH_PROJECT")

        rows.append(
            {
                "local_file": str(path.relative_to(root)),
                "root_run_id": root_run_id,
                "thread_id": metadata.get("thread_id"),
                "project_name": project,
                "api_url": metadata.get("LANGSMITH_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT"),
                "workspace_id": (langsmith.get("workspace") or {}).get("id"),
                "organization_id": (langsmith.get("organization") or {}).get("id"),
                "tracing_project_id": (langsmith.get("tracing_project") or {}).get("id"),
                "query": outputs.get("query"),
                "validation_passed": outputs.get("validation_passed"),
                "validation_attempts": outputs.get("validation_attempts"),
                "travelplan_title": (outputs.get("travelplan") or {}).get("title"),
                "days": len((outputs.get("travelplan") or {}).get("days") or []),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_to_dict(run: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, Any]:
    extra = run.get("extra") or {}
    metadata = extra.get("metadata") if isinstance(extra, dict) else None
    return {
        "thread_id": manifest_row.get("thread_id"),
        "local_root_run_id": manifest_row.get("root_run_id"),
        "id": str(run.get("id") or ""),
        "trace_id": str(run.get("trace_id")) if run.get("trace_id") else None,
        "parent_run_id": str(run.get("parent_run_id")) if run.get("parent_run_id") else None,
        "name": run.get("name"),
        "run_type": run.get("run_type"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "error": run.get("error"),
        "tags_json": safe_json(run.get("tags")),
        "metadata_json": safe_json(metadata),
        "inputs_json": safe_json(run.get("inputs")),
        "outputs_json": safe_json(run.get("outputs")),
    }


def walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)


def extract_messages(run_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field in ["inputs_json", "outputs_json"]:
        try:
            obj = json.loads(run_row[field])
        except Exception:
            continue

        for item in walk_values(obj):
            msg_type = item.get("type") or item.get("role")
            content = item.get("content")
            if msg_type and (content is not None or item.get("tool_calls") or item.get("tool_call_id")):
                rows.append(
                    {
                        "thread_id": run_row["thread_id"],
                        "run_id": run_row["id"],
                        "run_name": run_row["name"],
                        "run_type": run_row["run_type"],
                        "source_field": field.removesuffix("_json"),
                        "message_type": msg_type,
                        "name": item.get("name"),
                        "tool_call_id": item.get("tool_call_id"),
                        "content": content if isinstance(content, str) else safe_json(content),
                        "tool_calls_json": safe_json(item.get("tool_calls")),
                        "usage_json": safe_json(item.get("usage_metadata") or (item.get("response_metadata") or {}).get("token_usage")),
                    }
                )
    return rows


def extract_tool_calls(run_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field in ["inputs_json", "outputs_json"]:
        try:
            obj = json.loads(run_row[field])
        except Exception:
            continue

        for item in walk_values(obj):
            for call in item.get("tool_calls") or []:
                rows.append(
                    {
                        "thread_id": run_row["thread_id"],
                        "run_id": run_row["id"],
                        "run_name": run_row["name"],
                        "run_type": run_row["run_type"],
                        "source_field": field.removesuffix("_json"),
                        "tool_name": call.get("name"),
                        "tool_call_id": call.get("id"),
                        "tool_args_json": safe_json(call.get("args")),
                        "tool_args_text": safe_json(call.get("args")),
                    }
                )
    return rows


def fetch_langsmith_runs(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before using --fetch-langsmith")

    run_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    tool_call_rows: list[dict[str, Any]] = []

    for row in manifest:
        project = row.get("project_name")
        thread_id = row.get("thread_id")
        project_id = row.get("tracing_project_id")
        if not thread_id or not (project_id or project):
            continue

        filter_string = (
            'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), '
            f'eq(metadata_value, "{thread_id}"))'
        )
        api_url = (row.get("api_url") or "https://api.smith.langchain.com").rstrip("/")
        headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
        if row.get("workspace_id"):
            headers["X-Tenant-Id"] = row["workspace_id"]

        # Keep full run pages small; large LangSmith responses can intermittently
        # return 5xx from the EU endpoint.
        payload: dict[str, Any] = {"filter": filter_string, "limit": 20}
        if project_id:
            payload["session"] = [project_id]
        else:
            payload["session_name"] = project

        runs: list[dict[str, Any]] = []
        while True:
            for attempt in range(6):
                response = requests.post(f"{api_url}/runs/query", headers=headers, json=payload, timeout=60)
                if response.status_code != 429 and response.status_code < 500:
                    break
                time.sleep(5 * (attempt + 1))
            response.raise_for_status()
            body = response.json()
            page = body.get("runs") or []
            runs.extend(page)
            next_cursor = (body.get("cursors") or {}).get("next")
            if not page or not next_cursor:
                break
            payload["cursor"] = next_cursor

        # Fallback: fetch the root run directly if metadata filtering misses children.
        if not runs:
            payload = {"id": [row["root_run_id"]], "limit": 100}
            for attempt in range(6):
                response = requests.post(f"{api_url}/runs/query", headers=headers, json=payload, timeout=60)
                if response.status_code != 429 and response.status_code < 500:
                    break
                time.sleep(5 * (attempt + 1))
            response.raise_for_status()
            runs = response.json().get("runs") or []

        for run in runs:
            run_row = run_to_dict(run, row)
            run_rows.append(run_row)
            message_rows.extend(extract_messages(run_row))
            tool_call_rows.extend(extract_tool_calls(run_row))

    return run_rows, message_rows, tool_call_rows


def fetch_langsmith_tool_runs(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before using --fetch-langsmith")

    run_rows: list[dict[str, Any]] = []
    tool_call_rows: list[dict[str, Any]] = []

    for row in manifest:
        thread_id = row.get("thread_id")
        project_id = row.get("tracing_project_id")
        if not thread_id or not project_id:
            continue

        api_url = (row.get("api_url") or "https://api.smith.langchain.com").rstrip("/")
        headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
        if row.get("workspace_id"):
            headers["X-Tenant-Id"] = row["workspace_id"]

        filter_string = (
            'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), '
            f'eq(metadata_value, "{thread_id}"))'
        )
        payload: dict[str, Any] = {
            "session": [project_id],
            "filter": filter_string,
            "run_type": "tool",
            "limit": 100,
        }

        while True:
            for attempt in range(6):
                response = requests.post(f"{api_url}/runs/query", headers=headers, json=payload, timeout=60)
                if response.status_code != 429 and response.status_code < 500:
                    break
                time.sleep(3 * (attempt + 1))
            response.raise_for_status()
            body = response.json()
            page = body.get("runs") or []

            for run in page:
                run_row = run_to_dict(run, row)
                run_rows.append(run_row)
                tool_call_rows.append(
                    {
                        "thread_id": row["thread_id"],
                        "run_id": run_row["id"],
                        "run_name": run_row["name"],
                        "run_type": run_row["run_type"],
                        "source_field": "tool_run",
                        "tool_name": run_row["name"],
                        "tool_call_id": run_row["id"],
                        "tool_args_json": run_row["inputs_json"],
                        "tool_args_text": run_row["inputs_json"],
                    }
                )

            next_cursor = (body.get("cursors") or {}).get("next")
            if not page or not next_cursor:
                break
            payload["cursor"] = next_cursor

    return run_rows, [], tool_call_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export travel-agent LangGraph/LangSmith thread data into notebook-friendly files.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root used for relative local_file paths")
    parser.add_argument("--travel-dir", type=Path, help="Override local travel_agent trace directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--run-id", help="Optional full/partial local root run id to export")
    parser.add_argument("--offline", action="store_true", help="Only write local manifest; no LangSmith API calls")
    parser.add_argument("--fetch-langsmith", action="store_true", help="Fetch LangSmith runs for each local thread_id")
    parser.add_argument("--tool-runs-only", action="store_true", help="Fast mode: fetch only LangSmith tool runs and skip message expansion")
    args = parser.parse_args()

    root = args.root.resolve()
    travel_dir = (args.travel_dir or root / "travel_agent").resolve()
    out_dir = (args.output_dir or root / "thread_analysis" / "travel_agent").resolve()
    if not travel_dir.exists():
        raise SystemExit(f"Travel Agent directory not found: {travel_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = local_manifest(travel_dir, root, args.run_id)
    if not manifest:
        raise SystemExit("No matching travel_agent traces found")

    write_jsonl(out_dir / "manifest.jsonl", manifest)
    write_csv(out_dir / "manifest.csv", manifest)

    if args.fetch_langsmith:
        if args.tool_runs_only:
            run_rows, message_rows, tool_call_rows = fetch_langsmith_tool_runs(manifest)
        else:
            run_rows, message_rows, tool_call_rows = fetch_langsmith_runs(manifest)
        write_jsonl(out_dir / "langsmith_runs.jsonl", run_rows)
        write_csv(out_dir / "langsmith_runs.csv", run_rows)
        write_jsonl(out_dir / "messages.jsonl", message_rows)
        write_csv(out_dir / "messages.csv", message_rows)
        write_jsonl(out_dir / "tool_calls.jsonl", tool_call_rows)
        write_csv(out_dir / "tool_calls.csv", tool_call_rows)

    print(f"Wrote {len(manifest)} manifest row(s) to {out_dir}")
    if args.fetch_langsmith:
        print("Wrote langsmith_runs/messages/tool_calls tables")


if __name__ == "__main__":
    main()
