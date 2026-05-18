#!/usr/bin/env python3
"""Export notebook-friendly data for local baseline LangSmith traces.

Offline mode reads `baseline/run-*.json` and writes:

- `thread_analysis/baseline/manifest.csv`
- `thread_analysis/baseline/local_messages.csv`
- `thread_analysis/baseline/local_tool_calls.csv`

Online mode additionally fetches LangSmith runs for each baseline `thread_id` and
writes `langsmith_runs/messages/tool_calls` tables. Use the EU endpoint already
stored in the local trace metadata.

Usage:
    uv run python ie686_llm_project/langsmith_traces_analysis/scripts/export_baseline_threads.py --offline
    LANGSMITH_API_KEY=... uv run python ie686_llm_project/langsmith_traces_analysis/scripts/export_baseline_threads.py --fetch-langsmith
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


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


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def extract_query(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("type") != "human":
            continue
        text = message_text(message.get("content"))
        match = re.search(r"USER QUERY\s*(.*?)\s*CONSTRAINTS", text, re.S)
        if match:
            return match.group(1).strip()
    return None


def local_manifest_and_tables(
    baseline_dir: Path,
    root: Path,
    run_id_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    tool_call_rows: list[dict[str, Any]] = []

    for path in sorted(baseline_dir.glob("run-*.json")):
        root_run_id = run_id_from_path(path)
        if run_id_filter and run_id_filter not in root_run_id and run_id_filter not in path.name:
            continue

        obj = load_json(path)
        outputs = obj.get("outputs") or {}
        messages = outputs.get("messages") or []
        metadata = obj.get("metadata") or {}
        langsmith = obj.get("langsmith") or {}
        project = (langsmith.get("tracing_project") or {}).get("name") or metadata.get("LANGSMITH_PROJECT")

        requested = 0
        executed = 0
        final_markdown = ""
        for index, message in enumerate(messages):
            msg_type = message.get("type") or message.get("role")
            content = message_text(message.get("content"))
            tool_calls = message.get("tool_calls") or []
            requested += len(tool_calls)
            if msg_type == "tool":
                executed += 1
            if msg_type == "ai" and content.strip():
                final_markdown = content.strip()

            message_rows.append(
                {
                    "thread_id": metadata.get("thread_id"),
                    "local_root_run_id": root_run_id,
                    "local_file": str(path.relative_to(root)),
                    "message_index": index,
                    "message_type": msg_type,
                    "name": message.get("name"),
                    "id": message.get("id"),
                    "tool_call_id": message.get("tool_call_id"),
                    "status": message.get("status"),
                    "content": content,
                    "tool_calls_json": safe_json(tool_calls),
                    "usage_json": safe_json(
                        message.get("usage_metadata")
                        or (message.get("response_metadata") or {}).get("token_usage")
                    ),
                }
            )

            for call_index, call in enumerate(tool_calls):
                tool_call_rows.append(
                    {
                        "thread_id": metadata.get("thread_id"),
                        "local_root_run_id": root_run_id,
                        "local_file": str(path.relative_to(root)),
                        "source": "ai_tool_call_request",
                        "message_index": index,
                        "call_index": call_index,
                        "tool_name": call.get("name"),
                        "tool_call_id": call.get("id"),
                        "tool_args_json": safe_json(call.get("args")),
                        "tool_args_text": safe_json(call.get("args")),
                    }
                )

            if msg_type == "tool":
                tool_call_rows.append(
                    {
                        "thread_id": metadata.get("thread_id"),
                        "local_root_run_id": root_run_id,
                        "local_file": str(path.relative_to(root)),
                        "source": "tool_message_result",
                        "message_index": index,
                        "call_index": None,
                        "tool_name": message.get("name"),
                        "tool_call_id": message.get("tool_call_id"),
                        "tool_args_json": "",
                        "tool_args_text": "",
                        "tool_output": content,
                    }
                )

        manifest_rows.append(
            {
                "local_file": str(path.relative_to(root)),
                "root_run_id": root_run_id,
                "thread_id": metadata.get("thread_id"),
                "project_name": project,
                "api_url": metadata.get("LANGSMITH_ENDPOINT")
                or os.getenv("LANGSMITH_ENDPOINT")
                or os.getenv("LANGCHAIN_ENDPOINT"),
                "workspace_id": (langsmith.get("workspace") or {}).get("id"),
                "organization_id": (langsmith.get("organization") or {}).get("id"),
                "tracing_project_id": (langsmith.get("tracing_project") or {}).get("id"),
                "query": extract_query(messages),
                "message_count": len(messages),
                "requested_tool_calls": requested,
                "executed_tool_messages": executed,
                "final_markdown_chars": len(final_markdown),
                "has_error": obj.get("error") is not None,
                "error": obj.get("error"),
            }
        )

    return manifest_rows, message_rows, tool_call_rows


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
                        "usage_json": safe_json(
                            item.get("usage_metadata")
                            or (item.get("response_metadata") or {}).get("token_usage")
                        ),
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


def fetch_langsmith_runs(
    manifest: list[dict[str, Any]],
    *,
    tool_runs_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before using --fetch-langsmith")

    run_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    tool_call_rows: list[dict[str, Any]] = []

    for row in manifest:
        thread_id = row.get("thread_id")
        project_id = row.get("tracing_project_id")
        project = row.get("project_name")
        if not thread_id or not (project_id or project):
            continue

        api_url = (row.get("api_url") or "https://api.smith.langchain.com").rstrip("/")
        headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
        if row.get("workspace_id"):
            headers["X-Tenant-Id"] = row["workspace_id"]

        filter_string = (
            'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), '
            f'eq(metadata_value, "{thread_id}"))'
        )
        payload: dict[str, Any] = {"filter": filter_string, "limit": 100}
        if project_id:
            payload["session"] = [project_id]
        else:
            payload["session_name"] = project
        if tool_runs_only:
            payload["run_type"] = "tool"

        while True:
            for attempt in range(6):
                response = requests.post(
                    f"{api_url}/runs/query",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if response.status_code != 429:
                    break
                time.sleep(3 * (attempt + 1))
            response.raise_for_status()
            body = response.json()
            page = body.get("runs") or []

            for run in page:
                run_row = run_to_dict(run, row)
                run_rows.append(run_row)
                if tool_runs_only:
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
                else:
                    message_rows.extend(extract_messages(run_row))
                    tool_call_rows.extend(extract_tool_calls(run_row))

            next_cursor = (body.get("cursors") or {}).get("next")
            if not page or not next_cursor:
                break
            payload["cursor"] = next_cursor

    return run_rows, message_rows, tool_call_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export baseline LangGraph/LangSmith trace data into notebook-friendly files."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root containing baseline/")
    parser.add_argument("--baseline-dir", type=Path, help="Override baseline trace directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--run-id", help="Optional full/partial local root run id to export")
    parser.add_argument("--offline", action="store_true", help="Only write local tables; no LangSmith API calls")
    parser.add_argument("--fetch-langsmith", action="store_true", help="Fetch LangSmith runs for each local thread_id")
    parser.add_argument("--tool-runs-only", action="store_true", help="Fetch only LangSmith tool runs")
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_dir = (args.baseline_dir or root / "baseline").resolve()
    out_dir = (args.output_dir or root / "thread_analysis" / "baseline").resolve()
    if not baseline_dir.exists():
        raise SystemExit(f"Baseline directory not found: {baseline_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, local_messages, local_tool_calls = local_manifest_and_tables(
        baseline_dir,
        root,
        args.run_id,
    )
    if not manifest:
        raise SystemExit("No matching baseline traces found")

    write_jsonl(out_dir / "manifest.jsonl", manifest)
    write_csv(out_dir / "manifest.csv", manifest)
    write_jsonl(out_dir / "local_messages.jsonl", local_messages)
    write_csv(out_dir / "local_messages.csv", local_messages)
    write_jsonl(out_dir / "local_tool_calls.jsonl", local_tool_calls)
    write_csv(out_dir / "local_tool_calls.csv", local_tool_calls)

    if args.fetch_langsmith:
        run_rows, message_rows, tool_call_rows = fetch_langsmith_runs(
            manifest,
            tool_runs_only=args.tool_runs_only,
        )
        write_jsonl(out_dir / "langsmith_runs.jsonl", run_rows)
        write_csv(out_dir / "langsmith_runs.csv", run_rows)
        write_jsonl(out_dir / "messages.jsonl", message_rows)
        write_csv(out_dir / "messages.csv", message_rows)
        write_jsonl(out_dir / "tool_calls.jsonl", tool_call_rows)
        write_csv(out_dir / "tool_calls.csv", tool_call_rows)

    print(f"Wrote {len(manifest)} baseline manifest row(s) to {out_dir}")
    print(f"Wrote {len(local_messages)} local message row(s)")
    print(f"Wrote {len(local_tool_calls)} local tool-call/tool-result row(s)")
    if args.fetch_langsmith:
        print("Wrote langsmith_runs/messages/tool_calls tables")


if __name__ == "__main__":
    main()
