"""Fully export all LangSmith runs for travel_agent threads with progress logs.

This is intentionally verbose and incremental. It writes rows as they are fetched
so long exports are inspectable while running.

Outputs:
  thread_analysis/travel_agent_full/manifest.csv
  thread_analysis/travel_agent_full/langsmith_runs.csv/jsonl
  thread_analysis/travel_agent_full/messages.csv/jsonl
  thread_analysis/travel_agent_full/tool_calls.csv/jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
TRAVEL_DIR = ROOT / "travel_agent"
OUT_DIR = ROOT / "thread_analysis" / "travel_agent_full"

RUN_FIELDS = [
    "thread_id",
    "local_root_run_id",
    "id",
    "trace_id",
    "parent_run_id",
    "name",
    "run_type",
    "start_time",
    "end_time",
    "error",
    "tags_json",
    "metadata_json",
    "inputs_json",
    "outputs_json",
]
MESSAGE_FIELDS = [
    "thread_id",
    "run_id",
    "run_name",
    "run_type",
    "source_field",
    "message_type",
    "name",
    "tool_call_id",
    "content",
    "tool_calls_json",
    "usage_json",
]
TOOL_CALL_FIELDS = [
    "thread_id",
    "run_id",
    "run_name",
    "run_type",
    "source_field",
    "tool_name",
    "tool_call_id",
    "tool_args_json",
    "tool_args_text",
]


def log(message: str) -> None:
    print(message, flush=True)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def local_manifest(travel_dir: Path, root: Path, run_id_filter: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(travel_dir.glob("run-*.json")):
        root_run_id = path.stem.removeprefix("run-")
        if run_id_filter and run_id_filter not in root_run_id and run_id_filter not in path.name:
            continue

        obj = load_json(path)
        outputs = obj.get("outputs") or {}
        metadata = obj.get("metadata") or {}
        langsmith = obj.get("langsmith") or {}
        rows.append(
            {
                "local_file": str(path.relative_to(root)),
                "root_run_id": root_run_id,
                "thread_id": metadata.get("thread_id"),
                "project_name": (langsmith.get("tracing_project") or {}).get("name") or metadata.get("LANGSMITH_PROJECT"),
                "api_url": metadata.get("LANGSMITH_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
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


def open_csv(path: Path, fields: list[str]):
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return f, writer


def write_jsonl(f, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


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


def walk_values(value: Any) -> Iterable[dict[str, Any]]:
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


def post_runs_query(api_url: str, headers: dict[str, str], payload: dict[str, Any], max_retries: int) -> dict[str, Any]:
    for attempt in range(max_retries):
        response = requests.post(f"{api_url}/runs/query", headers=headers, json=payload, timeout=90)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        wait = 5 * (attempt + 1)
        log(f"  rate limited (429), sleeping {wait}s")
        time.sleep(wait)
    response.raise_for_status()
    return response.json()


def export_full(
    manifest: list[dict[str, Any]],
    out_dir: Path,
    *,
    limit: int,
    max_pages: int | None,
    expand_messages: bool,
    expand_tool_calls: bool,
) -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY first, e.g. `set -a && source .env && set +a`")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest files.
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in manifest for k in row}))
        writer.writeheader()
        writer.writerows(manifest)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            write_jsonl(f, row)

    run_csv_f, run_csv = open_csv(out_dir / "langsmith_runs.csv", RUN_FIELDS)
    msg_csv_f, msg_csv = open_csv(out_dir / "messages.csv", MESSAGE_FIELDS)
    tc_csv_f, tc_csv = open_csv(out_dir / "tool_calls.csv", TOOL_CALL_FIELDS)
    run_jsonl = (out_dir / "langsmith_runs.jsonl").open("w", encoding="utf-8")
    msg_jsonl = (out_dir / "messages.jsonl").open("w", encoding="utf-8")
    tc_jsonl = (out_dir / "tool_calls.jsonl").open("w", encoding="utf-8")

    totals = {"runs": 0, "messages": 0, "tool_calls": 0}
    try:
        for ti, row in enumerate(manifest, start=1):
            thread_id = row.get("thread_id")
            project_id = row.get("tracing_project_id")
            api_url = (row.get("api_url") or "https://api.smith.langchain.com").rstrip("/")
            log(f"THREAD {ti}/{len(manifest)} root={row.get('root_run_id')[:8]} thread={thread_id} title={row.get('travelplan_title')}")
            if not thread_id or not project_id:
                log("  skipped: missing thread_id or project_id")
                continue

            headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
            if row.get("workspace_id"):
                headers["X-Tenant-Id"] = row["workspace_id"]

            payload: dict[str, Any] = {
                "session": [project_id],
                "filter": 'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "' + thread_id + '"))',
                "limit": limit,
            }

            thread_counts = {"runs": 0, "messages": 0, "tool_calls": 0}
            page_no = 0
            while True:
                page_no += 1
                if max_pages is not None and page_no > max_pages:
                    log(f"  stopped at max_pages={max_pages}")
                    break

                started = time.time()
                body = post_runs_query(api_url, headers, payload, max_retries=8)
                page = body.get("runs") or []
                elapsed = time.time() - started
                log(f"  page {page_no}: fetched {len(page)} runs in {elapsed:.1f}s")

                if not page:
                    break

                page_messages = 0
                page_tool_calls = 0
                for run in page:
                    run_row = run_to_dict(run, row)
                    run_csv.writerow(run_row)
                    write_jsonl(run_jsonl, run_row)
                    totals["runs"] += 1
                    thread_counts["runs"] += 1

                    if expand_messages:
                        messages = extract_messages(run_row)
                        for msg in messages:
                            msg_csv.writerow(msg)
                            write_jsonl(msg_jsonl, msg)
                        page_messages += len(messages)
                        totals["messages"] += len(messages)
                        thread_counts["messages"] += len(messages)

                    if expand_tool_calls:
                        tool_calls = extract_tool_calls(run_row)
                        for tc in tool_calls:
                            tc_csv.writerow(tc)
                            write_jsonl(tc_jsonl, tc)
                        page_tool_calls += len(tool_calls)
                        totals["tool_calls"] += len(tool_calls)
                        thread_counts["tool_calls"] += len(tool_calls)

                run_csv_f.flush(); msg_csv_f.flush(); tc_csv_f.flush()
                run_jsonl.flush(); msg_jsonl.flush(); tc_jsonl.flush()
                log(f"  page {page_no}: wrote runs={len(page)}, messages={page_messages}, tool_calls={page_tool_calls}")

                next_cursor = (body.get("cursors") or {}).get("next")
                if not next_cursor:
                    break
                payload["cursor"] = next_cursor

            log(f"  thread done: {thread_counts}")
            log(f"  totals so far: {totals}")
    finally:
        for f in [run_csv_f, msg_csv_f, tc_csv_f, run_jsonl, msg_jsonl, tc_jsonl]:
            f.close()

    log(f"DONE. Wrote full export to {out_dir} with totals={totals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full verbose export of all travel_agent LangSmith thread runs.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root used for relative local_file paths")
    parser.add_argument("--travel-dir", type=Path, default=TRAVEL_DIR, help="Directory containing local travel_agent run-*.json files")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Directory for exported tables")
    parser.add_argument("--run-id", help="Optional full/partial local root run id")
    parser.add_argument("--limit", type=int, default=100, help="Runs per API page")
    parser.add_argument("--max-pages", type=int, help="Debug option: stop after N pages per thread")
    parser.add_argument("--no-messages", action="store_true", help="Do not recursively extract messages")
    parser.add_argument("--no-tool-calls", action="store_true", help="Do not recursively extract AI message tool_calls")
    args = parser.parse_args()

    root = args.root.resolve()
    travel_dir = args.travel_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not travel_dir.exists():
        raise SystemExit(f"Travel Agent directory not found: {travel_dir}")

    manifest = local_manifest(travel_dir, root, args.run_id)
    log(f"Found {len(manifest)} travel_agent trace(s)")
    log(f"Reading local traces from {travel_dir}")
    log(f"Writing export to {out_dir}")
    export_full(
        manifest,
        out_dir,
        limit=args.limit,
        max_pages=args.max_pages,
        expand_messages=not args.no_messages,
        expand_tool_calls=not args.no_tool_calls,
    )


if __name__ == "__main__":
    main()
