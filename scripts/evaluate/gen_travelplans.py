"""Generate travel plans for the curated query set in ``data/travel_queries.json``.

Pipeline:
1. Read queries from ``data/travel_queries.json``.
2. Run each ``query`` through the main workflow at
   ``travelplanner.workflows.task_planning.run``.
3. Persist the resulting ``TravelPlan`` (plus minimal metadata) as JSON files
   under ``eval_output/travelplans/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langgraph.types import Command

load_dotenv(find_dotenv(usecwd=True))

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAVELPLANNER_SRC = REPO_ROOT / "travelplanner" / "src"
if str(TRAVELPLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAVELPLANNER_SRC))

from travelplanner.config import get_setting
from travelplanner.schema.system_state import StateContractModel
from travelplanner.utils.checkpoint import make_memory_checkpointer
from travelplanner.workflows.task_planning import make_graph as make_task_planning_graph


def _stream_items(data: Any):
    if isinstance(data, dict):
        return data.items()
    if hasattr(data, "model_dump"):
        return data.model_dump().items()
    return ()


async def run_task_planning(
    query: str,
    model_name: str,
    temperature: float = 0.0,
) -> StateContractModel:
    """Run the task-planning workflow end-to-end and return the final state.

    Compiles the graph locally because ``task_planning.run`` calls ``.invoke()``
    on an uncompiled ``StateGraph``.
    """
    graph = make_task_planning_graph(
        model_name=model_name,
        temperature=temperature,
    ).compile(checkpointer=make_memory_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    current_input: dict[str, str] | Command = {"query": query}

    while True:
        interrupted = False
        async for part in graph.astream(
            current_input,
            config=config,
            stream_mode=["values", "updates", "messages", "custom"],
            version="v2",
        ):
            if part["type"] == "values":
                # ValuesStreamPart — full state snapshot after each step
                for node_name, state in _stream_items(part["data"]):
                    if node_name == "__interrupt__":
                        resume_value = random.choice(("ok", "skip"))
                        print(f"Graph interrupted: {state}. Resuming with {resume_value!r}.")
                        current_input = Command(resume=resume_value)
                        interrupted = True
                        break
                print(f"State: topic={part['data']}")
            elif part["type"] == "updates":
                # UpdatesStreamPart — only the changed keys from each node
                for node_name, state in _stream_items(part["data"]):
                    if node_name == "__interrupt__":
                        resume_value = random.choice(("ok", "skip"))
                        print(f"Graph interrupted: {state}. Resuming with {resume_value!r}.")
                        current_input = Command(resume=resume_value)
                        interrupted = True
                        break
                    print(f"Node `{node_name}` updated: {state}")
            elif part["type"] == "messages":
                # MessagesStreamPart — (message_chunk, metadata) from LLM calls
                msg, metadata = part["data"]
                print(msg.content, end="", flush=True)
            elif part["type"] == "custom":
                # CustomStreamPart — arbitrary data from get_stream_writer()
                print(f"Progress: {part['data']['progress']}%")

            if interrupted:
                break

        if interrupted:
            continue

        snapshot = graph.get_state(config)
        return StateContractModel.model_validate(snapshot.values)


DEFAULT_QUERIES_PATH = REPO_ROOT / "data" / "travel_queries.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_output" / "travelplans"


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def persist_result(
    output_dir: Path,
    record: dict[str, Any],
    result: Any,
    model_name: str,
    duration_seconds: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_id = record.get("id") or f"query_{int(time.time())}"
    out_path = output_dir / f"{query_id}.json"

    payload = {
        "id": query_id,
        "description": record.get("description"),
        "query": record.get("query"),
        "hard_constraints": record.get("hard_constraints"),
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "travelplan": result.travelplan.model_dump(mode="json"),
        "validation_passed": result.validation_passed,
        "validation_attempts": result.validation_attempts,
        "validation_feedback": result.validation_feedback,
        "task_list": [task.model_dump(mode="json") for task in result.task_list],
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return out_path


def persist_error(
    output_dir: Path,
    record: dict[str, Any],
    model_name: str,
    error: BaseException,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_id = record.get("id") or f"query_{int(time.time())}"
    out_path = output_dir / f"{query_id}.error.json"

    payload = {
        "id": query_id,
        "description": record.get("description"),
        "query": record.get("query"),
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to queries JSON (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plans (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override the workflow model_name. Default comes from config.yaml.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the workflow temperature. Default comes from config.yaml.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap on the number of queries to process (default: all).",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Only process queries whose id matches one of these values.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip queries whose output file already exists.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_name = args.model_name or str(
        get_setting("models.workflows.task_planning.model_name")
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(get_setting("models.workflows.task_planning.temperature", 0.0))
    )

    queries = load_queries(args.queries)
    if args.ids:
        wanted = set(args.ids)
        queries = [q for q in queries if q.get("id") in wanted]
    if args.max_examples is not None:
        queries = queries[: args.max_examples]

    print("=" * 80)
    print("gen_travelplans — generating travel plans for curated queries")
    print("=" * 80)
    print(f"Queries file : {args.queries}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Model        : {model_name}")
    print(f"Temperature  : {temperature}")
    print(f"To process   : {len(queries)} query/queries")
    print()

    total = len(queries)
    successes = 0
    failures = 0

    for idx, record in enumerate(queries, start=1):
        query_id = record.get("id", f"<no-id-{idx}>")
        query_text = record.get("query")
        if not query_text:
            print(f"[{idx}/{total}] {query_id}: missing 'query' — skipping")
            failures += 1
            continue

        out_path = args.output_dir / f"{query_id}.json"
        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{total}] {query_id}: output exists — skipping")
            continue

        print(f"[{idx}/{total}] {query_id}: running workflow...")
        t0 = time.monotonic()
        try:
            import asyncio
            result = asyncio.run(
                run_task_planning(
                    query=query_text,
                    model_name=model_name,
                    temperature=temperature,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface any agent failure per-query
            duration = time.monotonic() - t0
            err_path = persist_error(args.output_dir, record, model_name, exc)
            print(
                f"          {query_id}: FAILED after {duration:.1f}s "
                f"({type(exc).__name__}: {exc}) → {err_path}"
            )
            failures += 1
            continue

        duration = time.monotonic() - t0
        saved = persist_result(
            output_dir=args.output_dir,
            record=record,
            result=result,
            model_name=model_name,
            duration_seconds=duration,
        )
        successes += 1
        print(
            f"          {query_id}: done in {duration:.1f}s "
            f"(validation_passed={result.validation_passed}) → {saved}"
        )

    print()
    print("=" * 80)
    print(f"Finished: {successes} succeeded, {failures} failed, {total} total")
    print("=" * 80)


if __name__ == "__main__":
    main()
