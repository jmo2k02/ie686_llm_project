"""
Live integration test: run the general_web_search agent with real API and save output.

This script runs the agent end-to-end with the real Tavily API and saves
the full artifact to a timestamped JSON file for later review and comparison.

Usage:
    cd /home/lukas/uni/ie686_llm_project/travelplanner
    uv run python test_live_search.py

Output:
    .sisyphus/evidence/live-search-results/{timestamp}_{slug}.json

Re-run anytime to get fresh results and compare with previous runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from travelplanner.agents.general_web_search_agent import (
    make_graph as make_general_web_search_graph,
)
from travelplanner.schema.general_web_search_artifact import (
    GeneralWebSearchArtifactContentModel,
)
from travelplanner.schema.system_state import TaskModel


# Queries to run — real planning-relevant tasks
LIVE_QUERIES = [
    {
        "slug": "barcelona-beach-trip",
        "query": "Plan a family beach trip to Barcelona in June.",
        "task": (
            "Best beach zones in Barcelona for families with young children. "
            "Include info on water quality, lifeguards, nearby food options, "
            "and opening hour caveats for attractions."
        ),
    },
    {
        "slug": "tokyo-food-trip",
        "query": "Plan a 3-day food-focused trip to Tokyo.",
        "task": (
            "Best areas for food lovers in Tokyo with budget-friendly options, "
            "local izakaya recommendations, and market food highlights. "
            "Include transit access notes."
        ),
    },
]


def _slugify(text: str) -> str:
    """Make a filesystem-safe slug from text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"^-|-$", "", text)
    return text[:60]


def _ensure_env() -> None:
    """Verify required env vars are set."""
    missing = [name for name in ("TAVILY_API_KEY",) if not os.getenv(name, "").strip()]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        print("Set them in your shell or .env file.")
        sys.exit(1)


def _run_agent(*, query: str, task: str, query_slug: str) -> dict[str, Any]:
    """Run the agent and return the parsed artifact as a dict."""
    graph = make_general_web_search_graph()
    result: dict[str, Any] = graph.invoke(
        {
            "query": query,
            "task_list": [
                TaskModel(
                    name=f"live-test-{query_slug}",
                    type="general-web-search",
                    text=task,
                    is_valid=True,
                    validation_comment=None,
                )
            ],
            "agent_artifacts": {},
        }
    )
    artifacts = result.get("agent_artifacts", {}).get("general_web_search_agent", [])
    if not artifacts:
        return {"error": "No artifact returned", "query": query, "task": task}

    artifact = artifacts[-1]
    raw_content = (
        artifact["content"] if isinstance(artifact, dict) else artifact.content
    )
    parsed = GeneralWebSearchArtifactContentModel.model_validate(raw_content)
    return parsed.model_dump(mode="json")


def _save_result(result: dict[str, Any], slug: str) -> Path:
    """Save a result dict to the evidence directory with a timestamp."""
    evidence_dir = (
        Path(__file__).parent.parent / ".output" / "tests" / "live-search-results"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{slug}.json"
    out_path = evidence_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_path


def main() -> None:
    print("=" * 60)
    print("Live Search Agent Integration Test")
    print("=" * 60)
    print()

    _ensure_env()

    results: list[dict[str, Any]] = []

    for i, q in enumerate(LIVE_QUERIES):
        slug = q["slug"]
        query = q["query"]
        task = q["task"]

        print(f"[{i + 1}/{len(LIVE_QUERIES)}] Running: {slug}")
        print(f"  Query:  {query}")
        print(f"  Task:   {task[:80]}...")
        print()

        try:
            result = _run_agent(query=query, task=task, query_slug=slug)
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc), "query": query, "task": task}
            print(f"  ERROR: {exc}")
        else:
            status = result.get("status", "unknown")
            final_answer = str(result.get("final_answer", "") or "")[:100]
            proof_count = len(result.get("proof_points") or [])
            errors = result.get("errors") or []
            print(f"  Status: {status}")
            print(f"  Final answer (first 100 chars): {final_answer}")
            print(f"  Proof points: {proof_count}")
            if errors:
                print(f"  Errors: {errors}")
            print(f"  Raw output saved to: {result}")

        out_path = _save_result(result, slug)
        print(f"  Saved:  {out_path}")
        print()
        results.append(
            {"slug": slug, "path": str(out_path), "status": result.get("status")}
        )

        # Small delay between queries
        if i < len(LIVE_QUERIES) - 1:
            time.sleep(1)

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  [{r['status']}] {r['slug']} -> {r['path']}")

    evidence_dir = (
        Path(__file__).parent.parent / ".output" / "tests" / "live-search-results"
    )
    print()
    print(f"Previous runs (for comparison):")
    existing = sorted(evidence_dir.glob("*.json"))
    if existing:
        for p in existing[: -len(LIVE_QUERIES)]:  # show older runs
            print(f"  {p.name}")
    else:
        print("  (no previous runs)")


if __name__ == "__main__":
    main()
