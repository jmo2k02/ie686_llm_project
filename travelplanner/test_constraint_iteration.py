"""
Quick interactive test for the constraint iteration agent.
Run from the travelplanner/ directory:

    uv run python test_constraint_iteration.py

Requires a .env file in the repo root with OPENAI_API_KEY set.
See .env.example for the expected format.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv

# load .env from repo root (two levels up from travelplanner/)
load_dotenv(Path(__file__).parent.parent / ".env")

from langgraph.types import Command

from travelplanner.agents.constraint_iteration_agent import (
    ConstraintIterationState,
    get_constraint_list,
    get_message_history,
    make_graph,
)


def _print_summary(result: dict) -> None:
    SEP = "=" * 60
    THIN = "-" * 60

    # ── Session replay ────────────────────────────────────────────
    history = get_message_history(result)
    print(f"\n{SEP}")
    print("SESSION REPLAY")
    print(SEP)
    for msg in history.messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        label = "[Agent]" if role == "assistant" else "[You]  "
        # indent continuation lines
        lines = content.splitlines()
        print(f"\n{label} {lines[0]}")
        for line in lines[1:]:
            print(f"         {line}")

    # ── Artifact ──────────────────────────────────────────────────
    artifacts = result.get("agent_artifacts", {})
    artifact_list = artifacts.get("constraint_agent", [])
    if artifact_list:
        content_raw = artifact_list[0].content
        print(f"\n{SEP}")
        print("ARTIFACT — constraint-extraction-result")
        print(SEP)
        print(f"  Status   : {content_raw['status']}")
        print(f"  Model    : {content_raw.get('model') or '—'}")
        print(f"  Turns    : {content_raw.get('interaction_turns', 0)}")

        print(f"\n{THIN}")
        print("  Hard constraints")
        print(THIN)
        for i, c in enumerate(content_raw.get("hard_constraints", []), 1):
            print(f"  {i:2d}. {c['text']}")

        print(f"\n{THIN}")
        print("  Commonsense constraints")
        print(THIN)
        for i, c in enumerate(content_raw.get("commonsense_constraints", []), 1):
            skipped = "  [skipped]" if c.get("user_skipped") else ""
            print(f"  {i:2d}. {c['text']}{skipped}")

        missing = content_raw.get("categories_missing", [])
        skipped_cats = content_raw.get("categories_skipped_by_user", [])
        if missing:
            print(f"\n  Categories missing    : {', '.join(missing)}")
        if skipped_cats:
            print(f"  Skipped by user       : {', '.join(skipped_cats)}")

    # ── Final constraint list ─────────────────────────────────────
    constraints = get_constraint_list(result)
    print(f"\n{SEP}")
    print("FINAL CONSTRAINT LIST")
    print(SEP)
    for i, c in enumerate(constraints, 1):
        skipped = "  [skipped]" if c.user_skipped else ""
        print(f"  [{c.type:12s}] {i:2d}. {c.text}{skipped}")

    active = sum(1 for c in constraints if not c.user_skipped)
    skipped_total = sum(1 for c in constraints if c.user_skipped)
    print(f"\n  Total: {len(constraints)}  |  active: {active}  |  skipped: {skipped_total}")
    print()


def run_interactive(query: str, model_name: str = "openai:gpt-4o-mini") -> None:
    graph = make_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n" + "=" * 60)
    print("Constraint Iteration Agent — interactive test")
    print("=" * 60)
    print(f"Query: {query}\n")

    initial_state = ConstraintIterationState(
        query=query,
        model_name=model_name,
    )

    result = graph.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        agent_message = result["__interrupt__"][0].value
        print(f"\n[Agent]\n{agent_message}\n")
        user_input = input("[You] > ").strip()
        result = graph.invoke(Command(resume=user_input), config=config)

    _print_summary(result)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TravelPlanner — Constraint Iteration Agent")
    print("=" * 60)
    print("Describe your trip (press Enter twice when done):\n")

    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)

    query = " ".join(lines).strip()
    if not query:
        print("No input provided. Exiting.")
    else:
        run_interactive(query)
