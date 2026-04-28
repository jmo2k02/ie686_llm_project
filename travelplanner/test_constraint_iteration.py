"""
Quick interactive test for the constraint iteration agent.
Run from the travelplanner/ directory:

    uv run python test_constraint_iteration.py
"""
from __future__ import annotations

import os
import uuid

# führe folgendes aus um den Test zu starten: 
# cd "c:\Users\nava2\Documents\LLM & Agents\ie686_llm_project\travelplanner"
# uv run python test_constraint_iteration.py


# ── Trage deinen API-Key hier ein (nur für lokales Testen, nicht committen!) ──
os.environ.setdefault("OPENAI_API_KEY", "ADD YOUR KEY HERE")

from langgraph.types import Command

from travelplanner.agents.constraint_iteration_agent import (
    ConstraintIterationState,
    get_constraint_list,
    get_message_history,
    make_graph,
)


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

    # Graph finished — print results
    constraints = get_constraint_list(result)
    history = get_message_history(result)

    print("\n" + "=" * 60)
    print("DONE — Final constraint list")
    print("=" * 60)
    for i, c in enumerate(constraints, 1):
        skipped = " [SKIPPED]" if c.user_skipped else ""
        print(f"  [{c.type:12s}] {i:2d}. {c.text}{skipped}")

    print(f"\nTotal: {len(constraints)} constraints "
          f"({sum(1 for c in constraints if not c.user_skipped)} active, "
          f"{sum(1 for c in constraints if c.user_skipped)} skipped)")
    print(f"Message turns: {len(history.messages)}")


if __name__ == "__main__":
    import sys

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "I want to go to Barcelona from June 15 to June 22."
    )
    run_interactive(query)
