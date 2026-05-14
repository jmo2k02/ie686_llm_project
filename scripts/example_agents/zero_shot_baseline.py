"""Zero-shot baseline agent for the TravelPlanner benchmark.

Single LLM call with the raw user query — no tools, no retrieval, no reference
information. Serves as a lower-bound baseline.

Compatible with the evaluation pipeline via ``--graph_spec``.

Usage via pipeline:
    cd scripts/
    uv run python evaluate/pipeline.py \
        --graph_spec example_agents/zero_shot_baseline.py:make_graph \
        --model_name gpt-4o-mini \
        --set_type validation \
        --output_dir ./outputs \
        --mode sole-planning \
        --strategy zero-shot

Standalone quick test:
    cd scripts/
    MODEL_NAME=gpt-4o-mini uv run python example_agents/zero_shot_baseline.py \
        --set_type validation --max_examples 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

try:
    from evaluate.generate import load_split, persist_plan
except ImportError:  # pragma: no cover
    from generate import load_split, persist_plan  # type: ignore


_SYSTEM_PROMPT = (
    "You are a travel planning assistant. "
    "Given a user request, produce a detailed day-by-day travel itinerary. "
    "Include transportation between cities, accommodation for each night, "
    "breakfast, lunch, dinner, and attractions or activities for each day. "
    "Respect all constraints mentioned in the request (budget, dates, number of people, preferences). "
    "Return only the itinerary."
)


def _call_llm(system_prompt: str, user_prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Minimal LLM call using the same ChatOpenAI client as langgraph_minimal."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content or ""


def make_graph(model_name: str, temperature: float = 0.0, **_: Any) -> Pregel:
    """Build a zero-shot graph: one LLM call, no tools, no retrieval."""

    def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        plan = _call_llm(_SYSTEM_PROMPT, query, model_name, temperature)
        state["plan"] = plan.strip()
        return state

    graph = StateGraph(dict)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()


# ─── Stand-alone runner (quick smoke-tests) ───────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot baseline runner")
    parser.add_argument(
        "--set_type", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME"))
    parser.add_argument(
        "--mode", default="sole-planning", choices=["two-stage", "sole-planning"]
    )
    parser.add_argument("--strategy", default="zero-shot")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=None)
    return parser


def main() -> None:  # pragma: no cover
    parser = _build_parser()
    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name must be provided via flag or MODEL_NAME env var")

    graph = make_graph(args.model_name, args.temperature)
    dataset = load_split(args.set_type)
    base_dir = Path(args.output_dir)
    total = (
        len(dataset)
        if args.max_examples in (None, 0)
        else min(len(dataset), args.max_examples)
    )

    for idx in range(total):
        record = dataset[idx]
        state: Dict[str, Any] = {"query": record.get("query", "")}
        result = graph.invoke(state)
        persist_plan(
            base_dir=base_dir,
            set_type=args.set_type,
            example_id=idx + 1,
            model_name=args.model_name,
            mode=args.mode,
            strategy=args.strategy,
            plan_text=result.get("plan", ""),
        )


if __name__ == "__main__":
    main()
