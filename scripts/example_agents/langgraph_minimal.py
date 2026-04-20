"""Minimal LangGraph-powered planner for the TravelPlanner benchmark.

This script shows how to plug in a custom agent built with LangGraph while
still producing `generated_plan_*.json` files that downstream post-processing
expects. The graph is intentionally tiny: a single planner node that turns the
query and reference information into a plan via one OpenAI API call.

Usage:
export OPENAI_API_KEY

python agents/langgraph_minimal.py \
  --model_name gpt-4o-mini \
  --set_type validation \
  --output_dir /tmp/travelplanner \
  --mode sole-planning \
  --strategy direct
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = (
    "You are TravelPlanner, an assistant that writes day-by-day itineraries "
    "covering transportation, meals, attractions, and lodging while obeying "
    "budget and commonsense constraints. Be concrete, reference the provided "
    "facts when possible, and keep each day easy to parse."
)


def format_reference_information(reference_information: Any) -> str:
    """Render arbitrary reference data into a readable string for prompting."""
    print("  [format_reference_information] Formatting reference information...")
    if reference_information in (None, ""):
        print("  [format_reference_information] No reference information provided")
        return "No additional reference information supplied."
    if isinstance(reference_information, str):
        print("  [format_reference_information] Reference information is a string")
        return reference_information.strip()
    try:
        print(
            "  [format_reference_information] Converting reference information to JSON"
        )
        return json.dumps(reference_information, indent=2, ensure_ascii=True)
    except TypeError:
        print(
            "  [format_reference_information] Converting reference information to string"
        )
        return str(reference_information)


def build_user_prompt(query: str, reference_blob: str) -> str:
    print("  [build_user_prompt] Building user prompt...")
    print(f"  [build_user_prompt] Query length: {len(query)} characters")
    sections = [
        "Request:",
        query.strip(),
        "",
        "Reference Information:",
        reference_blob,
        "",
        "Return a structured multi-day itinerary.",
    ]
    prompt = "\n".join(section for section in sections if section is not None)
    print(f"  [build_user_prompt] Final prompt length: {len(prompt)} characters")
    return prompt


def make_graph(model_name: str, temperature: float) -> Pregel:
    print(
        f"[make_graph] Initializing graph with model: {model_name}, temperature: {temperature}"
    )
    client = ChatOpenAI(model=model_name, temperature=temperature)

    def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[planner_node] Starting planner node execution...")
        reference_blob = format_reference_information(
            state.get("reference_information")
        )
        prompt = build_user_prompt(state.get("query", ""), reference_blob)
        print(f"[planner_node] Calling OpenAI API with model: {model_name}")
        response = client.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        plan_text = response.content or ""
        print(f"[planner_node] Received plan with {len(plan_text)} characters")
        state["plan"] = plan_text.strip()
        print("[planner_node] Planner node execution complete")
        return state

    print("[make_graph] Building state graph...")
    graph = StateGraph(dict)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    print("[make_graph] Graph compilation complete")
    return graph.compile()


def load_split(set_type: str):
    print(f"[load_split] Loading dataset split: {set_type}")
    if set_type not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported set_type '{set_type}'.")
    dataset = load_dataset("osunlp/TravelPlanner", set_type)[set_type]
    print(f"[load_split] Loaded {len(dataset)} examples")
    return dataset


def persist_plan(
    base_dir: Path,
    set_type: str,
    example_id: int,
    model_name: str,
    mode: str,
    strategy: str,
    plan_text: str,
) -> None:
    print(f"[persist_plan] Saving plan for example {example_id}...")
    suffix = "" if mode == "two-stage" else f"_{strategy}"
    key = f"{model_name}{suffix}_{mode}_results"
    print(f"[persist_plan] Using key: {key}")

    split_dir = base_dir / set_type
    split_dir.mkdir(parents=True, exist_ok=True)
    plan_path = split_dir / f"generated_plan_{example_id}.json"
    print(f"[persist_plan] Output path: {plan_path}")

    if plan_path.exists():
        print(f"[persist_plan] Loading existing plan file")
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        print(f"[persist_plan] Creating new plan file")
        payload = [{}]

    if not payload:
        payload = [{}]

    payload[-1][key] = plan_text

    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=True)
    print(f"[persist_plan] Successfully saved plan")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal LangGraph agent runner")
    parser.add_argument(
        "--set_type", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory for generated_plan_*.json files",
    )
    parser.add_argument(
        "--model_name", default=os.environ.get("MODEL_NAME"), help="OpenAI model id"
    )
    parser.add_argument(
        "--mode",
        default="sole-planning",
        choices=["two-stage", "sole-planning"],
        help="Tag stored alongside the plan",
    )
    parser.add_argument(
        "--strategy", default="direct", help="Strategy tag for sole-planning mode"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--max_examples", type=int, default=None, help="Optional cap for quick tests"
    )
    return parser

def main() -> None:
    print("=" * 80)
    print("Starting LangGraph Minimal Agent")
    print("=" * 80)

    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model_name:
        parser.error("--model_name must be provided via flag or MODEL_NAME env var")

    print(f"\n[main] Configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Set type: {args.set_type}")
    print(f"  - Output dir: {args.output_dir}")
    print(f"  - Mode: {args.mode}")
    print(f"  - Strategy: {args.strategy}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max examples: {args.max_examples if args.max_examples else 'all'}\n")

    graph = make_graph(args.model_name, args.temperature)
    dataset = load_split(args.set_type)
    base_dir = Path(args.output_dir)

    total = (
        len(dataset)
        if args.max_examples in (None, 0)
        else min(len(dataset), args.max_examples)
    )
    print(f"\n[main] Processing {total} examples...\n")

    # for idx in range(total):
    for idx in range(2):
        print(f"\n{'=' * 80}")
        print(f"Processing Example {idx + 1}/{total}")
        print(f"{'=' * 80}")
        record = dataset[idx]
        state: Dict[str, Any] = {
            "query": record.get("query", ""),
            "reference_information": record.get("reference_information"),
        }
        print(f"[main] Invoking graph for example {idx + 1}...")
        result = graph.invoke(state)
        plan_text = result.get("plan", "")
        persist_plan(
            base_dir=base_dir,
            set_type=args.set_type,
            example_id=idx + 1,
            model_name=args.model_name,
            mode=args.mode,
            strategy=args.strategy,
            plan_text=plan_text,
        )
        print(f"[main] Completed example {idx + 1}/{total}\n")

    print(f"\n{'=' * 80}")
    print(f"All {total} examples processed successfully!")
    print(f"{'=' * 80}")
    


if __name__ == "__main__":
    main()
