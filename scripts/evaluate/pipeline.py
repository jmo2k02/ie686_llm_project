"""
Command-line interface for evaluating LangGraph agents via generation helpers.

Usage:

uv run python evaluate/pipeline.py   --model_name gpt-4o-mini   --set_type train   --output_dir /tmp/travelplanner   --mode sole-planning   --strategy direct
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .generate import (
        DEFAULT_GRAPH_SPEC,
        generation_loop,
        instantiate_graph,
        load_split,
    )
except ImportError:  # pragma: no cover - fallback when run directly
    from generate import (  # type: ignore
        DEFAULT_GRAPH_SPEC,
        generation_loop,
        instantiate_graph,
        load_split,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph agent runner")
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
    parser.add_argument("--strategy", default="direct", help="Strategy tag")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--max_examples", type=int, default=None, help="Optional cap for quick tests"
    )
    parser.add_argument(
        "--graph_spec",
        default=str(DEFAULT_GRAPH_SPEC),
        help=(
            "Import string pointing to a callable that returns a compiled graph. "
            "Format: /path/to/module.py:function_name"
        ),
    )
    return parser


def main() -> None:
    print("=" * 80)
    print("Starting LangGraph Agent Runner")
    print("=" * 80)

    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model_name:
        parser.error("--model_name must be provided via flag or MODEL_NAME env var")

    print(f"\n[pipeline] Configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Set type: {args.set_type}")
    print(f"  - Output dir: {args.output_dir}")
    print(f"  - Mode: {args.mode}")
    print(f"  - Strategy: {args.strategy}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Graph spec: {args.graph_spec}")
    print(f"  - Max examples: {args.max_examples if args.max_examples else 'all'}\n")

    graph = instantiate_graph(
        args.graph_spec, model_name=args.model_name, temperature=args.temperature
    )
    dataset = load_split(args.set_type)
    base_dir = Path(args.output_dir)

    total = (
        len(dataset)
        if args.max_examples in (None, 0)
        else min(len(dataset), args.max_examples)
    )

    generation_loop(
        graph=graph,
        dataset=dataset,
        base_dir=base_dir,
        set_type=args.set_type,
        model_name=args.model_name,
        mode=args.mode,
        strategy=args.strategy,
        total_examples=total,
    )


if __name__ == "__main__":
    main()
