"""Utility for running a LangGraph-powered agent against the TravelPlanner benchmark.

Unlike the demo script in ``scripts/example_agents/langgraph_minimal.py`` this module does not
define a graph itself. Instead, callers may point to any Python file containing a callable that
returns a compiled LangGraph graph (or compatible object with ``invoke``). The callable is
referenced via an import string that looks like ``/path/to/module.py:make_graph``. The generation
loop mirrors the logic found in ``langgraph_minimal.py`` so downstream tooling can expect the same
outputs.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict

from datasets import load_dataset


GraphBuilder = Callable[..., Any]

DEFAULT_GRAPH_PATH = (
    Path(__file__).resolve().parents[1] / "example_agents" / "langgraph_minimal.py"
)
DEFAULT_GRAPH_SPEC = f"{DEFAULT_GRAPH_PATH}:make_graph"


def parse_import_string(import_string: str) -> tuple[Path, str]:
    """Split ``path:callable`` strings into a filesystem path and attribute name."""

    if ":" not in import_string:
        raise ValueError(
            "Graph import string must be in the format '/path/to/module.py:function_name'."
        )
    path_str, attr = import_string.split(":", 1)
    module_path = Path(path_str.strip()).expanduser()
    if not module_path.is_absolute():
        module_path = (Path.cwd() / module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module file '{module_path}' does not exist")
    attr = attr.strip()
    if not attr:
        raise ValueError("Function name in import string cannot be empty")
    return module_path, attr


def load_graph_builder(import_string: str) -> GraphBuilder:
    """Load and return the graph-building callable referenced by ``import_string``."""

    module_path, attr = parse_import_string(import_string)
    module_name = f"graph_module_{module_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module specification from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        builder = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_path}' does not define a callable named '{attr}'"
        ) from exc

    if not callable(builder):
        raise TypeError(f"Attribute '{attr}' in '{module_path}' is not callable")

    return builder


def _select_supported_kwargs(builder: GraphBuilder, candidate_kwargs: Dict[str, Any]):
    """Return kwargs filtered to the ones accepted by ``builder`` when possible."""

    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        # Builtins or callables without signatures; best effort means pass everything
        return candidate_kwargs

    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kw:
        return candidate_kwargs

    supported = {
        name
        for name, param in signature.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in candidate_kwargs.items() if key in supported}


def instantiate_graph(import_string: str, **builder_kwargs: Any):
    """Instantiate a graph by loading the specified callable and invoking it."""

    builder = load_graph_builder(import_string)
    filtered_kwargs = _select_supported_kwargs(builder, builder_kwargs)
    return builder(**filtered_kwargs)


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


def generation_loop(
    *,
    graph: Any,
    dataset,
    base_dir: Path,
    set_type: str,
    model_name: str,
    mode: str,
    strategy: str,
    total_examples: int,
) -> None:
    """Run the generation process for ``total_examples`` dataset entries."""

    print(f"\n[generation_loop] Processing {total_examples} examples...\n")
    for idx in range(total_examples):
        print(f"\n{'=' * 80}")
        print(f"Processing Example {idx + 1}/{total_examples}")
        print(f"{'=' * 80}")
        record = dataset[idx]
        state: Dict[str, Any] = {
            "query": record.get("query", ""),
            "reference_information": record.get("reference_information"),
        }
        print(f"[generation_loop] Invoking graph for example {idx + 1}...")
        result = graph.invoke(state)
        plan_text = result.get("plan", "")
        persist_plan(
            base_dir=base_dir,
            set_type=set_type,
            example_id=idx + 1,
            model_name=model_name,
            mode=mode,
            strategy=strategy,
            plan_text=plan_text,
        )
        print(f"[generation_loop] Completed example {idx + 1}/{total_examples}\n")

    print(f"\n{'=' * 80}")
    print(f"All {total_examples} examples processed successfully!")
    print(f"{'=' * 80}")
