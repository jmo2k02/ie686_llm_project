from __future__ import annotations

from pathlib import Path

import nbformat as nbf


"""Generate a self-contained notebook for one TravelPlanner LangSmith thread.

The notebook loads clean tool-run data for actual execution counts, uses the
full export for graph/message context, and annotates trace rows with code refs.
"""


def clean(s: str) -> str:
    lines = s.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    margin = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    prefix = " " * margin
    return "\n".join(line[margin:] if line.startswith(prefix) else line for line in lines)


def md(s: str):
    return nbf.v4.new_markdown_cell(clean(s))


def code(s: str):
    return nbf.v4.new_code_cell(clean(s))


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (uv)",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

nb.cells = [
    md(
        """
        # Single-Thread TravelPlanner Agent-System Analysis

        This notebook reconstructs one LangSmith/LangGraph thread and maps the observed events back to the TravelPlanner implementation in `ie686_llm_project/`.

        Default thread:

        - Root run id: `019e31a3-434b-7083-8b8c-e909e9119ac5`
        - Thread id: `77d47fbb-7830-4f91-8694-8fcbbab780d6`
        - Scenario: Rome hidden history and street food itinerary

        The notebook uses two trace views:

        - `thread_analysis/travel_agent/`: clean real `run_type=tool` rows. Treat this as ground truth for executed tools.
        - `thread_analysis/travel_agent_full/`: verbose LangSmith/LangGraph state snapshots. Use this to inspect messages and graph nesting, but deduplicate because state snapshots repeat prior messages and tool calls.

        The code mapping focuses on the actual agent system:

        - high-level workflow: `travelplanner/workflows/task_planning.py`
        - constraint agent: `travelplanner/agents/constraint_iteration_agent.py`
        - planner/reviewer: `travelplanner/agents/planner/graph.py`
        - execution DeepAgent: `travelplanner/agents/execution/graph.py`
        - subagent tool wiring: `travelplanner/agents/tools.py`
        - TravelPlan mutation tools: `travelplanner/travelplan/tools.py`
        - final validator: `travelplanner/agents/itinerary_validator_agent.py`
        """
    ),
    md("## 0. Setup"),
    code(
        """
        from __future__ import annotations

        import json
        import math
        import re
        from collections import Counter
        from pathlib import Path
        from textwrap import shorten
        from urllib.parse import urlparse

        import numpy as np
        import pandas as pd
        import plotly.express as px
        from IPython.display import Markdown, display

        ROOT = Path.cwd()
        AGENT_ROOT = ROOT / "ie686_llm_project"
        CLEAN_THREAD_DIR = ROOT / "thread_analysis" / "travel_agent"
        FULL_THREAD_DIR = ROOT / "thread_analysis" / "travel_agent_full"
        LOCAL_TRAVEL_DIR = ROOT / "travel_agent"

        # Change this to another root_run_id fragment or thread_id to analyze a different thread.
        TARGET = "019e31a3"

        pd.set_option("display.max_columns", 140)
        pd.set_option("display.max_colwidth", 220)

        ERROR_RE = re.compile(
            r"timeout|timed out|error|failed|failure|exception|traceback|rate limit|unavailable|could not|unable|no reliable|missing info|missing_info",
            re.I,
        )
        UNCERTAIN_RE = re.compile(
            r"verify|re-verify|estimated|estimate|uncertain|unavailable|not available|backup|fallback|could not|depends|must be checked|not confirmed|approx",
            re.I,
        )
        FRAGILE_DOMAIN_RE = re.compile(
            r"google\\.com|maps\\.google|instagram\\.com|facebook\\.com|tripadvisor\\.com|booking\\.com|expedia\\.com|agoda\\.com|turbopass\\.com",
            re.I,
        )

        SEARCH_TOOLS = {
            "search_web",
            "search_restaurants",
            "search_attractions",
            "search_flights",
            "search_hotels",
            "check_route_timing",
            "build_place_distance_graph",
            "distance_between_places",
            "closest_places_to_target",
        }
        PLAN_MUTATION_TOOLS = {"init_plan", "add_day", "remove_day", "add_slot", "insert_slot", "delete_slot"}
        PLAN_READ_TOOLS = {"view_plan", "cost_summary"}
        STATE_TOOLS = {"write_todos"}

        def load_json(path: Path):
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def safe_loads(value, default=None):
            if default is None:
                default = {}
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            if isinstance(value, (dict, list)):
                return value
            try:
                return json.loads(value)
            except Exception:
                return default

        def compact_json(value, width=180):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
            if isinstance(value, str):
                parsed = safe_loads(value, default=value)
            else:
                parsed = value
            if not isinstance(parsed, str):
                parsed = json.dumps(parsed, ensure_ascii=False, sort_keys=True, default=str)
            return shorten(parsed.replace("\\n", " "), width=width, placeholder=" ...")

        def canonical(obj) -> str:
            if obj is None or (isinstance(obj, float) and np.isnan(obj)):
                return ""
            if not isinstance(obj, str):
                obj = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
            obj = obj.lower()
            obj = re.sub(r"https?://\\S+", " ", obj)
            obj = re.sub(r"[^\\w\\s€$£.,:-]", " ", obj)
            return re.sub(r"\\s+", " ", obj).strip()

        def query_like(args: dict) -> str:
            if not isinstance(args, dict):
                return canonical(args)
            for key in ["query", "request", "text", "prompt", "origin", "destination", "place", "location", "city"]:
                if key in args and args[key]:
                    return canonical(args[key])
            return canonical(args)

        def duration_seconds(start, end):
            try:
                return (pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)).total_seconds()
            except Exception:
                return np.nan

        def output_text(outputs_json: object) -> str:
            parsed = safe_loads(outputs_json, default={})
            if isinstance(parsed, dict):
                # LangSmith tool runs commonly store the tool message under output.content.
                out = parsed.get("output", parsed)
                if isinstance(out, dict):
                    return str(out.get("content") or out.get("output") or compact_json(out, width=800))
                return str(out)
            return str(outputs_json or "")

        def tool_role(name: str) -> str:
            if name in PLAN_MUTATION_TOOLS:
                return "TravelPlan mutation"
            if name in PLAN_READ_TOOLS:
                return "TravelPlan read"
            if name in SEARCH_TOOLS:
                return "Domain/search subagent"
            if name in STATE_TOOLS:
                return "DeepAgents state"
            return "LLM/LangGraph/internal"
        """
    ),
    md("## 1. Select The Thread And Load All Inputs"),
    code(
        """
        # Resolve the target through the manifest so either a root run fragment
        # or a LangSmith thread_id can select the thread.
        manifest = pd.read_csv(CLEAN_THREAD_DIR / "manifest.csv")
        full_manifest = pd.read_csv(FULL_THREAD_DIR / "manifest.csv")

        selector = (
            manifest["root_run_id"].astype(str).str.contains(TARGET, na=False)
            | manifest["thread_id"].astype(str).str.contains(TARGET, na=False)
        )
        if not selector.any():
            raise ValueError(f"No clean manifest row matched TARGET={TARGET!r}")
        thread_row = manifest.loc[selector].iloc[0]
        THREAD_ID = thread_row.thread_id
        ROOT_RUN_ID = thread_row.root_run_id

        # Clean export: authoritative actual tool executions.
        clean_runs = pd.read_csv(CLEAN_THREAD_DIR / "langsmith_runs.csv")
        clean_tool_calls = pd.read_csv(CLEAN_THREAD_DIR / "tool_calls.csv")

        # Full export: graph/message context, not execution counts, because
        # serialized LangGraph state repeats prior messages and tool calls.
        full_runs = pd.read_csv(FULL_THREAD_DIR / "langsmith_runs.csv")
        full_messages = pd.read_csv(FULL_THREAD_DIR / "messages.csv")
        full_tool_calls = pd.read_csv(FULL_THREAD_DIR / "tool_calls.csv")

        runs_t = clean_runs[clean_runs.thread_id == THREAD_ID].copy()
        tool_calls_t = clean_tool_calls[clean_tool_calls.thread_id == THREAD_ID].copy()
        full_runs_t = full_runs[full_runs.thread_id == THREAD_ID].copy()
        messages_t = full_messages[full_messages.thread_id == THREAD_ID].copy()
        full_tool_calls_t = full_tool_calls[full_tool_calls.thread_id == THREAD_ID].copy()

        local_path = ROOT / str(thread_row.local_file)
        if not local_path.exists():
            local_path = LOCAL_TRAVEL_DIR / f"run-{ROOT_RUN_ID}.json"
        # Local run JSON contains the final user-facing artifacts.
        local_run = load_json(local_path)
        outputs = local_run.get("outputs") or {}
        final_plan = outputs.get("travelplan") or {}

        display(Markdown(f'''
        **Selected thread**

        - Root run id: `{ROOT_RUN_ID}`
        - Thread id: `{THREAD_ID}`
        - Query: {thread_row.query}
        - Final title: `{thread_row.travelplan_title}`
        - Final validation: `{thread_row.validation_passed}` after `{thread_row.validation_attempts}` attempt(s)
        - Local final artifact: `{local_path.relative_to(ROOT)}`
        - Clean actual tool runs: `{len(runs_t):,}`
        - Full run rows: `{len(full_runs_t):,}`
        - Full extracted message rows: `{len(messages_t):,}`
        - Full extracted tool-call rows: `{len(full_tool_calls_t):,}`
        '''))

        display(pd.DataFrame([thread_row]))
        """
    ),
    md("## 2. Static Code Map For The Agent System"),
    code(
        """
        # Static implementation map used to annotate trace rows with the code
        # responsible for each observed event.
        CODE_MAP = {
            "workflow_entry": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/workflows/task_planning.py",
                "lines": "28-79",
                "symbol": "make_graph",
                "role": "Orchestrates constraint_agent -> planner_agent -> execution_agent -> itinerary_validator and validator retry routing.",
            },
            "workflow_run": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/workflows/task_planning.py",
                "lines": "82-86",
                "symbol": "run",
                "role": "Builds the graph, creates StateContractModel(query=...), invokes the graph, validates final state.",
            },
            "shared_state": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/schema/system_state.py",
                "lines": "109-160",
                "symbol": "StateContractModel",
                "role": "Shared LangGraph state: query, constraints, task_list, travelplan, todos, validation flags.",
            },
            "constraint_agent": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/constraint_iteration_agent.py",
                "lines": "77-116, 202-229",
                "symbol": "ConstraintIterationState / prompts",
                "role": "Extracts hard constraints, asks/records clarification, normalizes constraints.",
            },
            "planner_agent": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/planner/graph.py",
                "lines": "66-98, 101-168, 179-215",
                "symbol": "planner_node / reviewer_node / make_graph",
                "role": "Drafts tasks, reviewer validates them, then finalizes task_list for execution.",
            },
            "execution_node": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/execution/graph.py",
                "lines": "156-204",
                "symbol": "make_node.execution_node",
                "role": "Composes execution prompt, streams DeepAgent, mirrors todos, writes final tp.json/tp.md/tp.ics, returns travelplan/todos.",
            },
            "execution_graph": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/execution/graph.py",
                "lines": "52-96, 99-111, 114-153",
                "symbol": "make_graph / make_validation_graph / _compose_user_prompt",
                "role": "Creates DeepAgent with subagent tools plus TravelPlan tools; validation mode removes init_plan and changes prompt.",
            },
            "subagent_tool_wiring": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/tools.py",
                "lines": "35-118",
                "symbol": "make_subagent_tools",
                "role": "Wraps flight, hotel, restaurant, attraction, web, and routing helpers as StructuredTools.",
            },
            "travelplan_tools": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/travelplan/tools.py",
                "lines": "28-264",
                "symbol": "make_travelplan_tools",
                "role": "Closure-binds a TravelPlan instance into mutating/read-only StructuredTools.",
            },
            "travelplan_model": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/travelplan/plan.py",
                "lines": "27-90",
                "symbol": "TravelPlan",
                "role": "Owns days and delegates slot mutations to Day; computes cost summaries.",
            },
            "day_model": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/travelplan/day.py",
                "lines": "11-79",
                "symbol": "Day",
                "role": "Stores ordered slots; validates 1-based positions and prevents overlapping slots.",
            },
            "slot_model": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/travelplan/slot.py",
                "lines": "9-63",
                "symbol": "Slot",
                "role": "Defines slot fields, valid categories, strict end_time > start_time, overlap semantics.",
            },
            "validator": {
                "file": "ie686_llm_project/travelplanner/src/travelplanner/agents/itinerary_validator_agent.py",
                "lines": "77-144, 147-152",
                "symbol": "validator_node / make_graph",
                "role": "Checks final TravelPlan against constraints/tasks and returns pass/fail feedback.",
            },
        }

        # Map LangSmith tool names back to the implementation layer that created
        # or handled them.
        TOOL_CODE_MAP = {
            "init_plan": "travelplan_tools",
            "add_day": "travelplan_tools",
            "remove_day": "travelplan_tools",
            "add_slot": "travelplan_tools",
            "insert_slot": "travelplan_tools",
            "delete_slot": "travelplan_tools",
            "view_plan": "travelplan_tools",
            "cost_summary": "travelplan_tools",
            "search_flights": "subagent_tool_wiring",
            "search_hotels": "subagent_tool_wiring",
            "search_restaurants": "subagent_tool_wiring",
            "search_attractions": "subagent_tool_wiring",
            "search_web": "subagent_tool_wiring",
            "check_route_timing": "subagent_tool_wiring",
            "build_place_distance_graph": "subagent_tool_wiring",
            "distance_between_places": "subagent_tool_wiring",
            "closest_places_to_target": "subagent_tool_wiring",
            "write_todos": "execution_node",
        }

        code_map_df = pd.DataFrame.from_dict(CODE_MAP, orient="index").reset_index(names="key")
        display(code_map_df)
        """
    ),
    md("## 3. How The High-Level Workflow Explains This Thread"),
    code(
        """
        workflow_steps = pd.DataFrame([
            {
                "step": 1,
                "observed_trace_area": "constraint_agent / constraint messages",
                "state_written": "constraint_list, normalized_constraints, message_histories['key']",
                "code": "task_planning.py:63-70 calls constraint graph; constraint_iteration_agent.py extracts/normalizes constraints",
                "why_it_matters": "This turns the raw Rome request into explicit constraints such as no flights/hotel needed, dates, budget, interests.",
            },
            {
                "step": 2,
                "observed_trace_area": "planner_agent and planner_reviewer_agent histories",
                "state_written": "task_list, planner review feedback/history",
                "code": "planner/graph.py:66-168 drafts and reviews TaskModel rows",
                "why_it_matters": "The execution agent receives task suggestions, but execution/graph.py says they are guidelines, not a strict checklist.",
            },
            {
                "step": 3,
                "observed_trace_area": "clean run_type=tool rows: search_* / add_* / write_todos / view_plan",
                "state_written": "travelplan mutated in place, todos mirrored",
                "code": "execution/graph.py:173-202 streams a DeepAgent; tools.py:35-118 and travelplan/tools.py:28-264 define callable tools",
                "why_it_matters": "This is where the itinerary is actually built and most observable events occur.",
            },
            {
                "step": 4,
                "observed_trace_area": "itinerary_validator / ChatOpenAI runs",
                "state_written": "validation_passed, validation_feedback, validation_attempts, validator history",
                "code": "itinerary_validator_agent.py:99-144 validates and task_planning.py:56-61 decides retry/end",
                "why_it_matters": "This thread passed in one validation attempt, so no execution repair loop followed.",
            },
        ])
        display(workflow_steps)

        display(Markdown(
            "The key architectural point is that `execution/graph.py:61-63` closure-binds a single `TravelPlan` instance into all TravelPlan tools. "
            "Therefore every `add_day`, `add_slot`, `delete_slot`, or `insert_slot` row in LangSmith mutates the same object that later appears in `outputs.travelplan`."
        ))
        """
    ),
    md("## 4. Actual Tool Timeline From Clean LangSmith Tool Runs"),
    code(
        """
        # Normalize clean tool runs into a readable execution timeline.
        runs_t["duration_s"] = [duration_seconds(s, e) for s, e in zip(runs_t.start_time, runs_t.end_time)]
        runs_t["tool_args"] = runs_t.inputs_json.map(safe_loads)
        runs_t["args_norm"] = runs_t.tool_args.map(canonical)
        runs_t["query_norm"] = runs_t.tool_args.map(query_like)
        runs_t["output_text"] = runs_t.outputs_json.map(output_text)
        runs_t["has_error_language"] = runs_t.output_text.str.contains(ERROR_RE, na=False)
        runs_t["has_uncertainty_language"] = runs_t.output_text.str.contains(UNCERTAIN_RE, na=False)
        runs_t = runs_t.sort_values(["start_time", "id"]).reset_index(drop=True)
        runs_t["call_index"] = np.arange(1, len(runs_t) + 1)
        runs_t["role"] = runs_t.name.map(tool_role)
        runs_t["code_key"] = runs_t.name.map(lambda n: TOOL_CODE_MAP.get(n, "execution_node"))
        runs_t["code_ref"] = runs_t.code_key.map(lambda k: f"{CODE_MAP[k]['file']}:{CODE_MAP[k]['lines']}" if k in CODE_MAP else "")
        runs_t["args_preview"] = runs_t.tool_args.map(lambda x: compact_json(x, width=160))
        runs_t["output_preview"] = runs_t.output_text.map(lambda x: shorten(str(x).replace("\\n", " "), width=180, placeholder=" ..."))

        timeline_cols = [
            "call_index", "start_time", "duration_s", "name", "role", "args_preview", "output_preview",
            "has_error_language", "has_uncertainty_language", "code_ref",
        ]
        display(runs_t[timeline_cols])

        counts = runs_t.groupby(["role", "name"]).agg(
            calls=("id", "size"),
            p50_duration_s=("duration_s", "median"),
            max_duration_s=("duration_s", "max"),
            error_language=("has_error_language", "sum"),
            uncertainty_language=("has_uncertainty_language", "sum"),
        ).reset_index().sort_values(["role", "calls"], ascending=[True, False])
        display(counts)

        fig = px.bar(counts, x="name", y="calls", color="role", title="Actual executed tools in this thread")
        fig.update_layout(height=500, xaxis_tickangle=-35)
        fig.show()
        """
    ),
    md("## 5. TravelPlan Mutation Timeline"),
    code(
        """
        # Focus on tools that directly read or mutate the closure-bound
        # TravelPlan instance.
        mutation_t = runs_t[runs_t.name.isin(PLAN_MUTATION_TOOLS | PLAN_READ_TOOLS)].copy()
        mutation_t["mutation_semantics"] = mutation_t.name.map({
            "init_plan": "sets title and clears days",
            "add_day": "TravelPlan.add_day -> appends Day(index=len(days)+1)",
            "remove_day": "TravelPlan.remove_day -> pops day and renumbers",
            "add_slot": "builds Slot, then TravelPlan.add_slot -> Day.append_slot with overlap validation",
            "insert_slot": "builds Slot, then TravelPlan.insert_slot -> Day.insert_slot with overlap validation",
            "delete_slot": "TravelPlan.delete_slot -> Day.delete_slot by 1-based position",
            "view_plan": "TravelPlan.to_markdown read-only rendering",
            "cost_summary": "TravelPlan.cost_summary read-only total/per-day costs",
        })
        display(mutation_t[[
            "call_index", "name", "args_preview", "output_preview", "mutation_semantics", "code_ref"
        ]])

        exact_repeats = (
            mutation_t.groupby(["name", "args_norm"])
            .agg(count=("id", "size"), first_call=("call_index", "min"), last_call=("call_index", "max"), example_args=("args_preview", "first"))
            .reset_index()
            .query("count > 1")
            .sort_values("count", ascending=False)
        )
        display(Markdown("### Exact repeated TravelPlan tool calls in this thread"))
        display(exact_repeats)

        adjacent = mutation_t[["call_index", "name", "args_norm", "args_preview"]].copy()
        adjacent["prev_name"] = adjacent.name.shift(1)
        adjacent["prev_args_norm"] = adjacent.args_norm.shift(1)
        adjacent_repeats = adjacent[(adjacent.name == adjacent.prev_name) & (adjacent.args_norm == adjacent.prev_args_norm)]
        display(Markdown("### Consecutive exact repeats"))
        display(adjacent_repeats)
        """
    ),
    md("## 6. Final TravelPlan Structural Check"),
    code(
        """
        # Flatten the final TravelPlan artifact so every slot can be checked and
        # compared with the earlier mutation timeline.
        plan_rows = []
        for day in final_plan.get("days", []):
            for pos, slot in enumerate(day.get("slots", []), start=1):
                links = slot.get("links") or []
                notes = slot.get("notes") or ""
                desc = slot.get("description") or ""
                plan_rows.append({
                    "day_index": day.get("index"),
                    "day_label": day.get("label"),
                    "calendar_date": day.get("calendar_date"),
                    "position": pos,
                    "name": slot.get("name"),
                    "category": slot.get("category"),
                    "start_time": slot.get("start_time"),
                    "end_time": slot.get("end_time"),
                    "location": slot.get("location"),
                    "cost": slot.get("cost"),
                    "link_count": len(links),
                    "links": "; ".join(links),
                    "fragile_link": any(FRAGILE_DOMAIN_RE.search(url or "") for url in links),
                    "uncertainty_text": bool(UNCERTAIN_RE.search(" ".join([notes, desc]))),
                    "notes": notes,
                })
        slots_df = pd.DataFrame(plan_rows)
        if not slots_df.empty:
            slots_df["start_dt"] = pd.to_datetime(slots_df.start_time, errors="coerce")
            slots_df["end_dt"] = pd.to_datetime(slots_df.end_time, errors="coerce")
            slots_df["invalid_time"] = slots_df.end_dt <= slots_df.start_dt
        display(slots_df.drop(columns=["start_dt", "end_dt"], errors="ignore"))

        overlap_rows = []
        if not slots_df.empty:
            for day_index, group in slots_df.groupby("day_index"):
                g = group.sort_values("start_dt")
                records = g.to_dict("records")
                for i, a in enumerate(records):
                    for b in records[i + 1:]:
                        if pd.notna(a["start_dt"]) and pd.notna(a["end_dt"]) and pd.notna(b["start_dt"]) and pd.notna(b["end_dt"]):
                            if a["start_dt"] < b["end_dt"] and b["start_dt"] < a["end_dt"]:
                                overlap_rows.append({
                                    "day_index": day_index,
                                    "slot_a": a["name"],
                                    "slot_b": b["name"],
                                    "a_time": f"{a['start_time']} -> {a['end_time']}",
                                    "b_time": f"{b['start_time']} -> {b['end_time']}",
                                })
        structural_summary = {
            "days": len(final_plan.get("days", [])),
            "slots": len(slots_df),
            "computed_total_cost": float(slots_df.cost.fillna(0).sum()) if not slots_df.empty else 0.0,
            "invalid_time_slots": int(slots_df.invalid_time.sum()) if not slots_df.empty else 0,
            "overlapping_slot_pairs": len(overlap_rows),
            "slots_missing_links": int((slots_df.link_count == 0).sum()) if not slots_df.empty else 0,
            "fragile_link_slots": int(slots_df.fragile_link.sum()) if not slots_df.empty else 0,
            "uncertainty_text_slots": int(slots_df.uncertainty_text.sum()) if not slots_df.empty else 0,
        }
        display(pd.DataFrame([structural_summary]))
        display(Markdown("### Overlaps"))
        display(pd.DataFrame(overlap_rows))
        """
    ),
    md("## 7. Domain/Search Evidence Timeline"),
    code(
        """
        # Domain/search calls are evidence-producing steps before or between
        # plan mutations.
        search_t = runs_t[runs_t.name.isin(SEARCH_TOOLS)].copy()
        search_t["domain"] = search_t.name.str.removeprefix("search_")
        display(search_t[[
            "call_index", "name", "duration_s", "query_norm", "output_preview", "has_error_language", "has_uncertainty_language", "code_ref"
        ]])

        repeated_queries = (
            search_t.groupby(["name", "query_norm"])
            .agg(count=("id", "size"), first_call=("call_index", "min"), last_call=("call_index", "max"), example_args=("args_preview", "first"))
            .reset_index()
            .query("query_norm != '' and count > 1")
            .sort_values("count", ascending=False)
        )
        display(Markdown("### Repeated domain/search queries"))
        display(repeated_queries)

        if not search_t.empty:
            fig = px.scatter(
                search_t,
                x="call_index",
                y="duration_s",
                color="name",
                hover_data=["query_norm", "output_preview", "has_error_language", "has_uncertainty_language"],
                title="Search/subagent calls over execution time",
            )
            fig.update_layout(height=500)
            fig.show()
        """
    ),
    md("## 8. Full LangSmith Graph Nesting And State Snapshots"),
    code(
        """
        # Full runs show LangGraph nesting, middleware, and model calls; they are
        # context, not the source of truth for tool execution counts.
        full_runs_t["duration_s"] = [duration_seconds(s, e) for s, e in zip(full_runs_t.start_time, full_runs_t.end_time)]
        full_runs_t = full_runs_t.sort_values(["start_time", "id"]).reset_index(drop=True)
        full_runs_t["run_index"] = np.arange(1, len(full_runs_t) + 1)
        full_runs_t["parent_short"] = full_runs_t.parent_run_id.fillna("").astype(str).str.slice(0, 8)
        full_runs_t["id_short"] = full_runs_t.id.astype(str).str.slice(0, 8)
        full_runs_t["metadata"] = full_runs_t.metadata_json.map(safe_loads)
        full_runs_t["langgraph_node"] = full_runs_t.metadata.map(lambda m: m.get("langgraph_node") if isinstance(m, dict) else None)
        full_runs_t["langgraph_step"] = full_runs_t.metadata.map(lambda m: m.get("langgraph_step") if isinstance(m, dict) else None)
        full_runs_t["ls_run_depth"] = full_runs_t.metadata.map(lambda m: m.get("ls_run_depth") if isinstance(m, dict) else None)

        display(full_runs_t[[
            "run_index", "id_short", "parent_short", "name", "run_type", "start_time", "duration_s", "langgraph_node", "langgraph_step", "ls_run_depth"
        ]].head(200))

        run_counts = full_runs_t.groupby(["run_type", "name"]).size().reset_index(name="rows").sort_values("rows", ascending=False)
        display(run_counts.head(50))

        display(Markdown(
            "The full export has graph, model, middleware, subagent, and tool rows. It is the right source for nesting, but not for counting actual tool executions. "
            "Use the clean `run_type=tool` export for execution counts."
        ))
        """
    ),
    md("## 9. Message Reconstruction With Deduplication"),
    code(
        """
        # Exact deduplication makes repeated serialized state snapshots easier to
        # inspect without losing the original row counts.
        messages_t = messages_t.copy()
        messages_t["content_preview"] = messages_t.content.fillna("").astype(str).map(lambda x: shorten(x.replace("\\n", " "), width=260, placeholder=" ..."))
        messages_t["tool_calls_count"] = messages_t.tool_calls_json.fillna("[]").map(lambda x: len(safe_loads(x, default=[])) if isinstance(safe_loads(x, default=[]), list) else 0)
        dedup_messages = messages_t.drop_duplicates([
            "run_id", "source_field", "message_type", "name", "tool_call_id", "content", "tool_calls_json"
        ]).copy()
        display(Markdown(f"Full message rows for this thread: **{len(messages_t):,}**. Deduplicated exact message rows: **{len(dedup_messages):,}**."))
        display(dedup_messages[[
            "run_id", "run_name", "run_type", "source_field", "message_type", "name", "tool_call_id", "tool_calls_count", "content_preview"
        ]].head(200))

        message_counts = dedup_messages.groupby(["run_name", "message_type", "name"]).size().reset_index(name="rows").sort_values("rows", ascending=False)
        display(message_counts.head(50))
        """
    ),
    md("## 10. Tool-Call Overcount Check: Full Snapshots vs Real Tool Runs"),
    code(
        """
        # Quantify the overcount from recursively extracted state snapshots.
        full_tc = full_tool_calls_t.copy()
        full_tc["args_norm"] = full_tc.tool_args_json.map(lambda x: canonical(safe_loads(x)))
        full_tc["dedup_key"] = full_tc.apply(
            lambda r: r.tool_call_id if isinstance(r.tool_call_id, str) and r.tool_call_id else f"{r.tool_name}|{r.args_norm}",
            axis=1,
        )
        full_tc_dedup = full_tc.drop_duplicates(["dedup_key", "tool_name", "args_norm"])

        overcount = pd.DataFrame([
            {"source": "clean real tool-run export", "rows": len(runs_t), "unique_by_id_or_call": runs_t.id.nunique()},
            {"source": "full extracted tool_calls", "rows": len(full_tc), "unique_by_id_or_call": full_tc_dedup.dedup_key.nunique()},
        ])
        display(overcount)

        by_tool_compare = pd.concat([
            runs_t.name.value_counts().rename_axis("tool").reset_index(name="clean_real_runs").set_index("tool"),
            full_tc.tool_name.value_counts().rename_axis("tool").reset_index(name="full_snapshot_rows").set_index("tool"),
            full_tc_dedup.tool_name.value_counts().rename_axis("tool").reset_index(name="full_dedup_estimate").set_index("tool"),
        ], axis=1).fillna(0).astype(int).reset_index().sort_values("clean_real_runs", ascending=False)
        display(by_tool_compare)

        display(Markdown(
            "Use `clean_real_runs` for actual execution counts. The full snapshot rows are much larger because LangGraph state serializes previous messages and tool calls repeatedly."
        ))
        """
    ),
    md("## 11. Local Final Artifacts: Constraints, Tasks, Histories, Todos"),
    code(
        """
        display(Markdown("### Constraint list"))
        display(pd.DataFrame(outputs.get("constraint_list") or []))

        display(Markdown("### Normalized constraints"))
        display(pd.json_normalize(outputs.get("normalized_constraints") or {}))

        display(Markdown("### Planner task list"))
        display(pd.DataFrame(outputs.get("task_list") or []))

        display(Markdown("### Final todos mirrored from DeepAgents"))
        display(pd.DataFrame(outputs.get("todos") or []))

        histories = outputs.get("message_histories") or {}
        hist_rows = []
        for key, hist in histories.items():
            messages = hist.get("messages") or []
            hist_rows.append({
                "history_key": key,
                "user_agent": hist.get("user_agent"),
                "model": hist.get("model"),
                "agent_ref": hist.get("agent_ref"),
                "message_count": len(messages),
                "first_message": shorten(str(messages[0]) if messages else "", width=220, placeholder=" ..."),
                "last_message": shorten(str(messages[-1]) if messages else "", width=220, placeholder=" ..."),
            })
        display(Markdown("### Message histories saved in final local JSON"))
        display(pd.DataFrame(hist_rows))
        """
    ),
    md("## 12. Event-To-Code Narrative For This Thread"),
    code(
        """
        narrative = []
        narrative.append({
            "phase": "Input and constraints",
            "what_happened": "The raw Rome request entered StateContractModel.query. The constraint agent extracted/normalized dates, destination, budget, and 'no flights/hotel needed'.",
            "trace_evidence": "local outputs.constraint_list and outputs.normalized_constraints; full messages under constraint-related histories",
            "code_mapping": "task_planning.py:69-70 -> constraint_iteration_agent.py prompts/state -> system_state.py:122-123",
        })
        narrative.append({
            "phase": "Task planning",
            "what_happened": "The planner produced a task list and the reviewer approved or corrected it before execution.",
            "trace_evidence": "local outputs.task_list plus message_histories['planner_agent'] and ['planner_reviewer_agent']",
            "code_mapping": "planner/graph.py:66-168; route_after_review at 165-168 controls planner retries",
        })
        narrative.append({
            "phase": "Execution agent setup",
            "what_happened": "The execution node built a DeepAgent, composed a prompt from query/constraints/tasks, and exposed subagent search tools plus TravelPlan mutation tools.",
            "trace_evidence": "full_runs_t contains DeepAgents/LangGraph/model/tool nesting; clean runs start with execution tool rows",
            "code_mapping": "execution/graph.py:173-181 builds agent; 181 calls _compose_user_prompt; 185-202 streams and returns travelplan/todos",
        })
        narrative.append({
            "phase": "Evidence gathering",
            "what_happened": "The agent called search_web/search_restaurants/search_attractions/build_place_distance_graph for Rome details and restaurants before or between itinerary mutations.",
            "trace_evidence": "search_t timeline, query_norm, output_preview, uncertainty/error-language flags",
            "code_mapping": "agents/tools.py:53-118 creates StructuredTools; subagent_tools/*.py adapters run the domain agents",
        })
        narrative.append({
            "phase": "Plan mutation",
            "what_happened": "The agent initialized the plan, added three days, and added slots. Each tool mutated the same closure-bound TravelPlan instance.",
            "trace_evidence": "mutation_t timeline and final outputs.travelplan slots",
            "code_mapping": "travelplan/tools.py:63-160 tool bodies; plan.py:41-90 model operations; day.py:29-75 slot validation/sorting; slot.py:49-63 time and overlap semantics",
        })
        narrative.append({
            "phase": "Validation and end",
            "what_happened": "The validator checked the final plan and passed it in one attempt, so task_planning.route_after_validator ended the graph instead of looping back to execution_agent.",
            "trace_evidence": "manifest.validation_passed=True, validation_attempts=1, message_histories['itinerary_validator']",
            "code_mapping": "itinerary_validator_agent.py:99-144 and task_planning.py:56-61",
        })
        display(pd.DataFrame(narrative))
        """
    ),
    md("## 13. Focus Questions To Ask While Inspecting The Thread"),
    md(
        """
        Use the tables above to answer concrete questions about the agent system:

        - Did the execution agent follow the planner's task list, or did it merge/drop/add work? Compare `outputs.task_list` with the actual clean tool timeline.
        - Which evidence-producing tool calls preceded each itinerary slot? Compare `search_t.call_index` with later `add_slot.call_index` and the final slot links/notes.
        - Did the agent repair mistakes? Look for repeated exact mutation calls, adjacent repeats, `delete_slot`, `insert_slot`, and validator attempts greater than one.
        - Did uncertainty propagate into the final artifact? Compare `has_uncertainty_language` in search outputs with final slot `notes`, missing links, or fragile links.
        - Where would you change behavior? Use the `code_ref` column: prompt composition is in `execution/graph.py`, tool availability is in `agents/tools.py`, mutation semantics are in `travelplan/tools.py`, and validation retry routing is in `task_planning.py`.
        """
    ),
]

out_path = Path("single_thread_agent_system_analysis.ipynb")
nbf.write(nb, out_path)
print(f"Wrote {out_path}")
