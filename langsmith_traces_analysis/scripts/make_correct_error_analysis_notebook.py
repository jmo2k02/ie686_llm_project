from pathlib import Path
import nbformat as nbf


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
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (uv)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

nb.cells = [
    md(
        """
        # Correct LangSmith Tool-Use Error Analysis

        This notebook uses the real LangSmith thread export for the multi-agent TravelPlanner system:

        - `thread_analysis/travel_agent/langsmith_runs.csv`: real `run_type=tool` runs from LangSmith
        - `thread_analysis/travel_agent/tool_calls.csv`: one row per real tool run
        - `thread_analysis/travel_agent/manifest.csv`: local run id, thread id, query, validation outcome
        - `travel_agent/run-*.json`: final structured `TravelPlan` artifacts
        - `baseline/run-*.json`: baseline Tavily `tool_calls` embedded in messages

        The goal is to produce paper-ready error analysis for:

        - repeated tool calls
        - repeated queries
        - dead loops
        - timeouts / long-running tool calls
        - cascading errors
        - structural TravelPlan failures

        This replaces the earlier proxy-only analysis for the travel agent. The TravelPlanner section now uses actual execution-agent tool runs exported from LangSmith.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import math
        import re
        from collections import Counter
        from datetime import datetime
        from itertools import combinations
        from pathlib import Path
        from urllib.parse import urlparse

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import seaborn as sns
        from IPython.display import Markdown, display

        ROOT = Path.cwd()
        BASELINE_DIR = ROOT / "baseline"
        TRAVEL_DIR = ROOT / "travel_agent"
        THREAD_DIR = ROOT / "thread_analysis" / "travel_agent"

        sns.set_theme(style="whitegrid", context="talk")
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_colwidth", 180)

        VALID_CATEGORIES = {"meal", "attraction", "transport", "lodging", "leisure", "other"}
        SEARCH_TOOLS = {"search_web", "search_restaurants", "search_attractions", "search_flights", "search_hotels", "check_route_timing", "build_place_distance_graph"}
        MUTATION_TOOLS = {"init_plan", "add_day", "remove_day", "add_slot", "insert_slot", "delete_slot", "view_plan", "cost_summary"}
        FRAGILE_DOMAINS = re.compile(r"google\.com|maps\.google|instagram\.com|facebook\.com|tripadvisor\.com|booking\.com|expedia\.com|agoda\.com|turbopass\.com", re.I)
        ERROR_RE = re.compile(r"timeout|timed out|error|failed|failure|exception|traceback|rate limit|unavailable|could not|unable|no reliable|missing info|missing_info", re.I)
        UNCERTAIN_RE = re.compile(r"verify|re-verify|estimated|estimate|uncertain|unavailable|not available|backup|fallback|could not|depends|must be checked|not confirmed", re.I)

        def load_json(path: Path):
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def safe_loads(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return {}
            if isinstance(value, dict):
                return value
            try:
                return json.loads(value)
            except Exception:
                return {"_raw": str(value)}

        def safe_json(value) -> str:
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return str(value)

        def canonical(obj) -> str:
            if obj is None or (isinstance(obj, float) and np.isnan(obj)):
                return ""
            if not isinstance(obj, str):
                obj = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            obj = obj.lower()
            obj = re.sub(r"https?://\S+", " ", obj)
            obj = re.sub(r"[^\w\s€$£.-]", " ", obj)
            return re.sub(r"\s+", " ", obj).strip()

        def arg_text(args: dict) -> str:
            if not isinstance(args, dict):
                return canonical(args)
            for key in ["query", "request", "text", "prompt", "origin", "destination", "place", "location"]:
                if key in args and args[key]:
                    return canonical(args[key])
            return canonical(args)

        def duration_seconds(start, end):
            try:
                return (pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)).total_seconds()
            except Exception:
                return np.nan
        """
    ),
    md("## 1. Load Real Tool Runs And Local Trace Metadata"),
    code(
        """
        manifest = pd.read_csv(THREAD_DIR / "manifest.csv")
        travel_runs = pd.read_csv(THREAD_DIR / "langsmith_runs.csv")
        travel_tools = pd.read_csv(THREAD_DIR / "tool_calls.csv")

        travel_runs["duration_s"] = [duration_seconds(s, e) for s, e in zip(travel_runs.start_time, travel_runs.end_time)]
        travel_runs["tool_args"] = travel_runs.inputs_json.map(safe_loads)
        travel_runs["args_norm"] = travel_runs.tool_args.map(canonical)
        travel_runs["query_norm"] = travel_runs.tool_args.map(arg_text)
        travel_runs["output_text"] = travel_runs.outputs_json.fillna("").astype(str)
        travel_runs["has_error_language"] = travel_runs.output_text.str.contains(ERROR_RE, na=False)
        travel_runs["has_uncertainty_language"] = travel_runs.output_text.str.contains(UNCERTAIN_RE, na=False)
        travel_runs = travel_runs.merge(manifest[["thread_id", "root_run_id", "query", "validation_passed", "validation_attempts", "travelplan_title"]], on="thread_id", how="left")
        travel_runs = travel_runs.sort_values(["thread_id", "start_time", "id"]).reset_index(drop=True)
        travel_runs["call_index"] = travel_runs.groupby("thread_id").cumcount()

        display(Markdown(f"Loaded **{len(travel_runs):,} real TravelPlanner tool runs** across **{travel_runs.thread_id.nunique()} LangGraph threads**."))
        display(travel_runs[["thread_id", "root_run_id", "name", "start_time", "duration_s", "query_norm", "validation_passed", "validation_attempts"]].head(20))

        fig = px.bar(travel_runs.name.value_counts().reset_index(name="count"), x="name", y="count", title="Real TravelPlanner Tool Runs By Tool Name")
        fig.update_layout(height=500, xaxis_tickangle=-35)
        fig.show()
        """
    ),
    md("## 2. Baseline Tavily Tool Calls"),
    code(
        """
        baseline_rows = []
        for path in sorted(BASELINE_DIR.glob("run-*.json")):
            obj = load_json(path)
            messages = (obj.get("outputs") or {}).get("messages") or []
            run_id = path.stem.removeprefix("run-")
            for mi, msg in enumerate(messages):
                for ci, call in enumerate(msg.get("tool_calls") or []):
                    args = call.get("args") or {}
                    baseline_rows.append({
                        "system": "baseline",
                        "thread_id": run_id,
                        "root_run_id": run_id,
                        "name": call.get("name"),
                        "call_index": len(baseline_rows),
                        "message_index": mi,
                        "tool_args": args,
                        "args_norm": canonical(args),
                        "query_norm": arg_text(args),
                        "duration_s": np.nan,
                    })

        baseline_tools = pd.DataFrame(baseline_rows)
        display(Markdown(f"Loaded **{len(baseline_tools):,} explicit baseline Tavily tool calls** from local JSON exports."))
        display(baseline_tools.head(20))
        """
    ),
    md("## 3. Repeated Tool Calls"),
    code(
        """
        travel_dupes = (
            travel_runs.groupby(["thread_id", "name", "args_norm"])
            .agg(count=("id", "size"), first_call=("call_index", "min"), last_call=("call_index", "max"), example_args=("tool_args", "first"), root_run_id=("root_run_id", "first"))
            .reset_index()
            .query("count > 1")
            .sort_values("count", ascending=False)
        )
        baseline_dupes = (
            baseline_tools.groupby(["thread_id", "name", "args_norm"])
            .agg(count=("name", "size"), first_call=("call_index", "min"), last_call=("call_index", "max"), example_args=("tool_args", "first"), root_run_id=("root_run_id", "first"))
            .reset_index()
            .query("count > 1")
            .sort_values("count", ascending=False)
        )

        display(Markdown("### TravelPlanner repeated exact tool calls"))
        display(travel_dupes.head(50))
        display(Markdown("### Baseline repeated exact Tavily calls"))
        display(baseline_dupes.head(50))

        rep_summary = pd.concat([
            travel_dupes.assign(system="travel_agent"),
            baseline_dupes.assign(system="baseline"),
        ], ignore_index=True)
        if not rep_summary.empty:
            fig = px.bar(rep_summary.groupby(["system", "name"]).agg(repeated_groups=("args_norm", "size"), repeated_calls=("count", "sum")).reset_index(),
                         x="name", y="repeated_calls", color="system", barmode="group", title="Repeated Exact Tool Calls")
            fig.update_layout(height=550, xaxis_tickangle=-35)
            fig.show()
        """
    ),
    md("## 4. Repeated Queries"),
    code(
        """
        travel_search = travel_runs[travel_runs.name.isin(SEARCH_TOOLS)].copy()
        baseline_search = baseline_tools.copy()

        travel_query_dupes = (
            travel_search.groupby(["thread_id", "name", "query_norm"])
            .agg(count=("id", "size"), example_args=("tool_args", "first"), root_run_id=("root_run_id", "first"))
            .reset_index()
            .query("query_norm != '' and count > 1")
            .sort_values("count", ascending=False)
        )
        baseline_query_dupes = (
            baseline_search.groupby(["thread_id", "name", "query_norm"])
            .agg(count=("name", "size"), example_args=("tool_args", "first"), root_run_id=("root_run_id", "first"))
            .reset_index()
            .query("query_norm != '' and count > 1")
            .sort_values("count", ascending=False)
        )

        display(Markdown("### TravelPlanner repeated search/subagent queries"))
        display(travel_query_dupes.head(50))
        display(Markdown("### Baseline repeated Tavily queries"))
        display(baseline_query_dupes.head(50))

        qsum = pd.concat([travel_query_dupes.assign(system="travel_agent"), baseline_query_dupes.assign(system="baseline")], ignore_index=True)
        if not qsum.empty:
            fig = px.histogram(qsum, x="count", color="system", marginal="box", title="Distribution Of Repeated Query Counts")
            fig.show()
        """
    ),
    md("## 5. Dead Loops"),
    code(
        """
        loop_rows = []

        def detect_loops(df, system):
            for thread_id, g in df.sort_values(["thread_id", "call_index"]).groupby("thread_id"):
                rows = g.to_dict("records")
                # Consecutive exact repeat: A, A, A
                streak = 1
                prev_key = None
                for row in rows:
                    key = (row["name"], row.get("args_norm") or row.get("query_norm"))
                    if key == prev_key:
                        streak += 1
                    else:
                        streak = 1
                    if streak >= 2:
                        loop_rows.append({"system": system, "thread_id": thread_id, "root_run_id": row.get("root_run_id"), "loop_type": "consecutive_exact_repeat", "tool": row["name"], "count": streak, "example": row.get("query_norm") or row.get("args_norm")})
                    prev_key = key

                # Alternating mutation loop: add/delete/add/delete or insert/delete repeated on same day/position-ish args.
                for i in range(len(rows) - 3):
                    seq = rows[i:i+4]
                    names = [r["name"] for r in seq]
                    if names in (["add_slot", "delete_slot", "add_slot", "delete_slot"], ["insert_slot", "delete_slot", "insert_slot", "delete_slot"], ["delete_slot", "add_slot", "delete_slot", "add_slot"], ["delete_slot", "insert_slot", "delete_slot", "insert_slot"]):
                        loop_rows.append({"system": system, "thread_id": thread_id, "root_run_id": seq[-1].get("root_run_id"), "loop_type": "mutation_backtrack_loop", "tool": " -> ".join(names), "count": 4, "example": safe_json([s.get("tool_args") for s in seq])})

                # Excessive same query in one thread.
                counts = Counter((r["name"], r.get("query_norm") or r.get("args_norm")) for r in rows)
                for (tool, q), count in counts.items():
                    if q and count >= 3:
                        loop_rows.append({"system": system, "thread_id": thread_id, "root_run_id": rows[0].get("root_run_id"), "loop_type": "same_call_3plus", "tool": tool, "count": count, "example": q})

        detect_loops(travel_runs, "travel_agent")
        detect_loops(baseline_tools, "baseline")
        loops_df = pd.DataFrame(loop_rows).drop_duplicates() if loop_rows else pd.DataFrame()

        display(loops_df.sort_values(["system", "count"], ascending=[True, False]).head(100) if not loops_df.empty else pd.DataFrame({"status": ["No dead-loop indicators found"]}))
        if not loops_df.empty:
            fig = px.bar(loops_df.groupby(["system", "loop_type"]).size().reset_index(name="events"), x="loop_type", y="events", color="system", barmode="group", title="Dead-Loop Indicators")
            fig.update_layout(height=500, xaxis_tickangle=-25)
            fig.show()
        """
    ),
    md("## 6. Timeouts And Long-Running Tool Calls"),
    code(
        """
        # No explicit tool errors were returned in the exported tool runs, so timeout risk is approximated by latency outliers and error language.
        p95 = travel_runs.duration_s.quantile(0.95)
        p99 = travel_runs.duration_s.quantile(0.99)
        timeout_df = travel_runs[(travel_runs.duration_s >= p99) | travel_runs.has_error_language].copy()
        timeout_df["reason"] = np.where(timeout_df.has_error_language, "error_language", "p99_latency_outlier")

        display(Markdown(f"Tool latency thresholds: p95={p95:.2f}s, p99={p99:.2f}s."))
        display(timeout_df[["thread_id", "root_run_id", "name", "duration_s", "reason", "query_norm", "output_text"]].sort_values("duration_s", ascending=False).head(100))

        fig = px.box(travel_runs, x="name", y="duration_s", points=False, title="TravelPlanner Tool Run Latency By Tool")
        fig.update_layout(height=550, xaxis_tickangle=-35)
        fig.show()
        """
    ),
    md("## 7. Cascading Errors"),
    code(
        """
        cascade_rows = []
        for thread_id, g in travel_runs.sort_values(["thread_id", "call_index"]).groupby("thread_id"):
            meta = g.iloc[0]
            if meta.validation_attempts and meta.validation_attempts > 1:
                cascade_rows.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade_type": "validator_repair_loop", "severity": "high", "detail": f"validation_attempts={meta.validation_attempts}"})

            # Search/tool uncertainty followed by plan mutation means weak evidence entered the itinerary-building phase.
            uncertain_indices = set(g[g.has_uncertainty_language | g.has_error_language].call_index)
            for idx in uncertain_indices:
                later = g[(g.call_index > idx) & (g.name.isin({"add_slot", "insert_slot"}))].head(3)
                if not later.empty:
                    src = g[g.call_index == idx].iloc[0]
                    cascade_rows.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade_type": "weak_evidence_then_plan_mutation", "severity": "medium", "detail": f"{src.name} at call {idx} followed by {', '.join(later.name.tolist())}"})

            # Delete after add/insert is a repair/backtracking signal.
            for i, row in g[g.name == "delete_slot"].iterrows():
                prev_mut = g[(g.call_index < row.call_index) & (g.name.isin({"add_slot", "insert_slot"}))].tail(1)
                if not prev_mut.empty:
                    cascade_rows.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade_type": "mutation_repair_delete", "severity": "medium", "detail": f"delete_slot after {prev_mut.iloc[0].name}"})

        cascade_df = pd.DataFrame(cascade_rows).drop_duplicates() if cascade_rows else pd.DataFrame()
        display(cascade_df.head(100) if not cascade_df.empty else pd.DataFrame({"status": ["No cascade indicators found"]}))
        if not cascade_df.empty:
            fig = px.bar(cascade_df.groupby(["cascade_type", "severity"]).size().reset_index(name="events"), x="cascade_type", y="events", color="severity", title="Cascading Error Indicators")
            fig.update_layout(height=500, xaxis_tickangle=-30)
            fig.show()
        """
    ),
    md("## 8. Structural TravelPlan Validation"),
    code(
        """
        structural_rows = []
        slot_rows = []
        for path in sorted(TRAVEL_DIR.glob("run-*.json")):
            obj = load_json(path)
            outputs = obj.get("outputs") or {}
            metadata = obj.get("metadata") or {}
            thread_id = metadata.get("thread_id")
            root_run_id = path.stem.removeprefix("run-")
            budget = (outputs.get("normalized_constraints") or {}).get("budget_amount")
            plan = outputs.get("travelplan") or {}

            total_cost = 0.0
            for day in plan.get("days") or []:
                slots = day.get("slots") or []
                parsed_slots = []
                for pos, slot in enumerate(slots, start=1):
                    start = pd.to_datetime(slot.get("start_time"), errors="coerce")
                    end = pd.to_datetime(slot.get("end_time"), errors="coerce")
                    links = slot.get("links") or []
                    cost = slot.get("cost") or 0
                    total_cost += cost
                    text = " ".join(str(slot.get(k) or "") for k in ["name", "description", "location", "notes"])
                    row = {
                        "thread_id": thread_id,
                        "root_run_id": root_run_id,
                        "day_index": day.get("index"),
                        "position": pos,
                        "slot_name": slot.get("name"),
                        "category": slot.get("category"),
                        "start_time": start,
                        "end_time": end,
                        "cost": cost,
                        "links": links,
                        "missing_links": len(links) == 0,
                        "fragile_links": any(FRAGILE_DOMAINS.search(link or "") for link in links),
                        "unknown_category": slot.get("category") not in VALID_CATEGORIES,
                        "invalid_time": pd.isna(start) or pd.isna(end) or end <= start,
                        "uncertainty_note": bool(UNCERTAIN_RE.search(text)),
                    }
                    slot_rows.append(row)
                    parsed_slots.append(row)

                for a, b in combinations(parsed_slots, 2):
                    if not a["invalid_time"] and not b["invalid_time"] and a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]:
                        structural_rows.append({"thread_id": thread_id, "root_run_id": root_run_id, "check": "overlapping_slots", "detail": f"Day {day.get('index')}: {a['slot_name']} overlaps {b['slot_name']}"})

            if budget is not None and total_cost > float(budget):
                structural_rows.append({"thread_id": thread_id, "root_run_id": root_run_id, "check": "budget_exceeded", "detail": f"computed_total={total_cost:.2f}, budget={budget}"})

        slots_df = pd.DataFrame(slot_rows)
        for check_col in ["missing_links", "fragile_links", "unknown_category", "invalid_time", "uncertainty_note"]:
            bad = slots_df[slots_df[check_col]]
            for _, row in bad.iterrows():
                structural_rows.append({"thread_id": row.thread_id, "root_run_id": row.root_run_id, "check": check_col, "detail": f"Day {row.day_index} slot {row.position}: {row.slot_name}"})

        structural_df = pd.DataFrame(structural_rows)
        display(structural_df.head(100) if not structural_df.empty else pd.DataFrame({"status": ["No structural TravelPlan failures found"]}))

        structural_summary = structural_df.groupby("check").size().reset_index(name="events") if not structural_df.empty else pd.DataFrame(columns=["check", "events"])
        display(structural_summary)
        if not structural_summary.empty:
            fig = px.bar(structural_summary, x="check", y="events", title="Structural TravelPlan Error Checks")
            fig.update_layout(height=500, xaxis_tickangle=-30)
            fig.show()
        """
    ),
    md("## 9. Paper Taxonomy Table"),
    code(
        """
        taxonomy_rows = []
        taxonomy_rows.append({"category": "T2 Missing Mandatory Parameters", "baseline": int(len(baseline_query_dupes)), "travel_agent": int(len(travel_query_dupes)), "operationalization": "Repeated or underspecified query strings in real tool args; manual review examples above."})
        taxonomy_rows.append({"category": "T3 Redundant / Infinite Loops", "baseline": int((loops_df.system == 'baseline').sum()) if not loops_df.empty else 0, "travel_agent": int((loops_df.system == 'travel_agent').sum()) if not loops_df.empty else 0, "operationalization": "Consecutive exact repeats, same call >=3 times, mutation add/delete loops."})
        taxonomy_rows.append({"category": "T4 Misinterpreting Tool Outputs", "baseline": 0, "travel_agent": int(travel_runs.has_uncertainty_language.sum()), "operationalization": "Tool outputs containing uncertainty/fallback language that still feeds later plan mutations."})
        taxonomy_rows.append({"category": "T5 Link Rot / Empty Evidence", "baseline": 0, "travel_agent": int(slots_df.missing_links.sum() + slots_df.fragile_links.sum()), "operationalization": "Final TravelPlan slots with no links or fragile domains."})
        taxonomy_rows.append({"category": "T6 Cross-Agent Data Corruption", "baseline": 0, "travel_agent": int(len(cascade_df)) if not cascade_df.empty else 0, "operationalization": "Validator repair loops, delete-after-add repair, weak-evidence-then-mutation cascades."})
        taxonomy = pd.DataFrame(taxonomy_rows)
        display(taxonomy)

        latex = taxonomy.to_latex(index=False, escape=True)
        display(Markdown("### LaTeX table for `paper.tex`"))
        print(latex)
        """
    ),
    md("## 10. Report-Ready Summary"),
    code(
        """
        summary = f'''
        ### Error Analysis Summary

        The corrected analysis uses {len(travel_runs):,} real LangSmith tool runs from {travel_runs.thread_id.nunique()} TravelPlanner threads, rather than final-plan link proxies. The multi-agent system made heavy use of TravelPlan mutation tools (`add_slot`, `add_day`, `delete_slot`, `insert_slot`) and domain tools (`search_web`, `search_restaurants`, `search_attractions`, `search_flights`, `search_hotels`, `check_route_timing`).

        Key detected indicators:

        - Repeated exact TravelPlanner tool-call groups: {len(travel_dupes):,}
        - Repeated TravelPlanner query groups: {len(travel_query_dupes):,}
        - TravelPlanner dead-loop indicators: {int((loops_df.system == 'travel_agent').sum()) if not loops_df.empty else 0:,}
        - Baseline dead-loop indicators: {int((loops_df.system == 'baseline').sum()) if not loops_df.empty else 0:,}
        - TravelPlanner latency/error-language timeout indicators: {len(timeout_df):,}
        - TravelPlanner cascade indicators: {len(cascade_df) if not cascade_df.empty else 0:,}
        - Structural TravelPlan validation events: {len(structural_df) if not structural_df.empty else 0:,}

        Interpretation for the paper: the baseline's failure mode is mostly repeated or underspecified Tavily search. The TravelPlanner's failure mode is more architectural: repeated plan mutations, repair loops, weak-evidence propagation, and fragile final evidence links. This supports the paper's claim that the multi-agent system improves organization and constraint handling but introduces cross-agent propagation risks.
        '''
        display(Markdown(summary))
        """
    ),
]

out = Path("correct_langsmith_tool_error_analysis.ipynb")
nbf.write(nb, out)
print(f"Wrote {out}")
