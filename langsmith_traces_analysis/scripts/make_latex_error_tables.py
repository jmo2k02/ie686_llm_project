from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_error_analysis_tables.tex"
THREAD_DIR = ROOT / "thread_analysis" / "travel_agent"
BASELINE_DIR = ROOT / "baseline"
TRAVEL_DIR = ROOT / "travel_agent"

SEARCH_TOOLS = {"search_web", "search_restaurants", "search_attractions", "search_flights", "search_hotels", "check_route_timing", "build_place_distance_graph"}
MUTATION_TOOLS = {"init_plan", "add_day", "remove_day", "add_slot", "insert_slot", "delete_slot", "view_plan", "cost_summary"}
STATE_TOOLS = {"write_todos"}
VALID_CATEGORIES = {"meal", "attraction", "transport", "lodging", "leisure", "other"}
FRAGILE_DOMAINS = re.compile(r"google\.com|maps\.google|instagram\.com|facebook\.com|tripadvisor\.com|booking\.com|expedia\.com|agoda\.com|turbopass\.com", re.I)
ERROR_RE = re.compile(r"timeout|timed out|error|failed|failure|exception|traceback|rate limit|unavailable|could not|unable|no reliable|missing info|missing_info", re.I)
UNCERTAIN_RE = re.compile(r"verify|re-verify|estimated|estimate|uncertain|unavailable|not available|backup|fallback|could not|depends|must be checked|not confirmed", re.I)


def safe_loads(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {"_raw": str(value)}


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


def tex_table(df: pd.DataFrame, caption: str, label: str, float_spec: str = "htbp") -> str:
    return df.to_latex(index=False, escape=True, caption=caption, label=label, position=float_spec)


manifest = pd.read_csv(THREAD_DIR / "manifest.csv")
runs = pd.read_csv(THREAD_DIR / "langsmith_runs.csv")
runs["duration_s"] = [duration_seconds(s, e) for s, e in zip(runs.start_time, runs.end_time)]
runs["tool_args"] = runs.inputs_json.map(safe_loads)
runs["args_norm"] = runs.tool_args.map(canonical)
runs["query_norm"] = runs.tool_args.map(arg_text)
runs["output_text"] = runs.outputs_json.fillna("").astype(str)
runs["has_error_language"] = runs.output_text.str.contains(ERROR_RE, na=False)
runs["has_uncertainty_language"] = runs.output_text.str.contains(UNCERTAIN_RE, na=False)
runs = runs.merge(manifest[["thread_id", "root_run_id", "query", "validation_attempts", "validation_passed"]], on="thread_id", how="left")
runs = runs.sort_values(["thread_id", "start_time", "id"]).reset_index(drop=True)
runs["call_index"] = runs.groupby("thread_id").cumcount()

def tool_group(name: str) -> str:
    if name in SEARCH_TOOLS:
        return "Domain/search tools"
    if name in MUTATION_TOOLS:
        return "TravelPlan mutation tools"
    if name in STATE_TOOLS:
        return "State-management tools"
    return "Other"

runs["agent_group"] = runs.name.map(tool_group)

dupes = (
    runs.groupby(["thread_id", "name", "args_norm"])
    .agg(count=("id", "size"))
    .reset_index()
    .query("count > 1")
)
dupe_counts = dupes.groupby("name").agg(repeated_groups=("args_norm", "size"), repeated_calls=("count", "sum")).reset_index()

query_dupes = (
    runs[runs.name.isin(SEARCH_TOOLS)]
    .groupby(["thread_id", "name", "query_norm"])
    .agg(count=("id", "size"))
    .reset_index()
    .query("query_norm != '' and count > 1")
)
query_dupe_counts = query_dupes.groupby("name").agg(repeated_query_groups=("query_norm", "size"), repeated_query_calls=("count", "sum")).reset_index()

loop_rows = []
for thread_id, g in runs.groupby("thread_id"):
    rows = g.sort_values("call_index").to_dict("records")
    streak = 1
    prev_key = None
    for row in rows:
        key = (row["name"], row.get("args_norm") or row.get("query_norm"))
        if key == prev_key:
            streak += 1
        else:
            streak = 1
        if streak >= 2:
            loop_rows.append({"thread_id": thread_id, "name": row["name"], "loop_type": "consecutive repeat"})
        prev_key = key
    for i in range(len(rows) - 3):
        names = [r["name"] for r in rows[i:i+4]]
        if names in (["add_slot", "delete_slot", "add_slot", "delete_slot"], ["insert_slot", "delete_slot", "insert_slot", "delete_slot"], ["delete_slot", "add_slot", "delete_slot", "add_slot"], ["delete_slot", "insert_slot", "delete_slot", "insert_slot"]):
            loop_rows.append({"thread_id": thread_id, "name": "mutation repair", "loop_type": "add/delete backtrack"})
    counts = Counter((r["name"], r.get("query_norm") or r.get("args_norm")) for r in rows)
    for (name, q), count in counts.items():
        if q and count >= 3:
            loop_rows.append({"thread_id": thread_id, "name": name, "loop_type": "same call >=3"})
loops = pd.DataFrame(loop_rows)
loop_counts = loops.groupby("name").size().reset_index(name="dead_loop_indicators") if not loops.empty else pd.DataFrame(columns=["name", "dead_loop_indicators"])

p99 = runs.duration_s.quantile(0.99)
latency = runs[(runs.duration_s >= p99) | runs.has_error_language]
latency_counts = latency.groupby("name").size().reset_index(name="timeout_or_error_indicators")

tool_summary = (
    runs.groupby(["agent_group", "name"])
    .agg(calls=("id", "size"), threads=("thread_id", "nunique"), mean_latency_s=("duration_s", "mean"), p95_latency_s=("duration_s", lambda s: s.quantile(0.95)), uncertainty_outputs=("has_uncertainty_language", "sum"))
    .reset_index()
    .merge(dupe_counts, on="name", how="left")
    .merge(query_dupe_counts, on="name", how="left")
    .merge(loop_counts, on="name", how="left")
    .merge(latency_counts, on="name", how="left")
    .fillna(0)
)
for col in ["repeated_groups", "repeated_calls", "repeated_query_groups", "repeated_query_calls", "dead_loop_indicators", "timeout_or_error_indicators", "uncertainty_outputs"]:
    tool_summary[col] = tool_summary[col].astype(int)
tool_summary["mean_latency_s"] = tool_summary["mean_latency_s"].round(2)
tool_summary["p95_latency_s"] = tool_summary["p95_latency_s"].round(2)

paper_tool_summary = tool_summary[["agent_group", "name", "calls", "threads", "repeated_calls", "repeated_query_calls", "dead_loop_indicators", "timeout_or_error_indicators", "uncertainty_outputs"]]
paper_tool_summary = paper_tool_summary.sort_values(["dead_loop_indicators", "repeated_calls", "calls"], ascending=False).head(16)
paper_tool_summary = paper_tool_summary.rename(columns={
    "agent_group": "Agent/tool group",
    "name": "Tool",
    "calls": "Calls",
    "threads": "Threads",
    "repeated_calls": "Repeated calls",
    "repeated_query_calls": "Repeated queries",
    "dead_loop_indicators": "Loop indicators",
    "timeout_or_error_indicators": "Timeout/error indicators",
    "uncertainty_outputs": "Uncertainty outputs",
})

group_summary = (
    tool_summary.groupby("agent_group")
    .agg(calls=("calls", "sum"), repeated_calls=("repeated_calls", "sum"), repeated_queries=("repeated_query_calls", "sum"), loop_indicators=("dead_loop_indicators", "sum"), timeout_error_indicators=("timeout_or_error_indicators", "sum"), uncertainty_outputs=("uncertainty_outputs", "sum"))
    .reset_index()
    .rename(columns={"agent_group": "Agent/tool group", "calls": "Calls", "repeated_calls": "Repeated calls", "repeated_queries": "Repeated queries", "loop_indicators": "Loop indicators", "timeout_error_indicators": "Timeout/error indicators", "uncertainty_outputs": "Uncertainty outputs"})
)

slot_events = []
cascade_events = []
for path in sorted(TRAVEL_DIR.glob("run-*.json")):
    obj = json.loads(path.read_text(encoding="utf-8"))
    outputs = obj.get("outputs") or {}
    thread_id = (obj.get("metadata") or {}).get("thread_id")
    root_run_id = path.stem.removeprefix("run-")
    budget = (outputs.get("normalized_constraints") or {}).get("budget_amount")
    plan = outputs.get("travelplan") or {}
    total_cost = 0.0
    for day in plan.get("days") or []:
        parsed = []
        for pos, slot in enumerate(day.get("slots") or [], start=1):
            start = pd.to_datetime(slot.get("start_time"), errors="coerce")
            end = pd.to_datetime(slot.get("end_time"), errors="coerce")
            links = slot.get("links") or []
            cost = slot.get("cost") or 0
            total_cost += cost
            text = " ".join(str(slot.get(k) or "") for k in ["name", "description", "location", "notes"])
            event_base = {"thread_id": thread_id, "root_run_id": root_run_id, "category": slot.get("category") or "unknown"}
            if len(links) == 0:
                slot_events.append({**event_base, "error": "missing evidence link"})
            if any(FRAGILE_DOMAINS.search(link or "") for link in links):
                slot_events.append({**event_base, "error": "fragile evidence link"})
            if slot.get("category") not in VALID_CATEGORIES:
                slot_events.append({**event_base, "error": "unknown category"})
            if pd.isna(start) or pd.isna(end) or end <= start:
                slot_events.append({**event_base, "error": "invalid time interval"})
            if UNCERTAIN_RE.search(text):
                slot_events.append({**event_base, "error": "uncertainty in final slot"})
            parsed.append((slot.get("name"), start, end, event_base))
        for a, b in combinations(parsed, 2):
            if pd.notna(a[1]) and pd.notna(a[2]) and pd.notna(b[1]) and pd.notna(b[2]) and a[1] < b[2] and b[1] < a[2]:
                slot_events.append({**a[3], "error": "overlapping slots"})
    if budget is not None and total_cost > float(budget):
        slot_events.append({"thread_id": thread_id, "root_run_id": root_run_id, "category": "plan", "error": "budget exceeded"})

for thread_id, g in runs.groupby("thread_id"):
    meta = g.iloc[0]
    if meta.validation_attempts > 1:
        cascade_events.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade": "validator repair loop"})
    weak = set(g[g.has_uncertainty_language | g.has_error_language].call_index)
    for idx in weak:
        if not g[(g.call_index > idx) & (g.name.isin({"add_slot", "insert_slot"}))].empty:
            cascade_events.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade": "weak evidence then plan mutation"})
    for _, row in g[g.name == "delete_slot"].iterrows():
        if not g[(g.call_index < row.call_index) & (g.name.isin({"add_slot", "insert_slot"}))].empty:
            cascade_events.append({"thread_id": thread_id, "root_run_id": meta.root_run_id, "cascade": "delete after prior insertion"})

slot_df = pd.DataFrame(slot_events)
structural_summary = slot_df.groupby(["category", "error"]).size().reset_index(name="Events").sort_values("Events", ascending=False).head(14)
structural_summary = structural_summary.rename(columns={"category": "Slot category", "error": "Structural error"})

cascade_df = pd.DataFrame(cascade_events)
cascade_summary = cascade_df.groupby("cascade").agg(Events=("thread_id", "size"), Threads=("thread_id", "nunique")).reset_index().rename(columns={"cascade": "Cascade indicator"}) if not cascade_df.empty else pd.DataFrame(columns=["Cascade indicator", "Events", "Threads"])

latex = []
latex.append("% Auto-generated by scripts/make_latex_error_tables.py\n")
latex.append(tex_table(group_summary, "TravelPlanner tool-use error indicators grouped by agent/tool category.", "tab:tp-error-by-agent-group"))
latex.append("\n")
latex.append(tex_table(paper_tool_summary, "Most error-prone TravelPlanner tools by repeated calls, repeated queries, loop indicators, timeout/error indicators, and uncertainty-bearing outputs.", "tab:tp-error-by-tool"))
latex.append("\n")
latex.append(tex_table(structural_summary, "Structural TravelPlan errors in final itinerary slots, grouped by slot category.", "tab:tp-structural-errors"))
latex.append("\n")
latex.append(tex_table(cascade_summary, "Cascading-error indicators in the TravelPlanner execution traces.", "tab:tp-cascade-indicators"))

paragraph = f"""
% Suggested report text
The corrected LangSmith export contains {len(runs):,} real TravelPlanner tool runs across {runs.thread_id.nunique()} LangGraph threads. Grouping tools by architectural role shows that most error indicators concentrate in the TravelPlan mutation layer rather than in a single domain-search agent. This is expected for the multi-agent design: subagents retrieve evidence, but the execution agent repeatedly mutates the shared TravelPlan object while repairing budget, timing, or validation issues. We therefore interpret repeated add/delete or insert/delete patterns as repair-loop indicators rather than simple duplicate search calls. Domain/search tools still contribute repeated-query and uncertainty signals, especially when retrieved evidence is weak or later embedded into final slots. Structural validation of the final TravelPlan adds a second view of error propagation: missing or fragile links and uncertainty-bearing slot notes reveal cases where retrieval uncertainty survived into the final itinerary artifact.
"""
latex.append("\n" + paragraph)

OUT.write_text("\n".join(latex), encoding="utf-8")
print(f"Wrote {OUT}")
print(group_summary.to_string(index=False))
print(paper_tool_summary.to_string(index=False))
