from pathlib import Path
import nbformat as nbf


def clean(source: str) -> str:
    lines = source.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    margin = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    prefix = " " * margin
    return "\n".join(line[margin:] if line.startswith(prefix) else line for line in lines)


def md(source: str):
    return nbf.v4.new_markdown_cell(clean(source))


def code(source: str):
    return nbf.v4.new_code_cell(clean(source))


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
        # LangSmith Trace Error Analysis: Baseline vs. Multi-Agent TravelPlanner

        This notebook analyzes the local LangSmith JSON exports in `baseline/` and `travel_agent/` for the TravelPlanner project described in `paper.tex`.

        The analysis focuses on:

        - repeated tool calls and repeated queries
        - dead-loop behavior
        - timeouts, incomplete traces, and explicit runtime errors
        - cascading errors in the multi-agent pipeline
        - semantic failures such as missing mandatory parameters, weak evidence, contradictions, and unverifiable claims

        Important data caveat: these local files are not full nested LangSmith run trees. The baseline exports include `outputs.messages` with explicit `tool_calls`. The travel-agent exports include structured final artifacts, task lists, travel plans, validation fields, and per-agent message histories, but no explicit `tool_calls`. For the travel agent, this notebook therefore analyzes tool-use proxies: approved task types, generated evidence links, validation feedback, and failure language in plan slots.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import math
        import re
        from collections import Counter, defaultdict
        from dataclasses import dataclass
        from datetime import datetime
        from pathlib import Path
        from urllib.parse import parse_qs, unquote_plus, urlparse

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import seaborn as sns
        from IPython.display import Markdown, display
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        sns.set_theme(style="whitegrid", context="talk")
        pd.set_option("display.max_colwidth", 180)
        pd.set_option("display.max_columns", 80)

        ROOT = Path.cwd()
        BASELINE_DIR = ROOT / "baseline"
        TRAVEL_DIR = ROOT / "travel_agent"

        ERROR_KEYWORDS = re.compile(
            r"timeout|timed out|exception|traceback|failed|failure|error|rate limit|api failed|tool failed|could not|unable to|no reliable|not available|empty|missing_info|missing info",
            re.I,
        )
        UNCERTAINTY_KEYWORDS = re.compile(
            r"verify|re-verify|must be checked|should be checked|uncertain|assumption|estimate|estimated|not confirmed|unavailable|not available|backup|fallback|if not available|exact .* unavailable",
            re.I,
        )
        MISSING_PARAM_HINTS = re.compile(
            r"restaurant|hotel|flight|opening|hours|ticket|transport|route|train|ferry|car rental|villa|museum|attraction",
            re.I,
        )
        DATE_HINT = re.compile(r"\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|mon|tue|wed|thu|fri|sat|sun)\b", re.I)
        LOCATION_HINT = re.compile(r"\b(from|to|in|near|at)\b", re.I)

        def read_json(path: Path):
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def canonical_text(value) -> str:
            if value is None:
                return ""
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            value = value.lower()
            value = unquote_plus(value)
            value = re.sub(r"https?://\S+", " ", value)
            value = re.sub(r"[^\w\s€$£.-]", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
            return value

        def short_run_id(path: Path) -> str:
            return path.stem.replace("run-", "")[:8]

        def flatten_messages(obj: dict, system: str, run_id: str, file: str) -> list[dict]:
            rows = []
            if system == "baseline":
                messages = (obj.get("outputs") or {}).get("messages") or []
                for i, msg in enumerate(messages):
                    rows.append({
                        "system": system,
                        "run_id": run_id,
                        "file": file,
                        "agent": "baseline_agent",
                        "message_index": i,
                        "role": msg.get("type") or msg.get("role"),
                        "content": msg.get("content") or "",
                        "has_tool_calls": bool(msg.get("tool_calls")),
                        "raw": msg,
                    })
            else:
                histories = (obj.get("outputs") or {}).get("message_histories") or {}
                for agent, history in histories.items():
                    for i, msg in enumerate(history.get("messages") or []):
                        rows.append({
                            "system": system,
                            "run_id": run_id,
                            "file": file,
                            "agent": agent,
                            "message_index": i,
                            "role": msg.get("type") or msg.get("role"),
                            "content": msg.get("content") or "",
                            "has_tool_calls": bool(msg.get("tool_calls")),
                            "raw": msg,
                        })
            return rows

        def extract_user_query(obj: dict, system: str) -> str:
            if system == "travel_agent":
                outputs = obj.get("outputs") or {}
                return outputs.get("query") or ((obj.get("inputs") or {}).get("input") or {}).get("resume") or ""

            for msg in (obj.get("outputs") or {}).get("messages") or []:
                content = msg.get("content") or ""
                match = re.search(r"USER QUERY\\s*(.*?)\\n\\nCONSTRAINTS", content, re.S)
                if match:
                    return match.group(1).strip()
            return ""

        def extract_tool_calls(obj: dict, system: str, run_id: str, file: str) -> tuple[list[dict], list[dict]]:
            calls = []
            outputs = []
            if system != "baseline":
                return calls, outputs

            for i, msg in enumerate((obj.get("outputs") or {}).get("messages") or []):
                for j, call in enumerate(msg.get("tool_calls") or []):
                    args = call.get("args") or {}
                    query = args.get("query") or args.get("url") or json.dumps(args, ensure_ascii=False, sort_keys=True)
                    calls.append({
                        "system": system,
                        "run_id": run_id,
                        "file": file,
                        "message_index": i,
                        "call_index": j,
                        "tool_call_id": call.get("id"),
                        "tool_name": call.get("name"),
                        "query": query,
                        "query_norm": canonical_text(query),
                        "args": args,
                    })
                if msg.get("type") == "tool" or msg.get("role") == "tool":
                    outputs.append({
                        "system": system,
                        "run_id": run_id,
                        "file": file,
                        "message_index": i,
                        "tool_call_id": msg.get("tool_call_id"),
                        "tool_name": msg.get("name"),
                        "status": msg.get("status"),
                        "content": msg.get("content") or "",
                    })
            return calls, outputs

        def extract_travel_tasks(obj: dict, run_id: str, file: str) -> list[dict]:
            rows = []
            outputs = obj.get("outputs") or {}
            for i, task in enumerate(outputs.get("task_list") or []):
                rows.append({
                    "system": "travel_agent",
                    "run_id": run_id,
                    "file": file,
                    "task_index": i,
                    "task_name": task.get("name"),
                    "task_type": task.get("type"),
                    "task_text": task.get("text") or "",
                    "is_valid": task.get("is_valid"),
                    "validation_comment": task.get("validation_comment"),
                    "query_norm": canonical_text(task.get("text") or task.get("name")),
                })
            return rows

        def normalize_url(url: str) -> str:
            if not isinstance(url, str) or not url.strip():
                return ""
            parsed = urlparse(url.strip())
            host = parsed.netloc.lower().replace("www.", "")
            path = re.sub(r"/$", "", parsed.path.lower())
            return f"{host}{path}"

        def extract_slots(obj: dict, system: str, run_id: str, file: str) -> list[dict]:
            rows = []
            outputs = obj.get("outputs") or {}
            if system == "travel_agent":
                plan = outputs.get("travelplan") or {}
                for day in plan.get("days") or []:
                    for slot_i, slot in enumerate(day.get("slots") or []):
                        links = slot.get("links") or []
                        text = " ".join(str(slot.get(k, "")) for k in ["name", "description", "location", "notes"])
                        rows.append({
                            "system": system,
                            "run_id": run_id,
                            "file": file,
                            "day_index": day.get("index"),
                            "date": day.get("calendar_date"),
                            "slot_index": slot_i,
                            "name": slot.get("name"),
                            "category": slot.get("category"),
                            "location": slot.get("location"),
                            "cost": slot.get("cost"),
                            "links": links,
                            "link_count": len(links),
                            "domains": [urlparse(link).netloc.lower().replace("www.", "") for link in links if isinstance(link, str)],
                            "text": text,
                            "uncertainty_hits": len(UNCERTAINTY_KEYWORDS.findall(text)),
                            "error_hits": len(ERROR_KEYWORDS.findall(text)),
                        })
            else:
                final = "\\n".join(
                    msg.get("content") or ""
                    for msg in (outputs.get("messages") or [])
                    if (msg.get("type") == "ai" or msg.get("role") == "assistant") and (msg.get("content") or "").strip().startswith("#")
                )
                for i, line in enumerate(final.splitlines()):
                    if "|" not in line or line.strip().startswith("|---") or "Time | Type" in line:
                        continue
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    if len(cells) >= 6:
                        links = re.findall(r"https?://[^)\s]+", line)
                        rows.append({
                            "system": system,
                            "run_id": run_id,
                            "file": file,
                            "day_index": np.nan,
                            "date": None,
                            "slot_index": i,
                            "name": cells[2],
                            "category": cells[1],
                            "location": cells[3],
                            "cost": cells[4],
                            "links": links,
                            "link_count": len(links),
                            "domains": [urlparse(link).netloc.lower().replace("www.", "") for link in links],
                            "text": line,
                            "uncertainty_hits": len(UNCERTAINTY_KEYWORDS.findall(line)),
                            "error_hits": len(ERROR_KEYWORDS.findall(line)),
                        })
            return rows

        def missing_mandatory_parameter(query: str) -> bool:
            q = canonical_text(query)
            if not q or not MISSING_PARAM_HINTS.search(q):
                return False
            needs_date = bool(re.search(r"restaurant|opening|hours|flight|hotel|ticket|museum|attraction", q, re.I))
            needs_location = bool(re.search(r"restaurant|hotel|flight|route|transport|museum|attraction|villa", q, re.I))
            return (needs_date and not DATE_HINT.search(q)) or (needs_location and not LOCATION_HINT.search(q))

        def final_answer_text(obj: dict, system: str) -> str:
            outputs = obj.get("outputs") or {}
            if system == "travel_agent":
                return json.dumps(outputs.get("travelplan") or outputs, ensure_ascii=False)
            messages = outputs.get("messages") or []
            for msg in reversed(messages):
                content = msg.get("content") or ""
                if content.strip().startswith("#"):
                    return content
            return ""
        """
    ),
    md("## Load And Normalize Local Trace Exports"),
    code(
        """
        all_runs = []
        all_messages = []
        all_tool_calls = []
        all_tool_outputs = []
        all_travel_tasks = []
        all_slots = []

        for system, folder in [("baseline", BASELINE_DIR), ("travel_agent", TRAVEL_DIR)]:
            for path in sorted(folder.glob("*.json")):
                obj = read_json(path)
                run_id = short_run_id(path)
                outputs = obj.get("outputs")
                inputs = obj.get("inputs")
                query = extract_user_query(obj, system)
                final_text = final_answer_text(obj, system)
                messages = flatten_messages(obj, system, run_id, path.name)
                tool_calls, tool_outputs = extract_tool_calls(obj, system, run_id, path.name)
                tasks = extract_travel_tasks(obj, run_id, path.name) if system == "travel_agent" else []
                slots = extract_slots(obj, system, run_id, path.name)

                all_runs.append({
                    "system": system,
                    "run_id": run_id,
                    "file": path.name,
                    "has_inputs": inputs is not None,
                    "has_outputs": outputs is not None,
                    "root_error": obj.get("error"),
                    "query": query,
                    "query_norm": canonical_text(query),
                    "message_count": len(messages),
                    "explicit_tool_call_count": len(tool_calls),
                    "tool_output_count": len(tool_outputs),
                    "task_count": len(tasks),
                    "slot_count": len(slots),
                    "validation_passed": (outputs or {}).get("validation_passed") if system == "travel_agent" else None,
                    "validation_attempts": (outputs or {}).get("validation_attempts") if system == "travel_agent" else None,
                    "validation_feedback": (outputs or {}).get("validation_feedback") if system == "travel_agent" else None,
                    "final_text_len": len(final_text),
                    "uncertainty_hits": len(UNCERTAINTY_KEYWORDS.findall(final_text)),
                    "error_keyword_hits": len(ERROR_KEYWORDS.findall(final_text)),
                })
                all_messages.extend(messages)
                all_tool_calls.extend(tool_calls)
                all_tool_outputs.extend(tool_outputs)
                all_travel_tasks.extend(tasks)
                all_slots.extend(slots)

        runs_df = pd.DataFrame(all_runs)
        messages_df = pd.DataFrame(all_messages)
        tool_calls_df = pd.DataFrame(all_tool_calls)
        tool_outputs_df = pd.DataFrame(all_tool_outputs)
        travel_tasks_df = pd.DataFrame(all_travel_tasks)
        slots_df = pd.DataFrame(all_slots)

        display(Markdown(f"Loaded **{len(runs_df)}** exports: **{(runs_df.system == 'baseline').sum()} baseline** and **{(runs_df.system == 'travel_agent').sum()} travel-agent**."))
        display(runs_df.sort_values(["system", "file"]).reset_index(drop=True))
        """
    ),
    md("## Dataset Health And Export Completeness"),
    code(
        """
        health_cols = [
            "system", "run_id", "has_inputs", "has_outputs", "message_count", "explicit_tool_call_count",
            "task_count", "slot_count", "validation_passed", "validation_attempts", "final_text_len", "root_error"
        ]
        display(runs_df[health_cols].sort_values(["system", "run_id"]))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.countplot(data=runs_df, x="system", hue="has_outputs", ax=axes[0])
        axes[0].set_title("Exports With Outputs")
        sns.boxplot(data=runs_df, x="system", y="message_count", ax=axes[1])
        axes[1].set_title("Message Count By System")
        sns.boxplot(data=runs_df, x="system", y="slot_count", ax=axes[2])
        axes[2].set_title("Extracted Plan Slots By System")
        plt.tight_layout()
        plt.show()

        incomplete = runs_df.loc[~runs_df.has_outputs | (runs_df.final_text_len == 0), ["system", "run_id", "file", "has_outputs", "final_text_len"]]
        display(Markdown("### Incomplete Or Empty Exports"))
        display(incomplete if not incomplete.empty else pd.DataFrame({"status": ["No incomplete exports detected"]}))
        """
    ),
    md("## Repeated Tool Calls And Query Redundancy"),
    code(
        """
        display(Markdown("### Explicit Baseline Tool Calls"))
        display(tool_calls_df[["run_id", "message_index", "call_index", "tool_name", "query"]].head(30))

        if not tool_calls_df.empty:
            per_run_tool = tool_calls_df.groupby(["system", "run_id", "tool_name"]).size().reset_index(name="calls")
            display(per_run_tool.sort_values("calls", ascending=False))

            fig = px.bar(per_run_tool, x="run_id", y="calls", color="tool_name", facet_col="system", title="Explicit Tool Calls Per Trace")
            fig.update_layout(height=450)
            fig.show()

            duplicate_queries = (
                tool_calls_df.groupby(["system", "run_id", "tool_name", "query_norm"])
                .agg(count=("query", "size"), example_query=("query", "first"))
                .reset_index()
                .query("count > 1")
                .sort_values("count", ascending=False)
            )
            display(Markdown("### Exact Repeated Queries Within The Same Trace"))
            display(duplicate_queries)

            cross_run_repeats = (
                tool_calls_df.groupby(["tool_name", "query_norm"])
                .agg(count=("query", "size"), run_count=("run_id", "nunique"), example_query=("query", "first"))
                .reset_index()
                .query("count > 1")
                .sort_values(["count", "run_count"], ascending=False)
            )
            display(Markdown("### Repeated Queries Across All Baseline Traces"))
            display(cross_run_repeats.head(30))
        """
    ),
    md("## Travel-Agent Tool-Use Proxies: Task Types And Evidence Links"),
    code(
        """
        display(Markdown("Because travel-agent exports do not contain explicit `tool_calls`, task types and evidence links are used as comparable proxies for subagent/tool activity."))
        display(travel_tasks_df.head(30))

        task_counts = travel_tasks_df.groupby(["run_id", "task_type"]).size().reset_index(name="tasks")
        fig = px.bar(task_counts, x="run_id", y="tasks", color="task_type", title="Travel-Agent Planned/Approved Task Types Per Trace")
        fig.update_layout(height=500)
        fig.show()

        domain_rows = []
        for _, row in slots_df.loc[slots_df.system == "travel_agent"].iterrows():
            for domain in row["domains"]:
                domain_rows.append({"run_id": row.run_id, "category": row.category, "domain": domain})
        domains_df = pd.DataFrame(domain_rows)
        display(Markdown("### Most Common Evidence Domains In Travel-Agent Plans"))
        display(domains_df.value_counts("domain").rename("count").reset_index().head(25) if not domains_df.empty else domains_df)

        if not domains_df.empty:
            top_domains = domains_df.value_counts("domain").head(15).reset_index(name="count")
            fig = px.bar(top_domains, x="count", y="domain", orientation="h", title="Top Evidence Domains Used By Travel Agent")
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            fig.show()
        """
    ),
    md("## Dead Loops And Repeated-Query Risk"),
    code(
        """
        loop_events = []
        if not tool_calls_df.empty:
            ordered = tool_calls_df.sort_values(["run_id", "message_index", "call_index"])
            for run_id, group in ordered.groupby("run_id"):
                prev_query = None
                streak = 0
                for _, row in group.iterrows():
                    if row.query_norm == prev_query:
                        streak += 1
                    else:
                        streak = 1
                    prev_query = row.query_norm
                    total_same = (group.query_norm == row.query_norm).sum()
                    if streak >= 2 or total_same >= 3:
                        loop_events.append({
                            "system": row.system,
                            "run_id": run_id,
                            "tool_name": row.tool_name,
                            "query": row.query,
                            "event_type": "consecutive_repeat" if streak >= 2 else "same_query_3plus",
                            "streak": streak,
                            "total_same_query_in_run": int(total_same),
                        })

        # Travel-agent proxy: repeated evidence links and repeated slot names can indicate local reuse or propagation.
        for run_id, group in slots_df.loc[slots_df.system == "travel_agent"].groupby("run_id"):
            link_counter = Counter(normalize_url(link) for links in group.links for link in links)
            for link, count in link_counter.items():
                if link and count >= 3:
                    loop_events.append({
                        "system": "travel_agent",
                        "run_id": run_id,
                        "tool_name": "evidence_link_proxy",
                        "query": link,
                        "event_type": "same_link_3plus",
                        "streak": np.nan,
                        "total_same_query_in_run": count,
                    })

        loop_df = pd.DataFrame(loop_events)
        display(loop_df.sort_values(["system", "total_same_query_in_run"], ascending=[True, False]) if not loop_df.empty else pd.DataFrame({"status": ["No exact dead-loop indicators detected"]}))

        if not loop_df.empty:
            fig = px.histogram(loop_df, x="system", color="event_type", title="Dead-Loop / Reuse Indicators")
            fig.show()
        """
    ),
    md("## Near-Duplicate Query Analysis"),
    code(
        """
        query_sources = []
        for _, row in tool_calls_df.iterrows():
            query_sources.append({"system": "baseline", "run_id": row.run_id, "source": "explicit_tool_call", "text": row.query, "text_norm": row.query_norm})
        for _, row in travel_tasks_df.iterrows():
            query_sources.append({"system": "travel_agent", "run_id": row.run_id, "source": "task_proxy", "text": row.task_text, "text_norm": row.query_norm})

        query_df = pd.DataFrame(query_sources)
        near_pairs = []
        if len(query_df) >= 2:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            X = vectorizer.fit_transform(query_df.text_norm.fillna(""))
            sims = cosine_similarity(X)
            for i in range(len(query_df)):
                for j in range(i + 1, len(query_df)):
                    if sims[i, j] >= 0.82 and query_df.iloc[i].text_norm != query_df.iloc[j].text_norm:
                        near_pairs.append({
                            "similarity": round(float(sims[i, j]), 3),
                            "system_a": query_df.iloc[i].system,
                            "run_a": query_df.iloc[i].run_id,
                            "source_a": query_df.iloc[i].source,
                            "text_a": query_df.iloc[i].text,
                            "system_b": query_df.iloc[j].system,
                            "run_b": query_df.iloc[j].run_id,
                            "source_b": query_df.iloc[j].source,
                            "text_b": query_df.iloc[j].text,
                        })

        near_df = pd.DataFrame(near_pairs).sort_values("similarity", ascending=False) if near_pairs else pd.DataFrame()
        display(Markdown("Near duplicates use TF-IDF cosine similarity >= 0.82 over explicit baseline queries and travel-agent task descriptions."))
        display(near_df.head(50) if not near_df.empty else pd.DataFrame({"status": ["No near-duplicate query pairs above threshold"]}))
        """
    ),
    md("## Timeouts, Runtime Errors, And Incomplete Runs"),
    code(
        """
        runtime_events = []
        for _, row in runs_df.iterrows():
            if not row.has_outputs:
                runtime_events.append({"system": row.system, "run_id": row.run_id, "severity": "high", "event_type": "missing_outputs", "detail": row.file})
            if row.root_error:
                runtime_events.append({"system": row.system, "run_id": row.run_id, "severity": "high", "event_type": "root_error", "detail": str(row.root_error)})

        for _, row in messages_df.iterrows():
            content = row.content or ""
            matches = ERROR_KEYWORDS.findall(content)
            if matches:
                runtime_events.append({
                    "system": row.system,
                    "run_id": row.run_id,
                    "severity": "medium",
                    "event_type": "error_keyword_in_message",
                    "agent": row.agent,
                    "detail": content[:500],
                    "match_count": len(matches),
                })

        for _, row in tool_outputs_df.iterrows():
            if row.status and str(row.status).lower() not in {"success", "ok"}:
                runtime_events.append({
                    "system": row.system,
                    "run_id": row.run_id,
                    "severity": "high",
                    "event_type": "tool_output_non_success",
                    "agent": row.tool_name,
                    "detail": row.content[:500],
                })

        runtime_df = pd.DataFrame(runtime_events)
        display(runtime_df.sort_values(["severity", "system", "run_id"]) if not runtime_df.empty else pd.DataFrame({"status": ["No runtime error keywords or non-success tool outputs detected"]}))

        if not runtime_df.empty:
            summary = runtime_df.groupby(["system", "event_type"]).size().reset_index(name="events")
            fig = px.bar(summary, x="event_type", y="events", color="system", barmode="group", title="Runtime Error And Timeout Indicators")
            fig.update_layout(height=500, xaxis_tickangle=-35)
            fig.show()
        """
    ),
    md("## Cascading Errors"),
    code(
        """
        cascade_events = []

        # Invalid or questionable planner tasks are upstream signals in the travel-agent workflow.
        for _, row in travel_tasks_df.iterrows():
            if row.is_valid is False or pd.notna(row.validation_comment):
                cascade_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "stage": "planner_task",
                    "signal": "invalid_or_warned_task",
                    "detail": f"{row.task_name}: {row.validation_comment or row.task_text}",
                    "severity": "medium",
                })

        # Validation failure means errors survived synthesis and reached the reviewer.
        for _, row in runs_df.loc[runs_df.system == "travel_agent"].iterrows():
            if row.validation_passed is False:
                cascade_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "stage": "validator",
                    "signal": "validation_failed",
                    "detail": row.validation_feedback,
                    "severity": "high",
                })
            if isinstance(row.validation_feedback, str) and ERROR_KEYWORDS.search(row.validation_feedback):
                cascade_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "stage": "validator",
                    "signal": "validator_error_language",
                    "detail": row.validation_feedback,
                    "severity": "medium",
                })

        # Slot-level evidence or assumption language indicates upstream retrieval weakness propagated into final itinerary.
        for _, row in slots_df.loc[slots_df.system == "travel_agent"].iterrows():
            if row.uncertainty_hits or row.error_hits or row.link_count == 0:
                severity = "high" if row.error_hits or row.link_count == 0 else "medium"
                cascade_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "stage": "itinerary_slot",
                    "signal": "weak_or_missing_evidence",
                    "detail": f"{row.category}: {row.name} | links={row.link_count} | {row.text[:250]}",
                    "severity": severity,
                })

        cascade_df = pd.DataFrame(cascade_events)
        display(cascade_df.head(80) if not cascade_df.empty else pd.DataFrame({"status": ["No cascade indicators detected"]}))

        if not cascade_df.empty:
            cascade_summary = cascade_df.groupby(["run_id", "stage", "severity"]).size().reset_index(name="events")
            fig = px.bar(cascade_summary, x="run_id", y="events", color="stage", facet_row="severity", title="Travel-Agent Cascade Indicators By Trace")
            fig.update_layout(height=750)
            fig.show()

            stage_counts = cascade_df.groupby(["stage", "signal"]).size().reset_index(name="events")
            fig = px.sunburst(stage_counts, path=["stage", "signal"], values="events", title="Cascade Error Taxonomy")
            fig.show()
        """
    ),
    md("## Semantic Failure Heuristics"),
    code(
        """
        semantic_events = []

        # T2-style missing mandatory parameters in explicit baseline queries.
        for _, row in tool_calls_df.iterrows():
            if missing_mandatory_parameter(row.query):
                semantic_events.append({
                    "system": "baseline",
                    "run_id": row.run_id,
                    "category": "T2_missing_mandatory_parameters",
                    "source": "tool_query",
                    "detail": row.query,
                    "severity": "medium",
                })

        # Travel task descriptions should carry location/date/constraint detail for domain tools.
        for _, row in travel_tasks_df.iterrows():
            if missing_mandatory_parameter(row.task_text):
                semantic_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "category": "T2_missing_mandatory_parameters",
                    "source": "task_proxy",
                    "detail": row.task_text,
                    "severity": "medium",
                })

        # T4/T5-style weak evidence, link fragility, and hallucination risk in final artifacts.
        fragile_domains = re.compile(r"google\.com/maps|maps\.google|instagram\.com|facebook\.com|tripadvisor\.com|booking\.com|expedia\.com|agoda\.com", re.I)
        for _, row in slots_df.iterrows():
            if row.uncertainty_hits:
                semantic_events.append({
                    "system": row.system,
                    "run_id": row.run_id,
                    "category": "T4_misinterpretation_or_uncertainty",
                    "source": "plan_slot",
                    "detail": row.text[:400],
                    "severity": "medium",
                })
            if row.link_count == 0:
                semantic_events.append({
                    "system": row.system,
                    "run_id": row.run_id,
                    "category": "T5_missing_evidence_link",
                    "source": "plan_slot",
                    "detail": row.text[:400],
                    "severity": "medium",
                })
            if any(fragile_domains.search(link or "") for link in row.links):
                semantic_events.append({
                    "system": row.system,
                    "run_id": row.run_id,
                    "category": "T5_fragile_or_deep_link",
                    "source": "plan_slot_link",
                    "detail": ", ".join(row.links[:3]),
                    "severity": "low",
                })

        # Contradictory planning signals in travel-agent message histories.
        for _, row in messages_df.loc[messages_df.system == "travel_agent"].iterrows():
            text = row.content or ""
            if "No flights needed" in text and "Missing likely task types: flight" in text:
                semantic_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "category": "T6_cross_agent_data_corruption_or_contradiction",
                    "source": row.agent,
                    "detail": "Planner/reviewer mentions missing flight task despite user saying no flights needed.",
                    "severity": "high",
                })
            if "No hotel needed" in text and "Missing likely task types: flight, hotel" in text:
                semantic_events.append({
                    "system": "travel_agent",
                    "run_id": row.run_id,
                    "category": "T6_cross_agent_data_corruption_or_contradiction",
                    "source": row.agent,
                    "detail": "Planner/reviewer mentions missing hotel task despite user saying no hotel needed.",
                    "severity": "high",
                })

        semantic_df = pd.DataFrame(semantic_events)
        display(semantic_df.sort_values(["system", "run_id", "severity"]) if not semantic_df.empty else pd.DataFrame({"status": ["No semantic failure indicators detected"]}))

        if not semantic_df.empty:
            sem_summary = semantic_df.groupby(["system", "category"]).size().reset_index(name="events")
            fig = px.bar(sem_summary, x="category", y="events", color="system", barmode="group", title="Semantic Failure Indicators By Category")
            fig.update_layout(height=600, xaxis_tickangle=-35)
            fig.show()
        """
    ),
    md("## Error Taxonomy Heatmap Aligned With `paper.tex`"),
    code(
        """
        # Paper taxonomy from Section 5 tool-use analysis.
        taxonomy = {
            "T1_invalid_arguments": "Malformed query/URL or syntactically invalid call",
            "T2_missing_mandatory_parameters": "Key parameters such as dates, coordinates, travelers, or locations absent",
            "T3_redundant_or_infinite_loops": "Same query/evidence repeated without backtracking",
            "T4_misinterpretation_or_uncertainty": "Snippet/page output likely misread or over-interpreted",
            "T5_link_rot_empty_or_fragile_evidence": "Empty, missing, brittle, or hard-to-crawl evidence links",
            "T6_cross_agent_data_corruption_or_contradiction": "Bad upstream output or contradiction propagates downstream",
            "Runtime_incomplete_or_error": "Timeout, empty export, tool error, or explicit failure language",
        }

        taxonomy_events = []
        for _, row in semantic_df.iterrows() if 'semantic_df' in globals() and not semantic_df.empty else []:
            cat = row.category
            if cat.startswith("T5"):
                cat = "T5_link_rot_empty_or_fragile_evidence"
            taxonomy_events.append({"system": row.system, "run_id": row.run_id, "category": cat, "detail": row.detail})
        for _, row in loop_df.iterrows() if 'loop_df' in globals() and not loop_df.empty and "system" in loop_df.columns else []:
            taxonomy_events.append({"system": row.system, "run_id": row.run_id, "category": "T3_redundant_or_infinite_loops", "detail": row.query})
        for _, row in runtime_df.iterrows() if 'runtime_df' in globals() and not runtime_df.empty and "system" in runtime_df.columns else []:
            taxonomy_events.append({"system": row.system, "run_id": row.run_id, "category": "Runtime_incomplete_or_error", "detail": row.detail})
        for _, row in cascade_df.iterrows() if 'cascade_df' in globals() and not cascade_df.empty and "system" in cascade_df.columns else []:
            taxonomy_events.append({"system": row.system, "run_id": row.run_id, "category": "T6_cross_agent_data_corruption_or_contradiction" if row.stage != "itinerary_slot" else "T5_link_rot_empty_or_fragile_evidence", "detail": row.detail})

        taxonomy_df = pd.DataFrame(taxonomy_events)
        if taxonomy_df.empty:
            display(pd.DataFrame({"status": ["No taxonomy events detected"]}))
        else:
            counts = taxonomy_df.groupby(["system", "category"]).size().reset_index(name="events")
            display(counts.pivot_table(index="category", columns="system", values="events", fill_value=0).reindex(taxonomy.keys()))

            heat = counts.pivot_table(index="category", columns="system", values="events", fill_value=0).reindex(taxonomy.keys()).fillna(0)
            plt.figure(figsize=(10, 7))
            sns.heatmap(heat, annot=True, fmt=".0f", cmap="mako")
            plt.title("Detected Error Indicators By Paper Taxonomy")
            plt.ylabel("")
            plt.xlabel("")
            plt.tight_layout()
            plt.show()

            display(Markdown("### Taxonomy Definitions"))
            display(pd.DataFrame([{"category": k, "definition": v} for k, v in taxonomy.items()]))
        """
    ),
    md("## Per-Trace Error Scorecard"),
    code(
        """
        score_parts = []
        for df_name, df in [("semantic", semantic_df if 'semantic_df' in globals() else pd.DataFrame()), ("runtime", runtime_df if 'runtime_df' in globals() else pd.DataFrame()), ("loop", loop_df if 'loop_df' in globals() else pd.DataFrame()), ("cascade", cascade_df if 'cascade_df' in globals() else pd.DataFrame())]:
            if df is not None and not df.empty and {"system", "run_id"}.issubset(df.columns):
                part = df.groupby(["system", "run_id"]).size().reset_index(name=f"{df_name}_events")
                score_parts.append(part)

        scorecard = runs_df[["system", "run_id", "file", "explicit_tool_call_count", "task_count", "slot_count", "validation_passed", "uncertainty_hits", "error_keyword_hits"]].copy()
        for part in score_parts:
            scorecard = scorecard.merge(part, on=["system", "run_id"], how="left")
        for col in [c for c in scorecard.columns if c.endswith("_events")]:
            scorecard[col] = scorecard[col].fillna(0).astype(int)
        event_cols = [c for c in scorecard.columns if c.endswith("_events")]
        scorecard["total_error_indicators"] = scorecard[event_cols].sum(axis=1) + scorecard["error_keyword_hits"].fillna(0)

        display(scorecard.sort_values("total_error_indicators", ascending=False))

        fig = px.bar(scorecard, x="run_id", y="total_error_indicators", color="system", title="Overall Error Indicator Load Per Trace", hover_data=["file", "validation_passed"])
        fig.update_layout(height=500)
        fig.show()

        fig = px.scatter(scorecard, x="slot_count", y="total_error_indicators", color="system", size="uncertainty_hits", hover_name="run_id", title="Plan Size vs. Error Indicator Load")
        fig.show()
        """
    ),
    md("## Evidence Coverage And Link-Rot Risk"),
    code(
        """
        slot_summary = slots_df.copy()
        slot_summary["has_link"] = slot_summary.link_count > 0
        cov = slot_summary.groupby(["system", "run_id"]).agg(
            slots=("name", "count"),
            slots_with_links=("has_link", "sum"),
            avg_links=("link_count", "mean"),
            uncertainty_hits=("uncertainty_hits", "sum"),
            error_hits=("error_hits", "sum"),
        ).reset_index()
        cov["link_coverage"] = cov.slots_with_links / cov.slots.replace(0, np.nan)
        display(cov.sort_values(["system", "link_coverage"]))

        fig = px.box(cov, x="system", y="link_coverage", points="all", title="Evidence Link Coverage Per Trace")
        fig.update_yaxes(tickformat=".0%")
        fig.show()

        cat_cov = slot_summary.groupby(["system", "category"]).agg(slots=("name", "count"), link_coverage=("has_link", "mean"), avg_uncertainty=("uncertainty_hits", "mean")).reset_index()
        fig = px.scatter(cat_cov, x="link_coverage", y="avg_uncertainty", size="slots", color="system", hover_name="category", title="Evidence Coverage vs. Uncertainty By Slot Category")
        fig.update_xaxes(tickformat=".0%")
        fig.show()
        """
    ),
    md("## Concrete Examples To Inspect Manually"),
    code(
        """
        examples = []
        if 'duplicate_queries' in globals() and not duplicate_queries.empty:
            for _, row in duplicate_queries.head(5).iterrows():
                examples.append({"theme": "Repeated exact query", "system": row.system, "run_id": row.run_id, "example": row.example_query})
        if 'semantic_df' in globals() and not semantic_df.empty:
            for _, row in semantic_df.query("severity in ['high', 'medium']").head(10).iterrows():
                examples.append({"theme": row.category, "system": row.system, "run_id": row.run_id, "example": row.detail})
        if 'cascade_df' in globals() and not cascade_df.empty:
            for _, row in cascade_df.query("severity == 'high'").head(10).iterrows():
                examples.append({"theme": f"Cascade: {row.signal}", "system": row.system, "run_id": row.run_id, "example": row.detail})

        examples_df = pd.DataFrame(examples)
        display(examples_df if not examples_df.empty else pd.DataFrame({"status": ["No examples selected"]}))
        """
    ),
    md("## Summary Narrative For Project Report"),
    code(
        """
        baseline_runs = runs_df.query("system == 'baseline'")
        travel_runs = runs_df.query("system == 'travel_agent'")
        baseline_tool_calls = len(tool_calls_df)
        repeated_exact = len(duplicate_queries) if 'duplicate_queries' in globals() and not duplicate_queries.empty else 0
        empty_baseline = int((baseline_runs.has_outputs == False).sum())
        travel_valid_rate = travel_runs.validation_passed.mean() if not travel_runs.empty else np.nan
        avg_travel_slots = travel_runs.slot_count.mean() if not travel_runs.empty else np.nan
        avg_baseline_slots = baseline_runs.slot_count.mean() if not baseline_runs.empty else np.nan
        sem_counts = semantic_df.groupby("system").size().to_dict() if 'semantic_df' in globals() and not semantic_df.empty else {}

        summary = f'''
        ### Generated Findings

        - The dataset contains {len(baseline_runs)} baseline exports and {len(travel_runs)} travel-agent exports.
        - Baseline exports expose {baseline_tool_calls} explicit Tavily tool calls; travel-agent exports expose no raw `tool_calls`, so this notebook uses task lists, evidence links, validation fields, and plan slots as tool-use proxies.
        - Empty/incomplete exports: {empty_baseline} baseline trace(s) have missing outputs/final text.
        - Exact repeated baseline queries within the same trace: {repeated_exact} repeated-query groups.
        - Average extracted plan slots: baseline {avg_baseline_slots:.1f}, travel agent {avg_travel_slots:.1f}.
        - Travel-agent validation pass rate in the exported artifacts: {travel_valid_rate:.1%}.
        - Semantic/error indicators detected by heuristic: baseline {sem_counts.get('baseline', 0)}, travel agent {sem_counts.get('travel_agent', 0)}.

        Interpretation: the baseline is easier to inspect for direct search redundancy because it exposes raw Tavily calls. The travel agent produces richer structured artifacts and better traceable itineraries, but the artifacts show more opportunities for cascading evidence issues: fragile deep links, missing links, fallback language, and validation-stage contradictions can propagate into final slots. These findings are heuristic and should be paired with the paper's manual evaluation tables for final claims.
        '''
        display(Markdown(summary))
        """
    ),
]

out = Path("langsmith_trace_error_analysis.ipynb")
nbf.write(nb, out)
print(f"Wrote {out}")
