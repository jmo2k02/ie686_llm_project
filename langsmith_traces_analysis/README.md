# LangSmith Trace Analysis

This workspace analyzes TravelPlanner LangGraph and LangSmith traces. It contains notebooks, local trace exports, scripts for normalizing thread-level data, and a LaTeX table generator for reporting error patterns.

## Setup

Run commands from this directory:

```bash
uv sync
```

Start JupyterLab for notebook analysis:

```bash
uv run jupyter lab
```

## Contents

| Path | Purpose |
| --- | --- |
| `langsmith_trace_error_analysis.ipynb` | Exploratory trace and error-analysis notebook. |
| `correct_langsmith_tool_error_analysis.ipynb` | Corrected tool-level trace analysis. |
| `travelagent_error_counts_by_agent.ipynb` | Error counts grouped by agent/tool role. |
| `travelplanner_langsmith_error_analysis_new.ipynb` | Updated TravelPlanner trace-analysis notebook. |
| `scripts/export_travel_agent_threads.py` | Builds manifest, run, message, and tool-call tables from local exports and LangSmith. |
| `scripts/make_latex_error_tables.py` | Generates report-ready LaTeX tables from normalized trace tables. |
| `scripts/make_error_analysis_notebook.py` | Generates an error-analysis notebook. |
| `scripts/make_correct_error_analysis_notebook.py` | Generates the corrected error-analysis notebook. |
| `traces/manual_dashboard_exports/` | Local JSON exports from dashboard/LangSmith runs. |
| `traces/thread_analysis/` | Normalized CSV/JSONL outputs for analysis. |
| `traces/example_traces/` | Example normalized trace tables. |

## Data Directories

The analysis scripts use these root-level working directories:

| Path | Expected Contents |
| --- | --- |
| `travel_agent/` | TravelPlanner run JSON exports named `run-*.json`. |
| `baseline/` | Baseline run JSON exports named `run-*.json`. |
| `thread_analysis/travel_agent/` | Normalized manifest, run, message, and tool-call tables. |

The repository also includes sample/exported data under `traces/`. If you want the scripts to operate on those files without changing code, create local symlinks or copy the relevant folders into the root-level paths above.

## Offline Export

Offline mode does not call LangSmith. It reads local JSON exports from `travel_agent/` and writes a manifest to `thread_analysis/travel_agent/`:

```bash
uv run python scripts/export_travel_agent_threads.py --offline
```

Filter to a single root run with a full or partial run id:

```bash
uv run python scripts/export_travel_agent_threads.py --offline --run-id 019e31a3
```

## LangSmith Export

Set a LangSmith API key to fetch run, message, and tool-call tables for each local thread id:

```bash
export LANGSMITH_API_KEY="..."
uv run python scripts/export_travel_agent_threads.py --fetch-langsmith
```

For faster tool-only analysis:

```bash
export LANGSMITH_API_KEY="..."
uv run python scripts/export_travel_agent_threads.py --fetch-langsmith --tool-runs-only
```

The export script writes:

```text
thread_analysis/travel_agent/manifest.csv
thread_analysis/travel_agent/manifest.jsonl
thread_analysis/travel_agent/langsmith_runs.csv
thread_analysis/travel_agent/langsmith_runs.jsonl
thread_analysis/travel_agent/messages.csv
thread_analysis/travel_agent/messages.jsonl
thread_analysis/travel_agent/tool_calls.csv
thread_analysis/travel_agent/tool_calls.jsonl
```

## Generate Report Tables

After exporting LangSmith tool/run data, generate LaTeX tables:

```bash
uv run python scripts/make_latex_error_tables.py
```

The script writes `paper_error_analysis_tables.tex` and prints grouped summaries. It expects normalized thread-analysis files under `thread_analysis/travel_agent/` and local run exports under `travel_agent/`.

## Analysis Focus

The current analysis scripts look for:

- repeated tool calls and repeated search queries
- add/delete or insert/delete repair-loop indicators
- timeout and error-language indicators
- uncertainty language in tool outputs
- structural itinerary issues such as missing evidence links, fragile links, invalid intervals, overlapping slots, and budget overrun

Use the generated CSV/JSONL files as the stable input format for notebooks so notebook state does not become the source of truth.
