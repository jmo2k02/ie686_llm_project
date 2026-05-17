# TravelPlanner Package

`travelplanner` is the main Python package for the IE686 TravelPlanner project. It provides LangGraph workflows, specialized travel-planning agents, typed state schemas, tool integrations, and a Typer CLI for interactive planning and evaluation.

## Setup

Run package commands from this directory:

```bash
uv sync
uv run tp --help
```

Compile-check source files:

```bash
uv run python -m compileall src
```

## CLI Commands

```bash
uv run tp --help
uv run tp planner run
uv run tp eval list-workflows
uv run tp eval list-datasets
uv run tp eval run --workflow task-planning --dataset travel_queries --limit 3
```

The planner command launches a Rich dashboard that asks for the workflow and trip details, then displays workflow progress, messages, tool activity, token usage, and elapsed time.

## Package Layout

| Path | Purpose |
| --- | --- |
| `src/travelplanner/workflows/` | Top-level LangGraph workflow definitions. |
| `src/travelplanner/agents/` | Constraint, planner, execution, validator, search, and routing agents. |
| `src/travelplanner/agents/execution/` | DeepAgents-based execution agent that mutates a shared `TravelPlan`. |
| `src/travelplanner/agents/subagent_tools/` | Tool wrappers used by search and execution agents. |
| `src/travelplanner/travelplan/` | `TravelPlan` domain model, day/slot models, and mutation/export tools. |
| `src/travelplanner/schema/` | Pydantic contracts for shared state, artifacts, routing, and search results. |
| `src/travelplanner/cli/` | Typer CLI entrypoints. |
| `src/travelplanner/config/` | YAML and environment-based settings loader. |
| `src/travelplanner/evaluation/` | Evaluation registry, judging, and error-analysis helpers. |

## Main Workflow

`travelplanner.workflows.task_planning:make_graph` wires the primary application graph:

```text
constraint_agent -> planner_agent -> execution_agent -> itinerary_validator
```

The workflow uses `StateContractModel` as the shared state contract. The execution agent builds or repairs a `TravelPlan`, while `itinerary_validator` can route back to `execution_agent` until validation passes or `TRAVELPLANNER_MAX_VALIDATION_RETRIES` is reached. The default retry limit is `3`.

Programmatic usage:

```python
from travelplanner.workflows.task_planning import run

state = run(
    "Plan a 3 day trip to Rome",
    model_name="openrouter:google/gemini-3-flash-preview",
    temperature=0.0,
)
print(state.travelplan.to_markdown())
```

## Evaluation

List registered workflows and datasets:

```bash
uv run tp eval list-workflows
uv run tp eval list-datasets
```

Run the registered task-planning graph:

```bash
uv run tp eval run --workflow task-planning --dataset travel_queries --limit 3
```

Run a direct graph import path:

```bash
uv run tp eval run --graph travelplanner.workflows.task_planning:make_graph --limit 3
```

Registered defaults live in `src/travelplanner/evaluation/__init__.py`.

## Configuration

Settings are loaded from the repository root:

1. `config.yaml`
2. `local.config.yaml`

Local config overrides base config. Override config paths with:

```bash
export TRAVELPLANNER_GLOBAL_CONFIG_PATH=config.yaml
export TRAVELPLANNER_LOCAL_CONFIG_PATH=local.config.yaml
```

The loader also reads `.env` from the repository root when `travelplanner.config` is imported.

## Model Syntax

Use provider-aware model names:

```text
<model>
<provider>:<model>
```

Examples:

```text
gpt-4o-mini
openai:gpt-4o-mini
openrouter:google/gemini-3-flash-preview
groq:llama-3.3-70b-versatile
ollama:nemotron-3-super
```

Bare model names default to OpenAI. Shared model construction lives in `src/travelplanner/utils/llm.py`; do not create provider-specific clients inside individual agents unless changing the architecture deliberately.

## Environment Variables

| Variable | Used For |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI models. |
| `OPENAI_ORG_ID` | Optional OpenAI organization. |
| `OPENROUTER_API_KEY` | OpenRouter models. |
| `OPENROUTER_BASE_URL` | Optional OpenRouter-compatible endpoint override. |
| `GROQ_API_KEY` | Groq models. |
| `GROQ_BASE_URL` | Optional Groq-compatible endpoint override. |
| `OLLAMA_API_KEY` | Ollama Cloud/native Ollama calls. |
| `OLLAMA_BASE_URL` | Optional Ollama endpoint override. |
| `TAVILY_API_KEY` | Tavily-backed web-search tools. |
| `SERPAPI_API_KEY` | SerpAPI-backed flight, hotel, restaurant, and attraction search tools. |
| `TRAVELPLANNER_MAX_VALIDATION_RETRIES` | Maximum validator repair loops for the main workflow. |

## Search And Tool Agents

The package includes typed artifact contracts and tool-backed agents for:

- general web search
- flight search
- hotel search
- restaurant search
- attraction search
- routing checks and route-matrix integrations

Search agents should return typed artifacts with explicit schema fields and evidence, not anonymous text blobs.

## Generated Outputs

The execution agent currently writes these files in the current working directory after a run:

```text
tp.json
tp.md
tp.ics
```

These contain the generated travel plan in JSON, Markdown, and iCalendar formats.

## Testing And Validation

Run the available test suite when changing agent behavior or schemas:

```bash
uv run pytest
```

For Python-only edits, at minimum run:

```bash
uv run python -m compileall src
```
