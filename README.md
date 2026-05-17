# IE686 LLM TravelPlanner Project

Automated travel and event planning with LangGraph agents. This repository contains the main `travelplanner` Python package, evaluation scripts, course materials, sample data, and LangSmith trace-analysis notebooks for benchmarking agent behavior on TravelPlanner-style trip-planning tasks.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `travelplanner/` | Main Python package with LangGraph workflows, CLI commands, agents, schemas, tools, and evaluation helpers. |
| `scripts/` | Separate utility package for benchmark generation/evaluation scripts and example agents. |
| `langsmith_traces_analysis/` | Separate analysis workspace for exported LangSmith traces, notebooks, and paper-table generation. |
| `data/` | Local datasets, sample travel queries, and generated travel plans. |
| `docs/` | Architecture notes, workflow design, and lecture materials. |
| `config.yaml` | Repository-level default configuration for models, agents, and testing knobs. |
| `local.config.yaml` | Optional local override file; ignored by git. |

## Quick Start

Use `uv` for all Python environments.

```bash
cd travelplanner
uv sync
uv run tp --help
```

Run the interactive planner dashboard:

```bash
cd travelplanner
uv run tp planner run
```

Compile-check the package after Python edits:

```bash
cd travelplanner
uv run python -m compileall src
```

## Main Workflow

The primary workflow is `travelplanner.workflows.task_planning:make_graph`:

```text
constraint_agent -> execution_agent -> itinerary_validator
```

The workflow extracts constraints, drafts planning tasks, builds a `TravelPlan` through DeepAgents and tool calls, then validates and repairs the itinerary with a bounded retry loop.

## CLI Reference

Run commands from `travelplanner/`:

```bash
uv run tp --help
uv run tp planner run
uv run tp eval list-workflows
uv run tp eval list-datasets
uv run tp eval run --workflow task-planning --dataset travel_queries --limit 3
```

The evaluation CLI also accepts direct graph references:

```bash
uv run tp eval run --graph travelplanner.workflows.task_planning:make_graph --limit 3
```

## Configuration

Settings are loaded from:

1. `config.yaml`
2. `local.config.yaml`

Local values override base values. You can point to custom config files with:

```bash
export TRAVELPLANNER_GLOBAL_CONFIG_PATH=config.yaml
export TRAVELPLANNER_LOCAL_CONFIG_PATH=local.config.yaml
```

Model names use either `<model>` or `<provider>:<model>` syntax. Bare model names default to OpenAI.

Supported OpenAI-compatible providers include `openai`, `openrouter`, `groq`, and `ollama`. Common environment variables are:

| Provider | Required Variable | Optional Variables |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_ORG_ID` |
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| `groq` | `GROQ_API_KEY` | `GROQ_BASE_URL` |
| `ollama` | `OLLAMA_API_KEY` | `OLLAMA_BASE_URL` |

Tool integrations may also need `TAVILY_API_KEY`, `SERPAPI_API_KEY`, or LangSmith variables depending on the workflow being run.

## Other Workspaces

The repository has multiple Python packages. Run commands from the package directory you are working in.

Evaluation utilities:

```bash
cd scripts
uv run python evaluate/pipeline.py --help
uv run python example_agents/langgraph_minimal.py --help
```

LangSmith trace analysis:

```bash
cd langsmith_traces_analysis
uv sync
```

See `langsmith_traces_analysis/README.md` before running exports. The analysis scripts expect local working folders such as `travel_agent/`, `baseline/`, and `thread_analysis/`.

## More Documentation

- `travelplanner/README.md` documents the main package.
- `langsmith_traces_analysis/README.md` documents the trace-analysis workflow.
- `docs/workflow.md` describes the multi-agent architecture and design constraints.
