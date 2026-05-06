# TravelPlanner

This README documents the exact LLM configuration changes that were just made in this repository.

## Summary Of Prior Changes

The code previously instantiated `ChatOpenAI` directly inside multiple files and accepted only a plain `model_name` string. That made provider selection, API key handling, and base URL selection implicit and hard to extend.

The following changes were made:

1. Added a central chat model factory in `src/travelplanner/agents/llm_utils.py`.
2. Switched model naming to a provider-aware format: `<provider>:<model>`.
3. Added environment-variable based provider resolution for OpenAI-compatible backends.
4. Replaced direct model construction in agent code with the shared factory.
5. Replaced the deprecated `langchain_community.chat_models.ChatOpenAI` usage with `langchain_openai.ChatOpenAI`.
6. Added `langchain-openai` as a project dependency and removed the now-unused `langchain-community` dependency.

## Exact File Changes

### `src/travelplanner/agents/llm_utils.py`

Added:

- `OpenAICompatibleProvider` dataclass to describe per-provider configuration.
- `OPENAI_COMPATIBLE_PROVIDERS` registry for supported providers.
- `_get_env_value()` to read and normalize environment variables.
- `_parse_model_name()` to support both `<model>` and `<provider>:<model>` input.
- `make_chat_model()` as the single construction point for chat models.

Behavior added in `make_chat_model()`:

- Bare model names like `gpt-4o-mini` default to provider `openai`.
- Provider-prefixed names like `openrouter:anthropic/claude-3.7-sonnet` are split into provider and provider model.
- Unsupported providers raise a `ValueError` with the supported provider list.
- Missing required API keys raise a `ValueError` with the exact environment variable name to set.
- Ollama gets a default local base URL and a harmless fallback API key because the client expects one.

The returned client is now constructed with `langchain_openai.ChatOpenAI` using:

- `model`
- `temperature`
- `openai_api_key`
- `openai_api_base`
- `openai_organization`

### `src/travelplanner/agents/minimal_agent.py`

Changed:

- Removed direct `ChatOpenAI(...)` construction.
- Imported and used `make_chat_model(...)` instead.

Result:

- The minimal agent now inherits the same provider and API key behavior as the structured agent utilities.

### `src/travelplanner/agents/llm_utils.py`

Changed:

- `invoke_structured_model()` now calls `make_chat_model(...)` instead of constructing a model inline.

Result:

- `constraint_agent`, `planner_agent`, and `reviewer_agent` all now use the same provider-aware configuration path because they all call `invoke_structured_model()`.

### `pyproject.toml`

Changed:

- Added `langchain-openai>=1.2.0`.
- Removed the unused `langchain-community` dependency.

### `uv.lock`

Changed:

- Lockfile updated by `uv` to reflect the dependency change.

## Supported Model Syntax

Use either:

- `<model>`
- `<provider>:<model>`

Examples:

- `gpt-4o-mini`
- `openai:gpt-4o-mini`
- `openrouter:anthropic/claude-3.7-sonnet`
- `groq:llama-3.3-70b-versatile`
- `ollama:llama3.1`

If the provider is omitted, TravelPlanner assumes `openai`.

## Supported Providers And Environment Variables

The current implementation supports OpenAI-compatible providers only.

- `openai`
  Uses `OPENAI_API_KEY`
  Optional org variable: `OPENAI_ORG_ID`
- `openrouter`
  Uses `OPENROUTER_API_KEY`
  Default base URL: `https://openrouter.ai/api/v1`
  Optional override: `OPENROUTER_BASE_URL`
- `groq`
  Uses `GROQ_API_KEY`
  Default base URL: `https://api.groq.com/openai/v1`
  Optional override: `GROQ_BASE_URL`
- `ollama`
  Default base URL: `http://localhost:11434/v1`
  Optional override: `OLLAMA_BASE_URL`
  Optional key: `OLLAMA_API_KEY`

## What Did Not Change

These parts of the public code path were preserved:

- `task_planning_workflow.run(query, model_name, temperature)` still accepts a `model_name` string.
- Agent prompts and workflow structure were not changed.
- State models in `src/travelplanner/schema/system_state.py` were not changed.
- Core workflow entrypoints were preserved, but the evaluation CLI now accepts direct graph import strings.

## Evaluation CLI Graph Loading

The evaluation CLI can now run either:

- a workflow label from `AVAILABLE_WORKFLOWS`
- a direct graph reference such as `package.module:make_graph`

Examples:

```bash
uv run tp eval run --workflow task-planning --model openai:gpt-4o-mini
uv run tp eval run --graph travelplanner.workflows.task_planning:make_graph --model openai:gpt-4o-mini
uv run tp eval run --graph travelplanner.agents.constraint_agent:make_graph --graph-input-factory your_package.eval_inputs:build_constraint_input
```

Notes:

- If the imported object is callable, the CLI will pass `model_name` and `temperature` only when that builder accepts them.
- If the compiled graph exposes a Pydantic input schema, the CLI auto-fills common fields like `query`, `model_name`, and `temperature` from the dataset record and CLI options.
- For graphs with custom input contracts, pass `--graph-input-factory` to adapt each dataset record into the expected graph input.

## Example Usage

OpenAI:

```bash
export OPENAI_API_KEY="..."
uv run python -c "from travelplanner.agents.task_planning_workflow import run; print(run('Plan a 3 day trip to Rome', model_name='openai:gpt-4o-mini'))"
```

OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
uv run python -c "from travelplanner.agents.task_planning_workflow import run; print(run('Plan a 3 day trip to Rome', model_name='openrouter:anthropic/claude-3.7-sonnet'))"
```

Ollama:

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
uv run python -c "from travelplanner.agents.task_planning_workflow import run; print(run('Plan a 3 day trip to Rome', model_name='ollama:llama3.1'))"
```

## Interactive CLI Dashboard

The interactive CLI now uses a Rich live dashboard modeled after the TradingAgents CLI.

- `Progress` table shows all workflow agents and their current status.
- `Messages & Tools` table shows recent agent messages plus Tavily tool activity.
- Bottom footer shows run summary stats in the format:
  `Agents: x/x | LLM: x | Tools: x | Tokens: xk xk | Time elapsed: mm:ss`

You can launch it with the planner entrypoint:

```bash
uv run tp planner run
```

Workflow selection, the initial travel query, and any workflow follow-up questions are now asked inside the dashboard instead of dropping back out to a plain terminal prompt.

## Verification Performed

The changes were validated with:

```bash
uv run python -m compileall src
```

And with a direct factory sanity check to confirm:

- `ollama:llama3.1` builds a client with the expected base URL
- missing `OPENROUTER_API_KEY` raises the expected error

## Agent Docs

This repository now includes scoped `AGENTS.md` files to help coding agents work safely:

- `AGENTS.md`
- `src/travelplanner/agents/AGENTS.md`
- `src/travelplanner/schema/AGENTS.md`

If you later add native Anthropic or Gemini support, extend `make_chat_model()` in `src/travelplanner/agents/llm_utils.py` rather than reintroducing provider-specific client construction inside individual agents.

## General Web Search Agent

Task type `general-web-search` is now executable in the `task-planning` workflow.

- New agent: `src/travelplanner/agents/general_web_search_agent.py`
- Workflow wiring: `constraint_agent -> planner_agent -> reviewer_agent -> general_web_search_agent`
- Output location: `state.agent_artifacts["general_web_search_agent"]`

### Centralized Config

TravelPlanner now reads settings from repository-level YAML:

- `../config.yaml` (base config committed to repo)
- `../local.config.yaml` (optional local override, gitignored)

Config is merged as `base <- local`.
This is designed to be shared by all agents/workflows over time, not only web search.

The agent executes only tasks where:

- `task.type == "general-web-search"`
- `task.is_valid == true`

Set `TAVILY_API_KEY` in your environment to enable Tavily API calls.
If the key is missing, the agent stores an error artifact instead of failing the workflow.

Config knobs (env variables):

- `TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RESULTS` (default: `5`)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_TIMEOUT_SECONDS` (default: `30`)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_MAX_RETRIES` (default: `1`, bounded retry loop)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_DEPTH` (default: `basic`)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_INCLUDE_ANSWER` (default: `true`)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_MODEL` (default: `openrouter:minimax/minimax-m2.5:free`)
- `TRAVELPLANNER_GENERAL_WEB_SEARCH_ANSWER_TEMPERATURE` (default: `0.0`)

The same values can be configured in `config.yaml` under:

- `agents.general_web_search.*`

Iterative quality-gated search (1-3 searches based on result quality, not predetermined query plans):

The agent uses an iterative quality-gating loop instead of planning all subqueries upfront. After each search, the top result score is checked. If score >= 0.5 and results >= 2, synthesis begins. Otherwise a refined query is tried, with up to 3 total searches per task:

- Search 1: original query
- Search 2 (if needed): refined query targeting structured sources (site:wikidata.org OR site:openstreetmap.org)
- Search 3 (if still needed): alternative angle (latest news/events)

The `best_score_seen` variable tracks the highest result score across all searches. The loop stops as soon as quality thresholds are met or max searches is reached.

Prompt stack used by the web-search agent:

- `answer_system` + `answer_instruction`: synthesizes retrieved evidence into a planning-ready answer.

Prompt defaults live in `src/travelplanner/agents/general_web_search_agent.py`.

Answer format principle:

- The artifact is answer-first (`final_answer`) with explicit proof (`proof_points`).
- Main output is direct answer content, not raw website lists.

Typed artifact contract:

- Artifact type: `general-web-search-result`
- Validated payload schema: `GeneralWebSearchArtifactContentModel`
- Required contract fields: `task_ref`, `provider`, `status`, `attempt`, `result`, `final_answer`, `proof_points`, `errors`
- Status values: `success | partial | failed | skipped`

With `OPENROUTER_API_KEY` and default answer model above, each Tavily result is turned into an answer-with-proof artifact via OpenRouter so the agent is testable with:

- provider: `openrouter`
- model: `minimax/minimax-m2.5:free`

## Test Layout (lecture-style stages)

The web-search tests mirror the staged structure used in `docs/lecture`:

- `tests/general_web_search/research/` (config tests)
- `tests/general_web_search/outline/` (task selection tests)
- `tests/general_web_search/review/` (Tavily backend tests)
- `tests/general_web_search/final/` (graph integration tests)

Run:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

One-command validation:

```bash
uv run python test_general_web_search.py
```

Live LangGraph run (OpenRouter answer model + Tavily retrieval):

```bash
uv run python run_general_web_search_agent.py --query "Plan Barcelona beach + food trip" --task "Best beach zones in June with nearby food and opening-hour constraints"
```

Generic search-agent runner (extensible to future search agents from `docs/workflow.md`):

```bash
uv run python run_search_agent.py --agent general_web_search --query "Plan Barcelona beach + food trip" --task "Best beach zones in June with nearby food and opening-hour constraints"
```

### Testing Any Search Agent

The `run_search_agent.py` script doubles as a test runner for any registered search agent:

```bash
uv run python run_search_agent.py --agent <name> --run-tests
```

This runs the staged test suite for the specified agent and uses shared test infrastructure from `tests/shared/` and testing helpers from `src/travelplanner/testing/`.

## Extension pattern for future search agents:

1. Add a typed artifact schema in `src/travelplanner/schema/`.
2. Implement an agent graph in `src/travelplanner/agents/`.
3. Register the agent in `run_search_agent.py` (`SearchAgentSpec`).
4. Add staged tests mirroring `tests/general_web_search/`.
