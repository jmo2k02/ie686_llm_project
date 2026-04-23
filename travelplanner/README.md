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
- CLI behavior was not changed.

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
