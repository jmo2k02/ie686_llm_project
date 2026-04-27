# Repository Guide For AI Agents

## Scope

This file applies to the entire repository unless a deeper `AGENTS.md` overrides it.

## Project Shape

- Python project managed with `uv`
- Main package: `src/travelplanner`
- Agent orchestration: `src/travelplanner/agents`
- Shared schemas: `src/travelplanner/schema`
- CLI entrypoints: `src/travelplanner/cli`

## Primary Commands

- Install or sync environment: `uv sync`
- Run the CLI: `uv run tp --help`
- Compile-check Python source: `uv run python -m compileall src`
- Run a quick workflow invocation:
  `uv run python -c "from travelplanner.agents.task_planning_workflow import run; print(run('Plan a 3 day trip to Rome', model_name='openai:gpt-4o-mini'))"`

## Current LLM Configuration Contract

- Pass model names as either `<model>` or `<provider>:<model>`
- Bare model names default to OpenAI
- Provider resolution is centralized in `src/travelplanner/agents/llm_utils.py`
- Do not construct provider clients directly inside individual agents unless there is a deliberate architectural change

## Change Rules

- Prefer small, local edits
- Preserve the existing `task_planning_workflow.run(query, model_name, temperature)` interface unless there is a strong reason to change it
- Keep agent behavior deterministic by default unless a task explicitly calls for more creative settings
- When adding new model providers, extend the shared factory instead of branching provider logic across files
- Keep environment-variable names explicit and documented in `README.md`

## Verification Rules

- For Python-only edits, at minimum run `uv run python -m compileall src`
- If you change dependencies, use `uv add` or `uv remove` and let `uv.lock` update
- If you change prompts, workflow wiring, or schemas, verify the affected entrypoint still imports cleanly

## Documentation Rules

- Update `README.md` when behavior or setup changes
- Add or update deeper `AGENTS.md` files when a directory has special constraints or architecture that another agent should know about

## Safety Notes

- Do not commit real API keys or secret values
- Prefer environment variables over hardcoded credentials
- Avoid renaming schemas or task types casually because prompts and downstream code depend on them
