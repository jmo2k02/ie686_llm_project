# Agent Layer Guide

## Scope

This file applies to `src/travelplanner/agents` and its children.

## Architecture

- `task_planning_workflow.py` wires the high-level sequence:
  `constraint_agent -> planner_agent -> reviewer_agent`
- `llm_utils.py` owns shared LLM client construction and structured invocation
- `minimal_agent.py` is a simpler single-node planner path

## LLM Rules

- Reuse `make_chat_model()` for all chat model construction
- Reuse `invoke_structured_model()` for JSON-shaped responses when possible
- Keep provider parsing and environment handling in `llm_utils.py`
- If you add native non-OpenAI-compatible providers later, add them behind the shared factory boundary

## Prompt And Output Rules

- Keep prompts explicit about JSON-only output when the response is parsed into Pydantic models
- Preserve the current task type vocabulary unless the schema changes first
- If you change output shape, update both the Pydantic response model and the prompt examples together

## Message History Rules

- Preserve the current pattern of storing `query`, `user_prompt`, and raw model response in `MessageHistoryModel`
- Use stable history keys in `task_planning_workflow.py`
- Keep `agent_ref` values import-like and specific

## Editing Guidance

- Prefer changing shared utilities before duplicating logic across agents
- Avoid hidden provider-specific behavior in prompts or node functions
- Keep `temperature` threading explicit from workflow state into model construction
