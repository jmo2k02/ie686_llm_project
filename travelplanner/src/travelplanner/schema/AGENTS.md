# Schema Guide

## Scope

This file applies to `src/travelplanner/schema` and its children.

## Core Contract

- `StateContractModel` is the shared state passed through the workflow
- `ConstraintModel` and `TaskModel` are prompt-facing contracts as well as runtime models
- Changes here often require prompt updates in `src/travelplanner/agents`

## Change Rules

- Do not rename fields casually
- Do not change `Literal` values in `TaskModel.type` without updating prompts and any downstream consumers
- Preserve defaults when possible so existing workflow construction continues to work
- If you add new fields, consider whether message history, workflow nodes, and prompt examples also need updates

## Validation Guidance

- Prefer strict, explicit field types
- Keep descriptions useful because they serve as documentation for both humans and agents
- When changing schemas, re-check prompt examples for exact JSON shape alignment
