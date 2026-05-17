# ie686_llm_project

Automated Travel and Event Planner using LLM agents (LangGraph). Multi-package Python repo with a `travelplanner` core package and `scripts/` for evaluation and benchmarking against the TravelPlanner dataset.

## Quick Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `cd travelplanner`
3. `uv sync`
4. `source .venv/bin/activate`
5. Verify CLI: `tp --help`

> **Note:** `scripts/` and `travelplanner/` are separate packages. Run scripts from the `scripts/` directory using `uv run python <script>`.

## Architecture Overview

The system is a **multi-agent architecture (MAS)** built on LangGraph. It decomposes travel planning into specialized collaborative agents:

- **Constraint Agent**: Extracts and validates user constraints (budget, dates, accessibility, interests)
- **Planner Agent**: Provides high-level itinerary guidance (demoted from a rigid scheduler to a lightweight support module)
- **Execution Agent**: Spawns subagents dynamically (restaurant, transit, weather, budget-check) to synthesize a complete day-by-day plan
- **Verification Node**: Validates hard and commonsense constraints before finalization
