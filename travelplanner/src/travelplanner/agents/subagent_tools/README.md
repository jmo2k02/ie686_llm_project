# Subagent Tools — Integration Guide

Quick reference for wiring a new subagent tool into the execution agent.
Four steps, one file each.

## The Four Integration Points

| Step | File | What to add |
|------|------|-------------|
| 1 | `subagent_tools/` | Adapter that returns a `str` summary |
| 2 | `tool_args.py` | `BaseModel` args schema |
| 3 | `execution/prompts.py` | Bullet in `_SUBAGENT_TOOLS_DOCS` |
| 4 | `tools.py` | `StructuredTool.from_function` call |

## Step 1 — Adapter (`subagent_tools/`)

Return `str` from the tool function (not a raw artifact). Two patterns:

**Factory pattern** (model + temperature bound):

```python
"""Callable wrapper around the <name> agent graph.

Exposes ``make_<tool>_tool``: a factory that closure-binds model/temperature
/task_ref and returns a single-arg ``(query: str) -> str`` function suitable
for wrapping in a ``StructuredTool``.
"""

def make_<tool>_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    graph = make_<agent>_graph()

    def <tool_func>(query: str) -> str:
        # ... invoke graph, extract typed artifact, render summary ...
        return summary_str

    return <tool_func>
```

**Direct callable** (no model binding — e.g., routing helpers):

```python
def <tool_func>(...) -> dict:   # or str
    # directly wraps an integration helper
    result = integration_func(...)
    return result   # dict with ok/stage/error, or str summary
```

## Step 2 — Args Schema (`tool_args.py`)

```python
from pydantic import BaseModel, Field
from typing import Annotated

class <ToolName>Args(BaseModel):
    query: Annotated[
        str,
        Field(min_length=1, description="..."),
    ]
```

## Step 3 — Docs (`execution/prompts.py`)

Add a bullet to `_SUBAGENT_TOOLS_DOCS` describing:
- Tool name + args
- One-line what-it-does
- What it returns
- When to call it

## Step 4 — Registration (`tools.py`)

```python
from travelplanner.agents.subagent_tools.<module> import (
    make_<tool>_tool,  <TOOL>_DESCRIPTION
)
from travelplanner.agents.tool_args import <ToolName>Args

StructuredTool.from_function(
    func=make_<tool>_tool(model, temperature, task_ref),
    name="<tool_name>",
    description=<TOOL>_DESCRIPTION,
    args_schema=<ToolName>Args,
    handle_validation_error=True,
)
```

## Typed Artifact vs Tool-Facing Summary

- **Typed artifact** — internal representation (e.g., `GeneralWebSearchArtifactContentModel`). Rich, structured, used between agents.
- **Tool-facing summary** — what the `StructuredTool` returns to the caller. Always a `str` (or a JSON-friendly `dict` for routing tools). Render the artifact into human-readable text before returning.

Routing tools (`routing_check.py`) return `dict` with `ok/stage/error` keys — that is the tool-facing format. The internal `route_one_leg` helpers may return typed objects; the wrapper converts them.

## Validation Commands

```bash
# Syntax check
cd travelplanner && uv pip install -e . && python -c "from travelplanner.agents.tools import make_subagent_tools; print('OK')"

# List registered tools
python -c "from travelplanner.agents.tools import make_subagent_tools; tools = make_subagent_tools(); print([t.name for t in tools])"

# Check imports resolve
python -c "from travelplanner.agents.subagent_tools import general_web_search, routing_check; print('OK')"
```

## Examples

### `search_web` (factory pattern)

```python
# In tools.py — factory-bound, model + temperature closed over
StructuredTool.from_function(
    func=make_search_web_tool(model, temperature, task_ref),
    name="search_web",
    description=SEARCH_WEB_DESCRIPTION,
    args_schema=WebSearchArgs,
    handle_validation_error=True,
)

# Returns string like:
# "According to official EU entry requirements... [source: euraxess.ec.europa.eu]"
# or "Error: web search produced no artifact"
```

### Routing tools (direct callables)

```python
# In tools.py — direct callable, no model binding
StructuredTool.from_function(
    func=check_route_timing,
    name="check_route_timing",
    description=CHECK_ROUTE_TIMING_DESCRIPTION,
    args_schema=CheckRouteTimingArgs,
    handle_validation_error=True,
),

# Returns dict like:
# {"ok": true, "distance_km": 342.5, "duration_min": 217, "route_summary": "A96 to A9..."}
# or {"ok": false, "stage": "route_one_leg", "error": "Invalid address format"}
```