"""Callable wrapper around the general-web-search agent graph.

Exposes ``make_search_web_tool``: a factory that closure-binds model/temperature
/task_ref and returns a single-arg ``(query: str) -> str`` function suitable
for wrapping in a ``StructuredTool``. The graph (query Tavily → normalize →
artifact) is reused as-is so this module holds no web-search business logic —
only the adapter that turns a natural-language query into the graph's
task-shaped input and renders the resulting artifact.
"""

from __future__ import annotations

from typing import Callable

from travelplanner.agents.general_web_search_agent import (
    GeneralWebSearchAgentState,
    make_graph as make_general_web_search_graph,
)
from travelplanner.schema.general_web_search_artifact import (
    GeneralWebSearchArtifactContentModel,
)
from travelplanner.schema.system_state import AgentArtifactModel, TaskModel

SEARCH_WEB_DESCRIPTION = """General-purpose web search for factual information via Tavily.

**What it does**: Accepts a natural-language query for factual research, official sources, or current information. Queries the Tavily search API. Returns a source-backed answer with a list of source URLs and confidence scores. Use this ONLY for factual verification that other tools cannot provide — NOT for flight, hotel, or restaurant search (use the dedicated tools for those).

**When to call it**: Before filling any slot that requires verified factual details that cannot be obtained from structured tools. Examples: official museum hours and closing days, current road closures or strikes affecting a route, contact phone number or email for a company, exact address of a location, current event schedules, visa requirements for a destination. Call `search_web` and use the result to fill the slot.

**Args**:
  - query: str — Natural-language factual query. Must be a specific question seeking factual information. Not a general research topic.
  - Example valid queries:
    - "what are the opening hours for the Sagrada Familia in Barcelona in May 2026"
    - "is there a train strike in Germany on 2026-07-15 affecting Munich to Berlin route"
    - "contact phone number and email for the Louvre Museum in Paris"
    - "exact address of Munich Airport Terminal 2"
    - "current visa requirements for Indian citizens visiting Spain 2026"
    - "what is the dress code for dining at restaurant Pierre Gagnaire in Paris"

**Returns**: String containing: direct answer to the question, then a list of source URLs with confidence scores in brackets, e.g., "[confidence: 0.92] https://example.com/source". Returns "Error: ..." on failure.

**On failure**: If the returned string starts with "Error:", read the message. Common causes: Tavily API timeout (retry once), query too vague (be more specific), no results found (rephrase the question). If error persists, mark the information as unavailable and note the source could not be verified.

**IMPORTANT**: Do NOT use this tool for flight, hotel, or restaurant searches. Those have dedicated tools (search_flights, search_hotels, search_restaurants) that return structured data. Using search_web for those will return generic web results, not verified booking data.

**Example query**: "what are the opening hours for the Sagrada Familia in Barcelona in May 2026"
"""


def make_search_web_tool(
    model_name: str,
    temperature: float = 0.0,
    task_ref: str = "tool_call",
) -> Callable[[str], str]:
    """Return a ``search_web(query)`` callable bound to graph + config."""
    web_graph = make_general_web_search_graph().compile()

    def search_web(query: str) -> str:
        try:
            task = TaskModel(
                name=task_ref,
                type="general-web-search",
                text=query,
                is_valid=True,
            )
            agent_state = GeneralWebSearchAgentState(
                query=query,
                task_list=[task],
                agent_artifacts={},
            )
            result = web_graph.invoke(agent_state)
        except Exception as exc:
            return f"Error: {exc}"

        artifacts = result.get("agent_artifacts", {}).get("general_web_search_agent", [])
        if not artifacts:
            return "Error: web search produced no artifact"

        # Build answer + sources from typed artifacts
        lines = []
        for a in artifacts:
            if not isinstance(a, AgentArtifactModel):
                continue
            try:
                content = GeneralWebSearchArtifactContentModel.model_validate(a.content)
            except Exception:
                continue
            if content.answer:
                lines.append(content.answer)
            if content.sources:
                urls = [s.url for s in content.sources[:3] if s.url]
                if urls:
                    lines.append(f"Sources: {'; '.join(urls)}")

        if not lines:
            return "Error: web search produced no answer"
        return "\n\n".join(lines)

    return search_web