from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias, TypedDict, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from tavily import TavilyClient

from travelplanner.baseline_agent.config import BaselineAgentConfig, load_config_from_env
from travelplanner.utils.checkpoint import make_memory_checkpointer
from travelplanner.utils.llm import extract_token_usage, make_chat_model
from travelplanner.utils.runtime_monitor import record_llm_call, record_tool_call


JsonValue: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None


SYSTEM_PROMPT = """You are the Simple Baseline TravelPlanner Agent.

Purpose:
- Produce a single-agent baseline itinerary for later comparison against the
  multi-agent TravelPlanner system.
- You have exactly one tool: Tavily web search. You have no sub-agents and no
  specialized flight, hotel, restaurant, routing, or validation tools.

Operating rules:
1. Use Tavily web search before the final itinerary for current facts, links,
   opening hours, prices, neighborhoods, transit hints, and source URLs.
   The workflow may require a minimum number of completed Tavily searches before
   you may answer with final markdown only; if you receive a workflow nudge,
   complete the requested searches first.
   Use a small number of high-value searches, then synthesize the final answer.
2. Respect every hard constraint supplied by the user. If a constraint cannot be
   verified, state the assumption instead of silently ignoring it.
3. Prefer official tourism, venue, transit, accommodation, restaurant, and event
   pages when search results contain them. Use blogs and aggregators only as
   secondary evidence.
4. Do not invent exact prices, opening hours, flight numbers, hotel availability,
   or booking status. Mark uncertain values as estimates.
5. Order each day like the main execution agent: travel/check-in first, meals at
   plausible meal times, attractions during likely opening hours, transfers
   between distant places, and rest/free time when useful.
6. Return only markdown in the final answer.

Final markdown structure:
Use exactly these section headings, in this order:
1. `# Baseline itinerary: ...`
2. `## Goal and constraints understood`
3. `## Search summary`
4. `## Day 1 ...` and one section per day
5. `## Budget summary`
6. `## Unscheduled or uncertain items`
7. `## Assumptions and gaps`

Each day must be a markdown table with exactly these columns:
`Time | Type | Plan | Location | Cost | Evidence/Notes`.

Source discipline:
- Every non-obvious venue, attraction, transit pass, opening-time claim, and
  ticket price should have either a markdown link in Evidence/Notes or be marked
  as an estimate.
- If a requested item cannot be scheduled with available evidence, list it under
  "Unscheduled or uncertain items" rather than hiding it.
- Never include raw Tavily result dumps in the final answer.

Example final output shape:

# Baseline itinerary: 3 days in Lisbon

## Goal and constraints understood
- Origin: Berlin
- Dates: 2026-06-12 to 2026-06-14
- Budget: 800 EUR for 2 travelers

## Search summary
- Official tourism page for Belém Tower: https://...
- Transit pass information: https://...

## Day 1
| Time | Type | Plan | Location | Cost | Evidence/Notes |
|---|---|---|---|---:|---|
| 09:00-10:30 | transport | Arrive and transfer to hotel | Airport to Baixa | €8 | Metro estimate; verify live schedules |

## Budget summary
- Estimated total: €...

## Unscheduled or uncertain items
- Flight number unavailable because only Tavily web search is available.

## Assumptions and gaps
- Hotel prices require live booking confirmation.
"""


class BaselineState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class TavilySearchArgs(BaseModel):
    query: str = Field(description="Travel-planning search query for Tavily.")


@dataclass(frozen=True)
class BaselineRunResult:
    """Baseline run output.

    Attributes:
        markdown: Final itinerary or error stub.
        messages: LangGraph message list (best-effort on recursion errors).
        model_name: Chat model used.
        executed_tool_calls: Completed Tavily runs (tool result messages).
        requested_tool_calls: Tool-call slots on AI messages (may exceed executed
            when the model batches calls; trimming enforces the API budget).
    """

    markdown: str
    messages: list[AnyMessage]
    model_name: str
    executed_tool_calls: int
    requested_tool_calls: int


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _format_constraints(constraints: JsonValue) -> str:
    if constraints in (None, ""):
        return "No separate constraints supplied. Infer only from the query."
    if isinstance(constraints, str):
        return constraints.strip()
    return json.dumps(constraints, indent=2, ensure_ascii=False)


def _count_tool_calls(messages: list[AnyMessage]) -> int:
    return sum(len(getattr(message, "tool_calls", []) or []) for message in messages)


def _count_executed_tools(messages: list[AnyMessage]) -> int:
    """Number of completed tool runs (one Tavily invocation per tool message)."""

    return sum(1 for message in messages if getattr(message, "type", None) == "tool")


def _count_tool_requirement_nudges(messages: list[AnyMessage]) -> int:
    return sum(
        1
        for message in messages
        if getattr(message, "type", None) == "human"
        and _message_content_to_text(message.content).startswith("Workflow requirement:")
    )


def _cap_ai_tool_calls(response: AnyMessage, max_parallel: int) -> AnyMessage:
    """Ensure at most ``max_parallel`` tool calls are executed this turn (hard API cap)."""

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response
    if max_parallel <= 0:
        if isinstance(response, AIMessage):
            return response.model_copy(update={"tool_calls": []})
        return response
    if len(tool_calls) <= max_parallel:
        return response
    if isinstance(response, AIMessage):
        return response.model_copy(
            update={"tool_calls": list(tool_calls)[:max_parallel]},
        )
    return response


def _require_tavily_human_message(min_tool_calls: int) -> HumanMessage:
    if min_tool_calls <= 0:
        return HumanMessage(
            content="Continue: use Tavily when you need current facts, then produce the itinerary."
        )
    if min_tool_calls == 1:
        body = (
            "Workflow requirement: call `tavily_web_search` at least once with a "
            "focused query before writing the final itinerary. Do not reply with "
            "final markdown until you have completed the required Tavily search."
        )
    else:
        body = (
            f"Workflow requirement: call `tavily_web_search` at least {min_tool_calls} "
            "times with focused queries before writing the final itinerary. Do not reply "
            "with final markdown until you have completed the required Tavily searches."
        )
    return HumanMessage(content=body)


def _build_user_prompt(query: str, constraints: JsonValue) -> str:
    return (
        "GOAL\n"
        "Create a comparable baseline travel plan in markdown.\n\n"
        "USER QUERY\n"
        f"{query.strip()}\n\n"
        "CONSTRAINTS\n"
        f"{_format_constraints(constraints)}\n\n"
        "OUTPUT REQUIREMENT\n"
        "Return the final plan as markdown using the same practical timetable "
        "style as the main execution agent, but acknowledge that this baseline "
        "only has Tavily evidence. Include source links found via Tavily."
    )


def _format_tavily_result(raw_response: dict[str, Any]) -> str:
    lines: list[str] = []
    answer = raw_response.get("answer")
    if answer:
        lines.extend(["Tavily answer:", str(answer), ""])

    raw_results = raw_response.get("results") or []
    results = cast(list[dict[str, Any]], raw_results)
    if not results:
        return "No Tavily results returned."

    lines.append("Tavily results:")
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or ""
        content = str(result.get("content") or "").strip()
        if len(content) > 900:
            content = f"{content[:900].rstrip()}..."
        score = result.get("score")
        score_text = ""
        if isinstance(score, float):
            score_text = f" score={score:.2f}"
        elif score is not None:
            score_text = f" score={score}"
        lines.append(
            f"{index}. {title}{score_text}\n"
            f"   URL: {url}\n"
            f"   Snippet: {content}"
        )
    return "\n".join(lines)


def _make_tavily_tool(config: BaselineAgentConfig) -> StructuredTool:
    def tavily_web_search(query: str) -> str:
        """Search Tavily for current source-backed travel-planning information."""

        record_tool_call(
            tool_name="baseline_tavily_web_search",
            args={
                "query": query,
                "max_results": config.tavily_max_results,
                "search_depth": config.tavily_search_depth,
                "include_answer": config.tavily_include_answer,
            },
        )
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "Error: missing TAVILY_API_KEY environment variable."

        try:
            client = TavilyClient(api_key=api_key)
            raw_response = client.search(
                query=query,
                max_results=config.tavily_max_results,
                search_depth=config.tavily_search_depth,
                include_answer=config.tavily_include_answer,
            )
        except Exception as exc:  # External API boundary: expose failure to the agent.
            return f"Error: Tavily search failed for query {query!r}: {exc}"
        return _format_tavily_result(raw_response)

    return StructuredTool.from_function(
        func=tavily_web_search,
        name="tavily_web_search",
        description=(
            "Search Tavily for current, source-backed travel-planning facts. "
            "Prefer queries that can surface official URLs, opening hours, "
            "ticket prices, transit fares, neighborhoods, restaurants, hotels, "
            "events, and attractions."
        ),
        args_schema=TavilySearchArgs,
    )


def make_graph(
    config: BaselineAgentConfig | None = None,
    checkpointer: Any | None = None,
) -> Runnable[BaselineState, BaselineState]:
    effective_config = config or load_config_from_env()
    tavily_tool = _make_tavily_tool(effective_config)
    base_model = make_chat_model(
        model_name=effective_config.model_name,
        temperature=effective_config.temperature,
    )
    tool_model = base_model.bind_tools([tavily_tool])

    def agent_node(state: BaselineState) -> dict[str, list[AnyMessage]]:
        executed = _count_executed_tools(state["messages"])
        remaining_slots = max(0, effective_config.max_tool_calls - executed)
        response = tool_model.invoke(state["messages"])
        response = _cap_ai_tool_calls(response, remaining_slots)
        tokens_in, tokens_out = extract_token_usage(response)
        record_llm_call(
            model_name=effective_config.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return {"messages": [response]}

    def require_tools_node(_state: BaselineState) -> dict[str, list[AnyMessage]]:
        return {
            "messages": [
                _require_tavily_human_message(effective_config.min_tool_calls),
            ],
        }

    def final_node(state: BaselineState) -> dict[str, list[AnyMessage]]:
        response = base_model.invoke(
            [
                *state["messages"],
                HumanMessage(
                    content=(
                        "No more Tavily searches are available. Synthesize the "
                        "final answer now as markdown using the required daily "
                        "timetable structure and the search results already present. "
                        "Include Time, Type, Plan, Location, Cost, and Evidence/Notes "
                        "columns for each day."
                    )
                ),
            ]
        )
        tokens_in, tokens_out = extract_token_usage(response)
        record_llm_call(
            model_name=effective_config.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return {"messages": [response]}

    def route_after_agent(state: BaselineState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        executed = _count_executed_tools(state["messages"])
        if executed < effective_config.min_tool_calls:
            if _count_tool_requirement_nudges(state["messages"]) > 0:
                return "final"
            return "require_tools"
        if executed >= effective_config.max_tool_calls:
            return "final"
        return "end"

    def route_after_tools(state: BaselineState) -> str:
        executed = _count_executed_tools(state["messages"])
        if executed >= effective_config.max_tool_calls:
            return "final"
        return "agent"

    graph = StateGraph(BaselineState)
    graph.add_node("agent", agent_node)
    graph.add_node("require_tools", require_tools_node)
    graph.add_node("final", final_node)
    graph.add_node("tools", ToolNode([tavily_tool]))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "require_tools": "require_tools",
            "final": "final",
            "end": END,
        },
    )
    graph.add_edge("require_tools", "agent")
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "final": "final"},
    )
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)


def _recursion_limit_for_config(config: BaselineAgentConfig) -> int:
    """Return a LangGraph step budget large enough for the tool budget.

    A single Tavily search loop can consume both an agent node and a tool node;
    the configured default of 25 tool calls therefore cannot finish within a
    raw recursion limit of 20 if the model keeps searching until the cap.
    """

    required_for_tool_cap = (config.max_tool_calls * 2) + 3
    return max(config.recursion_limit, required_for_tool_cap)


def run_baseline(
    *,
    query: str,
    constraints: JsonValue = None,
    config: BaselineAgentConfig | None = None,
) -> BaselineRunResult:
    effective_config = config or load_config_from_env()
    graph = make_graph(effective_config, checkpointer=make_memory_checkpointer())
    input_state: BaselineState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(query, constraints)),
        ]
    }
    recursion_limit = _recursion_limit_for_config(effective_config)
    stream_config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }
    collected: list[AnyMessage] | None = None
    try:
        for state in graph.stream(
            input_state,
            stream_mode="values",
            config=stream_config,
        ):
            if isinstance(state, dict) and "messages" in state:
                collected = list(state["messages"])
    except GraphRecursionError:
        messages = list(collected) if collected is not None else []
        executed = _count_executed_tools(messages)
        requested = _count_tool_calls(messages)
        return BaselineRunResult(
            markdown=(
                "# Baseline incomplete — recursion limit reached\n\n"
                f"The baseline agent reached its recursion limit of "
                f"{recursion_limit} before producing a final "
                "markdown itinerary. Try increasing "
                "TRAVELPLANNER_BASELINE_AGENT_RECURSION_LIMIT, lowering "
                "TRAVELPLANNER_BASELINE_AGENT_MAX_TOOL_CALLS, or lowering "
                "TRAVELPLANNER_BASELINE_AGENT_MIN_TOOL_CALLS if the model "
                "ignored search nudges and exhausted steps."
            ),
            messages=messages,
            model_name=effective_config.model_name,
            executed_tool_calls=executed,
            requested_tool_calls=requested,
        )
    if collected is None:
        raise RuntimeError(
            "Baseline graph stream produced no state; this is unexpected for a "
            "compiled LangGraph run."
        )
    messages = list(collected)
    executed = _count_executed_tools(messages)
    requested = _count_tool_calls(messages)

    markdown = ""
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            markdown = _message_content_to_text(message.content)
            if markdown:
                break
    if not markdown:
        markdown = (
            "# Baseline incomplete — no final markdown response\n\n"
            f"The graph finished without a non-empty assistant message. "
            f"Tavily runs completed: {executed}. "
            f"Tool call slots on AI messages: {requested}."
        )

    return BaselineRunResult(
        markdown=markdown,
        messages=messages,
        model_name=effective_config.model_name,
        executed_tool_calls=executed,
        requested_tool_calls=requested,
    )
