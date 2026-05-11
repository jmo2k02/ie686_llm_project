from langchain.tools import tool
from langchain.agents import create_agent

from travelplanner.agents.subagent_tools.attraction_search import (
    SEARCH_ATTRACTIONS_DESCRIPTION,
    make_attraction_search_tool,
)
from travelplanner.agents.subagent_tools.flight_search import (
    SEARCH_FLIGHTS_DESCRIPTION,
    make_search_flights_tool,
)


## Agent 1 ##
## Example ##
# Create a subagent
subagent = create_agent(model="google_genai:gemini-3.1-pro-preview", tools=[...])

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_generic_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

## Flight Search Tool ##

_search_flights = make_search_flights_tool(
    model_name="google_genai:gemini-3.1-pro-preview",
    temperature=0.0,
)

@tool("search_flights", description=SEARCH_FLIGHTS_DESCRIPTION)
def call_flight_search(query: str) -> str:
    return _search_flights(query)


## Attraction Search Tool ##

_search_attractions = make_attraction_search_tool(
    model_name="google_genai:gemini-3.1-pro-preview",
    temperature=0.0,
    )

@tool("search_attractions", description=SEARCH_ATTRACTIONS_DESCRIPTION)
def call_attraction_search(query: str) -> str:
    return _search_attractions(query)


execution_agent = create_agent(model="google_genai:gemini-3.1-pro-preview", tools=[call_generic_agent, call_flight_search, call_attraction_search])
