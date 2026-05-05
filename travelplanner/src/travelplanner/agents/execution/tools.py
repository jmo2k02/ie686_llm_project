from langchain.tools import tool
from langchain.agents import create_agent



## Agent 1 ##
## Example ##
# Create a subagent
subagent = create_agent(model="google_genai:gemini-3.1-pro-preview", tools=[...])

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_generic_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


execution_agent = create_agent(model="google_genai:gemini-3.1-pro-preview", tools=[call_generic_agent])
