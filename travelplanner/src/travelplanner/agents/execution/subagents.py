from typing import Literal
from travelplanner.agents.attraction_search_agent import (
    make_graph as make_attraction_graph,
)
from travelplanner.agents.flight_search_agent import make_graph as make_flight_graph
from travelplanner.agents.general_web_search_agent import (
    make_graph as make_general_web_search_graph,
)

# from travelplanner.agents.hotel_search_agent import make_graph as make_hotel_graph
from travelplanner.agents.restaurant_search_agent import (
    make_graph as make_restaurant_graph,
)

TASK_AGENT_MAP = {
    "flight": ("flight_search_agent", make_flight_graph),
    # "hotel": ("hotel_search_agent", make_hotel_graph),
    "restaurant": ("restaurant_search_agent", make_restaurant_graph),
    "attraction": ("attraction_search_agent", make_attraction_graph),
    "general-web-search": ("general_web_search_agent", make_general_web_search_graph),
}


def spawn_search_agent_for_task(
    state,
    task,
    type_overwrite: Literal[
        "flight", "hotel", "restaurant", "attraction", "general-web-search"
    ] | None = None,
) -> dict:
    """Function to spawn a search agent with a task

    Args:
      state (AgentState): AgentState
      task (TaskModel): The task to be executed
      type_overwrite (str): Overwrite the task type to change to which agent it will be routed
    
    Returns:
      Result dictionary
    """
    if type_overwrite:
        task.type = type_overwrite
    agent_key, make_agent_graph = TASK_AGENT_MAP[task.type]
    agent_graph = make_agent_graph()

    # agent_input = {
    #     "query": state.query,
    #     "model_name": state.model_name,
    #     "temperature": state.temperature,
    #     "task_list": [task],
    #     "agent_artifacts": state.agent_artifacts,
    #     "message_history": None,
    # }
    text_input = ""

    result = agent_graph.invoke({"query": text_input})

    # {"description"
    # [
    #     {
    #         "name": "xx",
    #     }
    # ]}

    return {
        "task": task,
        "agent_key": agent_key,
        "artifacts": result.get("agent_artifacts", {}),
        "message_history": result.get("message_history"),
    }


# def spawn_search_agent_for_single_query(query: str, task_type: )
