


AVAILABLE_DATASETS = {
  "travel_queries": "data/travel_queries.json"
}

AVAILABLE_WORKFLOWS = {
  "task-planning": {
    "graph": "travelplanner.workflows.task_planning:make_graph",
  },
  "minimal": {
    "graph": "travelplanner.agents.minimal_agent:make_graph"
  }
}