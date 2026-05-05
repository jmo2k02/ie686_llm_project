import logging
from pydantic import BaseModel, Field

from travelplanner.agents.execution.subagents import (
    TASK_AGENT_MAP,
    spawn_search_agent_for_task,
)
from travelplanner.schema.system_state import TaskModel
from travelplanner.config import get_setting

logger = logging.getLogger("execution_agent")

MODEL_NAME = get_setting("agents.execution.model_name")


class ExecutionAgentState(BaseModel):
    query: str
    task_list: list[TaskModel] = Field(default_factory=list)


def dispatch_search_tasks(state):
    """"""
    for task in state.task_list:
        if task.type not in TASK_AGENT_MAP:
            msg = f"Task type not available in TASK_AGENT_MAP: '{task.type}'. Skipping task."
            logger.warning(msg)
            continue

        spawn_result = spawn_search_agent_for_task(state, task)

        print(spawn_result)
