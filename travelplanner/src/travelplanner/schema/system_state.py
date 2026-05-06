from typing import Annotated, Literal, get_args, get_origin
from pydantic import BaseModel, Field

from travelplanner.travelplan import TravelPlan
from travelplanner.schema.normalized_constraints import NormalizedConstraints

class MessageHistoryModel(BaseModel):
    """"""

    user_agent: Annotated[
        str | None,
        Field(default=None, description="The user agent that acted in this history"),
    ] = None
    model: Annotated[
        str | None,
        Field(
            default=None, description="The type of model that was used in the history"
        ),
    ] = None
    agent_ref: Annotated[
        str | None,
        Field(
            default=None, description="Reference to the Agent that created this history"
        ),
    ] = None
    messages: Annotated[
        list[dict], Field(description="Message history of a user and agent chat")
    ]


class AgentArtifactModel(BaseModel):
    """A generic artifact produced by an agent."""

    name: Annotated[str, Field(description="Identifier for this artifact")]
    type: Annotated[str, Field(description="Artifact type or category")]
    content: Annotated[
        dict | list | str | int | float | bool | None,
        Field(description="Artifact payload"),
    ]
    description: Annotated[
        str | None,
        Field(default=None, description="Optional human-readable artifact summary"),
    ] = None


class ConstraintModel(BaseModel):
    """Defines what a constraint in our system looks like"""

    type: Annotated[
        Literal["hard", "commonsense"], Field(description="The type of this constraint")
    ]
    user_skipped: Annotated[
        bool,
        Field(
            default=False,
            description="Denotation of whether a user specifically skipped this constraint",
        ),
    ]
    text: Annotated[str, Field(description="The textual definition of the constraint")]


class TaskModel(BaseModel):
    """An actionable task within the TravelPlanner System.
    It will be used by the Execution Agent to start different Agents and build the Timetable
    """

    name: Annotated[str, Field(description="Identifier for this task")]
    type: Annotated[
        Literal[
            "flight",
            "hotel",
            "restaurant",
            "attraction",
            "opening_times",
            "routing-check",
            "general-web-search",
        ],
        Field(
            description="The type of task that defines which Search Agent is supposed to be used for this task"
        ),
    ]
    text: Annotated[str, Field(description="The textual definition of the task")]
    is_valid: Annotated[
        bool,
        Field(
            default=False,
            description="Defines whether this task is valid and ready for a search agent",
        ),
    ]
    validation_comment: Annotated[
        str | None, Field(description="Comment on why this task is NOT valid")
    ] = None

def get_allowed_task_types() -> tuple[str, ...]:
    """Return the canonical task type values from TaskModel.type."""
    annotation = TaskModel.model_fields["type"].annotation
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) is not Literal:
        raise TypeError("TaskModel.type must be annotated as a Literal")
    return tuple(str(value) for value in get_args(annotation))


class TodoItem(BaseModel):
    title: str
    status: Literal["pending", "in_progress", "completed"]
    description: str

class StateContractModel(BaseModel):
    """This model defines all important states the system uses"""

    query: Annotated[
        str, Field(description="Initial user query of the Travel Planner system")
    ]
    message_histories: Annotated[
        dict[str, MessageHistoryModel],
        Field(
            default_factory=dict,
            description="Dictionary that maps different message histories of Chats to string keys",
        ),
    ]
    constraint_list: Annotated[list[ConstraintModel], Field(default_factory=list)]
    normalized_constraints: NormalizedConstraints | None = None
    task_list: Annotated[list[TaskModel], Field(default_factory=list)]
    travelplan: Annotated[
        TravelPlan,
        Field(
            default=None,
            description="The final output travelplan that will be used by the Execution Agent",
        ),
    ]
    todos: list[TodoItem] = []
    agent_artifacts: Annotated[
        dict[str, list[AgentArtifactModel]],
        Field(
            default_factory=dict,
            description="Artifacts produced by agents, grouped by agent key",
        ),
    ]
