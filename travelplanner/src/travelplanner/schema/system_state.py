from typing import Annotated, Literal
from pydantic import BaseModel, Field


class MessageHistoryModel(BaseModel):
    """"""

    user_agent: Annotated[
        str | None,
        Field(default=None, description="The user agent that acted in this history"),
    ]
    model: Annotated[
        str | None,
        Field(
            default=None, description="The type of model that was used in the history"
        ),
    ]
    agent_ref: Annotated[
        str | None,
        Field(
            default=None, description="Reference to the Agent that created this history"
        ),
    ]
    messages: Annotated[
        list[dict], Field(description="Message history of a user and agent chat")
    ]


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
        ]
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
    ]


class StateContractModel(BaseModel):
    """This model defines all important states the system uses"""

    query: Annotated[
        str, Field(description="Initial user query of the Travel Planner system")
    ]
    message_histories: Annotated[
        dict[str, MessageHistory],
        Field(
            default={},
            description="Dictionary that maps different message histories of Chats to string keys",
        ),
    ]
    constraint_list: Annotated[list[Constraint], Field(default=[])]
    task_list: Annotated[list[Task], Field(default=[])]
