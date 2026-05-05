from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from travelplanner.schema.calender import CalenderModel
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    ConstraintModel,
    MessageHistoryModel,
    StateContractModel,
    TaskModel,
)


def _default_msgpack_types() -> tuple[type[Any], ...]:
    from travelplanner.agents.constraint_iteration_agent import ViolationModel

    return (
        AgentArtifactModel,
        CalenderModel,
        ConstraintModel,
        MessageHistoryModel,
        StateContractModel,
        TaskModel,
        ViolationModel,
    )


def make_memory_checkpointer(
    *,
    extra_allowed_types: Iterable[type[Any]] = (),
) -> MemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist(
        [*_default_msgpack_types(), *extra_allowed_types]
    )
    return MemorySaver(serde=serde)
