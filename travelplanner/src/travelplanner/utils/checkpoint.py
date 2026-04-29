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


_DEFAULT_MSGPACK_TYPES: tuple[type[Any], ...] = (
    AgentArtifactModel,
    CalenderModel,
    ConstraintModel,
    MessageHistoryModel,
    StateContractModel,
    TaskModel,
)


def make_memory_checkpointer(
    *,
    extra_allowed_types: Iterable[type[Any]] = (),
) -> MemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist(
        [*_DEFAULT_MSGPACK_TYPES, *extra_allowed_types]
    )
    return MemorySaver(serde=serde)
