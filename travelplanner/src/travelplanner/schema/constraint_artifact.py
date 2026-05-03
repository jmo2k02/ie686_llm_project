from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ConstraintArtifactContentModel(BaseModel):
    query: Annotated[str, Field(description="Original user query")]
    status: Literal["success", "partial"]
    hard_constraints: Annotated[
        list[dict],
        Field(default_factory=list, description="Extracted hard constraints"),
    ]
    commonsense_constraints: Annotated[
        list[dict],
        Field(
            default_factory=list,
            description="Commonsense constraints checked during validation",
        ),
    ]
    categories_missing: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Hard constraint categories not present in the original request",
        ),
    ]
    categories_skipped_by_user: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Categories the user explicitly chose to skip",
        ),
    ]
    interaction_turns: Annotated[
        int,
        Field(
            default=0,
            description="Number of user input turns in the interactive flow",
        ),
    ]
    model: str | None = None
