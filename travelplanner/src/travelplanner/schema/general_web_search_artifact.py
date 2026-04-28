from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class GeneralWebSearchSourceModel(BaseModel):
    title: Annotated[str | None, Field(default=None, description="Source page title")]
    url: Annotated[str | None, Field(default=None, description="Source URL")]
    snippet: Annotated[
        str | None, Field(default=None, description="Content excerpt from source")
    ]
    score: Annotated[
        float | None, Field(default=None, description="Tavily relevance score")
    ]


class GeneralWebSearchResultModel(BaseModel):
    ok: bool
    query: str
    answer: str | None = None


class GeneralWebSearchProofPointModel(BaseModel):
    claim: Annotated[
        str,
        Field(
            description="Interpretive summary of what this source contributes for the query — phrased differently from evidence"
        ),
    ]
    evidence: Annotated[
        str,
        Field(
            description="Clean excerpt from source evidence (first 200 chars), stripped of noise"
        ),
    ]
    confidence: Annotated[
        float | None,
        Field(
            default=None,
            description="Confidence score 0.0-1.0 for how strongly this source supports the claim",
        ),
    ]
    source_url: str | None = None


class GeneralWebSearchErrorModel(BaseModel):
    code: Literal[
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "answer_error",
        "unknown_error",
    ]
    message: str


class GeneralWebSearchArtifactContentModel(BaseModel):
    task_ref: str
    query: Annotated[str, Field(description="Original search query")]
    provider: Literal["tavily"]
    status: Literal["success", "partial", "failed", "skipped"]
    attempt: int
    result: GeneralWebSearchResultModel
    answer: Annotated[str, Field(description="Synthesized answer text")] = ""
    model: str | None = None
    sources: Annotated[
        list[GeneralWebSearchSourceModel],
        Field(default_factory=list, description="All deduplicated sources retrieved"),
    ]
    proof_points: Annotated[
        list[GeneralWebSearchProofPointModel],
        Field(
            default_factory=list,
            description="Interpretive claims with evidence excerpts and confidence",
        ),
    ]
    errors: list[GeneralWebSearchErrorModel] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
