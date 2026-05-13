from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ConstraintVerdictModel(BaseModel):
    id: Annotated[str, Field(description="Constraint ID assigned by the pipeline, e.g. 'HC-1', 'CC-3'")]
    verdict: Annotated[
        Literal["PASS", "FAIL", "MISSING_INFO", "NA"],
        Field(
            description=(
                "PASS: plan satisfies the constraint. "
                "FAIL: plan violates it or required info is present but wrong. "
                "MISSING_INFO: plan lacks information needed to evaluate (counts as FAIL). "
                "NA: hard-constraint category was absent from the original query."
            )
        ),
    ]
    reasoning: Annotated[
        str,
        Field(description="Step-by-step chain-of-thought reasoning leading to the verdict"),
    ]


class JudgeOutputModel(BaseModel):
    """Structured output returned by a single judge via invoke_structured_model."""
    verdicts: Annotated[
        list[ConstraintVerdictModel],
        Field(description="One verdict per constraint, in the same order as presented"),
    ]


class JudgeResultModel(BaseModel):
    """Full result from one judge invocation, including raw response for audit."""
    model_name: str
    verdicts: list[ConstraintVerdictModel]
    raw_response: str
    retry_count: int = 0


class AggregatedConstraintModel(BaseModel):
    """Per-constraint majority-vote result across all judges."""
    id: str
    constraint_text: str
    constraint_type: Annotated[Literal["hard", "commonsense"], Field(description="'hard' or 'commonsense'")]
    final_verdict: Literal["PASS", "FAIL", "MISSING_INFO", "NA"]
    judge_verdicts: Annotated[list[str], Field(description="Raw verdict from each judge, in model order")]
    pass_count: int
    fail_count: int
    missing_count: int
    na_count: int


class RationaleVerificationOutputModel(BaseModel):
    """Structured LLM output for verifying a single slot's rationale."""
    verdict: Annotated[
        Literal["PASS", "FAIL", "MISSING_INFO"],
        Field(
            description=(
                "PASS: every verifiable claim about the slot is consistent with the evidence. "
                "FAIL: at least one claim is contradicted by the evidence. "
                "MISSING_INFO: the evidence does not contain enough information to confirm or deny the claims."
            )
        ),
    ]
    reasoning: Annotated[
        str,
        Field(description="Step-by-step reasoning that compares slot claims to retrieved evidence"),
    ]
    claims_checked: Annotated[
        list[str],
        Field(description="Specific factual claims about the slot that were checked"),
    ] = []


class RationaleVerificationModel(BaseModel):
    """Per-slot rationale verification result for a single slot in the plan."""
    day_index: Annotated[int, Field(description="1-based day index in the plan")]
    slot_position: Annotated[int, Field(description="1-based slot position within the day (after sorting by start_time)")]
    slot_name: Annotated[str, Field(description="Slot name copied from the plan, for human review")]
    source_type: Annotated[
        Literal["link", "web_search", "skipped"],
        Field(
            description=(
                "link: at least one slot.links URL was used as evidence. "
                "web_search: no link was provided so a Tavily web search was performed. "
                "skipped: verification was not attempted (e.g. slot has no meaningful claims)."
            )
        ),
    ]
    source_urls: Annotated[
        list[str],
        Field(description="URLs that were inspected to verify the slot"),
    ] = []
    verdict: Literal["PASS", "FAIL", "MISSING_INFO"]
    reasoning: str
    claims_checked: list[str] = []


class ScorecardModel(BaseModel):
    """Final evaluation scorecard with Xie et al. (2024) metrics."""
    plan_excerpt: Annotated[str, Field(description="First 300 characters of plan_text for reference")]
    judge_models: list[str]
    rationale_verifications: Annotated[
        list[RationaleVerificationModel],
        Field(description="Per-slot rationale verification results"),
    ] = []
    rationale_pass_count: Annotated[int, Field(description="Number of slots whose rationale was confirmed (PASS)")] = 0
    rationale_fail_count: Annotated[int, Field(description="Number of slots whose rationale was contradicted (FAIL)")] = 0
    rationale_missing_count: Annotated[int, Field(description="Number of slots whose rationale could not be verified (MISSING_INFO)")] = 0
    hc_micro_pass_rate: Annotated[float, Field(description="Fraction of applicable HC constraints that PASS")]
    cc_micro_pass_rate: Annotated[float, Field(description="Fraction of CC constraints that PASS")]
    hc_macro_pass_rate: Annotated[float, Field(description="1.0 if all applicable HCs pass, else 0.0")]
    cc_macro_pass_rate: Annotated[float, Field(description="1.0 if all CCs pass, else 0.0")]
    final_pass_rate: Annotated[float, Field(description="1.0 if both HC macro and CC macro are 1.0, else 0.0")]
    aggregated_constraints: list[AggregatedConstraintModel]
    timestamp: str
