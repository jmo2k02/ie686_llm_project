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


class UrlVerificationModel(BaseModel):
    """Fact-check result for a single URL found in the travel plan."""
    url: str
    fetched_title: str = ""
    verdict: Annotated[
        Literal["PASS", "FAIL", "MISSING_INFO"],
        Field(
            description=(
                "PASS: all verifiable claims in the plan about this URL match the fetched content. "
                "FAIL: at least one verifiable claim contradicts the fetched content. "
                "MISSING_INFO: URL content was not accessible or contained insufficient information."
            )
        ),
    ]
    reasoning: str
    claims_checked: Annotated[
        list[str],
        Field(description="Specific factual claims from the plan that were checked against this URL"),
    ] = []


class UrlVerificationOutputModel(BaseModel):
    """Structured output from the URL verification judge."""
    url: str
    verdict: Literal["PASS", "FAIL", "MISSING_INFO"]
    reasoning: str
    claims_checked: list[str] = []


class ScorecardModel(BaseModel):
    """Final evaluation scorecard with Xie et al. (2024) metrics."""
    plan_excerpt: Annotated[str, Field(description="First 300 characters of plan_text for reference")]
    judge_models: list[str]
    url_verifications: Annotated[
        list[UrlVerificationModel],
        Field(description="Fact-check results for each URL found in the travel plan"),
    ] = []
    url_pass_count: Annotated[int, Field(description="Number of URLs whose claims were confirmed (PASS)")] = 0
    url_fail_count: Annotated[int, Field(description="Number of URLs with at least one contradicted claim (FAIL)")] = 0
    url_missing_count: Annotated[int, Field(description="Number of URLs where content was insufficient to verify (MISSING_INFO)")] = 0
    hc_micro_pass_rate: Annotated[float, Field(description="Fraction of applicable HC constraints that PASS")]
    cc_micro_pass_rate: Annotated[float, Field(description="Fraction of CC constraints that PASS")]
    hc_macro_pass_rate: Annotated[float, Field(description="1.0 if all applicable HCs pass, else 0.0")]
    cc_macro_pass_rate: Annotated[float, Field(description="1.0 if all CCs pass, else 0.0")]
    final_pass_rate: Annotated[float, Field(description="1.0 if both HC macro and CC macro are 1.0, else 0.0")]
    aggregated_constraints: list[AggregatedConstraintModel]
    timestamp: str
