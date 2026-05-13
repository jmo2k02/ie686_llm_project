from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable LangSmith tracing when no API key is configured to avoid auth noise.
if not os.getenv("LANGCHAIN_API_KEY", "").strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import traceable
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from travelplanner.agents.general_web_search_agent import _extract_full_content
from travelplanner.schema.commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS
from travelplanner.schema.judge_artifact import (
    AggregatedConstraintModel,
    ConstraintVerdictModel,
    JudgeOutputModel,
    JudgeResultModel,
    ScorecardModel,
    UrlVerificationModel,
    UrlVerificationOutputModel,
)
from travelplanner.utils.llm import invoke_structured_model

from .systemprompt import (
    JUDGE_SYSTEM_PROMPT,
    URL_VERIFICATION_SYSTEM_PROMPT,
    build_judge_user_prompt_cc,
    build_judge_user_prompt_hc,
    build_url_verification_prompt,
)


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODELS: list[str] = [
    "openrouter:anthropic/claude-3.5-haiku",      # dot, not hyphen
    "openrouter:google/gemini-2.5-flash",
    "openrouter:mistralai/mistral-large",
    "openrouter:meta-llama/llama-3.3-70b-instruct",
]

_ALL_CC_AS_DICTS: list[dict] = [
    {"type": "commonsense", "text": c.text, "user_skipped": False}
    for c in ALL_COMMONSENSE_CONSTRAINT_DEFS
]

_MAX_JUDGE_RETRIES = 2

_URL_RE = re.compile(r'https?://[^\s\)\]\"\'\<\>]+')


# ─── State ────────────────────────────────────────────────────────────────────

class JudgeState(BaseModel):
    # Inputs
    plan_text: str
    hard_constraints: list[dict]
    commonsense_constraints: list[dict] = _ALL_CC_AS_DICTS
    judge_model_names: list[str] = DEFAULT_JUDGE_MODELS
    output_dir: str = "."
    # Pipeline state (populated by nodes)
    plan_urls: list[str] = []
    url_verifications: list[UrlVerificationModel] = []
    hc_judge_results: list[JudgeResultModel] = []
    cc_judge_results: list[JudgeResultModel] = []
    aggregated_constraints: list[AggregatedConstraintModel] = []
    scorecard: ScorecardModel | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── Node: load_inputs ────────────────────────────────────────────────────────

def load_inputs_node(state: JudgeState) -> dict[str, Any]:
    hc = [c for c in state.hard_constraints if not c.get("user_skipped", False)]
    cc = [c for c in state.commonsense_constraints if not c.get("user_skipped", False)]
    return {"hard_constraints": hc, "commonsense_constraints": cc}


# ─── Node: url_extraction ─────────────────────────────────────────────────────

def url_extraction_node(state: JudgeState) -> dict[str, Any]:
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in _URL_RE.findall(state.plan_text):
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return {"plan_urls": unique_urls}


# ─── Node: url_verification ───────────────────────────────────────────────────

def _plan_excerpt_for_url(plan_text: str, url: str, context_chars: int = 500) -> str:
    idx = plan_text.find(url)
    if idx == -1:
        return ""
    start = max(0, idx - context_chars // 2)
    end = min(len(plan_text), idx + len(url) + context_chars // 2)
    return plan_text[start:end]


@traceable(name="url_verification")
def _verify_single_url(url: str, plan_text: str, model_name: str) -> UrlVerificationModel:
    fetched_title = ""
    fetched_content = ""
    try:
        extracted = _extract_full_content([url], timeout=15, extract_depth="basic")
        if extracted:
            fetched_title = extracted[0].get("title", "")
            fetched_content = extracted[0].get("raw_content", "")
    except Exception:
        pass

    user_prompt = build_url_verification_prompt(
        url=url,
        fetched_title=fetched_title,
        fetched_content=fetched_content,
        plan_excerpt=_plan_excerpt_for_url(plan_text, url),
    )

    try:
        output, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=0.0,
            system_prompt=URL_VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=UrlVerificationOutputModel,
        )
        return UrlVerificationModel(
            url=url,
            fetched_title=fetched_title,
            verdict=output.verdict,
            reasoning=output.reasoning,
            claims_checked=output.claims_checked,
        )
    except Exception as exc:
        return UrlVerificationModel(
            url=url,
            fetched_title=fetched_title,
            verdict="MISSING_INFO",
            reasoning=f"Verification failed: {exc}",
        )


def url_verification_node(state: JudgeState) -> dict[str, Any]:
    if not state.plan_urls:
        return {"url_verifications": []}

    model_name = state.judge_model_names[0] if state.judge_model_names else DEFAULT_JUDGE_MODELS[0]
    order = {url: i for i, url in enumerate(state.plan_urls)}
    results: list[UrlVerificationModel] = []

    with ThreadPoolExecutor(max_workers=min(len(state.plan_urls), 4)) as pool:
        future_to_url = {
            pool.submit(_verify_single_url, url, state.plan_text, model_name): url
            for url in state.plan_urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(UrlVerificationModel(
                    url=url,
                    verdict="MISSING_INFO",
                    reasoning=f"Unexpected error: {exc}",
                ))

    results.sort(key=lambda r: order.get(r.url, 999))
    return {"url_verifications": results}


# ─── Judge helpers ────────────────────────────────────────────────────────────

def _make_fail_result(model_name: str, n: int, prefix: str, reason: str) -> JudgeResultModel:
    verdicts = [
        ConstraintVerdictModel(id=f"{prefix}-{i}", verdict="FAIL", reasoning=reason)
        for i in range(1, n + 1)
    ]
    return JudgeResultModel(
        model_name=model_name,
        verdicts=verdicts,
        raw_response=reason,
        retry_count=_MAX_JUDGE_RETRIES,
    )


@traceable(name="judge_invocation")
def _invoke_judge(model_name: str, user_prompt: str, n: int, prefix: str) -> JudgeResultModel:
    last_exc: Exception | None = None
    for attempt in range(_MAX_JUDGE_RETRIES + 1):
        try:
            output, _, raw = invoke_structured_model(
                model_name=model_name,
                temperature=0.0,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=JudgeOutputModel,
            )
            return JudgeResultModel(
                model_name=model_name,
                verdicts=output.verdicts,
                raw_response=raw,
                retry_count=attempt,
            )
        except Exception as exc:
            last_exc = exc

    reason = f"Judge failed after {_MAX_JUDGE_RETRIES + 1} attempts: {last_exc}"
    return _make_fail_result(model_name, n, prefix, reason)


def _run_judges_parallel(
    model_names: list[str],
    user_prompt: str,
    n: int,
    prefix: str,
) -> list[JudgeResultModel]:
    results: list[JudgeResultModel | None] = [None] * len(model_names)
    with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
        future_to_idx = {
            pool.submit(_invoke_judge, model, user_prompt, n, prefix): idx
            for idx, model in enumerate(model_names)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = _make_fail_result(model_names[idx], n, prefix, str(exc))
    return [r for r in results if r is not None]


# ─── Node: judge_hc ───────────────────────────────────────────────────────────

def judge_hc_node(state: JudgeState) -> dict[str, Any]:
    uv_dicts = [uv.model_dump() for uv in state.url_verifications]
    user_prompt = build_judge_user_prompt_hc(
        plan_text=state.plan_text,
        hard_constraints=state.hard_constraints,
        url_verifications=uv_dicts,
    )
    results = _run_judges_parallel(
        state.judge_model_names,
        user_prompt,
        n=len(state.hard_constraints),
        prefix="HC",
    )
    return {"hc_judge_results": results}


# ─── Node: judge_cc ───────────────────────────────────────────────────────────

def judge_cc_node(state: JudgeState) -> dict[str, Any]:
    uv_dicts = [uv.model_dump() for uv in state.url_verifications]
    user_prompt = build_judge_user_prompt_cc(
        plan_text=state.plan_text,
        commonsense_constraints=state.commonsense_constraints,
        url_verifications=uv_dicts,
        hard_constraints=state.hard_constraints,
    )
    results = _run_judges_parallel(
        state.judge_model_names,
        user_prompt,
        n=len(state.commonsense_constraints),
        prefix="CC",
    )
    return {"cc_judge_results": results}


# ─── Node: aggregate ──────────────────────────────────────────────────────────

def _majority_verdict(verdicts: list[str]) -> str:
    pass_count = verdicts.count("PASS")
    na_count = verdicts.count("NA")
    missing_count = verdicts.count("MISSING_INFO")
    fail_count = verdicts.count("FAIL")

    if na_count >= 3:
        return "NA"
    if pass_count >= 3:
        return "PASS"
    if missing_count > fail_count and missing_count > pass_count:
        return "MISSING_INFO"
    return "FAIL"


def aggregate_node(state: JudgeState) -> dict[str, Any]:
    aggregated: list[AggregatedConstraintModel] = []

    for i, c in enumerate(state.hard_constraints, start=1):
        cid = f"HC-{i}"
        judge_verdicts = [
            next((v.verdict for v in jr.verdicts if v.id == cid), "FAIL")
            for jr in state.hc_judge_results
        ]
        final = _majority_verdict(judge_verdicts)
        aggregated.append(AggregatedConstraintModel(
            id=cid,
            constraint_text=c.get("text", ""),
            constraint_type="hard",
            final_verdict=final,
            judge_verdicts=judge_verdicts,
            pass_count=judge_verdicts.count("PASS"),
            fail_count=judge_verdicts.count("FAIL"),
            missing_count=judge_verdicts.count("MISSING_INFO"),
            na_count=judge_verdicts.count("NA"),
        ))

    for i, c in enumerate(state.commonsense_constraints, start=1):
        cid = f"CC-{i}"
        judge_verdicts = [
            next((v.verdict for v in jr.verdicts if v.id == cid), "FAIL")
            for jr in state.cc_judge_results
        ]
        final = _majority_verdict(judge_verdicts)
        aggregated.append(AggregatedConstraintModel(
            id=cid,
            constraint_text=c.get("text", ""),
            constraint_type="commonsense",
            final_verdict=final,
            judge_verdicts=judge_verdicts,
            pass_count=judge_verdicts.count("PASS"),
            fail_count=judge_verdicts.count("FAIL"),
            missing_count=judge_verdicts.count("MISSING_INFO"),
            na_count=judge_verdicts.count("NA"),
        ))

    return {"aggregated_constraints": aggregated}


# ─── Node: score ──────────────────────────────────────────────────────────────

def score_node(state: JudgeState) -> dict[str, Any]:
    hc = [c for c in state.aggregated_constraints if c.constraint_type == "hard"]
    cc = [c for c in state.aggregated_constraints if c.constraint_type == "commonsense"]

    hc_applicable = [c for c in hc if c.final_verdict != "NA"]
    hc_pass = sum(1 for c in hc_applicable if c.final_verdict == "PASS")
    hc_micro = hc_pass / len(hc_applicable) if hc_applicable else 1.0
    hc_macro = 1.0 if hc_applicable and all(c.final_verdict == "PASS" for c in hc_applicable) else 0.0

    cc_pass = sum(1 for c in cc if c.final_verdict == "PASS")
    cc_micro = cc_pass / len(cc) if cc else 1.0
    cc_macro = 1.0 if cc and all(c.final_verdict == "PASS" for c in cc) else 0.0

    url_pass = sum(1 for u in state.url_verifications if u.verdict == "PASS")
    url_fail = sum(1 for u in state.url_verifications if u.verdict == "FAIL")
    url_missing = sum(1 for u in state.url_verifications if u.verdict == "MISSING_INFO")

    scorecard = ScorecardModel(
        plan_excerpt=state.plan_text[:300],
        judge_models=state.judge_model_names,
        url_verifications=state.url_verifications,
        url_pass_count=url_pass,
        url_fail_count=url_fail,
        url_missing_count=url_missing,
        hc_micro_pass_rate=round(hc_micro, 4),
        cc_micro_pass_rate=round(cc_micro, 4),
        hc_macro_pass_rate=hc_macro,
        cc_macro_pass_rate=cc_macro,
        final_pass_rate=1.0 if hc_macro == 1.0 and cc_macro == 1.0 else 0.0,
        aggregated_constraints=state.aggregated_constraints,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return {"scorecard": scorecard}


# ─── Node: report ─────────────────────────────────────────────────────────────

def report_node(state: JudgeState) -> dict[str, Any]:
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorecard = state.scorecard
    assert scorecard is not None

    (output_dir / "scorecard.json").write_text(
        scorecard.model_dump_json(indent=2), encoding="utf-8"
    )

    with (output_dir / "audit_log.jsonl").open("w", encoding="utf-8") as f:
        for jr in state.hc_judge_results + state.cc_judge_results:
            f.write(
                json.dumps({
                    "model_name": jr.model_name,
                    "retry_count": jr.retry_count,
                    "verdicts": [v.model_dump() for v in jr.verdicts],
                    "raw_response": jr.raw_response,
                }) + "\n"
            )

    return {}


# ─── Graph factory ────────────────────────────────────────────────────────────

def make_graph() -> Any:
    builder: StateGraph = StateGraph(JudgeState)
    builder.add_node("load_inputs", load_inputs_node)
    builder.add_node("url_extraction", url_extraction_node)
    builder.add_node("url_verification", url_verification_node)
    builder.add_node("judge_hc", judge_hc_node)
    builder.add_node("judge_cc", judge_cc_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("score", score_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("load_inputs")
    builder.add_edge("load_inputs", "url_extraction")
    builder.add_edge("url_extraction", "url_verification")
    builder.add_edge("url_verification", "judge_hc")
    builder.add_edge("judge_hc", "judge_cc")
    builder.add_edge("judge_cc", "aggregate")
    builder.add_edge("aggregate", "score")
    builder.add_edge("score", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ─── Public entry point ───────────────────────────────────────────────────────

def run_evaluation(
    plan_path: str,
    hard_constraints_path: str,
    output_dir: str = ".",
    judge_model_names: list[str] | None = None,
) -> ScorecardModel:
    """Evaluate a travel plan against hard and commonsense constraints.

    Workflow:
      1. Scan plan for URLs → fetch each via Tavily extract → LLM verifies plan claims.
      2. Four judges evaluate all hard constraints (parallel).
      3. Four judges evaluate all commonsense constraints (parallel).
      4. Majority-vote aggregation → Xie et al. (2024) metrics → scorecard.json + audit_log.jsonl.

    Args:
        plan_path: Path to the travel plan markdown file.
        hard_constraints_path: Path to JSON list of hard constraint dicts from
            constraint_iteration_agent (format: {type, text: "category: value", user_skipped}).
        output_dir: Directory for scorecard.json and audit_log.jsonl.
        judge_model_names: Override the default 4-judge model list.

    Returns:
        ScorecardModel with Xie et al. (2024) HC/CC Micro and Macro Pass Rates.
    """
    plan_text = Path(plan_path).read_text(encoding="utf-8")
    hard_constraints: list[dict] = json.loads(
        Path(hard_constraints_path).read_text(encoding="utf-8")
    )

    initial_state = JudgeState(
        plan_text=plan_text,
        hard_constraints=hard_constraints,
        judge_model_names=judge_model_names or DEFAULT_JUDGE_MODELS,
        output_dir=output_dir,
    )

    graph = make_graph()
    result = graph.invoke(initial_state)
    scorecard = result.get("scorecard")
    assert scorecard is not None, "Pipeline produced no scorecard"
    return scorecard
