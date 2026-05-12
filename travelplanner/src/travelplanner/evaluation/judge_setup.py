from __future__ import annotations

import json
import os
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

from travelplanner.agents.general_web_search_agent import _search_tavily
from travelplanner.schema.commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS
from travelplanner.schema.judge_artifact import (
    AggregatedConstraintModel,
    ConstraintVerdictModel,
    JudgeOutputModel,
    JudgeResultModel,
    ScorecardModel,
)
from travelplanner.utils.llm import invoke_structured_model

from .systemprompt import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODELS: list[str] = [
    "openrouter:anthropic/claude-3-5-haiku",
    "openrouter:google/gemini-flash-1.5",
    "openrouter:mistralai/mistral-small-3.1-24.09",
    "openrouter:meta-llama/llama-3.1-8b-instruct",
]

# Commonsense constraints are ground truth: always sourced from the canonical
# definition file, never from an external JSON.
_ALL_CC_AS_DICTS: list[dict] = [
    {"type": "commonsense", "text": c.text, "user_skipped": False}
    for c in ALL_COMMONSENSE_CONSTRAINT_DEFS
]

# Constraint texts that benefit from Tavily web lookups.
_GROUNDING_KEYWORDS = {
    "budget",
    "cost",
    "price",
    "geographically",
    "geographic",
    "overseas",
    "island",
    "transit",
    "opening hours",
    "advance booking",
    "origin city",
    "destination city",
}

_MAX_JUDGE_RETRIES = 2


# ─── State ────────────────────────────────────────────────────────────────────

class JudgeState(BaseModel):
    # Inputs
    plan_text: str
    # Hard constraints as {type, text, user_skipped} dicts, one per category,
    # with text formatted as "category: value" (matching constraint_iteration_agent output).
    hard_constraints: list[dict]
    # Commonsense constraints: defaults to all 23 from ALL_COMMONSENSE_CONSTRAINT_DEFS.
    commonsense_constraints: list[dict] = _ALL_CC_AS_DICTS
    user_query: str
    judge_model_names: list[str] = DEFAULT_JUDGE_MODELS
    output_dir: str = "."
    # Pipeline state (populated by nodes)
    tavily_evidence: dict[str, str] = {}
    judge_results: list[JudgeResultModel] = []
    aggregated_constraints: list[AggregatedConstraintModel] = []
    scorecard: ScorecardModel | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── Node: load_inputs ────────────────────────────────────────────────────────

def load_inputs_node(state: JudgeState) -> dict[str, Any]:
    hc = [c for c in state.hard_constraints if not c.get("user_skipped", False)]
    cc = [c for c in state.commonsense_constraints if not c.get("user_skipped", False)]
    return {"hard_constraints": hc, "commonsense_constraints": cc}


# ─── Node: tavily_grounding ───────────────────────────────────────────────────

def _needs_grounding(constraint_text: str) -> bool:
    lower = constraint_text.lower()
    return any(kw in lower for kw in _GROUNDING_KEYWORDS)


def tavily_grounding_node(state: JudgeState) -> dict[str, Any]:
    evidence: dict[str, str] = {}

    candidates: list[tuple[str, str]] = []
    for i, c in enumerate(state.hard_constraints, start=1):
        text = c.get("text", "")
        if _needs_grounding(text):
            candidates.append((f"HC-{i}", text))
    for i, c in enumerate(state.commonsense_constraints, start=1):
        text = c.get("text", "")
        if _needs_grounding(text):
            candidates.append((f"CC-{i}", text))

    for constraint_id, constraint_text in candidates:
        query = (
            f"Fact-check for travel plan evaluation — {constraint_id}: {constraint_text}. "
            f"Travel context: {state.user_query[:300]}"
        )
        result = _search_tavily(
            query,
            max_results=3,
            timeout=15,
            search_depth="basic",
            include_answer=True,
        )
        if result.get("ok"):
            snippet = result.get("answer") or ""
            if not snippet and result.get("results"):
                snippet = result["results"][0].get("content", "")[:400]
            if snippet:
                evidence[constraint_id] = snippet

    return {"tavily_evidence": evidence}


# ─── Node: judge_fan_out ──────────────────────────────────────────────────────

def _make_fail_result(model_name: str, n_hc: int, n_cc: int, reason: str) -> JudgeResultModel:
    verdicts = [
        ConstraintVerdictModel(id=f"HC-{i}", verdict="FAIL", reasoning=reason)
        for i in range(1, n_hc + 1)
    ] + [
        ConstraintVerdictModel(id=f"CC-{i}", verdict="FAIL", reasoning=reason)
        for i in range(1, n_cc + 1)
    ]
    return JudgeResultModel(
        model_name=model_name,
        verdicts=verdicts,
        raw_response=reason,
        retry_count=_MAX_JUDGE_RETRIES,
    )


@traceable(name="judge_invocation")
def _invoke_judge(
    model_name: str,
    user_prompt: str,
    n_hc: int,
    n_cc: int,
) -> JudgeResultModel:
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
    return _make_fail_result(model_name, n_hc, n_cc, reason)


def judge_fan_out_node(state: JudgeState) -> dict[str, Any]:
    user_prompt = build_judge_user_prompt(
        user_query=state.user_query,
        plan_text=state.plan_text,
        hard_constraints=state.hard_constraints,
        commonsense_constraints=state.commonsense_constraints,
        tavily_evidence=state.tavily_evidence,
    )
    n_hc = len(state.hard_constraints)
    n_cc = len(state.commonsense_constraints)

    results: list[JudgeResultModel | None] = [None] * len(state.judge_model_names)

    with ThreadPoolExecutor(max_workers=len(state.judge_model_names)) as pool:
        future_to_idx = {
            pool.submit(_invoke_judge, model, user_prompt, n_hc, n_cc): idx
            for idx, model in enumerate(state.judge_model_names)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                model = state.judge_model_names[idx]
                results[idx] = _make_fail_result(model, n_hc, n_cc, str(exc))

    return {"judge_results": [r for r in results if r is not None]}


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
    all_constraints: list[tuple[str, str, str]] = []
    for i, c in enumerate(state.hard_constraints, start=1):
        all_constraints.append((f"HC-{i}", c.get("text", ""), "hard"))
    for i, c in enumerate(state.commonsense_constraints, start=1):
        all_constraints.append((f"CC-{i}", c.get("text", ""), "commonsense"))

    aggregated: list[AggregatedConstraintModel] = []
    for cid, ctext, ctype in all_constraints:
        judge_verdicts = []
        for jr in state.judge_results:
            matched = next((v.verdict for v in jr.verdicts if v.id == cid), "FAIL")
            judge_verdicts.append(matched)

        final = _majority_verdict(judge_verdicts)
        aggregated.append(
            AggregatedConstraintModel(
                id=cid,
                constraint_text=ctext,
                constraint_type=ctype,  # type: ignore[arg-type]
                final_verdict=final,
                judge_verdicts=judge_verdicts,
                pass_count=judge_verdicts.count("PASS"),
                fail_count=judge_verdicts.count("FAIL"),
                missing_count=judge_verdicts.count("MISSING_INFO"),
                na_count=judge_verdicts.count("NA"),
            )
        )

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

    scorecard = ScorecardModel(
        user_query=state.user_query,
        plan_excerpt=state.plan_text[:300],
        judge_models=state.judge_model_names,
        hc_micro_pass_rate=round(hc_micro, 4),
        cc_micro_pass_rate=round(cc_micro, 4),
        hc_macro_pass_rate=hc_macro,
        cc_macro_pass_rate=cc_macro,
        final_pass_rate=1.0 if hc_macro == 1.0 and cc_macro == 1.0 else 0.0,
        aggregated_constraints=state.aggregated_constraints,
        tavily_evidence=state.tavily_evidence,
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
        for jr in state.judge_results:
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
    builder.add_node("tavily_grounding", tavily_grounding_node)
    builder.add_node("judge_fan_out", judge_fan_out_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("score", score_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("load_inputs")
    builder.add_edge("load_inputs", "tavily_grounding")
    builder.add_edge("tavily_grounding", "judge_fan_out")
    builder.add_edge("judge_fan_out", "aggregate")
    builder.add_edge("aggregate", "score")
    builder.add_edge("score", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ─── Public entry point ───────────────────────────────────────────────────────

def run_evaluation(
    plan_path: str,
    hard_constraints_path: str,
    user_query: str,
    output_dir: str = ".",
    judge_model_names: list[str] | None = None,
) -> ScorecardModel:
    """Evaluate a travel plan against hard and commonsense constraints.

    Hard constraints are read from a JSON file (ground truth, one dict per
    category with keys type/text/user_skipped, matching constraint_iteration_agent
    output format: text = "category: value").

    Commonsense constraints are always sourced from ALL_COMMONSENSE_CONSTRAINT_DEFS
    in commonsense_constraints.py — they are canonical and not configurable per run.

    Args:
        plan_path: Path to the travel plan markdown file.
        hard_constraints_path: Path to JSON file containing a list of hard
            constraint dicts produced by constraint_iteration_agent.
        user_query: The original user travel request.
        output_dir: Directory where scorecard.json and audit_log.jsonl are written.
        judge_model_names: Override the default 4-judge model list.

    Returns:
        The final ScorecardModel with Xie et al. (2024) metrics.
    """
    plan_text = Path(plan_path).read_text(encoding="utf-8")
    hard_constraints: list[dict] = json.loads(
        Path(hard_constraints_path).read_text(encoding="utf-8")
    )

    initial_state = JudgeState(
        plan_text=plan_text,
        hard_constraints=hard_constraints,
        user_query=user_query,
        judge_model_names=judge_model_names or DEFAULT_JUDGE_MODELS,
        output_dir=output_dir,
    )

    graph = make_graph()
    result = graph.invoke(initial_state)
    scorecard = result.get("scorecard")
    assert scorecard is not None, "Pipeline produced no scorecard"
    return scorecard
