from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable LangSmith tracing when no API key is configured to avoid auth noise.
if not os.getenv("LANGCHAIN_API_KEY", "").strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langsmith import traceable
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from travelplanner.agents.evaluation_web_search_agent import verify_all_slots
from travelplanner.schema.eval_commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS
from travelplanner.schema.judge_artifact import (
    AggregatedConstraintModel,
    ConstraintVerdictModel,
    JudgeOutputModel,
    JudgeResultModel,
    RationaleVerificationModel,
    ScorecardModel,
)
from travelplanner.travelplan.plan import TravelPlan
from travelplanner.utils.llm import invoke_structured_model

from .systemprompt import (
    JUDGE_SYSTEM_PROMPT,
    MARKDOWN_TO_TRAVELPLAN_SYSTEM_PROMPT,
    build_judge_user_prompt_cc,
    build_judge_user_prompt_hc,
    build_markdown_to_travelplan_prompt,
)


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODELS: list[str] = [
    "openrouter:minimax/minimax-m2.7",      # dot, not hyphen
    "openrouter:google/gemini-3-flash-preview",
    "openrouter:deepseek/deepseek-v4-flash",
    "openrouter:qwen/qwen3.5-397b-a17b",
]

# Model used to convert baseline markdown → TravelPlan and to run the per-slot
# rationale verifier. Kept separate from the judge ensemble so the same model
# is used consistently for plan extraction.
DEFAULT_EXTRACTION_MODEL = "openrouter:google/gemini-3-flash-preview"

_ALL_CC_AS_DICTS: list[dict] = [
    {"type": "commonsense", "text": c.text, "user_skipped": False}
    for c in ALL_COMMONSENSE_CONSTRAINT_DEFS
]

_MAX_JUDGE_RETRIES = 2
_JUDGE_TIMEOUT_SECS = 300  # 5 minutes per judge round

PlanInputFormat = Literal["json", "markdown"]


# ─── Progress logging ─────────────────────────────────────────────────────────

_RUN_START_MONO: float = time.monotonic()


def _log(msg: str) -> None:
    """Stderr-flushed progress line with wallclock + elapsed-since-run-start."""
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = time.monotonic() - _RUN_START_MONO
    print(f"[{now} +{elapsed:6.1f}s] {msg}", file=sys.stderr, flush=True)


def reset_log_clock() -> None:
    """Reset the elapsed-time counter — call at the start of each scenario."""
    global _RUN_START_MONO
    _RUN_START_MONO = time.monotonic()


# ─── State ────────────────────────────────────────────────────────────────────

class JudgeState(BaseModel):
    # Inputs
    plan_source_text: str
    plan_input_format: PlanInputFormat
    hard_constraints: list[dict]
    commonsense_constraints: list[dict] = _ALL_CC_AS_DICTS
    judge_model_names: list[str] = DEFAULT_JUDGE_MODELS
    extraction_model_name: str = DEFAULT_EXTRACTION_MODEL
    output_dir: str = "."
    # Pipeline state (populated by nodes)
    plan: TravelPlan | None = None
    plan_markdown: str = ""
    rationale_verifications: list[RationaleVerificationModel] = []
    hc_judge_results: list[JudgeResultModel] = []
    cc_judge_results: list[JudgeResultModel] = []
    aggregated_constraints: list[AggregatedConstraintModel] = []
    scorecard: ScorecardModel | None = None

    model_config = {"arbitrary_types_allowed": True}


# ─── Node: load_inputs ────────────────────────────────────────────────────────

def load_inputs_node(state: JudgeState) -> dict[str, Any]:
    _log(f"[node:load_inputs] start — format={state.plan_input_format}")
    t0 = time.monotonic()
    hc = [c for c in state.hard_constraints if not c.get("user_skipped", False)]
    cc = [c for c in state.commonsense_constraints if not c.get("user_skipped", False)]
    _log(
        f"[node:load_inputs] done ({time.monotonic() - t0:.1f}s) — "
        f"hc={len(hc)} cc={len(cc)}"
    )
    return {"hard_constraints": hc, "commonsense_constraints": cc}


# ─── Node: build_travelplan ───────────────────────────────────────────────────

@traceable(name="markdown_to_travelplan")
def _markdown_to_travelplan(markdown: str, model_name: str) -> TravelPlan:
    plan, _, _ = invoke_structured_model(
        model_name=model_name,
        #model_name="openrouter:openai/gpt-5.5",
        temperature=0.0,
        system_prompt=MARKDOWN_TO_TRAVELPLAN_SYSTEM_PROMPT,
        user_prompt=build_markdown_to_travelplan_prompt(markdown),
        response_model=TravelPlan,
    )
    return plan


def build_travelplan_node(state: JudgeState) -> dict[str, Any]:
    _log(f"[node:build_travelplan] start — format={state.plan_input_format}")
    t0 = time.monotonic()
    if state.plan_input_format == "json":
        raw = json.loads(state.plan_source_text)
        if "travelplan" in raw:
            raw = raw["travelplan"]
        plan = TravelPlan.model_validate(raw)
    elif state.plan_input_format == "markdown":
        _log(
            f"[node:build_travelplan] calling extraction LLM "
            f"(model={state.extraction_model_name})"
        )
        plan = _markdown_to_travelplan(
            state.plan_source_text, state.extraction_model_name
        )
    else:
        raise ValueError(f"Unsupported plan_input_format: {state.plan_input_format!r}")

    n_slots = sum(len(d.slots) for d in plan.days)
    _log(
        f"[node:build_travelplan] done ({time.monotonic() - t0:.1f}s) — "
        f"days={len(plan.days)} slots={n_slots}"
    )
    return {"plan": plan, "plan_markdown": plan.to_markdown()}


# ─── Node: rationale_verification ─────────────────────────────────────────────

def rationale_verification_node(state: JudgeState) -> dict[str, Any]:
    assert state.plan is not None, "build_travelplan_node must run first"
    n_slots = sum(len(d.slots) for d in state.plan.days)
    _log(
        f"[node:rationale_verification] start — verifying {n_slots} slot(s) "
        f"(extraction_model={state.extraction_model_name})"
    )
    t0 = time.monotonic()
    results = verify_all_slots(
        state.plan,
        state.extraction_model_name,
        progress_callback=lambda rv, done, total: _log(
            f"[node:rationale_verification] slot {done}/{total} done — "
            f"Day {rv.day_index}/slot {rv.slot_position} ({rv.source_type}) "
            f"→ {rv.verdict}"
        ),
    )
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_miss = sum(1 for r in results if r.verdict == "MISSING_INFO")
    _log(
        f"[node:rationale_verification] done ({time.monotonic() - t0:.1f}s) — "
        f"PASS={n_pass} FAIL={n_fail} MISSING={n_miss}"
    )
    return {"rationale_verifications": results}


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
            if attempt < _MAX_JUDGE_RETRIES:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
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
    total = len(model_names)
    done = 0
    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=len(model_names))
    try:
        future_to_idx = {
            pool.submit(_invoke_judge, model, user_prompt, n, prefix): idx
            for idx, model in enumerate(model_names)
        }
        try:
            for future in as_completed(future_to_idx, timeout=_JUDGE_TIMEOUT_SECS):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = _make_fail_result(model_names[idx], n, prefix, str(exc))
                done += 1
                jr = results[idx]
                short = model_names[idx].split("/")[-1]
                retry_info = (
                    f" retries={jr.retry_count}" if jr is not None and jr.retry_count else ""
                )
                _log(
                    f"[judges:{prefix}] {done}/{total} done — {short} "
                    f"({time.monotonic() - t0:.1f}s elapsed){retry_info}"
                )
        except concurrent.futures.TimeoutError:
            for future, idx in future_to_idx.items():
                if results[idx] is None:
                    short = model_names[idx].split("/")[-1]
                    _log(
                        f"[judges:{prefix}] TIMEOUT — {short} "
                        f"({_JUDGE_TIMEOUT_SECS}s elapsed)"
                    )
                    results[idx] = _make_fail_result(
                        model_names[idx], n, prefix,
                        f"timed out after {_JUDGE_TIMEOUT_SECS}s",
                    )
                    done += 1
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return [r for r in results if r is not None]


# ─── Node: judge_hc ───────────────────────────────────────────────────────────

def judge_hc_node(state: JudgeState) -> dict[str, Any]:
    _log(
        f"[node:judge_hc] start — {len(state.judge_model_names)} judges × "
        f"{len(state.hard_constraints)} hard constraints"
    )
    t0 = time.monotonic()
    rv_dicts = [rv.model_dump() for rv in state.rationale_verifications]
    user_prompt = build_judge_user_prompt_hc(
        plan_text=state.plan_markdown,
        hard_constraints=state.hard_constraints,
        rationale_verifications=rv_dicts,
    )
    results = _run_judges_parallel(
        state.judge_model_names,
        user_prompt,
        n=len(state.hard_constraints),
        prefix="HC",
    )
    _log(f"[node:judge_hc] done ({time.monotonic() - t0:.1f}s)")
    return {"hc_judge_results": results}


# ─── Node: judge_cc ───────────────────────────────────────────────────────────

def judge_cc_node(state: JudgeState) -> dict[str, Any]:
    _log(
        f"[node:judge_cc] start — {len(state.judge_model_names)} judges × "
        f"{len(state.commonsense_constraints)} commonsense constraints"
    )
    t0 = time.monotonic()
    rv_dicts = [rv.model_dump() for rv in state.rationale_verifications]
    user_prompt = build_judge_user_prompt_cc(
        plan_text=state.plan_markdown,
        commonsense_constraints=state.commonsense_constraints,
        rationale_verifications=rv_dicts,
        hard_constraints=state.hard_constraints,
    )
    results = _run_judges_parallel(
        state.judge_model_names,
        user_prompt,
        n=len(state.commonsense_constraints),
        prefix="CC",
    )
    _log(f"[node:judge_cc] done ({time.monotonic() - t0:.1f}s)")
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
    _log("[node:aggregate] start")
    t0 = time.monotonic()
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

    _log(
        f"[node:aggregate] done ({time.monotonic() - t0:.1f}s) — "
        f"{len(aggregated)} aggregated constraints"
    )
    return {"aggregated_constraints": aggregated}


# ─── Node: score ──────────────────────────────────────────────────────────────

def score_node(state: JudgeState) -> dict[str, Any]:
    _log("[node:score] start")
    t0 = time.monotonic()
    hc = [c for c in state.aggregated_constraints if c.constraint_type == "hard"]
    cc = [c for c in state.aggregated_constraints if c.constraint_type == "commonsense"]

    hc_applicable = [c for c in hc if c.final_verdict != "NA"]
    hc_pass = sum(1 for c in hc_applicable if c.final_verdict == "PASS")
    hc_micro = hc_pass / len(hc_applicable) if hc_applicable else 1.0
    hc_macro = 1.0 if hc_applicable and all(c.final_verdict == "PASS" for c in hc_applicable) else 0.0

    cc_pass = sum(1 for c in cc if c.final_verdict == "PASS")
    cc_micro = cc_pass / len(cc) if cc else 1.0
    cc_macro = 1.0 if cc and all(c.final_verdict == "PASS" for c in cc) else 0.0

    rv_pass = sum(1 for r in state.rationale_verifications if r.verdict == "PASS")
    rv_fail = sum(1 for r in state.rationale_verifications if r.verdict == "FAIL")
    rv_missing = sum(1 for r in state.rationale_verifications if r.verdict == "MISSING_INFO")

    scorecard = ScorecardModel(
        plan_excerpt=state.plan_markdown[:300],
        judge_models=state.judge_model_names,
        rationale_verifications=state.rationale_verifications,
        rationale_pass_count=rv_pass,
        rationale_fail_count=rv_fail,
        rationale_missing_count=rv_missing,
        hc_micro_pass_rate=round(hc_micro, 4),
        cc_micro_pass_rate=round(cc_micro, 4),
        hc_macro_pass_rate=hc_macro,
        cc_macro_pass_rate=cc_macro,
        final_pass_rate=1.0 if hc_macro == 1.0 and cc_macro == 1.0 else 0.0,
        aggregated_constraints=state.aggregated_constraints,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _log(
        f"[node:score] done ({time.monotonic() - t0:.1f}s) — "
        f"hc_macro={hc_macro:.0%} cc_macro={cc_macro:.0%} "
        f"final={scorecard.final_pass_rate:.0%}"
    )
    return {"scorecard": scorecard}


# ─── Node: report ─────────────────────────────────────────────────────────────

def report_node(state: JudgeState) -> dict[str, Any]:
    _log(f"[node:report] start — writing to {state.output_dir}")
    t0 = time.monotonic()
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorecard = state.scorecard
    assert scorecard is not None

    (output_dir / "scorecard.json").write_text(
        scorecard.model_dump_json(indent=2), encoding="utf-8"
    )

    with (output_dir / f"audit_log_{datetime.now(timezone.utc).isoformat()}.jsonl").open("w", encoding="utf-8") as f:
        for jr in state.hc_judge_results + state.cc_judge_results:
            f.write(
                json.dumps({
                    "model_name": jr.model_name,
                    "retry_count": jr.retry_count,
                    "verdicts": [v.model_dump() for v in jr.verdicts],
                    "raw_response": jr.raw_response,
                }) + "\n"
            )

    _log(f"[node:report] done ({time.monotonic() - t0:.1f}s)")
    return {}


# ─── Graph factory ────────────────────────────────────────────────────────────

def make_graph() -> Any:
    builder: StateGraph = StateGraph(JudgeState)
    builder.add_node("load_inputs", load_inputs_node)
    builder.add_node("build_travelplan", build_travelplan_node)
    builder.add_node("rationale_verification", rationale_verification_node)
    builder.add_node("judge_hc", judge_hc_node)
    builder.add_node("judge_cc", judge_cc_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("score", score_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("load_inputs")
    builder.add_edge("load_inputs", "build_travelplan")
    builder.add_edge("build_travelplan", "rationale_verification")
    builder.add_edge("rationale_verification", "judge_hc")
    builder.add_edge("judge_hc", "judge_cc")
    builder.add_edge("judge_cc", "aggregate")
    builder.add_edge("aggregate", "score")
    builder.add_edge("score", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ─── Public entry point ───────────────────────────────────────────────────────

def _infer_plan_format(plan_path: str, override: PlanInputFormat | None) -> PlanInputFormat:
    if override is not None:
        return override
    suffix = Path(plan_path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown", ".txt"}:
        return "markdown"
    raise ValueError(
        f"Cannot infer plan format from '{plan_path}'. Pass plan_format='json' or 'markdown'."
    )


def run_evaluation(
    plan_path: str,
    hard_constraints_path: str,
    output_dir: str = ".",
    judge_model_names: list[str] | None = None,
    plan_format: PlanInputFormat | None = None,
    extraction_model_name: str | None = None,
) -> ScorecardModel:
    """Evaluate a travel plan against hard and commonsense constraints.

    Workflow:
      1. Load the plan as a TravelPlan. JSON inputs are parsed directly; markdown
         inputs (e.g. from the baseline agent) are converted via a structured LLM call.
      2. Verify each Slot's factual rationale: use slot.links if present, otherwise
         fall back to a Tavily web search built from the slot's claims.
      3. Four judges evaluate all hard constraints (parallel).
      4. Four judges evaluate all commonsense constraints (parallel).
      5. Majority-vote aggregation → Xie et al. (2024) metrics → scorecard.json + audit_log.jsonl.

    Args:
        plan_path: Path to the travel plan. Either a TravelPlan .json file
            (travelplanner pipeline output) or a markdown .md file (baseline agent
            output).
        hard_constraints_path: Path to JSON list of hard constraint dicts from
            constraint_iteration_agent (format: {type, text: "category: value", user_skipped}).
        output_dir: Directory for scorecard.json and audit_log.jsonl.
        judge_model_names: Override the default 4-judge model list.
        plan_format: Force the plan format ("json" or "markdown"). When omitted,
            the format is inferred from the file suffix.
        extraction_model_name: Override the model used for markdown → TravelPlan
            extraction and per-slot rationale verification.

    Returns:
        ScorecardModel with Xie et al. (2024) HC/CC Micro and Macro Pass Rates.
    """
    plan_source_text = Path(plan_path).read_text(encoding="utf-8")
    plan_input_format = _infer_plan_format(plan_path, plan_format)
    hard_constraints: list[dict] = json.loads(
        Path(hard_constraints_path).read_text(encoding="utf-8")
    )

    initial_state = JudgeState(
        plan_source_text=plan_source_text,
        plan_input_format=plan_input_format,
        hard_constraints=hard_constraints,
        judge_model_names=judge_model_names or DEFAULT_JUDGE_MODELS,
        extraction_model_name=extraction_model_name or DEFAULT_EXTRACTION_MODEL,
        output_dir=output_dir,
    )

    graph = make_graph()
    result = graph.invoke(initial_state)
    scorecard = result.get("scorecard")
    assert scorecard is not None, "Pipeline produced no scorecard"
    return scorecard
