"""Run the LLM-as-a-Judge evaluation across all 30 queries × 2 sources.

Loads travel plans from `data/travelplans/{travel_agent,baseline}/{query_id}.{json,md}`,
evaluates each against the structured hard constraints in
`data/travel_queries.json`, and writes per-evaluation results to
`data/evaluation/{source}/{query_id}.json`. Per-id audit logs are written to a
sibling subdirectory.

Each evaluation persists immediately on completion so a crash never loses
progress — rerun the script and finished tasks are skipped. Use
`--aggregate-only` to rebuild `summary.json` from existing per-id results.

Usage:
    python scripts/evaluate/evaluate.py
    python scripts/evaluate/evaluate.py --max-workers 4
    python scripts/evaluate/evaluate.py --aggregate-only
    python scripts/evaluate/evaluate.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from travelplanner.evaluation.judge_setup import reset_log_clock, run_evaluation
from travelplanner.schema.judge_artifact import ScorecardModel


REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = REPO_ROOT / "data" / "travel_queries.json"
PLANS_DIR = REPO_ROOT / "data" / "travelplans"
EVAL_DIR = REPO_ROOT / "data" / "evaluation"
SUMMARY_PATH = EVAL_DIR / "summary.json"

DEFAULT_MAX_WORKERS = 4

SOURCES: dict[str, dict[str, str]] = {
    "travel_agent": {"plan_suffix": ".json", "plan_format": "json"},
    "baseline":     {"plan_suffix": ".md",   "plan_format": "markdown"},
}

SCORECARD_RATE_FIELDS = (
    "hc_micro_pass_rate",
    "hc_macro_pass_rate",
    "cc_micro_pass_rate",
    "cc_macro_pass_rate",
    "final_pass_rate",
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Task:
    source: str
    query_id: str
    plan_path: Path
    hc_items: list[dict]
    plan_format: str


def convert_hard_constraints(hc: dict) -> list[dict]:
    """Convert a `travel_queries.json` `hard_constraints` dict into the
    [{type, text, user_skipped}] list format that `run_evaluation` expects.

    Categories match `HARD_CONSTRAINT_CATEGORIES` in `constraint_iteration_agent`,
    so the judge prompts behave identically to the production agent's output.
    """
    out: list[dict] = []

    def add(category: str, value: str) -> None:
        out.append({"type": "hard", "text": f"{category}: {value}", "user_skipped": False})

    if hc.get("origin"):
        add("origin", str(hc["origin"]))
    if hc.get("destination"):
        add("destination", str(hc["destination"]))
    if hc.get("date_from") and hc.get("date_to"):
        add("travel_dates", f"{hc['date_from']} to {hc['date_to']}")

    travelers = hc.get("travelers") or {}
    parts: list[str] = []
    adults = int(travelers.get("adults") or 0)
    c_under_6 = int(travelers.get("children_under_6") or 0)
    c_6_to_16 = int(travelers.get("children_6_to_16") or 0)
    if adults:
        parts.append(f"{adults} adult{'s' if adults != 1 else ''}")
    if c_under_6:
        parts.append(f"{c_under_6} child{'ren' if c_under_6 != 1 else ''} under 6")
    if c_6_to_16:
        parts.append(f"{c_6_to_16} child{'ren' if c_6_to_16 != 1 else ''} 6 to 16")
    if parts:
        add("travelers", ", ".join(parts))

    if hc.get("budget_amount") is not None:
        currency = hc.get("budget_currency") or "EUR"
        add("budget", f"{hc['budget_amount']} {currency} total")

    if hc.get("accommodation"):
        add("accommodation", str(hc["accommodation"]))
    transport = hc.get("transport")
    if transport and transport != "No Preference":
        add("transport", str(transport))
    if hc.get("interests"):
        add("interests", str(hc["interests"]))
    return out


def _result_has_timeout(result_path: Path) -> bool:
    try:
        rec = json.loads(result_path.read_text(encoding="utf-8"))
        return bool(rec.get("has_timeout", False))
    except (json.JSONDecodeError, OSError):
        return False


def discover_tasks(queries: list[dict], *, force: bool, retry_timeouts: bool = False) -> list[Task]:
    tasks: list[Task] = []
    for q in queries:
        qid = q["id"]
        hc_items = convert_hard_constraints(q["hard_constraints"])
        for source, cfg in SOURCES.items():
            plan_path = PLANS_DIR / source / f"{qid}{cfg['plan_suffix']}"
            result_path = EVAL_DIR / source / f"{qid}.json"
            if not plan_path.exists():
                _log(f"[skip] no plan: {plan_path.relative_to(REPO_ROOT)}")
                continue
            if result_path.exists() and not force:
                if retry_timeouts and _result_has_timeout(result_path):
                    _log(f"[retry-timeout] re-queuing: {result_path.relative_to(REPO_ROOT)}")
                else:
                    _log(f"[resume] already done: {result_path.relative_to(REPO_ROOT)}")
                    continue
            tasks.append(
                Task(
                    source=source,
                    query_id=qid,
                    plan_path=plan_path,
                    hc_items=hc_items,
                    plan_format=cfg["plan_format"],
                )
            )
    return tasks


def evaluate_one(task: Task) -> dict:
    out_subdir = EVAL_DIR / task.source / task.query_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    hc_path = out_subdir / "hard_constraints.json"
    hc_path.write_text(json.dumps(task.hc_items, indent=2), encoding="utf-8")

    t0 = time.monotonic()
    try:
        scorecard: ScorecardModel = run_evaluation(
            plan_path=str(task.plan_path),
            hard_constraints_path=str(hc_path),
            output_dir=str(out_subdir),
            plan_format=task.plan_format,
        )
        has_timeout = bool(scorecard.timed_out_models) or any(
            "timed out after" in (rv.reasoning or "")
            for rv in scorecard.rationale_verifications
        )
        record: dict = {
            "id": task.query_id,
            "source": task.source,
            "status": "ok",
            "has_timeout": has_timeout,
            "timed_out_models": scorecard.timed_out_models,
            "duration_seconds": round(time.monotonic() - t0, 1),
            "plan_path": str(task.plan_path.relative_to(REPO_ROOT)),
            "scorecard": scorecard.model_dump(mode="json"),
        }
    except Exception as exc:
        record = {
            "id": task.query_id,
            "source": task.source,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.monotonic() - t0, 1),
            "plan_path": str(task.plan_path.relative_to(REPO_ROOT)),
        }

    result_path = EVAL_DIR / task.source / f"{task.query_id}.json"
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(result_path)
    return record


def load_all_results() -> list[dict]:
    records: list[dict] = []
    for source in SOURCES:
        src_dir = EVAL_DIR / source
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                _log(f"[warn] skipping malformed result {path}: {exc}")
    return records


def _rationale_pass_rate(scorecard: dict) -> float | None:
    p = scorecard.get("rationale_pass_count", 0)
    f = scorecard.get("rationale_fail_count", 0)
    m = scorecard.get("rationale_missing_count", 0)
    denom = p + f + m
    return p / denom if denom else None


def _source_metrics(group: list[dict]) -> dict:
    if not group:
        return {"n": 0}
    scs = [r["scorecard"] for r in group]
    means: dict[str, float | None] = {
        "hc_micro_mean": mean(s["hc_micro_pass_rate"] for s in scs),
        "hc_macro_mean": mean(s["hc_macro_pass_rate"] for s in scs),
        "cc_micro_mean": mean(s["cc_micro_pass_rate"] for s in scs),
        "cc_macro_mean": mean(s["cc_macro_pass_rate"] for s in scs),
        "final_pass_rate_mean": mean(s["final_pass_rate"] for s in scs),
    }
    rrs = [r for r in (_rationale_pass_rate(s) for s in scs) if r is not None]
    means["rationale_pass_rate_mean"] = mean(rrs) if rrs else None

    components = [
        means["hc_micro_mean"],
        means["hc_macro_mean"],
        means["cc_micro_mean"],
        means["cc_macro_mean"],
    ]
    if means["rationale_pass_rate_mean"] is not None:
        components.append(means["rationale_pass_rate_mean"])
    means["overall_score"] = mean(components)
    return {"n": len(group), **means}


def aggregate(records: list[dict]) -> dict:
    by_source = {s: [r for r in records if r["source"] == s and r["status"] == "ok"]
                 for s in SOURCES}

    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r["status"] == "ok":
            by_id[r["id"]][r["source"]] = r["scorecard"]

    comparison: list[dict] = []
    for qid in sorted(by_id):
        srcs = by_id[qid]
        if "travel_agent" in srcs and "baseline" in srcs:
            ta, base = srcs["travel_agent"], srcs["baseline"]
            row = {
                "id": qid,
                "travel_agent": {k: ta[k] for k in SCORECARD_RATE_FIELDS},
                "baseline":     {k: base[k] for k in SCORECARD_RATE_FIELDS},
                "delta_final_pass_rate": ta["final_pass_rate"] - base["final_pass_rate"],
            }
            ta_rr = _rationale_pass_rate(ta)
            base_rr = _rationale_pass_rate(base)
            row["travel_agent"]["rationale_pass_rate"] = ta_rr
            row["baseline"]["rationale_pass_rate"] = base_rr
            comparison.append(row)

    timeout_records = [r for r in records if r.get("has_timeout")]
    return {
        "n_total": len(records),
        "n_ok":    sum(1 for r in records if r["status"] == "ok"),
        "n_error": sum(1 for r in records if r["status"] == "error"),
        "n_timeout": len(timeout_records),
        "timeout_ids": [{"source": r["source"], "id": r["id"], "timed_out_models": r.get("timed_out_models", [])} for r in timeout_records],
        "summary_by_source": {
            "travel_agent": _source_metrics(by_source["travel_agent"]),
            "baseline":     _source_metrics(by_source["baseline"]),
        },
        "per_query_comparison": comparison,
    }


def write_summary(summary: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SUMMARY_PATH)


def _fmt_rate(x: Any) -> str:
    if x is None:
        return "  n/a"
    return f"{x:6.1%}"


def print_summary(summary: dict) -> None:
    print()
    print("=" * 72)
    print(f"EVALUATION SUMMARY — {summary['n_ok']}/{summary['n_total']} ok, "
          f"{summary['n_error']} error")
    print("=" * 72)
    header = (f"{'metric':<28} {'travel_agent':>14} {'baseline':>14} "
              f"{'Δ (ta−base)':>14}")
    print(header)
    print("-" * len(header))
    ta = summary["summary_by_source"]["travel_agent"]
    bs = summary["summary_by_source"]["baseline"]
    rows = [
        ("HC micro mean",          "hc_micro_mean"),
        ("HC macro mean",          "hc_macro_mean"),
        ("CC micro mean",          "cc_micro_mean"),
        ("CC macro mean",          "cc_macro_mean"),
        ("Rationale pass-rate mean", "rationale_pass_rate_mean"),
        ("Final pass-rate mean",   "final_pass_rate_mean"),
        ("OVERALL score",          "overall_score"),
    ]
    for label, key in rows:
        ta_v, bs_v = ta.get(key), bs.get(key)
        delta = (ta_v - bs_v) if (ta_v is not None and bs_v is not None) else None
        delta_s = f"{delta:+.1%}" if delta is not None else "   n/a"
        print(f"{label:<28} {_fmt_rate(ta_v):>14} {_fmt_rate(bs_v):>14} {delta_s:>14}")
    print()
    print(f"travel_agent: n={ta.get('n', 0)}    baseline: n={bs.get('n', 0)}")
    n_comp = len(summary["per_query_comparison"])
    print(f"per-query comparison rows: {n_comp}")

    n_timeout = summary.get("n_timeout", 0)
    if n_timeout:
        print()
        print(f"WARNING: {n_timeout} evaluation(s) had a model timeout — scores are unreliable.")
        print("Re-run with --retry-timeouts to fix them.")
        for entry in summary.get("timeout_ids", []):
            models = ", ".join(entry.get("timed_out_models") or ["(slot verification)"])
            print(f"  {entry['source']}/{entry['id']}  timed-out: {models}")

    print(f"summary written to: {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"parallel evaluations (default {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip evaluation, just rebuild summary.json from existing per-id files")
    parser.add_argument("--force", action="store_true",
                        help="re-evaluate even if a per-id result file already exists")
    parser.add_argument("--retry-timeouts", action="store_true",
                        help="re-evaluate plans whose previous run had a model timeout")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        records = load_all_results()
        _log(f"[aggregate-only] loaded {len(records)} record(s) from {EVAL_DIR}")
        summary = aggregate(records)
        write_summary(summary)
        print_summary(summary)
        return

    if not QUERIES_PATH.exists():
        sys.exit(f"error: queries file not found at {QUERIES_PATH}")
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    tasks = discover_tasks(queries, force=args.force, retry_timeouts=args.retry_timeouts)
    _log(f"Launching {len(tasks)} evaluation(s) with max_workers={args.max_workers}")
    if not tasks:
        _log("Nothing to do — aggregating existing results.")
        summary = aggregate(load_all_results())
        write_summary(summary)
        print_summary(summary)
        return

    reset_log_clock()
    t0 = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_task = {pool.submit(evaluate_one, t): t for t in tasks}
        for fut in as_completed(future_to_task):
            t = future_to_task[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"id": t.query_id, "source": t.source,
                       "status": "error", "error": repr(exc)}
            completed += 1
            status = rec.get("status", "?")
            dur = rec.get("duration_seconds", 0.0)
            _log(f"[done {completed}/{len(tasks)}] "
                 f"{t.source}/{t.query_id} → {status} ({dur:.1f}s)")

    _log(f"All tasks finished in {time.monotonic() - t0:.1f}s")

    summary = aggregate(load_all_results())
    write_summary(summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
