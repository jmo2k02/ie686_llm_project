"""Evaluate the real travel plan for query_1_couple_citytrip_Adrian.

Reads the TravelPlan JSON produced by the travel agent from
data/travelplans/travel_agent/query_1_couple_citytrip_Adrian.json, converts the
hard constraints from data/travel_queries.json, and runs the full LLM judge
pipeline.

Usage:
    python scratch_evaluation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from travelplanner.evaluation.judge_setup import reset_log_clock, run_evaluation
from travelplanner.schema.commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS
from travelplanner.schema.judge_artifact import ScorecardModel

REPO_ROOT = Path(__file__).resolve().parent
QUERY_ID = "query_1_couple_citytrip_Adrian"
PLAN_FILE = REPO_ROOT / "data" / "travelplans" / "travel_agent" / f"{QUERY_ID}.json"
QUERIES_FILE = REPO_ROOT / "data" / "travel_queries.json"
OUTPUT_DIR = REPO_ROOT / "eval_output" / "travel_agent"


def _convert_hard_constraints(hc: dict) -> list[dict]:
    out: list[dict] = []

    def add(category: str, value: str) -> None:
        out.append({"type": "hard", "text": f"{category}: {value}", "user_skipped": False})

    if hc.get("origin"):
        add("origin", hc["origin"])
    if hc.get("destination"):
        add("destination", hc["destination"])
    if hc.get("date_from") and hc.get("date_to"):
        add("travel_dates", f"{hc['date_from']} to {hc['date_to']}")
    travelers = hc.get("travelers") or {}
    parts: list[str] = []
    a = int(travelers.get("adults") or 0)
    c6 = int(travelers.get("children_under_6") or 0)
    c16 = int(travelers.get("children_6_to_16") or 0)
    if a:   parts.append(f"{a} adult{'s' if a != 1 else ''}")
    if c6:  parts.append(f"{c6} child{'ren' if c6 != 1 else ''} under 6")
    if c16: parts.append(f"{c16} child{'ren' if c16 != 1 else ''} 6 to 16")
    if parts:
        add("travelers", ", ".join(parts))
    if hc.get("budget_amount") is not None:
        add("budget", f"{hc['budget_amount']} {hc.get('budget_currency', 'EUR')} total")
    if hc.get("accommodation"):
        add("accommodation", hc["accommodation"])
    transport = hc.get("transport")
    if transport and transport != "No Preference":
        add("transport", transport)
    if hc.get("interests"):
        add("interests", hc["interests"])
    return out


def _print_scorecard(scorecard: ScorecardModel, label: str) -> None:
    print("=" * 64)
    print(f"SCORECARD — {label}")
    print("=" * 64)
    n_rv = len(scorecard.rationale_verifications)
    rv_summary = (
        f"{scorecard.rationale_pass_count} PASS, "
        f"{scorecard.rationale_fail_count} FAIL, "
        f"{scorecard.rationale_missing_count} MISSING_INFO"
        f"  (of {n_rv} slot{'s' if n_rv != 1 else ''})"
    ) if n_rv else "no slots verified"
    print(f"Rationale verification : {rv_summary}")
    print(f"HC  Micro Pass Rate    : {scorecard.hc_micro_pass_rate:.1%}")
    print(f"CC  Micro Pass Rate    : {scorecard.cc_micro_pass_rate:.1%}")
    print(f"HC  Macro Pass Rate    : {scorecard.hc_macro_pass_rate:.0%}")
    print(f"CC  Macro Pass Rate    : {scorecard.cc_macro_pass_rate:.0%}")
    print(f"Final Pass Rate        : {scorecard.final_pass_rate:.0%}")
    print()

    if scorecard.rationale_verifications:
        print("Rationale verification detail:")
        for rv in scorecard.rationale_verifications:
            tag = f"[{rv.verdict:12s}]"
            print(f"  {tag} Day {rv.day_index}/slot {rv.slot_position} — {rv.slot_name}  "
                  f"(source: {rv.source_type})")
            if rv.verdict != "PASS" and rv.reasoning:
                print(f"               reason: {rv.reasoning[:120]}")
        print()

    short_names = [m.split("/")[-1] for m in scorecard.judge_models]
    print("Per-constraint verdicts:")
    for c in scorecard.aggregated_constraints:
        votes = " | ".join(f"{name}={v}" for name, v in zip(short_names, c.judge_verdicts))
        print(f"  [{c.final_verdict:12s}] {c.id}: {c.constraint_text[:65]}")
        print(f"  {' ' * 15} {votes}")
    print()
    print(f"Judges    : {', '.join(scorecard.judge_models)}")
    print(f"Timestamp : {scorecard.timestamp}")
    print()


def main() -> None:
    # Load wrapper JSON and extract the nested TravelPlan dict
    if not PLAN_FILE.exists():
        sys.exit(f"error: plan file not found: {PLAN_FILE}")
    wrapper = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    travelplan_dict = wrapper.get("travelplan")
    if not travelplan_dict:
        sys.exit(f"error: no 'travelplan' key in {PLAN_FILE}")

    n_days = len(travelplan_dict.get("days", []))
    n_slots = sum(len(d.get("slots", [])) for d in travelplan_dict.get("days", []))

    # Load hard constraints from travel_queries.json
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    query = next((q for q in queries if q["id"] == QUERY_ID), None)
    if not query:
        sys.exit(f"error: '{QUERY_ID}' not found in {QUERIES_FILE}")
    hc_items = _convert_hard_constraints(query["hard_constraints"])

    print(f"Query  : {QUERY_ID}")
    print(f"Plan   : {n_days} days, {n_slots} slots")
    print(f"HC     : {len(hc_items)} hard constraints")
    print(f"CC     : {len(ALL_COMMONSENSE_CONSTRAINT_DEFS)} commonsense constraints")
    print(f"Output : {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_log_clock()
    print(f"===== evaluation start @ {datetime.now().strftime('%H:%M:%S')} =====",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        plan_path = tmp / "plan.json"
        plan_path.write_text(json.dumps(travelplan_dict, indent=2), encoding="utf-8")
        hc_path = tmp / "hard_constraints.json"
        hc_path.write_text(json.dumps(hc_items, indent=2), encoding="utf-8")

        scorecard = run_evaluation(
            plan_path=str(plan_path),
            hard_constraints_path=str(hc_path),
            output_dir=str(OUTPUT_DIR),
            plan_format="json",
        )

    print(f"===== done in {time.monotonic() - t0:.1f}s =====\n",
          file=sys.stderr, flush=True)

    _print_scorecard(scorecard, f"travel_agent — {QUERY_ID}")
    print(f"Scorecard : {OUTPUT_DIR}/scorecard.json")
    print(f"Audit log : {OUTPUT_DIR}/audit_log.jsonl")


if __name__ == "__main__":
    main()
