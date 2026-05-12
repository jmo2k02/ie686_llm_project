"""End-to-end smoke test for the LLM-as-a-Judge evaluation pipeline.

Creates two temporary files (plan + hard constraints), runs the full pipeline,
and prints the scorecard. The commonsense constraints are always taken from
ALL_COMMONSENSE_CONSTRAINT_DEFS in commonsense_constraints.py — no fixture needed.

Hard constraints follow the exact format produced by constraint_iteration_agent:
  {"type": "hard", "text": "category: value", "user_skipped": bool}
for all 8 categories: destination, origin, travel_dates, travelers, budget,
accommodation, transport, interests.

Usage:
    python scratch_evaluation.py
"""

import json
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from travelplanner.evaluation.judge_setup import run_evaluation
from travelplanner.schema.commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS

# ─── Dummy inputs ─────────────────────────────────────────────────────────────

# A plan with two deliberate violations:
#   1. Budget exceeded: 350 + 1200 + 200 = 1750 ... wait let me make it clearly exceed 2000
#      Flight €420 + Hotel €400×3 = €1200 + Meals/Activities €500 = €2120 > €2000
#   2. Within-city violation: Day 2 lists a Paris restaurant while the traveler is in Barcelona.
DUMMY_PLAN = """
# Barcelona Trip — 3 Days

## Day 1 — Arrival (Barcelona)
- **Flight**: Munich (MUC) → Barcelona (BCN), Lufthansa LH1234, departs 08:00, arrives 10:00. Cost: €420.
- **Hotel**: Hotel Arts Barcelona, check-in 14:00. Rate: €400/night × 3 nights = €1200.
- **Dinner**: Tickets (avant-garde tapas, Barcelona). €80.

## Day 2 — Sightseeing (Barcelona)
- **Breakfast**: Café de Flore, Paris. €20.  ← DELIBERATE CITY VIOLATION
- **Activity**: Sagrada Família guided tour, 10:00–12:30. €35.
- **Lunch**: Bar del Pla, Barcelona. €25.
- **Activity**: Picasso Museum, 15:00–17:00. €15.
- **Dinner**: Bodega Sepúlveda, Barcelona. €60.

## Day 3 — Departure
- **Breakfast**: Federal Café, Barcelona. €15.
- **Activity**: Montjuïc Cable Car, 10:00–11:30. €14.
- **Lunch**: La Boqueria market, Barcelona. €20.
- **Flight**: Barcelona (BCN) → Munich (MUC), Vueling VY6301, departs 17:00. Cost: €320.

## Cost Summary
- Flights: €740
- Hotel: €1200
- Meals & Activities: €284
- **Total: €2224**  ← exceeds 2000 EUR budget
""".strip()

# Hard constraints in constraint_iteration_agent format: "category: value"
# All 8 categories must be present; skipped ones have user_skipped=True.
DUMMY_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Barcelona, Spain", "user_skipped": False},
    {"type": "hard", "text": "origin: Munich, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-06-10 to 2026-06-13 (3 nights)", "user_skipped": False},
    {"type": "hard", "text": "travelers: 1 adult", "user_skipped": False},
    {"type": "hard", "text": "budget: 2000 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: food, architecture, local culture", "user_skipped": False},
]

DUMMY_QUERY = (
    "Plan a 3-day trip from Munich to Barcelona for 1 adult. "
    "Budget: 2000 EUR total. Depart 2026-06-10, return 2026-06-13. "
    "Interested in food, architecture, and local culture. Transport: flight."
)


# ─── Run ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.md"
        hc_path = Path(tmpdir) / "hard_constraints.json"
        out_dir = Path(tmpdir) / "results"

        plan_path.write_text(DUMMY_PLAN, encoding="utf-8")
        hc_path.write_text(json.dumps(DUMMY_HARD_CONSTRAINTS, indent=2), encoding="utf-8")

        print("Running evaluation pipeline...")
        print(f"  Plan: {len(DUMMY_PLAN)} chars")
        print(f"  Hard constraints: {len(DUMMY_HARD_CONSTRAINTS)} (all 8 categories)")
        n_cc = len(ALL_COMMONSENSE_CONSTRAINT_DEFS)
        print(f"  Commonsense constraints: {n_cc} (from commonsense_constraints.py)")
        print()

        scorecard = run_evaluation(
            plan_path=str(plan_path),
            hard_constraints_path=str(hc_path),
            user_query=DUMMY_QUERY,
            output_dir=str(out_dir),
        )

    # ─── Print results ────────────────────────────────────────────────────────

    print("=" * 60)
    print("SCORECARD")
    print("=" * 60)
    print(f"HC  Micro Pass Rate : {scorecard.hc_micro_pass_rate:.1%}")
    print(f"CC  Micro Pass Rate : {scorecard.cc_micro_pass_rate:.1%}")
    print(f"HC  Macro Pass Rate : {scorecard.hc_macro_pass_rate:.0%}")
    print(f"CC  Macro Pass Rate : {scorecard.cc_macro_pass_rate:.0%}")
    print(f"Final Pass Rate     : {scorecard.final_pass_rate:.0%}")
    print()

    short_names = [m.split("/")[-1] for m in scorecard.judge_models]
    print("Per-constraint verdicts:")
    for c in scorecard.aggregated_constraints:
        votes = " | ".join(f"{name}={v}" for name, v in zip(short_names, c.judge_verdicts))
        status = f"[{c.final_verdict:12s}]"
        print(f"  {status} {c.id}: {c.constraint_text[:65]}")
        print(f"  {' ' * 15} {votes}")

    if scorecard.tavily_evidence:
        print()
        print("Tavily evidence gathered:")
        for cid, snippet in scorecard.tavily_evidence.items():
            print(f"  {cid}: {snippet[:120]}...")

    print()
    print(f"Judges: {', '.join(scorecard.judge_models)}")
    print(f"Timestamp: {scorecard.timestamp}")
    print()
    print("Expected violations:")
    print("  HC-5 (budget) → FAIL: plan total €2224 exceeds €2000 limit")
    print("  CC-10 (Total estimated cost) → FAIL: €2224 > €2000")
    print("  CC-12 (geographically sensible per day) → FAIL: Paris café listed on Barcelona day")
    print("  CC-22 (Total final cost) → FAIL: €2224 > €2000")


if __name__ == "__main__":
    main()
