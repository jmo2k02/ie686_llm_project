"""End-to-end smoke test for the LLM-as-a-Judge evaluation pipeline.

Hard constraints follow constraint_iteration_agent format:
  {"type": "hard", "text": "category: value", "user_skipped": bool}

Intentional violations in the plan:
  - HC-5 (budget): total €2286 > €2000 limit
  - CC-12 (geographically sensible): Day 2 lists a Paris café while traveler is in Barcelona
  - CC-10 / CC-22 (total cost): €2286 > €2000

Audit log and scorecard are written to ./eval_output/ for post-run inspection.

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

# 4-day / 3-night trip with explicit dates in every day header.
# Deliberate violations:
#   - Day 2: Café de Flore (Paris) listed on a Barcelona day  → geographic CC fail
#   - Total: €2286 > €2000 budget                             → HC-5 + cost CC fail
DUMMY_PLAN = """
# Barcelona Trip — 4 Days / 3 Nights (2026-06-10 to 2026-06-13)

## Day 1 — 2026-06-10 (Arrival, Barcelona)
- **Flight**: Munich (MUC) → Barcelona (BCN), Lufthansa LH1234, departs 08:00, arrives 10:00. Cost: €420.
- **Hotel**: Hotel Arts Barcelona, check-in 14:00. Rate: €400/night × 3 nights = €1200.
  Book at: https://www.hotelartsbarcelona.com
- **Dinner**: Tickets (avant-garde tapas, Barcelona). €80.

## Day 2 — 2026-06-11 (Sightseeing, Barcelona)
- **Breakfast**: Café de Flore, Paris. €20.  ← DELIBERATE CITY VIOLATION
- **Activity**: Sagrada Família guided tour, 10:00–12:30. €35.
  Tickets: https://sagradafamilia.org/en/tickets
- **Lunch**: Bar del Pla, Barcelona. €25.
- **Activity**: Picasso Museum, 15:00–17:00. €15.
- **Dinner**: Bodega Sepúlveda, Barcelona. €60.

## Day 3 — 2026-06-12 (Sightseeing, Barcelona)
- **Breakfast**: Federal Café, Barcelona. €15.
- **Activity**: Montjuïc Cable Car, 10:00–11:30. €14.
- **Lunch**: La Boqueria market, Barcelona. €20.
- **Dinner**: La Mar Salada (seafood), Barcelona. €50.

## Day 4 — 2026-06-13 (Departure)
- **Breakfast**: Federal Café, Barcelona. €12.
- **Flight**: Barcelona (BCN) → Munich (MUC), Vueling VY6301, departs 17:00. Cost: €320.

## Cost Summary
- Flights: €740
- Hotel: €1200
- Meals & Activities: €346
- **Total: €2286**  ← exceeds 2000 EUR budget
""".strip()

# Hard constraints in constraint_iteration_agent format: "category: value"
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


# ─── Run ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Persist output for post-run audit log inspection
    out_dir = Path("eval_output")
    out_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.md"
        hc_path = Path(tmpdir) / "hard_constraints.json"

        plan_path.write_text(DUMMY_PLAN, encoding="utf-8")
        hc_path.write_text(json.dumps(DUMMY_HARD_CONSTRAINTS, indent=2), encoding="utf-8")

        print("Running evaluation pipeline...")
        print(f"  Plan: {len(DUMMY_PLAN)} chars")
        print(f"  Hard constraints: {len(DUMMY_HARD_CONSTRAINTS)} (all 8 categories)")
        n_cc = len(ALL_COMMONSENSE_CONSTRAINT_DEFS)
        print(f"  Commonsense constraints: {n_cc} (from commonsense_constraints.py)")
        print(f"  Audit log → {out_dir.resolve()}/audit_log.jsonl")
        print()

        scorecard = run_evaluation(
            plan_path=str(plan_path),
            hard_constraints_path=str(hc_path),
            output_dir=str(out_dir),
        )

    # ─── Print results ────────────────────────────────────────────────────────

    print("=" * 60)
    print("SCORECARD")
    print("=" * 60)
    n_urls = len(scorecard.url_verifications)
    if n_urls:
        url_summary = (
            f"{scorecard.url_pass_count} PASS, "
            f"{scorecard.url_fail_count} FAIL, "
            f"{scorecard.url_missing_count} MISSING_INFO"
            f"  (of {n_urls} URL{'s' if n_urls != 1 else ''})"
        )
    else:
        url_summary = "no URLs in plan"

    print(f"URL Verification    : {url_summary}")
    print(f"HC  Micro Pass Rate : {scorecard.hc_micro_pass_rate:.1%}   (fraction of individual HC constraints that PASS)")
    print(f"CC  Micro Pass Rate : {scorecard.cc_micro_pass_rate:.1%}   (fraction of individual CC constraints that PASS)")
    print(f"HC  Macro Pass Rate : {scorecard.hc_macro_pass_rate:.0%}    (1 only if every HC passes — the acceptance gate)")
    print(f"CC  Macro Pass Rate : {scorecard.cc_macro_pass_rate:.0%}    (1 only if every CC passes — the acceptance gate)")
    print(f"Final Pass Rate     : {scorecard.final_pass_rate:.0%}    (1 only if both HC Macro and CC Macro are 1)")
    print()

    if scorecard.url_verifications:
        print("URL Verification Results:")
        for uv in scorecard.url_verifications:
            title = f" ({uv.fetched_title})" if uv.fetched_title else ""
            print(f"  [{uv.verdict:12s}] {uv.url}{title}")
            if uv.verdict == "MISSING_INFO" and uv.claims_checked:
                print(f"               Reason : {uv.reasoning[:120]}")
                print(f"               Could not verify:")
                for claim in uv.claims_checked:
                    print(f"                 • {claim}")
            elif uv.verdict == "FAIL" and uv.claims_checked:
                print(f"               Reason : {uv.reasoning[:120]}")
                print(f"               Claims checked:")
                for claim in uv.claims_checked:
                    print(f"                 • {claim}")
            elif uv.verdict == "PASS" and uv.claims_checked:
                print(f"               Verified:")
                for claim in uv.claims_checked:
                    print(f"                 • {claim}")
        print()

    short_names = [m.split("/")[-1] for m in scorecard.judge_models]
    print("Per-constraint verdicts:")
    for c in scorecard.aggregated_constraints:
        votes = " | ".join(f"{name}={v}" for name, v in zip(short_names, c.judge_verdicts))
        status = f"[{c.final_verdict:12s}]"
        print(f"  {status} {c.id}: {c.constraint_text[:65]}")
        print(f"  {' ' * 15} {votes}")

    print()
    print(f"Judges: {', '.join(scorecard.judge_models)}")
    print(f"Timestamp: {scorecard.timestamp}")
    print()
    print("Expected violations:")
    print("  HC-5 (budget) → FAIL: plan total €2286 exceeds €2000 limit")
    print("  CC-12 (geographically sensible per day) → FAIL: Paris café on Barcelona day")
    print("  CC-10 / CC-22 (total cost) → FAIL: €2286 > €2000")
    print()
    print(f"Inspect raw judge output: {out_dir.resolve()}/audit_log.jsonl")


if __name__ == "__main__":
    main()
