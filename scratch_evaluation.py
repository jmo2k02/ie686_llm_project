"""End-to-end smoke test for the LLM-as-a-Judge evaluation pipeline.

Runs two scenarios so we exercise both input adapters:

1. **Baseline scenario** — feeds a markdown itinerary (the format produced by the
   baseline single-agent system) and lets the pipeline convert it to a TravelPlan
   via a structured LLM call.
2. **Travelplanner scenario** — feeds a TravelPlan JSON (the format produced by
   the multi-agent travelplanner pipeline) directly.

Hard constraints follow the constraint_iteration_agent format:
  {"type": "hard", "text": "category: value", "user_skipped": bool}

Intentional violations (both scenarios):
  - HC-5 (budget): total €2286 > €2000 limit
  - CC (geographically sensible): Day 2 lists a Paris café while traveler is in Barcelona
  - CC (total cost): €2286 > €2000

Per-scenario audit log and scorecard are written to ./eval_output/{baseline,travelplan}/.

Usage:
    python scratch_evaluation.py
"""

import json
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from travelplanner.evaluation.judge_setup import reset_log_clock, run_evaluation
from travelplanner.schema.commonsense_constraints import ALL_COMMONSENSE_CONSTRAINT_DEFS
from travelplanner.schema.judge_artifact import ScorecardModel
from travelplanner.travelplan.day import Day
from travelplanner.travelplan.plan import TravelPlan
from travelplanner.travelplan.slot import Slot

# ─── Dummy inputs ─────────────────────────────────────────────────────────────

# 4-day / 3-night trip with explicit dates in every day header.
# Deliberate violations:
#   - Day 2: Café de Flore (Paris) listed on a Barcelona day  → geographic CC fail
#   - Total: €2286 > €2000 budget                             → HC-5 + cost CC fail
DUMMY_PLAN_MARKDOWN = """
# Barcelona Trip — 4 Days / 3 Nights (2026-06-10 to 2026-06-13)

## Day 1 — 2026-06-10 (Arrival, Barcelona)
- **Flight**: Munich (MUC) → Barcelona (BCN), Lufthansa LH1234, departs 08:00, arrives 10:00. Cost: €420.
- **Hotel**: Hotel Arts Barcelona, check-in 14:00. Rate: €400/night × 3 nights = €1200.
  Book at: https://www.hotelartsbarcelona.com
- **Dinner**: Tickets (avant-garde tapas, Barcelona). €80.

## Day 2 — 2026-06-11 (Sightseeing, Barcelona)
- **Breakfast**: Café de Flore, Paris. €20.  ← DELIBERATE CITY VIOLATION
- **Activity**: Sagrada Família guided tour, 10:00–12:30. €30.
  Tickets: https://sagradafamilia.org/en/tickets-individuals
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


def _dt(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm)


def build_dummy_travelplan() -> TravelPlan:
    """Build a TravelPlan Pydantic instance mirroring DUMMY_PLAN_MARKDOWN."""
    d1 = date(2026, 6, 10)
    d2 = date(2026, 6, 11)
    d3 = date(2026, 6, 12)
    d4 = date(2026, 6, 13)

    day1 = Day(
        index=1,
        calendar_date=d1,
        label="Arrival, Barcelona",
        slots=[
            Slot(
                name="Flight MUC → BCN (Lufthansa LH1234)",
                description="Outbound flight Munich to Barcelona. Cost €420.",
                start_time=_dt(d1, 8),
                end_time=_dt(d1, 10),
                category="transport",
                location="Munich (MUC) → Barcelona (BCN)",
                cost=420.0,
                links=[],
            ),
            Slot(
                name="Hotel Arts Barcelona check-in",
                description="3-night stay at €400/night = €1200.",
                start_time=_dt(d1, 14),
                end_time=_dt(d1, 15),
                category="lodging",
                location="Barcelona",
                cost=1200.0,
                links=["https://www.hotelartsbarcelona.com"],
            ),
            Slot(
                name="Dinner at Tickets",
                description="Avant-garde tapas in Barcelona.",
                start_time=_dt(d1, 20),
                end_time=_dt(d1, 22),
                category="meal",
                location="Barcelona",
                cost=80.0,
            ),
        ],
    )

    day2 = Day(
        index=2,
        calendar_date=d2,
        label="Sightseeing, Barcelona",
        slots=[
            # DELIBERATE GEOGRAPHY VIOLATION
            Slot(
                name="Breakfast at Café de Flore",
                description="Café de Flore is located in Paris, Saint-Germain-des-Prés.",
                start_time=_dt(d2, 8),
                end_time=_dt(d2, 9),
                category="meal",
                location="Paris",  # ← violation: should be in Barcelona
                cost=20.0,
            ),
            Slot(
                name="Sagrada Família guided tour",
                description="Guided tour of Gaudí's basilica. Ticket price €30.",
                start_time=_dt(d2, 10),
                end_time=_dt(d2, 12, 30),
                category="attraction",
                location="Barcelona",
                cost=30.0,
                links=["https://sagradafamilia.org/en/tickets-individuals"],
            ),
            Slot(
                name="Lunch at Bar del Pla",
                description="Catalan tapas lunch.",
                start_time=_dt(d2, 13),
                end_time=_dt(d2, 14, 30),
                category="meal",
                location="Barcelona",
                cost=25.0,
            ),
            Slot(
                name="Picasso Museum",
                description="Permanent collection visit.",
                start_time=_dt(d2, 15),
                end_time=_dt(d2, 17),
                category="attraction",
                location="Barcelona",
                cost=15.0,
            ),
            Slot(
                name="Dinner at Bodega Sepúlveda",
                description="Traditional Catalan dinner.",
                start_time=_dt(d2, 20),
                end_time=_dt(d2, 22),
                category="meal",
                location="Barcelona",
                cost=60.0,
            ),
        ],
    )

    day3 = Day(
        index=3,
        calendar_date=d3,
        label="Sightseeing, Barcelona",
        slots=[
            Slot(
                name="Breakfast at Federal Café",
                description="Australian-style brunch spot in Sant Antoni.",
                start_time=_dt(d3, 8),
                end_time=_dt(d3, 9),
                category="meal",
                location="Barcelona",
                cost=15.0,
            ),
            Slot(
                name="Montjuïc Cable Car",
                description="Scenic cable car ride up Montjuïc.",
                start_time=_dt(d3, 10),
                end_time=_dt(d3, 11, 30),
                category="attraction",
                location="Barcelona",
                cost=14.0,
            ),
            Slot(
                name="Lunch at La Boqueria market",
                description="Market stalls lunch.",
                start_time=_dt(d3, 13),
                end_time=_dt(d3, 14),
                category="meal",
                location="Barcelona",
                cost=20.0,
            ),
            Slot(
                name="Dinner at La Mar Salada",
                description="Barceloneta seafood dinner.",
                start_time=_dt(d3, 20),
                end_time=_dt(d3, 22),
                category="meal",
                location="Barcelona",
                cost=50.0,
            ),
        ],
    )

    day4 = Day(
        index=4,
        calendar_date=d4,
        label="Departure",
        slots=[
            Slot(
                name="Breakfast at Federal Café",
                description="Quick breakfast before checkout.",
                start_time=_dt(d4, 8),
                end_time=_dt(d4, 9),
                category="meal",
                location="Barcelona",
                cost=12.0,
            ),
            Slot(
                name="Flight BCN → MUC (Vueling VY6301)",
                description="Return flight. Cost €320.",
                start_time=_dt(d4, 17),
                end_time=_dt(d4, 19),
                category="transport",
                location="Barcelona (BCN) → Munich (MUC)",
                cost=320.0,
            ),
        ],
    )

    return TravelPlan(
        title="Barcelona Trip — 4 Days / 3 Nights",
        days=[day1, day2, day3, day4],
    )


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

def _print_scorecard(scorecard: ScorecardModel, scenario_label: str) -> None:
    print("=" * 60)
    print(f"SCORECARD — {scenario_label}")
    print("=" * 60)
    n_rv = len(scorecard.rationale_verifications)
    if n_rv:
        rv_summary = (
            f"{scorecard.rationale_pass_count} PASS, "
            f"{scorecard.rationale_fail_count} FAIL, "
            f"{scorecard.rationale_missing_count} MISSING_INFO"
            f"  (of {n_rv} slot{'s' if n_rv != 1 else ''})"
        )
    else:
        rv_summary = "no slots verified"

    print(f"Rationale verification : {rv_summary}")
    print(f"HC  Micro Pass Rate    : {scorecard.hc_micro_pass_rate:.1%}   (fraction of individual HC constraints that PASS)")
    print(f"CC  Micro Pass Rate    : {scorecard.cc_micro_pass_rate:.1%}   (fraction of individual CC constraints that PASS)")
    print(f"HC  Macro Pass Rate    : {scorecard.hc_macro_pass_rate:.0%}    (1 only if every HC passes — the acceptance gate)")
    print(f"CC  Macro Pass Rate    : {scorecard.cc_macro_pass_rate:.0%}    (1 only if every CC passes — the acceptance gate)")
    print(f"Final Pass Rate        : {scorecard.final_pass_rate:.0%}    (1 only if both HC Macro and CC Macro are 1)")
    print()

    if scorecard.rationale_verifications:
        print("Rationale Verification Results:")
        for rv in scorecard.rationale_verifications:
            tag = f"[{rv.verdict:12s}]"
            print(f"  {tag} Day {rv.day_index}/slot {rv.slot_position} — {rv.slot_name}  (source: {rv.source_type})")
            if rv.source_urls:
                for url in rv.source_urls[:3]:
                    print(f"               url: {url}")
            if rv.claims_checked:
                print(f"               claims checked:")
                for claim in rv.claims_checked:
                    print(f"                 • {claim}")
            if rv.verdict != "PASS" and rv.reasoning:
                print(f"               reason: {rv.reasoning[:140]}")
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


def _run_baseline_scenario(hc_path: Path) -> ScorecardModel:
    out_dir = Path("eval_output/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_log_clock()
    print(
        f"\n===== BASELINE scenario start @ {datetime.now().strftime('%H:%M:%S')} =====",
        file=sys.stderr,
        flush=True,
    )
    scenario_t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.md"
        plan_path.write_text(DUMMY_PLAN_MARKDOWN, encoding="utf-8")

        print(">>> BASELINE scenario: markdown input")
        print(f"    Plan: {len(DUMMY_PLAN_MARKDOWN)} chars (markdown)")
        n_cc = len(ALL_COMMONSENSE_CONSTRAINT_DEFS)
        print(f"    Hard constraints   : {len(DUMMY_HARD_CONSTRAINTS)}")
        print(f"    Commonsense rules  : {n_cc}")
        print(f"    Audit log          → {out_dir.resolve()}/audit_log.jsonl")
        print()

        result = run_evaluation(
            plan_path=str(plan_path),
            hard_constraints_path=str(hc_path),
            output_dir=str(out_dir),
            plan_format="markdown",
        )
    print(
        f"===== BASELINE scenario done in {time.monotonic() - scenario_t0:.1f}s =====\n",
        file=sys.stderr,
        flush=True,
    )
    return result


def _run_travelplanner_scenario(hc_path: Path) -> ScorecardModel:
    out_dir = Path("eval_output/travelplan")
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_log_clock()
    print(
        f"\n===== TRAVELPLANNER scenario start @ {datetime.now().strftime('%H:%M:%S')} =====",
        file=sys.stderr,
        flush=True,
    )
    scenario_t0 = time.monotonic()
    plan = build_dummy_travelplan()
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        print(">>> TRAVELPLANNER scenario: TravelPlan JSON input")
        print(f"    Plan: {len(plan.days)} days, {sum(len(d.slots) for d in plan.days)} slots")
        n_cc = len(ALL_COMMONSENSE_CONSTRAINT_DEFS)
        print(f"    Hard constraints   : {len(DUMMY_HARD_CONSTRAINTS)}")
        print(f"    Commonsense rules  : {n_cc}")
        print(f"    Audit log          → {out_dir.resolve()}/audit_log.jsonl")
        print()

        result = run_evaluation(
            plan_path=str(plan_path),
            hard_constraints_path=str(hc_path),
            output_dir=str(out_dir),
            plan_format="json",
        )
    print(
        f"===== TRAVELPLANNER scenario done in {time.monotonic() - scenario_t0:.1f}s =====\n",
        file=sys.stderr,
        flush=True,
    )
    return result


def main() -> None:
    Path("eval_output").mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        hc_path = Path(tmpdir) / "hard_constraints.json"
        hc_path.write_text(json.dumps(DUMMY_HARD_CONSTRAINTS, indent=2), encoding="utf-8")

        baseline_scorecard = _run_baseline_scenario(hc_path)
        travelplan_scorecard = _run_travelplanner_scenario(hc_path)

    _print_scorecard(baseline_scorecard, "Baseline (markdown input)")
    _print_scorecard(travelplan_scorecard, "Travelplanner (JSON input)")

    print("Expected violations (both scenarios):")
    print("  HC-5 (budget) → FAIL: plan total €2286 exceeds €2000 limit")
    print("  CC (geographically sensible per day) → FAIL: Paris café on Barcelona day")
    print("  CC (total cost vs budget) → FAIL: €2286 > €2000")
    print()
    print(f"Inspect raw judge output:")
    print(f"  baseline      : {(Path('eval_output/baseline').resolve())}/audit_log.jsonl")
    print(f"  travelplanner : {(Path('eval_output/travelplan').resolve())}/audit_log.jsonl")


if __name__ == "__main__":
    main()
