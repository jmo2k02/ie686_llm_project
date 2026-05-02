# scratch_param_extraction.py — LLM parameter extraction test
# Tests whether the agent correctly parses natural language into FlightParamsModel.
# Run from project root: python scratch_param_extraction.py
from travelplanner.agents.flight_search_agent import (
    _extract_flight_params,
    load_config_from_env,
)

MODEL = "openrouter:minimax/minimax-m2.5"
TEMPERATURE = 0.0

config = load_config_from_env()

CASES = [
    {
        "label": "Round trip — cities, explicit return",
        "text": "Find a round trip flight from Munich to London on June 10 2026, returning June 17 2026, 2 adults.",
    },
    {
        "label": "One way — cities, no return",
        "text": "I need a one-way flight from Frankfurt to New York on July 5 2026 for 1 adult.",
    },
    {
        "label": "Multi-city — three legs",
        "text": "Book a multi-city trip: Berlin to Paris on May 20 2026, then Paris to Rome on May 25 2026, then Rome back to Berlin on June 1 2026.",
    },
    {
        "label": "Round trip — ambiguous city names",
        "text": "Fly from Vienna to Tallinn on August 3 2026, return August 10 2026.",
    },
    {
        "label": "One way — informal phrasing",
        "text": "Get me a flight to Barcelona from Amsterdam, departing 2026-09-15, just one passenger.",
    },
]


def fmt_params(p) -> str:
    lines = [
        f"  trip_type : {p.trip_type}  ({'round trip' if p.trip_type == 1 else 'one way' if p.trip_type == 2 else 'multi-city'})",
        f"  adults    : {p.adults}",
        f"  currency  : {p.currency}",
    ]
    for i, seg in enumerate(p.segments):
        prefix = f"  seg {i + 1}    :" if len(p.segments) > 1 else "  segment  :"
        lines.append(f"{prefix} {seg.departure_id} → {seg.arrival_id} on {seg.outbound_date}")
    if p.return_date:
        lines.append(f"  return    : {p.return_date}")
    return "\n".join(lines)


pass_count = 0
fail_count = 0

for case in CASES:
    print(f"\n{'─' * 60}")
    print(f"Case: {case['label']}")
    print(f"Input: \"{case['text']}\"")
    try:
        params = _extract_flight_params(case["text"], MODEL, TEMPERATURE, config)
        print(f"Result:\n{fmt_params(params)}")
        pass_count += 1
    except Exception as exc:
        print(f"FAILED: {exc}")
        fail_count += 1

print(f"\n{'═' * 60}")
print(f"Results: {pass_count} passed, {fail_count} failed out of {len(CASES)} cases.")
