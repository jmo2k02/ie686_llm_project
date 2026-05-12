# scratch_e2e.py — end-to-end: plain text → param extraction → flight search → results
# Edit QUERY below and run: python scratch_e2e.py

from travelplanner.agents.flight_search_agent import (
    _extract_flight_params,
    load_config_from_env,
    run_flight_search,
)

QUERY = "I want to travel from Munich to Sydney from June 24 2026 until July 16 2026."
MODEL = "openrouter:minimax/minimax-m2.5"
TEMPERATURE = 0.0

# ── Step 1: extract structured params from plain text ────────────────────────

config = load_config_from_env()

print(f'Query: "{QUERY}"\n')
print("Extracting parameters...")
params = _extract_flight_params(QUERY, MODEL, TEMPERATURE, config)

trip_label = {1: "round trip", 2: "one way"}.get(params.trip_type, str(params.trip_type))
seg = params.segments[0]
route = f"{seg.departure_id} → {seg.arrival_id} on {seg.outbound_date}"
if params.return_date:
    route += f"  (return {params.return_date})"
print(f"  {trip_label}  |  {route}  |  adults: {params.adults}  currency: {params.currency}\n")

# ── Step 2: search flights ────────────────────────────────────────────────────

print("Searching flights...")
result = run_flight_search(params, config)

if result.errors:
    for err in result.errors:
        print(f"  Error [{err.code}]: {err.message}")

# ── Step 3: selected flights (what the orchestrator receives) ─────────────────

def print_flight(flight, *, show_price: bool = True) -> None:
    if show_price:
        print(f"  Price: {flight.currency} {flight.price}  |  Duration: {flight.total_duration_minutes} min  |  Carbon: {flight.carbon_emissions_kg} kg")
    else:
        print(f"  Duration: {flight.total_duration_minutes} min  |  Carbon: {flight.carbon_emissions_kg} kg")
    for leg in flight.legs:
        print(f"    {leg.airline} {leg.flight_number}: {leg.departure_airport.id} {leg.departure_airport.time} → {leg.arrival_airport.id} {leg.arrival_airport.time}")
    for lv in flight.layovers:
        print(f"    Layover: {lv.name} ({lv.id}) {lv.duration_minutes} min")


print(f"\n{'━' * 60}")
print(f"SELECTED FLIGHTS  [{len(result.selected_flights)} flight(s) committed to orchestrator]")
print(f"{'━' * 60}")
for i, flight in enumerate(result.selected_flights):
    is_return = params.trip_type == 1 and i == 1
    if is_return:
        label = "Return (included in round-trip price)"
    elif params.trip_type == 1:
        label = "Round-trip total"
    else:
        label = "One-way fare"
    print(f"\n{label}:")
    print_flight(flight, show_price=not is_return)

if result.price_insights:
    print(f"\nPrice insights: {result.price_insights.price_level}  (typical range: {result.price_insights.typical_price_range})")

if result.google_flights_url:
    print(f"\nVerify / book: {result.google_flights_url}")
