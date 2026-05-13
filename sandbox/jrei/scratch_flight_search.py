# scratch_flight_search.py — round-trip test
# Route: MUC → LHR → MUC
# Run from project root: python scratch_flight_search.py
from travelplanner.agents.flight_search_agent import load_config_from_env, run_flight_search
from travelplanner.schema.flight_search_artifact import FlightParamsModel, FlightSegmentParams

config = load_config_from_env()

params = FlightParamsModel(
    trip_type=1,  # round trip
    segments=[
        FlightSegmentParams(departure_id="MUC", arrival_id="VNO", outbound_date="2026-06-24"),
    ],
    return_date="2026-06-26",
    adults=1,
    currency="EUR",
)

print(f"Searching round trip: {params.segments[0].departure_id} → {params.segments[0].arrival_id}")
print(f"Outbound: {params.segments[0].outbound_date}  Return: {params.return_date}")
print(f"Adults: {params.adults}  Currency: {params.currency}\n")

result = run_flight_search(params, config)

if result.errors:
    for err in result.errors:
        print(f"Error [{err.code}]: {err.message}")

print(f"Outbound [{len(result.best_flights)} option(s)]:")
for j, flight in enumerate(result.best_flights):
    print(f"  --- Option {j + 1} ---")
    print(f"  Price: {flight.currency} {flight.price}  |  Duration: {flight.total_duration_minutes} min  |  Carbon: {flight.carbon_emissions_kg} kg")
    for leg in flight.legs:
        print(f"    {leg.airline} {leg.flight_number}: {leg.departure_airport.id} {leg.departure_airport.time} → {leg.arrival_airport.id} {leg.arrival_airport.time}")
    for lv in flight.layovers:
        print(f"    Layover: {lv.name} ({lv.id}) {lv.duration_minutes} min")

if result.return_flights:
    print(f"\nReturn [{len(result.return_flights)} option(s)]:")
    for j, flight in enumerate(result.return_flights):
        print(f"  --- Option {j + 1} ---")
        print(f"  Price: {flight.currency} {flight.price}  |  Duration: {flight.total_duration_minutes} min  |  Carbon: {flight.carbon_emissions_kg} kg")
        for leg in flight.legs:
            print(f"    {leg.airline} {leg.flight_number}: {leg.departure_airport.id} {leg.departure_airport.time} → {leg.arrival_airport.id} {leg.arrival_airport.time}")
        for lv in flight.layovers:
            print(f"    Layover: {lv.name} ({lv.id}) {lv.duration_minutes} min")
else:
    print("\nNo return flights found (departure_token may be missing on outbound options).")

if result.price_insights:
    print(f"\nPrice insights: {result.price_insights.price_level} (typical range: {result.price_insights.typical_price_range})")
