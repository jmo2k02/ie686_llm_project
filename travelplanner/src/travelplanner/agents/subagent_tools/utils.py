"""Shared formatting helpers used by sub-agent tool wrappers."""

from __future__ import annotations

from travelplanner.schema.flight_search_artifact import (
    FlightSearchArtifactContentModel,
)
from travelplanner.schema.system_state import AgentArtifactModel


def summarize_flight_artifact(artifact: AgentArtifactModel) -> str:
    """Render a flight-search artifact as a compact text summary for an LLM.

    One block per artifact: route + status + selected flights (cheapest per
    direction) with legs, layovers, carbon, and price-insight context.
    """
    content = FlightSearchArtifactContentModel.model_validate(artifact.content)

    trip_type = content.config.get("trip_type")
    trip_label = {1: "round trip", 2: "one way", 3: "multi-city"}.get(
        trip_type, str(trip_type)
    )

    if trip_type == 3:
        route = " → ".join(
            f"{legs[0].legs[0].departure_airport.id}→{legs[0].legs[-1].arrival_airport.id}"
            for legs in content.multi_city_legs
            if legs
        ) or f"{content.departure_id} (multi-city)"
    else:
        route = f"{content.departure_id} → {content.arrival_id} on {content.outbound_date}"
        if content.return_date:
            route += f" (return {content.return_date})"

    lines = [
        f"{trip_label} | {route} | adults: {content.adults} | currency: {content.currency}",
        f"Status: {content.status}",
    ]
    for err in content.errors:
        lines.append(f"  Error [{err.code}]: {err.message}")

    lines.append(f"Selected flights ({len(content.selected_flights)}):")
    for i, flight in enumerate(content.selected_flights):
        if trip_type == 3:
            label = f"Leg {i + 1}"
        else:
            label = "Outbound" if i == 0 else "Return"
        carbon = (
            f" | carbon: {flight.carbon_emissions_kg} kg"
            if flight.carbon_emissions_kg is not None
            else ""
        )
        lines.append(
            f"  {label}: {flight.currency} {flight.price} | "
            f"{flight.total_duration_minutes} min{carbon}"
        )
        for leg in flight.legs:
            lines.append(
                f"    {leg.airline} {leg.flight_number}: "
                f"{leg.departure_airport.id} {leg.departure_airport.time} → "
                f"{leg.arrival_airport.id} {leg.arrival_airport.time}"
            )
        for lv in flight.layovers:
            lines.append(
                f"    Layover: {lv.name} ({lv.id}) {lv.duration_minutes} min"
            )

    if content.price_insights:
        lines.append(
            f"Price insights: {content.price_insights.price_level} "
            f"(typical range: {content.price_insights.typical_price_range})"
        )

    return "\n".join(lines)
