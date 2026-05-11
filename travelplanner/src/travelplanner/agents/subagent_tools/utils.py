"""Shared formatting helpers used by sub-agent tool wrappers."""

from __future__ import annotations

from travelplanner.schema.flight_search_artifact import (
    FlightSearchArtifactContentModel,
)
from travelplanner.schema.hotel_search_artifact import (
    HotelSearchArtifactContentModel,
)
from travelplanner.schema.restaurant_search_artifact import (
    RestaurantArtifactContentModel,
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


def summarize_hotel_artifact(artifact: AgentArtifactModel) -> str:
    """Render a hotel-search artifact as a compact text summary for an LLM."""
    content = HotelSearchArtifactContentModel.model_validate(artifact.content)

    params = content.search_parameters
    lines = [
        f"Hotel search | {params.location} | {params.check_in_date} to {params.check_out_date} ({params.nights} nights)",
        f"Guests: {params.guest_count} | Budget: {params.budget_max} EUR/night | Status: {content.status}",
    ]

    for err in content.errors:
        lines.append(f"  Error [{err.code}]: {err.message}")

    if content.options:
        lines.append(f"Top hotels ({len(content.options)}):")
        for hotel in content.options:
            budget_note = " (over budget)" if hotel.over_budget else ""
            lines.append(
                f"  {hotel.rank or '?'}. {hotel.name} – {hotel.currency} {hotel.nightly_rate:.0f}/night"
                f" | Rating: {hotel.rating}/10{budget_note}"
            )
            if hotel.area:
                lines.append(f"     Area: {hotel.area}")
            if hotel.facilities:
                lines.append(f"     Facilities: {', '.join(hotel.facilities[:8])}")

    if not content.options and content.status != "failed":
        lines.append("No hotels found for the given criteria.")

    return "\n".join(lines)


def summarize_restaurant_artifact(artifact: AgentArtifactModel) -> str:
    """Render a restaurant-search artifact as a compact text summary for an LLM."""
    content = RestaurantArtifactContentModel.model_validate(artifact.content)

    lines = [
        f"Restaurant search | {content.city}",
        f"Query: {content.query} | Cuisine: {content.cuisine or 'any'} | Budget: {content.budget or 'not specified'} | Meal: {content.meal_type or 'any'} | Status: {content.status}",
    ]

    for err in content.errors:
        lines.append(f"  Error [{err.code}]: {err.message}")

    if content.items:
        lines.append(f"Selected restaurants ({len(content.items)}):")
        for item in content.items:
            lines.append(
                f"  {item.name} | Rating: {item.rating or 'N/A'} | Price: {item.price_range or item.price_level or 'N/A'}"
            )
            if item.address:
                lines.append(f"     Address: {item.address}")
            if item.cuisine:
                lines.append(f"     Cuisine: {item.cuisine}")
            if item.opening_hours:
                lines.append(f"     Hours: {item.opening_hours}")
            if item.selection_reason:
                lines.append(f"     Why: {item.selection_reason}")
            if item.dietary_suitability:
                lines.append(f"     Dietary match: {', '.join(item.dietary_suitability)}")

    if not content.items and content.status != "failed":
        lines.append("No restaurants found for the given criteria.")

    return "\n".join(lines)
