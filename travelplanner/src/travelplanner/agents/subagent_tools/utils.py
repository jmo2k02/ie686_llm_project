"""Shared formatting helpers used by sub-agent tool wrappers."""

from __future__ import annotations

from travelplanner.schema.attraction_search_artifact import (
    AttractionArtifactContentModel,
)
from travelplanner.schema.constraint_artifact import ConstraintArtifactContentModel
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
    trip_label = {1: "round trip", 2: "one way"}.get(trip_type, str(trip_type))

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
        is_return = trip_type == 1 and i == 1
        carbon = (
            f" | carbon: {flight.carbon_emissions_kg} kg"
            if flight.carbon_emissions_kg is not None
            else ""
        )
        if is_return:
            lines.append(f"  Return: {flight.total_duration_minutes} min{carbon}")
        else:
            label = "Round-trip total" if trip_type == 1 else "One-way fare"
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

    if content.google_flights_url:
        lines.append(f"Verify / book: {content.google_flights_url}")

    return "\n".join(lines)


def summarize_attraction_artifact(artifact: AgentArtifactModel) -> str:
    """Render an attraction-search artifact as a compact text summary for an LLM."""
    content = AttractionArtifactContentModel.model_validate(artifact.content)

    lines = [
        f"Destination: {content.destination} | Budget: {content.budget} EUR | "
        f"Archetype: {content.selected_archetype} | Status: {content.status}",
    ]
    
    if content.item:
        item = content.item
        lines.append(
            f"Activity — Day {item.day} {item.time_slot.upper()}: {item.title}"
        )
        lines.append(f"  {item.description}")
        lines.append(f"  Local touchpoint: {item.local_touchpoint}")
        lines.append(
            f"  Duration: {item.estimated_duration_hours}h | "
            f"Est. price: {item.estimated_price_range} | "
            f"Place found: {item.place_found}"
        )
        if item.place_found:
            lines.append(f"  Place: {item.location_name}")
            if item.location_address:
                lines.append(f"    Address: {item.location_address}")
            if item.coordinates:
                lines.append(
                    f"    ({item.coordinates['lat']:.4f}, {item.coordinates['lng']:.4f})"
                )
            if item.place_rating is not None:
                reviews = f" ({item.place_review_count} reviews)" if item.place_review_count else ""
                lines.append(
                    f"    Rating: {item.place_rating}{reviews} | "
                    f"Price: {item.place_price_level or 'N/A'} | "
                    f"Type: {item.place_type or 'N/A'}"
                )
            if item.place_hours:
                lines.append(f"    Hours: {item.place_hours}")
            if item.selection_reason:
                lines.append(f"    Why: {item.selection_reason}")
        else:
            lines.append(f"  No specific place found (location: {item.location_name})")
        lines.append(f"  Provenance: {item.provenance}")
    
    if content.top_candidates:
        lines.append(f"Top candidates [{len(content.top_candidates)}]:")
        for i, c in enumerate(content.top_candidates):
            rating = f" | Rating: {c.rating}" if c.rating is not None else ""
            reviews = f" ({c.reviews} reviews)" if c.reviews else ""
            lines.append(f"  [{i}] {c.title} | {c.address or 'address unknown'}{rating}{reviews}")

    if content.google_maps_url:
        lines.append(f"Verify on Google Maps: {content.google_maps_url}")

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
            if hotel.booking_url:
                lines.append(f"     Book: {hotel.booking_url}")

    if not content.options and content.status != "failed":
        lines.append("No hotels found for the given criteria.")

    if content.booking_url:
        lines.append(f"Manual search (Nuitee): {content.booking_url}")

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
                f"  {item.name} | Rating: {item.rating or 'N/A'} | Price: {item.price_level or 'N/A'}"
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
            link = item.website or item.google_maps_url
            if link:
                lines.append(f"     Link: {link}")

    if not content.items and content.status != "failed":
        lines.append("No restaurants found for the given criteria.")

    return "\n".join(lines)


def summarize_constraint_artifact(
    artifact: AgentArtifactModel,
    violations: list | None = None,
) -> str:
    """Render a constraint-extraction artifact as a compact text summary for an LLM."""
    content = ConstraintArtifactContentModel.model_validate(artifact.content)

    lines = [
        f"Constraint extraction | Query: {content.query} | Status: {content.status}",
    ]

    if content.corrected_query and content.corrected_query != content.query:
        lines.append(f"  Spell-corrected to: {content.corrected_query}")

    if content.normalized_constraints is not None:
        nc = content.normalized_constraints.model_dump(exclude_none=True)
        if nc:
            lines.append("Normalized constraints:")
            for key, val in nc.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        lines.append(f"  {key}.{sub_key}: {sub_val}")
                else:
                    lines.append(f"  {key}: {val}")
    elif content.hard_constraints:
        lines.append("Hard constraints:")
        for c in content.hard_constraints:
            if not c.get("user_skipped"):
                lines.append(f"  - {c.get('text', '')}")

    if content.categories_missing:
        lines.append(f"Missing categories: {', '.join(content.categories_missing)}")

    if violations:
        lines.append(f"\nWarnings ({len(violations)} violation(s) detected):")
        for v in violations:
            lines.append(f"  - {v.violated_constraint}")
            lines.append(f"    Reason: {v.explanation}")
            for suggestion in v.suggestions:
                lines.append(f"    Suggestion: {suggestion}")
    else:
        lines.append("No constraint violations detected.")

    return "\n".join(lines)
