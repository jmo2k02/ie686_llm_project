from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from travelplanner.schema.system_state import ConstraintModel

CheckedBy = Literal[
    "constraint_agent",
    "flight_agent",
    "hotel_agent",
    "restaurant_agent",
    "attraction_agent",
    "routing_agent",
]
CheckType = Literal["deterministic", "heuristic"]


@dataclass(frozen=True)
class CommonsenseConstraintDef:
    text: str
    checked_by: CheckedBy
    required_categories: frozenset[str] = field(default_factory=frozenset)
    check_type: CheckType = "heuristic"


ALL_COMMONSENSE_CONSTRAINT_DEFS: list[CommonsenseConstraintDef] = [
    # ── Constraint Agent ───────────────────────────────────────────────────────
    # Checked before any domain agent runs, against the user's raw constraints.
    CommonsenseConstraintDef(
        text="Car or road trip transport is not feasible for island or overseas destinations — a flight or ferry is required.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    CommonsenseConstraintDef(
        text="No key information should be left out of the plan, such as the lack of accommodation during travel.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    CommonsenseConstraintDef(
        text="Transportation choices within the trip must be reasonable. For example, having both “self-driving” and “flight” would be considered a conflict.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    # ── Flight Agent ───────────────────────────────────────────────────────────
    # Checked when searching for and validating transport options.
    CommonsenseConstraintDef(
        text="Both outbound and return transport must appear in the plan.",
        checked_by="flight_agent",
        required_categories=frozenset({"transport"}),
    ),
    
    CommonsenseConstraintDef(
        text="Flight or train connections must allow sufficient layover time (≥60 min domestic, ≥90 min international).",
        checked_by="flight_agent",
        required_categories=frozenset({"transport"}),
    ),
    # ── Hotel Agent ────────────────────────────────────────────────────────────
    # Checked when searching for and validating accommodation options.
    CommonsenseConstraintDef(
        text="Accommodation must be planned for every night of the trip.",
        checked_by="hotel_agent",
        required_categories=frozenset({"accommodation", "travel_dates"}),
    ),
    
    CommonsenseConstraintDef(
        text="Hotel check-in must be scheduled after arrival transport reaches the destination.",
        checked_by="hotel_agent",
        required_categories=frozenset({"accommodation", "transport"}),
    ),
    # ── Restaurant Agent ───────────────────────────────────────────────────────
    # Checked when searching for and scheduling dining options.
    CommonsenseConstraintDef(
        text="Restaurants must be open on the planned day and time.",
        checked_by="restaurant_agent",
        required_categories=frozenset(),
    ),
    
    CommonsenseConstraintDef(
        text="Restaurant choices should not be repeated throughout the trip.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    # ── Attraction Agent ───────────────────────────────────────────────────────
    # Checked when searching for and scheduling activities and sights.
    CommonsenseConstraintDef(
        text="Activities should be geographically sensible per day (no unnecessary cross-city travel).",
        checked_by="attraction_agent",
        required_categories=frozenset({"destination"}),
    ),
    CommonsenseConstraintDef(
        text="Attractions must be open on the planned day and time.",
        checked_by="attraction_agent",
        required_categories=frozenset(),
    ),

    CommonsenseConstraintDef(
        text="Attraction choices should not be repeated throughout the trip.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    # ── Routing Agent ──────────────────────────────────────────────────────────
    # Checked when validating the assembled day-by-day schedule and overall costs.
    CommonsenseConstraintDef(
        text="Travel time between consecutive activities must account for realistic transit duration.",
        checked_by="routing_agent",
        required_categories=frozenset(),
    ),
]


def get_constraints_for(agent: CheckedBy) -> list[CommonsenseConstraintDef]:
    """Return all constraint definitions owned by the given agent."""
    return [c for c in ALL_COMMONSENSE_CONSTRAINT_DEFS if c.checked_by == agent]


# Constraint iteration agent constraints as ConstraintModel list (for state initialisation).
COMMONSENSE_CONSTRAINTS: list[ConstraintModel] = [
    ConstraintModel(type="commonsense", text=c.text, user_skipped=False)
    for c in get_constraints_for("constraint_agent")
]
