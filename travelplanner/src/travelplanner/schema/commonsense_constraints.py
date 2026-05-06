from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from travelplanner.schema.system_state import ConstraintModel

CheckedBy = Literal["constraint_agent", "planner_agent", "reviewer_agent", "domain_agent"]
CheckType = Literal["deterministic", "heuristic"]


@dataclass(frozen=True)
class CommonsenseConstraintDef:
    text: str
    checked_by: CheckedBy
    required_categories: frozenset[str] = field(default_factory=frozenset)
    check_type: CheckType = "heuristic"


ALL_COMMONSENSE_CONSTRAINT_DEFS: list[CommonsenseConstraintDef] = [
    # ── Constraint Iteration Agent ──────────────────────────────────────────────
    # These constraints are checked against the hard constraints gathered from the
    # user's query. Only evaluated when their required_categories are available.
    CommonsenseConstraintDef(
        text="Trip start date must be in the future.",
        checked_by="constraint_agent",
        required_categories=frozenset({"travel_dates"}),
        check_type="deterministic",
    ),
    CommonsenseConstraintDef(
        text="Trip end date must be after the trip start date.",
        checked_by="constraint_agent",
        required_categories=frozenset({"travel_dates"}),
        check_type="deterministic",
    ),
    CommonsenseConstraintDef(
        text="Trip duration must be at least 1 day.",
        checked_by="constraint_agent",
        required_categories=frozenset({"travel_dates"}),
        check_type="deterministic",
    ),
    CommonsenseConstraintDef(
        text="Budget must be a positive value.",
        checked_by="constraint_agent",
        required_categories=frozenset({"budget"}),
        check_type="deterministic",
    ),
    CommonsenseConstraintDef(
        text="Car or road trip transport is not feasible for island or overseas destinations — a flight or ferry is required.",
        checked_by="constraint_agent",
        required_categories=frozenset({"transport", "destination"}),
        check_type="heuristic",
    ),

    # ── Planner Agent ──────────────────────────────────────────────────────────
    # Checked once a day-by-day plan outline exists.
    CommonsenseConstraintDef(
        text="Both outbound and return transport must appear in the plan.",
        checked_by="planner_agent",
        required_categories=frozenset({"transport"}),
    ),
    CommonsenseConstraintDef(
        text="Accommodation must be planned for every night of the trip.",
        checked_by="planner_agent",
        required_categories=frozenset({"accommodation", "travel_dates"}),
    ),
    CommonsenseConstraintDef(
        text="Every day must contain at least one planned activity.",
        checked_by="planner_agent",
        required_categories=frozenset({"travel_dates"}),
    ),
    CommonsenseConstraintDef(
        text="Arrival day and departure day must have reduced activity load to account for travel time.",
        checked_by="planner_agent",
        required_categories=frozenset({"travel_dates", "transport"}),
    ),
    CommonsenseConstraintDef(
        text="Total estimated cost must not exceed the stated budget.",
        checked_by="planner_agent",
        required_categories=frozenset({"budget"}),
    ),
    CommonsenseConstraintDef(
        text="If no budget was stated, cost estimates must still be surfaced for user awareness.",
        checked_by="planner_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Activities should be geographically sensible per day (no unnecessary cross-city travel).",
        checked_by="planner_agent",
        required_categories=frozenset({"destination"}),
    ),
    CommonsenseConstraintDef(
        text="Group size must be respected in accommodation capacity.",
        checked_by="planner_agent",
        required_categories=frozenset({"accommodation"}),
    ),

    # ── Reviewer Agent ─────────────────────────────────────────────────────────
    # Checked once a full timestamped itinerary exists.
    CommonsenseConstraintDef(
        text="No two activities on the same day may overlap in time.",
        checked_by="reviewer_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Travel time between consecutive activities must account for realistic transit duration.",
        checked_by="reviewer_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Outbound transport must depart on the trip start date.",
        checked_by="reviewer_agent",
        required_categories=frozenset({"travel_dates", "transport"}),
    ),
    CommonsenseConstraintDef(
        text="Return transport must depart on the trip end date.",
        checked_by="reviewer_agent",
        required_categories=frozenset({"travel_dates", "transport"}),
    ),
    CommonsenseConstraintDef(
        text="Return transport must be scheduled after the last planned activity.",
        checked_by="reviewer_agent",
        required_categories=frozenset({"transport"}),
    ),
    CommonsenseConstraintDef(
        text="Flight or train connections must allow sufficient layover time (≥60 min domestic, ≥90 min international).",
        checked_by="reviewer_agent",
        required_categories=frozenset({"transport"}),
    ),
    CommonsenseConstraintDef(
        text="Attractions and restaurants must be open on the planned day and time.",
        checked_by="reviewer_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Popular venues that require advance booking must be flagged.",
        checked_by="reviewer_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Total final cost must not exceed the stated budget.",
        checked_by="reviewer_agent",
        required_categories=frozenset({"budget"}),
    ),

    # ── Domain Agents (future) ─────────────────────────────────────────────────
    # Checked at booking time by the respective domain agent.
    CommonsenseConstraintDef(
        text="Hotel check-in must be scheduled after arrival transport reaches the destination.",
        checked_by="domain_agent",
        required_categories=frozenset({"accommodation", "transport"}),
    ),
    CommonsenseConstraintDef(
        text="Hotel check-out must be scheduled before or at the time of departure transport.",
        checked_by="domain_agent",
        required_categories=frozenset({"accommodation", "transport"}),
    ),
    CommonsenseConstraintDef(
        text="Room capacity must accommodate the number of travelers.",
        checked_by="domain_agent",
        required_categories=frozenset({"accommodation"}),
    ),
    CommonsenseConstraintDef(
        text="Outbound transport must depart from the correct origin city.",
        checked_by="domain_agent",
        required_categories=frozenset({"transport", "destination"}),
    ),
    CommonsenseConstraintDef(
        text="Return transport must depart from the correct destination city.",
        checked_by="domain_agent",
        required_categories=frozenset({"transport", "destination"}),
    ),
    CommonsenseConstraintDef(
        text="Activities must fall within stated opening hours.",
        checked_by="domain_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Activities requiring advance booking must be reserved.",
        checked_by="domain_agent",
        required_categories=frozenset(),
    ),
    CommonsenseConstraintDef(
        text="Activities must be appropriate for all travelers (e.g. age restrictions with children in the group).",
        checked_by="domain_agent",
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
