from __future__ import annotations

from travelplanner.schema.system_state import ConstraintModel


COMMONSENSE_CONSTRAINTS: list[ConstraintModel] = [
    # ── Temporal ────────────────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="The trip start date must be in the future.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="The trip end date must be after the trip start date.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="The trip must last at least one full day.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Hotel check-in must be scheduled after the arrival transport reaches the destination.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Hotel check-out must be scheduled before or at the time of the departure transport.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="The return transport must be scheduled after the last planned activity of the trip.",
        user_skipped=False,
    ),
    # ── Sequencing ──────────────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="Outbound transport (flight, train, etc.) must be booked for the trip start date.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Return transport must be booked for the trip end date.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="No two activities on the same day may overlap in time.",
        user_skipped=False,
    ),
    # ── Budget ──────────────────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="The total estimated cost of all bookings must not exceed the stated budget.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="If no budget was specified, cost estimates must still be surfaced for user awareness.",
        user_skipped=False,
    ),
    # ── Feasibility ─────────────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="Travel time between consecutive activities must account for realistic transit duration.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Flight or train connections must allow sufficient transfer time (at least 60 min domestic, 90 min international).",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Restaurants and attractions must be visited within their stated opening hours.",
        user_skipped=False,
    ),
    # ── Completeness ────────────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="Accommodation must be booked for every night of the trip.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Every day of the trip must contain at least one planned activity.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Both outbound and return transport must be planned before the itinerary is finalized.",
        user_skipped=False,
    ),
    # ── Legal / Practical ───────────────────────────────────────────────────
    ConstraintModel(
        type="commonsense",
        text="For international trips, passport validity must extend at least 6 months beyond the return date.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Visa requirements for the destination country must be checked and flagged if applicable.",
        user_skipped=False,
    ),
    ConstraintModel(
        type="commonsense",
        text="Popular attractions or restaurants that require advance booking must be flagged accordingly.",
        user_skipped=False,
    ),
]
