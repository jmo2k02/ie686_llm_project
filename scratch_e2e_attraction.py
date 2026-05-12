# scratch_e2e_attraction.py — end-to-end: structured task text → param extraction → archetype → LLM activity → SERPAPI place
# Edit TASK_TEXT below and run: python scratch_e2e_attraction.py
# To constrain the time slot, mention it naturally in TASK_TEXT (e.g. "a morning activity").

from dotenv import load_dotenv

from travelplanner.agents.attraction_search_agent import (
    _extract_attraction_params,
    load_config_from_env,
    run_attraction_search,
)

load_dotenv()

TASK_TEXT = """\
Find anactivity for one person visiting Barcelona on Day 2 of their trip,
with a budget of 80 EUR. They are interested in the local startup scene and want
to blend remote work with exploration of creative and professional communities at a
slow pace. Previously, they had visited a co-working space in Poblenou, and the 
Barcelona faculty space.
"""

MODEL = "openrouter:minimax/minimax-m2.5"
TEMPERATURE = 0.0

# ── Step 1: extract structured params from task text ──────────────────────────

config = load_config_from_env()

print(f"Task text:\n{TASK_TEXT}\n")
print("Extracting parameters...")
params = _extract_attraction_params(TASK_TEXT, MODEL, TEMPERATURE)
print(f"  Destination : {params.destination}")
print(f"  Day         : {params.day}")
print(f"  Budget      : {params.budget} EUR")
print(f"  Profile     : {params.traveller_profile}")
print(f"  Time slot   : {params.time_slot or '(any)'}")
print(f"  Previous    : {params.previous_activities or '(none)'}")
print(f"  Hint        : {params.orchestrator_hint or '(none)'}")
print()

# ── Step 2: run full pipeline ─────────────────────────────────────────────────

print("Running attraction search (embedding → LLM generation → SERPAPI)...")
print()

result = run_attraction_search(
    params=params,
    model_name=MODEL,
    temperature=TEMPERATURE,
    config=config,
    task_ref="scratch_e2e",
)

# ── Summary header ────────────────────────────────────────────────────────────

print(f"{'━' * 60}")
print(f"STATUS: {result.status}  |  Archetype: {result.selected_archetype}")
print(f"{'━' * 60}")

if result.errors:
    print()
    print("Errors:")
    for err in result.errors:
        print(f"  [{err.code}] {err.message}")

# ── Committed item (what the orchestrator receives) ───────────────────────────

if result.item:
    item = result.item
    print()
    print(f"SELECTED ACTIVITY — Day {item.day} {item.time_slot.upper()}")
    print(f"  Title       : {item.title}")
    print(f"  Description : {item.description}")
    print(f"  Touchpoint  : {item.local_touchpoint}")
    print(f"  Duration    : {item.estimated_duration_hours}h  |  Budget: {item.estimated_price_range}  |  place_found: {item.place_found}")
    if item.place_found:
        print(f"  ▶ {item.location_name}")
        if item.location_address:
            print(f"    {item.location_address}")
        if item.coordinates:
            print(f"    ({item.coordinates['lat']:.4f}, {item.coordinates['lng']:.4f})")
        if item.place_rating:
            reviews = f"  ({item.place_review_count} reviews)" if item.place_review_count else ""
            print(f"    Rating: {item.place_rating}{reviews}  |  Price: {item.place_price_level or 'N/A'}  |  Type: {item.place_type or 'N/A'}")
        if item.place_hours:
            print(f"    Hours: {item.place_hours}")
        if item.selection_reason:
            print(f"    Why: {item.selection_reason}")
    else:
        print(f"  ▶ No specific place found  (location: {item.location_name})")
    print(f"  Provenance  : {item.provenance}")

# ── Top candidates (for routing agent) ───────────────────────────────────────

if result.top_candidates:
    print()
    print(f"TOP CANDIDATES  [{len(result.top_candidates)} available for routing agent]")
    for i, c in enumerate(result.top_candidates):
        rating = f"  Rating: {c.rating}" if c.rating else ""
        reviews = f" ({c.reviews} reviews)" if c.reviews else ""
        print(f"  [{i}] {c.title}  |  {c.address or 'address unknown'}{rating}{reviews}")
        if c.gps_coordinates:
            print(f"      ({c.gps_coordinates['lat']:.4f}, {c.gps_coordinates['lng']:.4f})")

if result.google_maps_url:
    print(f"\nVerify on Google Maps: {result.google_maps_url}")
