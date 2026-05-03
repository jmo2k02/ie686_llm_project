# scratch_e2e_attraction.py — end-to-end: profile → archetype selection → LLM activities → SERPAPI place resolution
# Edit the variables below and run: python scratch_e2e_attraction.py

from dotenv import load_dotenv

from travelplanner.agents.attraction_search_agent import (
    load_config_from_env,
    run_attraction_search,
)

load_dotenv()

DESTINATION = "Barcelona"
DAYS = 3
BUDGET = "medium"  # "low" | "medium" | "high"
TRAVELLER_PROFILE = "solo digital nomad interested in the local startup scene, wants to blend remote work with exploration of creative and professional communities, slow pace"
MODEL = "openrouter:minimax/minimax-m2.5"
TEMPERATURE = 0.0

# ── Run ───────────────────────────────────────────────────────────────────────

config = load_config_from_env()

print(f"Destination   : {DESTINATION}")
print(f"Days          : {DAYS}")
print(f"Budget        : {BUDGET}")
print(f"Profile       : {TRAVELLER_PROFILE}")
print(f"Model         : {MODEL}")
print()
print("Running attraction search (embedding → LLM generation → SERPAPI)...")
print()

result = run_attraction_search(
    destination=DESTINATION,
    days=DAYS,
    budget=BUDGET,
    traveller_profile=TRAVELLER_PROFILE,
    model_name=MODEL,
    temperature=TEMPERATURE,
    config=config,
    task_ref="scratch_e2e",
)

# ── Summary header ────────────────────────────────────────────────────────────

print(f"{'━' * 60}")
print(f"STATUS: {result.status}  |  Archetype: {result.selected_archetype}  |  Items: {len(result.items)}")
print(f"{'━' * 60}")

if result.errors:
    print()
    print("Errors:")
    for err in result.errors:
        print(f"  [{err.code}] {err.message}")

# ── Items (what the orchestrator receives) ────────────────────────────────────

for item in result.items:
    print()
    print(f"Day {item.day} — {item.time_slot.upper()}  |  {item.title}")
    print(f"  {item.description}")
    print(f"  Local touchpoint: {item.local_touchpoint}")
    print(f"  Duration: {item.estimated_duration_hours}h  |  Budget: {item.estimated_price_range}  |  place_found: {item.place_found}")
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
    print(f"  Provenance: {item.provenance}")
