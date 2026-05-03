"""Integration tests for intelligent hotel search agent.

Full end-to-end tests with live API calls to LiteAPI and LLM providers.
All test cases share the minimum requirements:
  countryCode=DE, cityName=Berlin, checkin=2026-07-01, checkout=2026-07-02,
  currency=EUR, guestNationality=DE, occupancies=[{adults: 2}]
but add different extra constraints/preferences.

Usage:
    # Run the default test case (case 1)
    uv run python test_hotel_search.py

    # Run a specific test case by number (1, 2, or 3)
    uv run python test_hotel_search.py --case 2

    # Run all test cases sequentially
    uv run python test_hotel_search.py --case all

    # Team testing with OpenRouter (default):
    uv run python test_hotel_search.py --case 3

    # Personal testing with Ollama:
    USE_OLLAMA=true uv run python test_hotel_search.py --case 1

Run from travelplanner/ directory:
    uv run python test_hotel_search.py

Requirements:
    - LITEAPI_API_KEY in .env
    - OPENROUTER_API_KEY in .env (for team testing)
    - OLLAMA_API_KEY in .env (for personal testing)
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from travelplanner.agents.hotel_search_agent import intelligent_hotel_search
from travelplanner.schema.system_state import StateContractModel

# Automatically use Ollama for local/personal testing
# Note: For Ollama, use the model name directly (no "ollama:" prefix)
MODEL_NAME = "nemotron-3-super"

# Base minimum requirements shared by all test cases
BASE_MINIMUM = {
    "countryCode": "DE",
    "cityName": "Berlin",
    "checkin": "2026-07-01",
    "checkout": "2026-07-02",
    "currency": "EUR",
    "guestNationality": "DE",
    "occupancies": [{"adults": 2}],
}


def _print_base():
    print("\n[Base minimum requirements]")
    for k, v in BASE_MINIMUM.items():
        print(f"  {k}: {v}")


def _run_search(query: str, agent_key: str = "hotel_search"):
    """Execute search and return updated state."""
    system_state = StateContractModel(query="Plan Berlin trip")
    updated_state = intelligent_hotel_search(
        query=query,
        system_state=system_state,
        model_name=MODEL_NAME,
        agent_key=agent_key,
    )
    return updated_state


def _print_artifacts(updated_state, agent_key: str = "hotel_search"):
    """Pretty-print the hotel search artifact."""
    artifacts = updated_state.agent_artifacts.get(agent_key, [])
    if not artifacts:
        print("\n✗ No artifacts found in SystemState!")
        return

    result = artifacts[0]
    content = result.content

    print(f"\nStatus: {content['status']}")
    print(f"Hotels found: {len(content.get('options', []))}")
    print(f"Artifacts in SystemState: {len(artifacts)}")

    print("\n" + "=" * 60)
    print("PARSED PARAMETERS:")
    print("=" * 60)
    search_params = content.get("search_parameters", {})
    print(f"Location: {search_params.get('location')}")
    print(f"Dates: {search_params.get('check_in_date')} to {search_params.get('check_out_date')}")
    print(f"Nights: {search_params.get('nights')}")
    print(f"Budget: €{search_params.get('budget_max')}/night")
    print(f"Guests: {search_params.get('guest_count')}")

    if content.get("recommendations"):
        print("\n" + "=" * 60)
        print("AI RECOMMENDATIONS:")
        print("=" * 60)
        print(content["recommendations"])

    if content.get("options"):
        print("\n" + "=" * 60)
        print("TOP 3 HOTELS (Raw Data):")
        print("=" * 60)
        for i, hotel in enumerate(content["options"][:3], 1):
            print(f"\n{i}. {hotel['name']}")
            print(f"   Price: {hotel['currency']} {hotel['nightly_rate']:.2f}/night")
            print(f"   Rating: {hotel['rating']}/10")
            print(f"   Over Budget: {hotel['over_budget']}")
            print(f"   Facilities: {', '.join(hotel['facilities'][:5])}...")


def test_case_1_budget_wifi():
    """Case 1: Berlin, tight budget, Wi-Fi required, no breakfast needed."""
    print("\n" + "=" * 60)
    print("TEST CASE 1: Budget + Wi-Fi (no breakfast)")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - max budget: 90 EUR/night")
    print("  - required facility: Wi-Fi")
    print("  - breakfast not needed")

    query = (
        "Find a hotel in Berlin for July 1–2, 2026. "
        "We are 2 adults from Germany. "
        "Must have Wi-Fi. Budget max 90 euros per night. "
        "No need for breakfast included."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    updated_state = _run_search(query, agent_key="hotel_search")
    _print_artifacts(updated_state, agent_key="hotel_search")


def test_case_2_central_pool_breakfast():
    """Case 2: Berlin city center, pool + breakfast, higher budget."""
    print("\n" + "=" * 60)
    print("TEST CASE 2: City Center + Pool + Breakfast")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - preferred area: city center / Mitte")
    print("  - required facilities: swimming pool, breakfast included")
    print("  - budget up to 180 EUR/night")

    query = (
        "Looking for a hotel in central Berlin (Mitte or near Alexanderplatz) "
        "for July 1–2, 2026. Two German adults. "
        "Must have a swimming pool and breakfast included. "
        "Budget up to 180 euros per night."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    updated_state = _run_search(query, agent_key="hotel_search")
    _print_artifacts(updated_state, agent_key="hotel_search")


def test_case_3_family_parking_gym():
    """Case 3: Berlin, family-friendly, parking + gym, flexible dates note."""
    print("\n" + "=" * 60)
    print("TEST CASE 3: Family + Parking + Gym")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - traveling with a small child (family-friendly preferred)")
    print("  - required facilities: parking, gym/fitness center")
    print("  - budget up to 150 EUR/night")

    query = (
        "Need a family-friendly hotel in Berlin for July 1–2, 2026. "
        "Two adults and a small child (German guests). "
        "Must offer parking and a gym. "
        "Budget up to 150 euros per night. "
        "Close to public transport is a plus."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    updated_state = _run_search(query, agent_key="hotel_search")
    _print_artifacts(updated_state, agent_key="hotel_search")


def main():
    parser = argparse.ArgumentParser(
        description="Run hotel search integration test cases for Berlin."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="1",
        help="Test case to run: 1, 2, 3, or 'all' (default: 1)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Hotel Search Agent - Integration Tests")
    print("=" * 60)
    print(f"\nUsing LLM: {MODEL_NAME}")
    print("(Testing mode with local Ollama)")
    print("=" * 60)

    case = args.case.strip().lower()

    if case == "1":
        test_case_1_budget_wifi()
    elif case == "2":
        test_case_2_central_pool_breakfast()
    elif case == "3":
        test_case_3_family_parking_gym()
    elif case == "all":
        test_case_1_budget_wifi()
        test_case_2_central_pool_breakfast()
        test_case_3_family_parking_gym()
    else:
        print(f"\nUnknown case '{args.case}'. Use 1, 2, 3, or all.")
        return

    print("\n" + "=" * 60)
    print("Integration Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
