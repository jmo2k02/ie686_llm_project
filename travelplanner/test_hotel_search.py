"""Integration tests for intelligent hotel search agent.

Full end-to-end tests with live API calls to LiteAPI and LLM providers.
All test cases share the minimum requirements:
  countryCode=DE, cityName=Berlin, checkin=2026-07-01, checkout=2026-07-02,
  currency=EUR, guestNationality=DE, occupancies=[{adults: 2}]
but add different extra constraints/preferences.

Provider dispatch (matches ``travelplanner.utils.llm``):
  - Default: OpenAI (gpt-5-mini)
  - ``--ollama``: Uses ollama:nemotron-3-super

Usage:
    # Run with default OpenAI provider
    uv run python test_hotel_search.py --case 1

    # Run with Ollama
    uv run python test_hotel_search.py --case 1 --ollama

    # Run all test cases
    uv run python test_hotel_search.py --case all

Run from travelplanner/ directory:
    uv run python test_hotel_search.py

Requirements:
    - LITEAPI_API_KEY in .env
    - OPENAI_API_KEY in .env (for OpenAI — default)
    - OLLAMA_API_KEY in .env (only if using --ollama)
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from travelplanner.agents.hotel_search_agent import intelligent_hotel_search
from travelplanner.schema.system_state import StateContractModel

DEFAULT_MODEL = "gpt-5-mini"
OLLAMA_MODEL = "ollama:nemotron-3-super"

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

TEST_CASES = {
    "1": {
        "title": "TEST CASE 1: Budget + Wi-Fi (no breakfast)",
        "extras": [
            "max budget: 90 EUR/night",
            "required facility: Wi-Fi",
            "breakfast not needed",
        ],
        "query": (
            "Find a hotel in Berlin for July 1–2, 2026. "
            "We are 2 adults from Germany. "
            "Must have Wi-Fi. Budget max 90 euros per night. "
            "No need for breakfast included."
        ),
    },
    "2": {
        "title": "TEST CASE 2: City Center + Pool + Breakfast",
        "extras": [
            "preferred area: city center / Mitte",
            "required facilities: swimming pool, breakfast included",
            "budget up to 180 EUR/night",
        ],
        "query": (
            "Looking for a hotel in central Berlin (Mitte or near Alexanderplatz) "
            "for July 1–2, 2026. Two German adults. "
            "Must have a swimming pool and breakfast included. "
            "Budget up to 180 euros per night."
        ),
    },
    "3": {
        "title": "TEST CASE 3: Family + Parking + Gym",
        "extras": [
            "traveling with a small child (family-friendly preferred)",
            "required facilities: parking, gym/fitness center",
            "budget up to 150 EUR/night",
        ],
        "query": (
            "Need a family-friendly hotel in Berlin for July 1–2, 2026. "
            "Two adults and a small child (German guests). "
            "Must offer parking and a gym. "
            "Budget up to 150 euros per night. "
            "Close to public transport is a plus."
        ),
    },
}


def _get_model_name(use_ollama: bool) -> str:
    return OLLAMA_MODEL if use_ollama else DEFAULT_MODEL


def _print_base():
    print("\n[Base minimum requirements]")
    for k, v in BASE_MINIMUM.items():
        print(f"  {k}: {v}")


def _run_search(query: str, model_name: str):
    """Execute search and return updated state."""
    system_state = StateContractModel(query="Plan Berlin trip")
    return intelligent_hotel_search(
        query=query,
        system_state=system_state,
        model_name=model_name,
        agent_key="hotel_search",
    )


def _print_artifacts(updated_state):
    """Pretty-print the hotel search artifact."""
    artifacts = updated_state.agent_artifacts.get("hotel_search", [])
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

    if content.get("booking_url"):
        print("\n" + "=" * 60)
        print("NUITEE BOOKING LINK:")
        print("=" * 60)
        print(content["booking_url"])

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


def run_test_case(case_id: str, model_name: str):
    """Run a single test case by ID."""
    case = TEST_CASES[case_id]
    print("\n" + "=" * 60)
    print(case["title"])
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    for extra in case["extras"]:
        print(f"  - {extra}")

    print(f"\nQuery: {case['query'].strip()}")
    print("\nProcessing...")

    updated_state = _run_search(case["query"], model_name)
    _print_artifacts(updated_state)


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
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama (ollama:nemotron-3-super) instead of OpenAI (gpt-5-mini)",
    )
    args = parser.parse_args()

    model_name = _get_model_name(args.ollama)
    provider_label = "Ollama" if args.ollama else "OpenAI"

    print("\n" + "=" * 60)
    print("Hotel Search Agent - Integration Tests")
    print("=" * 60)
    print(f"\nUsing LLM: {model_name}")
    print(f"(Provider: {provider_label})")
    print("=" * 60)

    case = args.case.strip().lower()

    if case == "all":
        for cid in TEST_CASES:
            run_test_case(cid, model_name)
    elif case in TEST_CASES:
        run_test_case(case, model_name)
    else:
        print(f"\nUnknown case '{args.case}'. Use 1, 2, 3, or all.")
        return

    print("\n" + "=" * 60)
    print("Integration Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
