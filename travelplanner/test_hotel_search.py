"""Integration tests for intelligent hotel search agent.

Full end-to-end tests with live API calls to LiteAPI and LLM providers.

Usage:
    # Team testing with OpenRouter (default):
    uv run python test_hotel_search_integration.py

    # Personal testing with Ollama:
    USE_OLLAMA=true uv run python test_hotel_search_integration.py

Run from travelplanner/ directory:
    uv run python test_hotel_search_integration.py

Requirements:
    - LITEAPI_API_KEY in .env
    - OPENROUTER_API_KEY in .env (for team testing)
    - OLLAMA_API_KEY in .env (for personal testing)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from travelplanner.agents.hotel_search_agent import intelligent_hotel_search

# Check if user wants to use Ollama for personal testing
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
MODEL_NAME = "ollama:gpt-oss:120b" if USE_OLLAMA else "openrouter:anthropic/claude-3.5-sonnet"

print(f"\nUsing LLM: {MODEL_NAME}")
if USE_OLLAMA:
    print("(Personal testing mode with local Ollama)")
else:
    print("(Team testing mode with OpenRouter credits)")
print("=" * 60)


def test_natural_language_query():
    """Test with a natural language query."""
    print("\n" + "=" * 60)
    print("TEST: Natural Language Hotel Search")
    print("=" * 60)

    query = """
    I'm planning a romantic honeymoon in Barcelona next month.
    We love swimming and need good wifi for work.
    Budget is around 200 euros per night, staying for a week.
    Would prefer a place with a gym too if possible.
    """

    print(f"\nQuery: {query.strip()}")
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)

    result = intelligent_hotel_search(query, model_name=MODEL_NAME)

    content = result.content

    print(f"\nStatus: {content['status']}")
    print(f"Hotels found: {len(content.get('options', []))}")

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
            print(f"   Amenities: {', '.join(hotel['facilities'][:5])}...")


def test_business_trip_query():
    """Test with a business trip query."""
    print("\n" + "=" * 60)
    print("TEST: Business Trip Query")
    print("=" * 60)

    query = "Business trip to Barcelona, June 15-18, need parking and wifi, max 150/night"

    print(f"\nQuery: {query}")
    print("\nProcessing...")

    result = intelligent_hotel_search(query, model_name=MODEL_NAME)

    content = result.content

    print(f"\nStatus: {content['status']}")
    print(f"Hotels found: {len(content.get('options', []))}")

    if content.get("recommendations"):
        print("\n" + "=" * 60)
        print("AI RECOMMENDATIONS:")
        print("=" * 60)
        print(content["recommendations"])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Hotel Search Agent - Integration Tests")
    print("=" * 60)

    test_natural_language_query()
    # test_business_trip_query()  # Uncomment to test

    print("\n" + "=" * 60)
    print("Integration Tests Complete!")
    print("=" * 60)
