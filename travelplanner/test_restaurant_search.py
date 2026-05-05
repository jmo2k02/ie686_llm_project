"""Integration tests for intelligent restaurant search agent.

Full end-to-end tests with live API calls to Google Places and LLM providers.
All test cases share the minimum requirement: city = "Berlin"
but add different extra constraints/preferences.

Provider dispatch:
  - Default: OpenAI (gpt-5-mini)
  - --ollama: Uses ollama:nemotron-3-super

Usage:
    # Run with default OpenAI provider
    uv run python test_restaurant_search.py --case 1

    # Run with Ollama
    uv run python test_restaurant_search.py --case 1 --ollama

    # Run all test cases
    uv run python test_restaurant_search.py --case all

Run from travelplanner/ directory:
    uv run python test_restaurant_search.py

Requirements:
    - GOOGLE_PLACES_API_KEY in .env
    - OPENAI_API_KEY in .env (for OpenAI — default)
    - OLLAMA_API_KEY in .env (only if using --ollama)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from travelplanner.agents.restaurant_search_agent import (
    intelligent_restaurant_search,
    load_config_from_env,
    _extract_restaurant_params,
    run_restaurant_search,
)
from travelplanner.schema.system_state import StateContractModel

DEFAULT_MODEL = "gpt-5-mini"
OLLAMA_MODEL = "ollama:nemotron-3-super"
BASE_MINIMUM = {"city": "Berlin"}


def _get_model_name(use_ollama: bool) -> str:
    if use_ollama:
        return OLLAMA_MODEL
    return DEFAULT_MODEL


def _print_base():
    print("\n[Base minimum requirement]")
    for k, v in BASE_MINIMUM.items():
        print(f"  {k}: {v}")


def _run_and_print(query: str, model_name: str, agent_key: str = "restaurant_search"):
    """Execute search via the convenience wrapper and pretty-print results."""
    system_state = StateContractModel(query="Plan Berlin trip")

    updated_state = intelligent_restaurant_search(
        query=query,
        system_state=system_state,
        model_name=model_name,
        agent_key=agent_key,
    )

    artifacts = updated_state.agent_artifacts.get(agent_key, [])
    if not artifacts:
        print("\n✗ No artifacts found in SystemState!")
        return

    artifact = artifacts[0]
    content = artifact.content

    print(f"\nStatus: {content.get('status')}")
    print(f"Restaurants found: {len(content.get('items', []))}")
    print(f"Artifacts in SystemState: {len(artifacts)}")

    print("\n" + "=" * 60)
    print("PARSED PARAMETERS:")
    print("=" * 60)
    print(f"City: {content.get('city')}")
    print(f"Cuisine: {content.get('cuisine')}")
    print(f"Budget: {content.get('budget')}")
    print(f"Meal type: {content.get('meal_type')}")
    print(f"Dietary restrictions: {content.get('dietary_restrictions', [])}")

    if content.get("items"):
        print("\n" + "=" * 60)
        print("SELECTED RESTAURANT:")
        print("=" * 60)
        item = content["items"][0]
        print(f"  Name: {item.get('name')}")
        print(f"  Address: {item.get('address')}")
        print(f"  Rating: {item.get('rating')}")
        print(f"  Price: {item.get('price_range')}")
        print(f"  Phone: {item.get('phone')}")
        print(f"  Website: {item.get('website')}")
        print(f"  Selection reason: {item.get('selection_reason')}")

    if content.get("errors"):
        print("\n" + "=" * 60)
        print("ERRORS:")
        print("=" * 60)
        for err in content["errors"]:
            print(f"  {err.get('code')}: {err.get('message')}")


def test_case_1_budget_lunch(model_name: str):
    """Case 1: Berlin, cheap lunch, no special cuisine."""
    print("\n" + "=" * 60)
    print("TEST CASE 1: Budget Lunch in Berlin")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - meal type: lunch")
    print("  - budget: low ($)")
    print("  - no special cuisine")

    query = (
        "Find a cheap lunch spot in Berlin. "
        "Just need something quick and affordable. "
        "Budget is around 10-15 euros per person."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    _run_and_print(query, model_name=model_name, agent_key="restaurant_search")


def test_case_2_central_dinner_vegetarian(model_name: str):
    """Case 2: Berlin city center, dinner, vegetarian-friendly, medium budget."""
    print("\n" + "=" * 60)
    print("TEST CASE 2: Vegetarian Dinner in Central Berlin")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - preferred area: city center / Mitte")
    print("  - meal type: dinner")
    print("  - dietary restriction: vegetarian")
    print("  - budget: medium ($$)")

    query = (
        "Looking for a nice vegetarian restaurant in central Berlin (Mitte or Kreuzberg) "
        "for dinner. We are 2 people. "
        "Must have good vegetarian options. "
        "Budget around 20-30 euros per person."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    _run_and_print(query, model_name=model_name, agent_key="restaurant_search")


def test_case_3_family_breakfast_halal(model_name: str):
    """Case 3: Berlin, family-friendly breakfast, halal, near park."""
    print("\n" + "=" * 60)
    print("TEST CASE 3: Family Halal Breakfast in Berlin")
    print("=" * 60)
    _print_base()
    print("\n[Extras]")
    print("  - meal type: breakfast / brunch")
    print("  - dietary restriction: halal")
    print("  - family-friendly (with kids)")
    print("  - near a park is a plus")

    query = (
        "Need a family-friendly breakfast place in Berlin. "
        "We are a family with 2 adults and 2 small children. "
        "Food must be halal. "
        "Would be great if it's near a park or playground. "
        "Casual atmosphere, not too fancy."
    )
    print(f"\nQuery: {query.strip()}")
    print("\nProcessing...")

    _run_and_print(query, model_name=model_name, agent_key="restaurant_search")


def test_smoke(model_name: str):
    """Quick smoke test using the low-level run_restaurant_search directly."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: Low-level API call")
    print("=" * 60)

    config = load_config_from_env()
    if not config.api_key:
        print("\n⚠️  Skipped — GOOGLE_PLACES_API_KEY not set")
        return

    task_text = "Find a nice Italian dinner spot in Berlin for 2 people, medium budget"

    print("\n==> Step 1: LLM parameter extraction...")
    try:
        params = _extract_restaurant_params(task_text, model_name=model_name, temperature=0.0)
        print(
            f"    extracted: city={params.city}, cuisine={params.cuisine}, "
            f"budget={params.budget}, meal={params.meal_type}"
        )
    except Exception as exc:
        print(f"\n❌ Step 1 failed — LLM parameter extraction error: {exc}")
        raise SystemExit(1)

    print("==> Step 2: Google Places search + LLM candidate selection...")
    result = run_restaurant_search(
        params=params,
        config=config,
        model_name=model_name,
        temperature=0.0,
        task_ref="smoke-test",
        query_text=task_text,
    )

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))

    if result.status == "success":
        print(f"\n✅ Smoke-test passed — {len(result.items)} restaurant(s) found")
    elif result.status == "partial":
        print(f"\n⚠️ Smoke-test partial — {len(result.errors)} error(s)")
    else:
        print(f"\n❌ Smoke-test failed — {result.errors[0].code}: {result.errors[0].message}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run restaurant search integration test cases for Berlin."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="1",
        help="Test case to run: 1, 2, 3, 'smoke', or 'all' (default: 1)",
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
    print("Restaurant Search Agent - Integration Tests")
    print("=" * 60)
    print(f"\nUsing LLM: {model_name}")
    print(f"(Provider: {provider_label})")
    print("=" * 60)

    case = args.case.strip().lower()

    if case == "1":
        test_case_1_budget_lunch(model_name)
    elif case == "2":
        test_case_2_central_dinner_vegetarian(model_name)
    elif case == "3":
        test_case_3_family_breakfast_halal(model_name)
    elif case == "smoke":
        test_smoke(model_name)
    elif case == "all":
        test_case_1_budget_lunch(model_name)
        test_case_2_central_dinner_vegetarian(model_name)
        test_case_3_family_breakfast_halal(model_name)
        test_smoke(model_name)
    else:
        print(f"\nUnknown case '{args.case}'. Use 1, 2, 3, smoke, or all.")
        return

    print("\n" + "=" * 60)
    print("Integration Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
