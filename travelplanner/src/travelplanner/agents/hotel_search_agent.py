"""Intelligent Hotel Search Agent with LLM-based query parsing.

This agent combines:
1. Deterministic hotel search functions (geocoding, API calls, filtering)
2. LLM-based natural language query parsing
3. LLM-based recommendation synthesis

Architecture:
    User Query (NL) → [LLM Parse] → Structured Params → [Search] → Hotels → [LLM Synthesize] → Output

The search functions are provider-agnostic and can be reused by other agents.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pycountry
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field

from travelplanner.schema.hotel_search_artifact import (
    HotelSearchArtifactContentModel,
    HotelSearchCoordinatesModel,
    HotelSearchMetadataModel,
    HotelSearchParametersModel,
    HotelSearchErrorModel,
    HotelOptionModel,
)
from travelplanner.schema.system_state import AgentArtifactModel
from travelplanner.utils.llm import make_chat_model

# Load environment variables from .env file at module initialization
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_env_path = os.path.join(_project_root, '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
    print(f"[hotel_search_agent] Loaded environment from: {_env_path}")
else:
    print(f"[hotel_search_agent] WARNING: No .env file found at: {_env_path}")

# API Configuration
LITEAPI_BASE_URL = "https://api.liteapi.travel/v3.0"
LITEAPI_BOOKING_URL = "https://book.liteapi.travel/v3.0"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

# Geocoding cache (simple dict - thread-safe enough for this use case)
_geocoding_cache: Dict[str, Tuple[float, float]] = {}
_last_nominatim_request: float = 0.0

# Amenity matching keywords for fuzzy matching
AMENITY_KEYWORDS = {
    "wifi": ["wifi", "wi-fi", "internet", "wireless"],
    "pool": ["pool", "swimming"],
    "gym": ["gym", "fitness", "workout", "exercise"],
    "parking": ["parking", "garage"],
    "breakfast": ["breakfast", "morning meal"],
    "spa": ["spa", "wellness", "massage"],
    "restaurant": ["restaurant", "dining"],
    "bar": ["bar", "lounge", "pub"],
    "pets": ["pet", "dog", "cat", "animal"],
    "kitchen": ["kitchen", "kitchenette", "cooking"],
}


# ============================================================================
# Utility Functions (Provider-Agnostic)
# ============================================================================


def geocode_location(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Convert location string to (latitude, longitude) using Nominatim.

    Args:
        location: Location string (e.g., "Eixample, Barcelona")

    Returns:
        Tuple of (latitude, longitude) or (None, None) if geocoding fails
    """
    global _last_nominatim_request

    # Check cache
    if location in _geocoding_cache:
        print(f"[geocode_location] Using cached coordinates for {location}")
        return _geocoding_cache[location]

    print(f"[geocode_location] Geocoding location: {location}")

    # Rate limiting (Nominatim requires 1 request per second)
    time_since_last = time.time() - _last_nominatim_request
    if time_since_last < 1.0:
        sleep_time = 1.0 - time_since_last
        print(f"[geocode_location] Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)

    _last_nominatim_request = time.time()

    try:
        headers = {"User-Agent": "TravelPlannerAgent/1.0 (educational project)"}
        params = {"q": location, "format": "json", "limit": 1}

        response = requests.get(
            f"{NOMINATIM_BASE}/search",
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        results = response.json()
        if not results:
            print(f"[geocode_location] No results found")
            return None, None

        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])

        # Cache result
        _geocoding_cache[location] = (lat, lon)

        print(f"[geocode_location] Found coordinates: ({lat}, {lon})")
        return lat, lon

    except (requests.exceptions.RequestException, KeyError, ValueError, IndexError) as e:
        print(f"[geocode_location] Error: {e}")
        return None, None


def calculate_nights(check_in: str, check_out: str) -> int:
    """Calculate number of nights between check-in and check-out dates.

    Args:
        check_in: Check-in date in ISO format (YYYY-MM-DD)
        check_out: Check-out date in ISO format (YYYY-MM-DD)

    Returns:
        Number of nights
    """
    try:
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (check_out_dt - check_in_dt).days
        print(f"[calculate_nights] Calculated {nights} nights")
        return max(1, nights)
    except Exception as e:
        print(f"[calculate_nights] Error calculating nights: {e}")
        return 1


def parse_location(location: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse location string into (city_name, country_code).

    Args:
        location: Location string (e.g., "Eixample, Barcelona, Spain")

    Returns:
        Tuple of (city_name, country_code) or (None, None) if parsing fails

    Examples:
        "Eixample, Barcelona, Spain" → ("Barcelona", "ES")
        "Westminster, London, UK" → ("London", "GB")
        "Barcelona, Spain" → ("Barcelona", "ES")
    """
    print(f"[parse_location] Parsing location: {location}")

    parts = [p.strip() for p in location.split(",")]

    if len(parts) < 2:
        print(f"[parse_location] Invalid location format, need at least city, country")
        return None, None

    country_str = parts[-1]
    city_name = parts[-2]

    try:
        country = pycountry.countries.search_fuzzy(country_str)[0]
        country_code = country.alpha_2
        print(f"[parse_location] Parsed: city={city_name}, country={country_code}")
        return city_name, country_code
    except LookupError:
        print(f"[parse_location] Country not found: '{country_str}'")
        return city_name, None
    except (IndexError, AttributeError) as e:
        print(f"[parse_location] Failed to extract country code: {e}")
        return city_name, None
    except Exception as e:
        print(f"[parse_location] Unexpected error: {e}")
        return city_name, None


# ============================================================================
# LiteAPI REST Integration
# ============================================================================


def _get_api_headers() -> Dict[str, str]:
    """Get headers for LiteAPI requests."""
    api_key = os.environ.get("LITEAPI_API_KEY")
    if not api_key:
        raise ValueError(
            "LITEAPI_API_KEY not found in environment. "
            "Please add it to .env file in project root."
        )
    return {
        "X-API-Key": api_key,
        "accept": "application/json",
        "content-type": "application/json"
    }


def get_hotel_details(hotel_id: str, timeout: int = 4) -> Dict[str, Any]:
    """Fetch full hotel details including amenities.

    Args:
        hotel_id: LiteAPI hotel ID
        timeout: Request timeout in seconds (default: 4)

    Returns:
        Dict with hotel details including hotelFacilities
    """
    print(f"[get_hotel_details] Fetching details for hotel: {hotel_id}")

    try:
        params = {"hotelId": hotel_id, "timeout": timeout}
        response = requests.get(
            f"{LITEAPI_BASE_URL}/data/hotel",
            headers=_get_api_headers(),
            params=params,
            timeout=timeout + 2  # Add buffer to timeout
        )
        response.raise_for_status()

        data = response.json()
        hotel_data = data.get("data", {})

        facilities = hotel_data.get("hotelFacilities", [])
        print(f"[get_hotel_details] Found {len(facilities)} facilities")

        return {
            "status": "success",
            "hotel": hotel_data
        }

    except requests.exceptions.Timeout:
        print(f"[get_hotel_details] Request timeout for hotel: {hotel_id}")
        return {"status": "failed", "error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"[get_hotel_details] Request error: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[get_hotel_details] Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


def search_places(text_query: str) -> Dict[str, Any]:
    """Search for places using LiteAPI autocomplete.

    Args:
        text_query: Search query (e.g., "Barcelona, Spain")

    Returns:
        Dict with placeId and displayName
    """
    print(f"[search_places] Searching for: {text_query}")

    try:
        params = {"textQuery": text_query}
        response = requests.get(
            f"{LITEAPI_BASE_URL}/data/places",
            headers=_get_api_headers(),
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        places = data.get("data", [])

        if not places:
            print(f"[search_places] No places found for: {text_query}")
            return {"status": "failed", "error": "no_results"}

        first_place = places[0]
        place_id = first_place.get("placeId", "")
        display_name = first_place.get("displayName", "")

        print(f"[search_places] Found place: {display_name} (ID: {place_id})")

        return {
            "status": "success",
            "placeId": place_id,
            "displayName": display_name
        }

    except requests.exceptions.Timeout:
        print(f"[search_places] Request timeout")
        return {"status": "failed", "error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"[search_places] Request error: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[search_places] Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


def search_hotels_via_api(
    place_id: Optional[str],
    city_name: Optional[str],
    country_code: Optional[str],
    check_in_date: str,
    check_out_date: str,
    guest_count: int,
    timeout: int = 8
) -> Dict[str, Any]:
    """Search for hotels using LiteAPI /data/hotels endpoint.

    Args:
        place_id: LiteAPI place ID from search_places()
        city_name: City name (fallback if place_id is None)
        country_code: Country code (fallback)
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        guest_count: Number of guests
        timeout: Request timeout in seconds

    Returns:
        Dict with search results including hotels and rates
    """
    print(f"[search_hotels_via_api] Searching hotels in {city_name}, {country_code}")

    try:
        if not place_id and city_name and country_code:
            place_result = search_places(f"{city_name}, {country_code}")
            if place_result.get("status") == "success":
                place_id = place_result.get("placeId")

        if not place_id:
            return {
                "status": "failed",
                "error": "Could not determine place_id for location"
            }

        occupancies = [
            {
                "rooms": 1,
                "adults": guest_count,
                "children": []
            }
        ]

        payload = {
            "placeId": place_id,
            "checkin": check_in_date,
            "checkout": check_out_date,
            "occupancies": occupancies,
            "currency": "EUR",
            "guestNationality": "US",
            "timeout": timeout
        }

        print(f"[search_hotels_via_api] Request payload: {json.dumps(payload, indent=2)}")

        start_time = time.time()
        response = requests.post(
            f"{LITEAPI_BASE_URL}/data/hotels",
            headers=_get_api_headers(),
            json=payload,
            timeout=timeout + 2
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        response.raise_for_status()

        data = response.json()

        hotels = data.get("data", [])
        print(f"[search_hotels_via_api] Found {len(hotels)} hotels in {elapsed_ms}ms")

        return {
            "status": "success",
            "data": hotels,
            "api_response_time_ms": elapsed_ms
        }

    except requests.exceptions.Timeout:
        print(f"[search_hotels_via_api] Request timeout")
        return {"status": "failed", "error": "timeout"}
    except requests.exceptions.HTTPError as e:
        print(f"[search_hotels_via_api] HTTP error: {e}")
        return {"status": "failed", "error": f"http_error: {e}"}
    except requests.exceptions.RequestException as e:
        print(f"[search_hotels_via_api] Request error: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[search_hotels_via_api] Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Hotel Filtering and Ranking
# ============================================================================


def _amenity_match(hotel_amenities: List[str], required_amenity: str) -> bool:
    """Check if hotel has required amenity using fuzzy keyword matching.

    Args:
        hotel_amenities: List of hotel amenities (lowercase)
        required_amenity: Required amenity (e.g., "wifi", "pool")

    Returns:
        True if hotel has amenity
    """
    keywords = AMENITY_KEYWORDS.get(required_amenity.lower(), [required_amenity.lower()])

    all_amenities_text = " ".join(hotel_amenities)

    for keyword in keywords:
        if keyword in all_amenities_text:
            return True

    return False


def filter_hotels_by_constraints(
    hotels: List[HotelOptionModel],
    required_amenities: Optional[List[str]] = None,
    preferred_amenities: Optional[List[str]] = None,
    min_rating: Optional[float] = None
) -> Tuple[List[HotelOptionModel], Dict[str, int]]:
    """Filter hotels by amenities and rating.

    Args:
        hotels: List of hotel options
        required_amenities: Must-have amenities (AND logic)
        preferred_amenities: Nice-to-have amenities (for scoring)
        min_rating: Minimum rating threshold

    Returns:
        Tuple of (filtered_hotels, preferred_amenity_counts)
    """
    print(f"[filter_hotels_by_constraints] Filtering {len(hotels)} hotels")
    print(f"  Required amenities: {required_amenities}")
    print(f"  Preferred amenities: {preferred_amenities}")
    print(f"  Min rating: {min_rating}")

    filtered = []
    preferred_counts = {}

    for hotel in hotels:
        # Apply min rating filter
        if min_rating is not None and hotel.rating < min_rating:
            print(f"[filter] {hotel.name} excluded: rating {hotel.rating} < {min_rating}")
            continue

        # Apply required amenities filter (ALL must be present)
        if required_amenities:
            hotel_amenities_lower = [a.lower() for a in hotel.amenities]
            has_all_required = True

            for req in required_amenities:
                if not _amenity_match(hotel_amenities_lower, req):
                    print(f"[filter] {hotel.name} excluded: missing required amenity '{req}'")
                    has_all_required = False
                    break

            if not has_all_required:
                continue

        # Count preferred amenities (for ranking)
        preferred_count = 0
        if preferred_amenities:
            hotel_amenities_lower = [a.lower() for a in hotel.amenities]
            for pref in preferred_amenities:
                if _amenity_match(hotel_amenities_lower, pref):
                    preferred_count += 1

        preferred_counts[hotel.accommodation_id] = preferred_count
        filtered.append(hotel)

    print(f"[filter_hotels_by_constraints] Filtered to {len(filtered)} hotels")
    return filtered, preferred_counts


def rank_hotels(
    hotels: List[HotelOptionModel],
    budget_max: float,
    preferred_counts: Optional[Dict[str, int]] = None,
    exclude_over_budget: bool = False
) -> List[HotelOptionModel]:
    """Rank hotels by criteria and return top 10.

    Ranking criteria (in order):
    1. Preferred amenity count (descending)
    2. Within budget status (within budget first)
    3. Rating (descending)
    4. Price (ascending)

    Args:
        hotels: List of filtered hotels
        budget_max: Maximum budget per night
        preferred_counts: Dict mapping hotel ID to preferred amenity count
        exclude_over_budget: If True, exclude over-budget hotels

    Returns:
        List of top 10 ranked hotels with rank field set
    """
    print(f"[rank_hotels] Ranking {len(hotels)} hotels")

    if exclude_over_budget:
        hotels = [h for h in hotels if not h.over_budget]
        print(f"[rank_hotels] After budget filter: {len(hotels)} hotels")

    if not hotels:
        return []

    if preferred_counts is None:
        preferred_counts = {}

    def sort_key(hotel: HotelOptionModel):
        pref_count = preferred_counts.get(hotel.accommodation_id, 0)
        within_budget = not hotel.over_budget
        return (-pref_count, not within_budget, -hotel.rating, hotel.nightly_rate)

    sorted_hotels = sorted(hotels, key=sort_key)

    top_hotels = sorted_hotels[:10]

    for i, hotel in enumerate(top_hotels, start=1):
        hotel.rank = i

    print(f"[rank_hotels] Ranked top {len(top_hotels)} hotels")
    return top_hotels


def _extract_price_from_rate(rate: Dict[str, Any]) -> Tuple[float, str]:
    """Extract price from LiteAPI rate structure.

    Args:
        rate: Rate dict from LiteAPI

    Returns:
        Tuple of (price, currency)
    """
    price = 0.0
    currency = "EUR"

    retail_rate = rate.get("retailRate", {})
    if isinstance(retail_rate, dict):
        total = retail_rate.get("total", [])
        if total and isinstance(total, list):
            try:
                price = float(total[0].get("amount", 0.0))
                currency = total[0].get("currency", "EUR")
            except (ValueError, TypeError, IndexError):
                pass

    return price, currency


def _build_hotel_option_from_data(
    hotel_info: Dict[str, Any],
    rate_data: Dict[str, Any],
    nights: int,
    budget_max: float
) -> Optional[HotelOptionModel]:
    """Build a HotelOptionModel from LiteAPI hotel info and rate data.

    Args:
        hotel_info: Hotel data from LiteAPI (name, address, rating, etc.)
        rate_data: Rate data containing room types and pricing
        nights: Number of nights
        budget_max: Maximum budget per night

    Returns:
        HotelOptionModel or None if data is invalid
    """
    hotel_id = hotel_info.get("id", "")
    hotel_name = hotel_info.get("name", "Unknown Hotel")

    if not hotel_id:
        return None

    room_types = rate_data.get("roomTypes", [])
    if not room_types:
        return None

    def get_room_price(room: Dict[str, Any]) -> Tuple[float, str]:
        rates = room.get("rates", [])
        if rates and isinstance(rates, list):
            return _extract_price_from_rate(rates[0])
        return 0.0, "EUR"

    cheapest_room = None
    cheapest_price = float('inf')
    cheapest_currency = "EUR"

    for room in room_types:
        price, currency = get_room_price(room)
        if price < cheapest_price:
            cheapest_price = price
            cheapest_currency = currency
            cheapest_room = room

    if cheapest_room is None or cheapest_price == float('inf') or cheapest_price <= 0:
        return None

    nightly_rate = cheapest_price / nights if nights > 0 else cheapest_price

    star_rating = float(hotel_info.get("star_rating", 0) or hotel_info.get("rating", 0) or 0)

    location = hotel_info.get("location", {})
    if isinstance(location, dict):
        hotel_lat = location.get("latitude", 0.0)
        hotel_lon = location.get("longitude", 0.0)
    else:
        hotel_lat = 0.0
        hotel_lon = 0.0

    address = hotel_info.get("address", "")

    amenities = hotel_info.get("hotelFacilities", [])
    if isinstance(amenities, list):
        amenities = [str(a) for a in amenities]
    else:
        amenities = []

    main_photo = hotel_info.get("main_photo", "")
    if not main_photo:
        hotel_images = hotel_info.get("hotelImages", [])
        if isinstance(hotel_images, list) and hotel_images:
            for img in hotel_images:
                if isinstance(img, dict) and img.get("defaultImage"):
                    main_photo = img.get("url", "")
                    break
            if not main_photo and hotel_images:
                main_photo = hotel_images[0].get("url", "") if isinstance(hotel_images[0], dict) else str(hotel_images[0])

    photos = [main_photo] if main_photo else []

    tags = hotel_info.get("tags", [])
    if isinstance(tags, list):
        amenities.extend([str(t) for t in tags[:5]])

    hotel_option = HotelOptionModel(
        search_result_id=cheapest_room.get("offerId", ""),
        accommodation_id=hotel_id,
        name=hotel_name,
        nightly_rate=nightly_rate,
        total_cost=cheapest_price,
        currency=cheapest_currency,
        area=hotel_info.get("location_type"),
        address=address if address else None,
        amenities=list(set(amenities)),
        rating=star_rating,
        reviews=0,
        check_in_time="15:00",
        check_out_time="11:00",
        latitude=hotel_lat,
        longitude=hotel_lon,
        photos=photos,
        booking_available=True,
        over_budget=nightly_rate > budget_max,
        over_budget_amount=max(0, nightly_rate - budget_max)
    )

    return hotel_option


def _create_failed_artifact(
    error_msg: str,
    search_params: Dict[str, Any],
    check_in_date: str,
    check_out_date: str,
    nights: int,
    latitude: Optional[float],
    longitude: Optional[float],
    task_id: Optional[int]
) -> AgentArtifactModel:
    """Create a failed artifact for error cases."""
    error_code = "unknown_error"
    if "api_key" in error_msg.lower():
        error_code = "missing_api_key"
    elif "timeout" in error_msg.lower():
        error_code = "timeout_error"
    elif "http" in error_msg.lower():
        error_code = "http_error"

    location = search_params.get("location", "")
    budget_max = float(search_params.get("budget_max", 0.0))
    guest_count = int(search_params.get("guest_count", 1))

    content = HotelSearchArtifactContentModel(
        task_ref=str(task_id) if task_id else "",
        status="failed",
        attempt=1,
        search_parameters=HotelSearchParametersModel(
            location=location,
            coordinates=HotelSearchCoordinatesModel(latitude=latitude, longitude=longitude) if latitude and longitude else None,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            nights=nights,
            budget_max=budget_max,
            guest_count=guest_count,
            rooms=1
        ),
        options=[],
        metadata=HotelSearchMetadataModel(total_results=0, returned_results=0),
        errors=[HotelSearchErrorModel(code=error_code, message=error_msg)]
    )
    return AgentArtifactModel(
        name="hotel_shortlist",
        type="hotel_search",
        content=content.model_dump(),
        description=f"Hotel search failed: {error_msg}"
    )


def _enrich_hotels_with_amenities(
    hotels: List[HotelOptionModel],
    required_amenities: List[str],
    preferred_amenities: List[str]
) -> List[HotelOptionModel]:
    """Enrich top hotels with full amenity details from hotel details API."""
    if not (required_amenities or preferred_amenities) or not hotels:
        return hotels

    print(f"[_enrich_hotels_with_amenities] Fetching details for top {min(20, len(hotels))} hotels")

    # Fetch details for top 20 candidates
    top_candidates = sorted(hotels, key=lambda h: (-h.rating, h.nightly_rate))[:20]

    for hotel in top_candidates:
        details_response = get_hotel_details(hotel.accommodation_id, timeout=4)

        if details_response.get("status") == "success":
            hotel_data = details_response.get("hotel", {})
            facilities = hotel_data.get("hotelFacilities", [])

            if facilities:
                hotel.amenities = list(set(hotel.amenities + facilities))
                print(f"[_enrich_hotels_with_amenities] Enriched {hotel.name} with {len(facilities)} facilities")

    return hotels


def build_hotel_artifact(
    api_response: Dict[str, Any],
    search_params: Dict[str, Any],
    check_in_date: str,
    check_out_date: str,
    nights: int,
    latitude: Optional[float],
    longitude: Optional[float],
    task_id: Optional[int],
    elapsed_ms: Optional[int] = None
) -> AgentArtifactModel:
    """Build hotel_shortlist artifact from LiteAPI API search results.

    Args:
        api_response: Response from search_hotels_via_api()
        search_params: Original search parameters from state
        check_in_date: Parsed check-in date
        check_out_date: Parsed check-out date
        nights: Calculated nights
        latitude: Geocoded latitude
        longitude: Geocoded longitude
        task_id: Optional task ID
        elapsed_ms: API response time in ms

    Returns:
        AgentArtifactModel with HotelSearchArtifactContentModel content
    """
    print(f"[build_hotel_artifact] Building artifact from API search results")

    # Extract parameters
    location = search_params.get("location", "")
    budget_max = float(search_params.get("budget_max", 0.0))
    guest_count = int(search_params.get("guest_count", 1))
    required_amenities = search_params.get("required_amenities", [])
    preferred_amenities = search_params.get("preferred_amenities", [])
    min_rating = search_params.get("min_rating")
    exclude_over_budget = search_params.get("exclude_over_budget", False)

    print(f"[build_hotel_artifact] Constraints: required={required_amenities}, "
          f"min_rating={min_rating}, exclude_over_budget={exclude_over_budget}")

    # Handle API failures
    if api_response.get("status") == "failed":
        error_msg = api_response.get("error", "unknown_error")
        return _create_failed_artifact(
            error_msg, search_params, check_in_date, check_out_date,
            nights, latitude, longitude, task_id
        )

    # Extract hotel data from API response
    rate_data = api_response.get("data", [])
    hotel_info_map = {}

    for rate_item in rate_data:
        hotel_id = rate_item.get("hotelId", "")
        if hotel_id:
            hotel_info_map[hotel_id] = {
                "id": hotel_id,
                "name": rate_item.get("hotelName", f"Hotel {hotel_id}"),
                "rating": rate_item.get("rating", 0),
                "location": rate_item.get("location", {}),
                "address": rate_item.get("address", ""),
            }

    # Build hotel list
    hotels = []
    for rate_item in rate_data:
        hotel_id = rate_item.get("hotelId", "")
        if not hotel_id:
            continue

        hotel_info = hotel_info_map.get(hotel_id, {"id": hotel_id, "name": f"Hotel {hotel_id}"})
        hotel_option = _build_hotel_option_from_data(
            hotel_info=hotel_info,
            rate_data=rate_item,
            nights=nights,
            budget_max=budget_max
        )

        if hotel_option:
            hotels.append(hotel_option)

    # Enrich with amenity details if needed
    hotels = _enrich_hotels_with_amenities(hotels, required_amenities, preferred_amenities)

    # Filter and rank
    filtered_hotels, preferred_counts = filter_hotels_by_constraints(
        hotels,
        required_amenities=required_amenities,
        preferred_amenities=preferred_amenities,
        min_rating=min_rating
    )

    ranked_hotels = rank_hotels(
        filtered_hotels,
        budget_max,
        preferred_counts=preferred_counts,
        exclude_over_budget=exclude_over_budget
    )

    # Build final artifact
    status = "failed" if not ranked_hotels else "success"
    content = HotelSearchArtifactContentModel(
        task_ref=str(task_id) if task_id else "",
        status=status,
        attempt=1,
        search_parameters=HotelSearchParametersModel(
            location=location,
            coordinates=HotelSearchCoordinatesModel(latitude=latitude, longitude=longitude) if latitude and longitude else None,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            nights=nights,
            budget_max=budget_max,
            guest_count=guest_count,
            rooms=1
        ),
        options=ranked_hotels,
        metadata=HotelSearchMetadataModel(
            total_results=len(rate_data),
            returned_results=len(ranked_hotels),
            api_response_time_ms=elapsed_ms
        )
    )

    print(f"[build_hotel_artifact] Created artifact with {len(ranked_hotels)} options")

    return AgentArtifactModel(
        name="hotel_shortlist",
        type="hotel_search",
        content=content.model_dump(),
        description=f"Found {len(ranked_hotels)} hotel options in {location}"
    )


# ============================================================================
# Intelligent Hotel Search Agent (LLM-based)
# ============================================================================


class IntelligentHotelSearchState(BaseModel):
    """State for intelligent hotel search agent."""

    query: str = Field(description="Natural language user query")
    task_id: Optional[int] = Field(default=None, description="Task ID")
    model_name: str = Field(
        default="openrouter:anthropic/claude-3.5-sonnet",
        description="LLM model to use (openrouter or ollama)"
    )

    # Intermediate state
    parsed_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="LLM-parsed search parameters"
    )
    raw_search_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw hotel search results"
    )

    # Output
    hotel_artifact: Optional[AgentArtifactModel] = Field(
        default=None, description="Final artifact with recommendations"
    )

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# LLM Helper Functions
# ============================================================================


def _call_llm(system_prompt: str, user_prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Call LLM with proper provider handling.

    Args:
        system_prompt: System message
        user_prompt: User message
        model_name: Model identifier (e.g., "openrouter:...", "ollama:...")
        temperature: Sampling temperature

    Returns:
        Response content as string
    """
    provider, model = model_name.split(":", 1) if ":" in model_name else ("openrouter", model_name)

    if provider == "ollama":
        # Use Ollama Cloud API
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not found in environment")

        client = OllamaClient(
            host="https://ollama.com",
            headers={'Authorization': f'Bearer {api_key}'}
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Non-streaming call
        response = client.chat(model=model, messages=messages, stream=False)
        return response['message']['content']

    else:
        # Use OpenRouter or other OpenAI-compatible provider
        llm = make_chat_model(model_name=model_name, temperature=temperature)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content


# ============================================================================
# LLM Parsing
# ============================================================================

QUERY_PARSER_SYSTEM_PROMPT = """You are a hotel search parameter extractor.

Your task: Parse natural language hotel queries into structured JSON parameters.

Today's date: {{today}}

Output JSON format:
{{
  "location": "City, Country",
  "dates": "YYYY-MM-DD to YYYY-MM-DD",
  "budget_max": float,
  "guest_count": int,
  "required_amenities": ["wifi", "pool"],
  "preferred_amenities": ["gym", "spa"],
  "min_rating": float,
  "purpose": "business|vacation|honeymoon|family",
  "special_requirements": "string or null"
}}

Rules:
1. **Location**: Extract city and country. If country is missing, infer from context.
2. **Dates**: Convert relative dates ("next month", "in 2 weeks") to absolute YYYY-MM-DD.
3. **Budget**: Extract max per night. If "total budget", divide by nights.
4. **Amenities**: Map natural language to standardized terms:
   - "wifi/internet" → "wifi"
   - "swimming/pool" → "pool"
   - "gym/fitness" → "gym"
   - "parking/garage" → "parking"
   - "breakfast" → "breakfast"
5. **Purpose**: Infer from context (honeymoon, business trip, family vacation).
6. **Guest count**: Default to 2 if not specified.
7. **Required vs Preferred**: "need/must have" = required, "would like/prefer" = preferred.

Examples:

Query: "Find a hotel in Paris for next week, need wifi and pool, max 200 per night"
Output: {{
  "location": "Paris, France",
  "dates": "2026-05-06 to 2026-05-13",
  "budget_max": 200.0,
  "guest_count": 2,
  "required_amenities": ["wifi", "pool"],
  "preferred_amenities": [],
  "purpose": "vacation"
}}

Query: "Business trip to Munich, June 15-18, hotel near airport with parking"
Output: {{
  "location": "Munich, Germany",
  "dates": "2026-06-15 to 2026-06-18",
  "budget_max": 300.0,
  "guest_count": 1,
  "required_amenities": ["parking"],
  "preferred_amenities": ["wifi"],
  "purpose": "business",
  "special_requirements": "near airport"
}}

Output ONLY the JSON object, no explanation."""


def parse_query_with_llm(
    query: str, model_name: str = "openrouter:anthropic/claude-3.5-sonnet"
) -> Dict[str, Any]:
    """Parse natural language query into structured search parameters.

    Args:
        query: Natural language hotel search query
        model_name: LLM model to use. Options:
            - "openrouter:anthropic/claude-3.5-sonnet" (default, team credits)
            - "ollama:gpt-oss:120b" (personal testing with Ollama Cloud)
    """
    print(f"[parse_query_with_llm] Parsing query with LLM: {model_name}")

    today = datetime.now().date().isoformat()
    system_prompt = QUERY_PARSER_SYSTEM_PROMPT.replace("{{today}}", today)

    try:
        content = _call_llm(
            system_prompt=system_prompt,
            user_prompt=f"Query: {query}",
            model_name=model_name,
            temperature=0.0
        )

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        params = json.loads(content)
        print(f"[parse_query_with_llm] Parsed: {params}")
        return params

    except json.JSONDecodeError as e:
        print(f"[parse_query_with_llm] JSON parse error: {e}")
        print(f"[parse_query_with_llm] Raw response: {content}")
        return {}
    except Exception as e:
        print(f"[parse_query_with_llm] Error: {e}")
        return {}


# ============================================================================
# Recommendation Synthesis
# ============================================================================

RECOMMENDATION_SYSTEM_PROMPT = """You are a hotel recommendation expert.

Given search results, generate personalized recommendations that help the user choose.

Format:
**Recommended Hotels**

1. **[Hotel Name]** (€[price]/night, [rating]/10)
   - **Why it fits**: [Specific reason based on user's purpose/requirements]
   - **Location**: [Address + transit info if available]
   - **Highlights**: [Top 3 amenities/features]
   - **Note**: [Any caveats, e.g. "over budget but exceptional value"]

2. [Repeat for top 3-5 hotels]

**Key Considerations**:
- [Any trade-offs the user should know about]
- [Alternative suggestions if criteria are too restrictive]

Rules:
- Focus on WHY each hotel matches the user's purpose (business, honeymoon, etc.)
- Mention over-budget options if they offer exceptional value
- Be concise - 2-3 sentences per hotel
- NO generic descriptions - be specific based on actual amenities/location"""


def synthesize_recommendations(
    query: str,
    parsed_params: Dict[str, Any],
    artifact_content: Dict[str, Any],
    model_name: str = "openrouter:anthropic/claude-3.5-sonnet",
) -> str:
    """Generate LLM-based recommendations from search results.

    Args:
        query: Original user query
        parsed_params: Parsed search parameters
        artifact_content: Hotel search results
        model_name: LLM model to use. Options:
            - "openrouter:anthropic/claude-3.5-sonnet" (default, team credits)
            - "ollama:gpt-oss:120b" (personal testing with Ollama Cloud)
    """
    print(f"[synthesize_recommendations] Generating recommendations with {model_name}")

    options = artifact_content.get("options", [])
    if not options:
        return "No hotels found matching your criteria. Try widening your search parameters."

    # Prepare hotel summaries for LLM
    hotel_summaries = []
    for hotel in options[:5]:  # Top 5
        summary = {
            "name": hotel["name"],
            "price": f"{hotel['currency']} {hotel['nightly_rate']:.0f}",
            "rating": hotel["rating"],
            "amenities": hotel["amenities"][:10],
            "over_budget": hotel["over_budget"],
            "address": hotel.get("address", ""),
        }
        hotel_summaries.append(summary)

    user_context = f"""
User Query: {query}

Parsed Requirements:
- Location: {parsed_params.get('location')}
- Dates: {parsed_params.get('dates')}
- Budget: {parsed_params.get('budget_max')}/night
- Purpose: {parsed_params.get('purpose', 'unspecified')}
- Required: {', '.join(parsed_params.get('required_amenities', []))}
- Preferred: {', '.join(parsed_params.get('preferred_amenities', []))}

Search Results ({len(options)} hotels):
{json.dumps(hotel_summaries, indent=2)}
"""

    try:
        recommendations = _call_llm(
            system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
            user_prompt=user_context,
            model_name=model_name,
            temperature=0.3
        )

        print(f"[synthesize_recommendations] Generated {len(recommendations)} chars")
        return recommendations

    except Exception as e:
        print(f"[synthesize_recommendations] Error: {e}")
        return f"Error generating recommendations: {str(e)}"


# ============================================================================
# LangGraph Nodes
# ============================================================================


def parse_node(state: IntelligentHotelSearchState) -> Dict[str, Any]:
    """Node 1: Parse natural language query into structured parameters."""
    print("\n[parse_node] Parsing user query with LLM")

    parsed = parse_query_with_llm(state.query, model_name=state.model_name)

    if not parsed:
        # Fallback: return error artifact
        return {
            "parsed_parameters": None,
            "hotel_artifact": AgentArtifactModel(
                name="hotel_shortlist",
                type="hotel_search",
                content={
                    "status": "failed",
                    "errors": [{"code": "parse_error", "message": "Could not parse query"}],
                },
                description="Failed to parse hotel search query",
            ),
        }

    return {"parsed_parameters": parsed}


def search_node(state: IntelligentHotelSearchState) -> Dict[str, Any]:
    """Node 2: Execute deterministic hotel search."""
    print("\n[search_node] Executing hotel search")

    params = state.parsed_parameters
    if not params:
        return {"raw_search_results": None}

    # Extract parameters
    location = params.get("location", "")
    dates = params.get("dates", "")
    budget_max = float(params.get("budget_max", 150.0))
    guest_count = int(params.get("guest_count", 2))
    required_amenities = params.get("required_amenities", [])
    preferred_amenities = params.get("preferred_amenities", [])
    min_rating = params.get("min_rating")

    # Parse dates
    if " to " in dates:
        check_in, check_out = dates.split(" to ")
        check_in = check_in.strip()
        check_out = check_out.strip()
    else:
        # Fallback
        today = datetime.now().date()
        check_in = (today + timedelta(days=7)).isoformat()
        check_out = (today + timedelta(days=14)).isoformat()

    nights = calculate_nights(check_in, check_out)

    # Geocode + search
    city_name, country_code = parse_location(location)
    latitude, longitude = geocode_location(location)

    place_id = None
    if city_name and country_code:
        place_result = search_places(f"{city_name}, {country_code}")
        place_id = place_result.get("placeId")

    # API search
    api_response = search_hotels_via_api(
        place_id=place_id,
        city_name=city_name,
        country_code=country_code,
        check_in_date=check_in,
        check_out_date=check_out,
        guest_count=guest_count,
    )

    # Build artifact with enrichment
    artifact = build_hotel_artifact(
        api_response=api_response,
        search_params={
            "location": location,
            "budget_max": budget_max,
            "guest_count": guest_count,
            "required_amenities": required_amenities,
            "preferred_amenities": preferred_amenities,
            "min_rating": min_rating,
        },
        check_in_date=check_in,
        check_out_date=check_out,
        nights=nights,
        latitude=latitude,
        longitude=longitude,
        task_id=state.task_id,
        elapsed_ms=api_response.get("api_response_time_ms"),
    )

    return {
        "raw_search_results": api_response,
        "hotel_artifact": artifact,
    }


def synthesize_node(state: IntelligentHotelSearchState) -> Dict[str, Any]:
    """Node 3: Generate LLM-based recommendations."""
    print("\n[synthesize_node] Generating recommendations")

    if not state.hotel_artifact:
        return {}

    content = state.hotel_artifact.content

    if content.get("status") == "failed":
        # Don't synthesize for failed searches
        return {}

    # Generate recommendations
    recommendations = synthesize_recommendations(
        query=state.query,
        parsed_params=state.parsed_parameters,
        artifact_content=content,
        model_name=state.model_name,
    )

    # Add recommendations to artifact
    content["recommendations"] = recommendations

    updated_artifact = AgentArtifactModel(
        name="hotel_shortlist",
        type="hotel_search",
        content=content,
        description=f"Found {len(content.get('options', []))} hotels with AI recommendations",
    )

    return {"hotel_artifact": updated_artifact}


# ============================================================================
# Graph Factory
# ============================================================================


def make_intelligent_hotel_graph():
    """Build intelligent hotel search graph with LLM integration."""
    print("[make_intelligent_hotel_graph] Building graph")

    graph = StateGraph(IntelligentHotelSearchState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)

    # Linear flow
    graph.set_entry_point("parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ============================================================================
# Convenience Function
# ============================================================================


def intelligent_hotel_search(
    query: str,
    task_id: Optional[int] = None,
    model_name: str = "openrouter:anthropic/claude-3.5-sonnet"
) -> AgentArtifactModel:
    """Execute intelligent hotel search from natural language query.

    Args:
        query: Natural language hotel search query
        task_id: Optional task ID
        model_name: LLM model to use. Options:
            - "openrouter:anthropic/claude-3.5-sonnet" (default, team credits)
            - "ollama:gpt-oss:120b" (personal testing with Ollama Cloud)

    Returns:
        AgentArtifactModel with hotel recommendations

    Examples:
        >>> # Team usage with OpenRouter (default)
        >>> result = intelligent_hotel_search(
        ...     "Find romantic hotel in Barcelona for honeymoon next month, need pool and wifi"
        ... )

        >>> # Personal testing with Ollama Cloud
        >>> result = intelligent_hotel_search(
        ...     "Find hotel in Munich with parking",
        ...     model_name="ollama:gpt-oss:120b"
        ... )
        >>> print(result.content["recommendations"])
    """
    graph = make_intelligent_hotel_graph()

    state = IntelligentHotelSearchState(
        query=query,
        task_id=task_id,
        model_name=model_name
    )

    result = graph.invoke(state)
    return result["hotel_artifact"]


if __name__ == "__main__":
    # Example usage
    result = intelligent_hotel_search(
        "Find me a nice hotel in Barcelona for next week, we love swimming and need wifi, max 150 per night"
    )

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print(result.content.get("recommendations", "No recommendations generated"))
