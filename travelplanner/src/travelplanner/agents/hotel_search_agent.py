"""Hotel Search Agent using LiteAPI REST API.

This agent searches for accommodation using the LiteAPI direct REST API and returns
a typed hotel_shortlist artifact. It follows the LangGraph StateGraph pattern
and system architecture established in travelplanner.agents.

Architecture:
- Uses Pydantic models for state and artifacts
- Returns AgentArtifactModel with HotelSearchArtifactContentModel payload
- Follows spawn-on-demand pattern (no LLM calls, pure API integration)
- Implements typed artifact schema with alternatives (3-10 hotel options)

Usage:
    from travelplanner.agents.hotel_search_agent import make_graph

    graph = make_graph()
    result = graph.invoke({
        "query": "Find hotels in Barcelona",
        "search_parameters": {
            "location": "Barcelona, Spain",
            "dates": "2026-06-01 to 2026-06-07",
            "budget_max": 150.0,
            "guest_count": 2
        },
        "task_id": 1
    })
    artifact = result["hotel_artifact"]
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import threading

import pycountry
import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
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

# Geocoding cache to avoid repeated calls (with max size limit)
_GEOCODING_CACHE_MAX_SIZE = 1000
_geocoding_cache: Dict[str, Tuple[float, float]] = {}
_last_nominatim_request: float = 0.0
_geocoding_lock = threading.Lock()


# ============================================================================
# State Models
# ============================================================================


class HotelSearchAgentState(BaseModel):
    """State model for hotel search agent following travelplanner conventions."""

    query: str = Field(description="User's original travel query")
    search_parameters: Dict[str, Any] = Field(
        description="Hotel search parameters (location, dates, budget, guests, amenities, etc.)"
    )
    task_id: Optional[int] = Field(
        default=None, description="Task ID if spawned by planner"
    )
    hotel_artifact: Optional[AgentArtifactModel] = Field(
        default=None, description="Output artifact with hotel shortlist"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# Expected search_parameters structure:
# {
#     "location": str,              # Required: "Barcelona, Spain"
#     "dates": str,                 # Required: "2026-06-01 to YYYY-MM-DD"
#     "budget_max": float,          # Required: Maximum per night
#     "guest_count": int,           # Required: Number of guests
#     "required_amenities": list[str],  # Optional: ["wifi", "pool", "parking"]
#     "preferred_amenities": list[str], # Optional: ["gym", "breakfast"]
#     "exclude_over_budget": bool,  # Optional: Default False (show alternatives)
#     "min_rating": float,          # Optional: Minimum rating (0-10)
# }


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
    print(f"[geocode_location] Geocoding location: {location}")

    with _geocoding_lock:
        if location in _geocoding_cache:
            print(f"[geocode_location] Using cached coordinates")
            return _geocoding_cache[location]

        global _last_nominatim_request
        time_since_last = time.time() - _last_nominatim_request
        if time_since_last < 1.0:
            sleep_time = 1.0 - time_since_last
            print(f"[geocode_location] Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        _last_nominatim_request = time.time()

        try:
            headers = {
                "User-Agent": "TravelPlannerAgent/1.0 (educational project)"
            }
            params = {
                "q": location,
                "format": "json",
                "limit": 1
            }

            response = requests.get(
                f"{NOMINATIM_BASE}/search",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            results = response.json()
            if not results:
                print(f"[geocode_location] No results found for location: {location}")
                return None, None

            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])

            if len(_geocoding_cache) >= _GEOCODING_CACHE_MAX_SIZE:
                first_key = next(iter(_geocoding_cache))
                del _geocoding_cache[first_key]
                print(f"[geocode_location] Cache evicted: {first_key}")

            _geocoding_cache[location] = (lat, lon)

            print(f"[geocode_location] Found coordinates: ({lat}, {lon})")
            return lat, lon

        except requests.exceptions.Timeout:
            print(f"[geocode_location] Request timeout for location: {location}")
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"[geocode_location] Request error: {e}")
            return None, None
        except (KeyError, ValueError, IndexError) as e:
            print(f"[geocode_location] Failed to parse response: {e}")
            return None, None
        except Exception as e:
            print(f"[geocode_location] Unexpected error: {e}")
            return None, None


def parse_date_range(date_string: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse date range string into check-in and check-out dates.

    Args:
        date_string: Date string in format "YYYY-MM-DD to YYYY-MM-DD"
                     or "YYYY-MM-DD - YYYY-MM-DD"

    Returns:
        Tuple of (check_in_date, check_out_date) or (None, None) if parsing fails
    """
    print(f"[parse_date_range] Parsing date string: {date_string}")

    try:
        if " to " in date_string:
            parts = date_string.split(" to ", maxsplit=1)
        elif " - " in date_string:
            parts = date_string.split(" - ", maxsplit=1)
        else:
            print(f"[parse_date_range] Invalid date format: {date_string}")
            return None, None

        if len(parts) != 2:
            print(f"[parse_date_range] Expected 2 parts, got {len(parts)}")
            return None, None

        check_in = parts[0].strip()
        check_out = parts[1].strip()

        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d")

        today = datetime.now().date()
        check_in_date = check_in_dt.date()
        check_out_date = check_out_dt.date()

        if check_in_date < today:
            print(f"[parse_date_range] Check-in date cannot be in the past")
            return None, None

        if check_out_date <= check_in_date:
            print(f"[parse_date_range] Check-out date must be after check-in date")
            return None, None

        nights = (check_out_date - check_in_date).days
        if nights > 365:
            print(f"[parse_date_range] Stay duration exceeds maximum (365 nights)")
            return None, None

        max_future = today + timedelta(days=730)
        if check_in_date > max_future:
            print(f"[parse_date_range] Check-in date too far in future (max 2 years)")
            return None, None

        print(f"[parse_date_range] Parsed dates: {check_in} to {check_out}")
        return check_in, check_out

    except ValueError as e:
        print(f"[parse_date_range] Date format error: {e}")
        return None, None
    except Exception as e:
        print(f"[parse_date_range] Unexpected error: {e}")
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

        if places:
            place = places[0]
            print(f"[search_places] Found: {place.get('displayName')} (placeId: {place.get('placeId')})")
            return {
                "placeId": place.get("placeId"),
                "displayName": place.get("displayName"),
                "formattedAddress": place.get("formattedAddress")
            }

        return {}

    except Exception as e:
        print(f"[search_places] Error: {e}")
        return {}


def search_hotels_via_api(
    place_id: Optional[str],
    city_name: Optional[str],
    country_code: str,
    check_in_date: str,
    check_out_date: str,
    guest_count: int,
    currency: str = "EUR",
    guest_nationality: str = "US",
) -> Dict[str, Any]:
    """Search for hotel rates using LiteAPI REST API.

    Uses LiteAPI endpoint: POST /hotels/rates

    Args:
        place_id: Optional place ID from search_places
        city_name: City name (used if no place_id)
        country_code: ISO 3166-1 alpha-2 country code
        check_in_date: Check-in date in ISO format (YYYY-MM-DD)
        check_out_date: Check-out date in ISO format (YYYY-MM-DD)
        guest_count: Number of adult guests
        currency: Currency code (default: EUR)
        guest_nationality: Guest nationality ISO code (default: US)

    Returns:
        Dict with 'status', 'data' (list of rate results), 'hotels' (hotel info), and optional 'error'
    """
    print(f"[search_hotels_via_api] Searching hotels in {city_name or place_id}, {country_code}")
    print(f"[search_hotels_via_api] Dates: {check_in_date} to {check_out_date}")
    print(f"[search_hotels_via_api] Guests: {guest_count}")

    start_time = time.time()

    try:
        body = {
            "occupancies": [{"adults": guest_count}],
            "currency": currency,
            "guestNationality": guest_nationality,
            "checkin": check_in_date,
            "checkout": check_out_date,
            "roomMapping": True,
            "maxRatesPerHotel": 1,
            "includeHotelData": True
        }

        if place_id:
            body["placeId"] = place_id
        else:
            body["cityName"] = city_name
            body["countryCode"] = country_code

        response = requests.post(
            f"{LITEAPI_BASE_URL}/hotels/rates",
            headers=_get_api_headers(),
            json=body,
            timeout=30
        )
        response.raise_for_status()

        elapsed_ms = int((time.time() - start_time) * 1000)
        print(f"[search_hotels_via_api] API response time: {elapsed_ms}ms")

        data = response.json()

        if data.get("error"):
            error = data.get("error", {})
            return {
                "status": "failed",
                "data": [],
                "hotels": [],
                "error": f"API error {error.get('code')}: {error.get('description', 'Unknown error')}"
            }

        rate_data = data.get("data", [])
        hotel_data = data.get("hotels", [])

        return {
            "status": "success",
            "data": rate_data,
            "hotels": hotel_data,
            "meta": {"total": len(rate_data)},
            "api_response_time_ms": elapsed_ms
        }

    except requests.exceptions.Timeout:
        print(f"[search_hotels_via_api] Request timeout")
        return {
            "status": "failed",
            "data": [],
            "hotels": [],
            "error": "Request timed out"
        }
    except requests.exceptions.RequestException as e:
        print(f"[search_hotels_via_api] Request error: {e}")
        return {
            "status": "failed",
            "data": [],
            "hotels": [],
            "error": str(e)
        }
    except Exception as e:
        print(f"[search_hotels_via_api] Unexpected error: {e}")
        return {
            "status": "failed",
            "data": [],
            "hotels": [],
            "error": str(e)
        }


def filter_hotels_by_constraints(
    hotels: List[HotelOptionModel],
    required_amenities: Optional[List[str]] = None,
    preferred_amenities: Optional[List[str]] = None,
    min_rating: Optional[float] = None,
) -> Tuple[List[HotelOptionModel], Dict[str, int]]:
    """Filter hotels by user constraints.

    Args:
        hotels: List of hotel models
        required_amenities: Must-have amenities (case-insensitive)
        preferred_amenities: Nice-to-have amenities (used for scoring)
        min_rating: Minimum rating threshold (0-10)

    Returns:
        Tuple of (filtered list of hotels, dict mapping hotel_id to preferred_amenity_count)
    """
    filtered = []
    preferred_counts = {}

    # Amenity keyword mapping for better fuzzy matching
    AMENITY_KEYWORDS = {
        "wifi": ["wifi", "wi-fi", "internet", "wireless"],
        "pool": ["pool", "swimming"],
        "parking": ["parking", "garage", "valet"],
        "gym": ["gym", "fitness", "workout", "exercise"],
        "breakfast": ["breakfast", "morning meal"],
        "spa": ["spa", "wellness", "massage"],
        "restaurant": ["restaurant", "dining"],
        "bar": ["bar", "lounge", "pub"],
        "ac": ["air conditioning", "aircon", "a/c"],
        "pets": ["pet", "dog", "cat", "animal"],
    }

    def normalize_amenity(amenity: str) -> str:
        return amenity.lower().strip().replace("-", "").replace("_", "")

    def amenity_matches(required: str, hotel_amenity: str) -> bool:
        """Check if hotel amenity satisfies required amenity using keyword matching."""
        required_norm = normalize_amenity(required)
        hotel_norm = normalize_amenity(hotel_amenity)

        # Direct substring match
        if required_norm in hotel_norm:
            return True

        # Keyword-based fuzzy match
        if required in AMENITY_KEYWORDS:
            keywords = AMENITY_KEYWORDS[required]
            for keyword in keywords:
                keyword_norm = normalize_amenity(keyword)
                if keyword_norm in hotel_norm:
                    return True

        return False

    required_normalized = [normalize_amenity(a) for a in (required_amenities or [])]
    preferred_normalized = [normalize_amenity(a) for a in (preferred_amenities or [])]

    for hotel in hotels:
        if min_rating is not None and hotel.rating < min_rating:
            print(f"[filter_hotels] {hotel.name} filtered: rating {hotel.rating} < {min_rating}")
            continue

        hotel_amenities_normalized = [normalize_amenity(a) for a in hotel.amenities]

        has_all_required = True
        if required_amenities:
            for req_amenity in required_amenities:
                if not any(amenity_matches(req_amenity, hotel_amenity) for hotel_amenity in hotel.amenities):
                    print(f"[filter_hotels] {hotel.name} filtered: missing required amenity '{req_amenity}'")
                    has_all_required = False
                    break

        if not has_all_required:
            continue

        preferred_count = 0
        if preferred_amenities:
            for pref_amenity in preferred_amenities:
                if any(amenity_matches(pref_amenity, hotel_amenity) for hotel_amenity in hotel.amenities):
                    preferred_count += 1

        preferred_counts[hotel.accommodation_id] = preferred_count

        filtered.append(hotel)

    print(f"[filter_hotels] Filtered {len(hotels)} → {len(filtered)} hotels")
    return filtered, preferred_counts


def rank_hotels(
    hotels: List[HotelOptionModel],
    budget_max: float,
    preferred_counts: Optional[Dict[str, int]] = None,
    exclude_over_budget: bool = False,
    min_results: int = 3,
    max_results: int = 10
) -> List[HotelOptionModel]:
    """Rank hotels by preference: budget fit, rating, preferred amenities, then price.

    Args:
        hotels: List of hotel models
        budget_max: Maximum budget per night
        preferred_counts: Dict mapping hotel_id to preferred amenity count
        exclude_over_budget: If True, completely exclude over-budget hotels
        min_results: Minimum number of results to return (default: 3)
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of ranked hotels (up to max_results, may be less if not enough hotels)
    """
    print(f"[rank_hotels] Ranking {len(hotels)} hotels with budget_max={budget_max}, exclude_over_budget={exclude_over_budget}")

    if not hotels:
        print(f"[rank_hotels] No hotels to rank")
        return []

    if preferred_counts is None:
        preferred_counts = {}

    within_budget = [h for h in hotels if h.nightly_rate <= budget_max]
    over_budget = [h for h in hotels if h.nightly_rate > budget_max]

    print(f"[rank_hotels] Within budget: {len(within_budget)}, Over budget: {len(over_budget)}")

    def sort_key_within_budget(h):
        preferred_count = preferred_counts.get(h.accommodation_id, 0)
        return (-preferred_count, -h.rating, h.nightly_rate)

    within_budget.sort(key=sort_key_within_budget)

    over_budget.sort(key=lambda h: h.nightly_rate)

    if exclude_over_budget:
        ranked = within_budget
        print(f"[rank_hotels] Excluding over-budget hotels")
    else:
        ranked = within_budget + over_budget

    count = min(max_results, len(ranked))
    result = ranked[:count]

    for i, hotel in enumerate(result):
        hotel.rank = i + 1

    print(f"[rank_hotels] Returning top {len(result)} hotels")
    return result


# ============================================================================
# Artifact Construction
# ============================================================================


def _extract_price_from_rate(rate: Dict[str, Any]) -> Tuple[float, str]:
    """Extract price and currency from a rate object.

    LiteAPI rate structure:
    {
        "retailRate": {
            "total": [{"amount": 131.54, "currency": "USD"}],
            "taxesAndFees": [{"included": true}]
        }
    }

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

    location = search_params.get("location", "")
    budget_max = float(search_params.get("budget_max", 0.0))
    guest_count = int(search_params.get("guest_count", 1))

    required_amenities = search_params.get("required_amenities", [])
    preferred_amenities = search_params.get("preferred_amenities", [])
    min_rating = search_params.get("min_rating")
    exclude_over_budget = search_params.get("exclude_over_budget", False)

    print(f"[build_hotel_artifact] Constraints: required_amenities={required_amenities}, "
          f"min_rating={min_rating}, exclude_over_budget={exclude_over_budget}")

    if api_response.get("status") == "failed":
        error_msg = api_response.get("error", "unknown_error")
        error_code = "unknown_error"
        if "api_key" in error_msg.lower():
            error_code = "missing_api_key"
        elif "timeout" in error_msg.lower():
            error_code = "timeout_error"
        elif "http" in error_msg.lower():
            error_code = "http_error"

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

    hotels = []

    rate_data = api_response.get("data", [])
    hotel_info_list = api_response.get("hotels", [])

    hotel_info_map = {h.get("id"): h for h in hotel_info_list if h.get("id")}

    # Step 1: Build initial hotel list from rate data
    for rate_item in rate_data:
        hotel_id = rate_item.get("hotelId", "")
        if not hotel_id:
            continue

        hotel_info = hotel_info_map.get(hotel_id, {})
        if not hotel_info:
            hotel_info = {"id": hotel_id, "name": f"Hotel {hotel_id}"}

        hotel_option = _build_hotel_option_from_data(
            hotel_info=hotel_info,
            rate_data=rate_item,
            nights=nights,
            budget_max=budget_max
        )

        if hotel_option is not None:
            hotels.append(hotel_option)

    # Step 2: Enrich top hotels with full details (amenities) if needed
    fetch_details = bool(required_amenities or preferred_amenities)

    if fetch_details and hotels:
        print(f"[build_hotel_artifact] Fetching details for top {min(20, len(hotels))} hotels to get amenities")

        # Fetch details for top 20 candidates (before filtering)
        top_candidates = sorted(hotels, key=lambda h: (-h.rating, h.nightly_rate))[:20]

        for hotel in top_candidates:
            details_response = get_hotel_details(hotel.accommodation_id, timeout=4)

            if details_response.get("status") == "success":
                hotel_data = details_response.get("hotel", {})
                facilities = hotel_data.get("hotelFacilities", [])

                if facilities:
                    # Update amenities with full details
                    hotel.amenities = list(set(hotel.amenities + facilities))
                    print(f"[build_hotel_artifact] Enriched {hotel.name} with {len(facilities)} facilities")

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
# LangGraph Node
# ============================================================================


def hotel_search_node(state: HotelSearchAgentState) -> dict[str, Any]:
    """LangGraph node that performs hotel search via LiteAPI REST API.

    Args:
        state: HotelSearchAgentState with search parameters

    Returns:
        Updated state dict with hotel_artifact
    """
    print("\n[hotel_search_node] Starting hotel search node execution...")

    search_params = state.search_parameters
    task_id = state.task_id

    location = search_params.get("location", "")
    dates = search_params.get("dates", "")
    budget_max = float(search_params.get("budget_max", 0.0))
    guest_count = int(search_params.get("guest_count", 1))

    print(f"[hotel_search_node] Location: {location}")
    print(f"[hotel_search_node] Dates: {dates}")
    print(f"[hotel_search_node] Budget: {budget_max} per night")
    print(f"[hotel_search_node] Guests: {guest_count}")

    check_in_date, check_out_date = parse_date_range(dates)
    if not check_in_date or not check_out_date:
        error_content = HotelSearchArtifactContentModel(
            task_ref=str(task_id) if task_id else "",
            status="failed",
            attempt=1,
            search_parameters=HotelSearchParametersModel(
                location=location,
                check_in_date="",
                check_out_date="",
                nights=1,  # Use 1 as placeholder for invalid dates
                budget_max=budget_max,
                guest_count=guest_count,
            ),
            options=[],
            metadata=HotelSearchMetadataModel(total_results=0, returned_results=0),
            errors=[HotelSearchErrorModel(code="parse_error", message=f"Invalid date format: {dates}")]
        )
        return {"hotel_artifact": AgentArtifactModel(
            name="hotel_shortlist",
            type="hotel_search",
            content=error_content.model_dump(),
            description=f"Hotel search failed: Invalid date format"
        )}

    nights = calculate_nights(check_in_date, check_out_date)

    city_name, country_code = parse_location(location)

    latitude, longitude = geocode_location(location)

    place_id = None
    if city_name and country_code:
        place_result = search_places(f"{city_name}, {country_code}")
        place_id = place_result.get("placeId")

    if not city_name or not country_code:
        if latitude is None or longitude is None:
            error_content = HotelSearchArtifactContentModel(
                task_ref=str(task_id) if task_id else "",
                status="failed",
                attempt=1,
                search_parameters=HotelSearchParametersModel(
                    location=location,
                    check_in_date=check_in_date,
                    check_out_date=check_out_date,
                    nights=nights,
                    budget_max=budget_max,
                    guest_count=guest_count,
                ),
                options=[],
                metadata=HotelSearchMetadataModel(total_results=0, returned_results=0),
                errors=[HotelSearchErrorModel(code="geocoding_error", message=f"Could not parse location: {location}")]
            )
            return {"hotel_artifact": AgentArtifactModel(
                name="hotel_shortlist",
                type="hotel_search",
                content=error_content.model_dump(),
                description=f"Hotel search failed: Could not parse location"
            )}

    api_response = search_hotels_via_api(
        place_id=place_id,
        city_name=city_name,
        country_code=country_code,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        guest_count=guest_count
    )

    if api_response.get("status") == "failed":
        error_content = HotelSearchArtifactContentModel(
            task_ref=str(task_id) if task_id else "",
            status="failed",
            attempt=1,
            search_parameters=HotelSearchParametersModel(
                location=location,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                nights=nights,
                budget_max=budget_max,
                guest_count=guest_count,
            ),
            options=[],
            metadata=HotelSearchMetadataModel(total_results=0, returned_results=0),
            errors=[HotelSearchErrorModel(code="http_error", message=api_response.get("error", "unknown_error"))]
        )
        return {"hotel_artifact": AgentArtifactModel(
            name="hotel_shortlist",
            type="hotel_search",
            content=error_content.model_dump(),
            description=f"Hotel search failed: {api_response.get('error')}"
        )}

    elapsed_ms = api_response.get("api_response_time_ms")

    artifact = build_hotel_artifact(
        api_response=api_response,
        search_params=search_params,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        nights=nights,
        latitude=latitude,
        longitude=longitude,
        task_id=task_id,
        elapsed_ms=elapsed_ms
    )

    print(f"[hotel_search_node] Hotel search node execution complete")
    return {"hotel_artifact": artifact}


# ============================================================================
# Graph Factory
# ============================================================================


def make_graph():
    """Build hotel search agent graph following travelplanner conventions.

    Returns:
        Compiled LangGraph Pregel object
    """
    print(f"[make_graph] Initializing hotel search graph")

    graph = StateGraph(HotelSearchAgentState)
    graph.add_node("hotel_search", hotel_search_node)
    graph.set_entry_point("hotel_search")
    graph.add_edge("hotel_search", END)

    print("[make_graph] Graph compilation complete")
    return graph.compile()