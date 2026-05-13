from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.config import get_setting
from travelplanner.schema.restaurant_search_artifact import (
    RestaurantArtifactContentModel,
    RestaurantCandidateModel,
    RestaurantItemModel,
    RestaurantLocationModel,
    RestaurantParamsModel,
    RestaurantSearchErrorModel,
    RestaurantSelectionModel,
)
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    StateContractModel,
    TaskModel,
)
from travelplanner.utils.llm import invoke_structured_model

# Google Places API (New) v1 – Text Search endpoint
_GOOGLE_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_RESULTS = 5

# Mapping from budget label to Google Places API price-level enum
_BUDGET_PRICE_LEVEL: dict[str, str] = {"low": "PRICE_LEVEL_INEXPENSIVE", "medium": "PRICE_LEVEL_MODERATE", "high": "PRICE_LEVEL_EXPENSIVE"}

_PARAM_EXTRACTION_SYSTEM_PROMPT = """\
You are a restaurant search parameter extractor for a travel planning assistant.

Given a natural-language restaurant task, extract structured search parameters.

Rules:
- Identify the destination city from the text.
- Identify cuisine type if explicitly mentioned (e.g. "Italian", "sushi", "vegan").
- Identify budget category if mentioned: low / medium / high.
- Identify meal type if mentioned: breakfast / brunch / lunch / dinner / any.
- List any dietary restrictions mentioned (e.g. "vegetarian", "gluten-free", "halal", "kosher").
- If a minimum rating is mentioned, extract it as a float; otherwise leave null.
- If number of diners is mentioned, use that value; otherwise default to 1.
- If a preferred time is mentioned, preserve the raw text (e.g. "19:30", "evening").
- Capture any additional free-text requirements in additional_notes.
- Return JSON only — no extra text, no markdown fences.
"""

_SELECTION_SYSTEM_PROMPT = """\
You are a restaurant selector. Given a traveller profile and a list of restaurant candidates, choose the best option.

Prefer venues that:
- Match the cuisine preference
- Fit the budget level
- Have high ratings and positive review signals
- Suit any dietary restrictions mentioned

Return JSON only: {"selected_index": int, "selection_reason": "one sentence"}
"""


@dataclass(frozen=True)
class RestaurantSearchConfig:
    api_key: str = ""
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    max_results: int = _DEFAULT_MAX_RESULTS


def _parse_int(value: str | None, *, default: int, minimum: int) -> int:
    if value is None or value.strip() == "":
        return default
    return max(int(value.strip()), minimum)


def load_config_from_env() -> RestaurantSearchConfig:
    cfg_prefix = "agents.restaurant_search"
    return RestaurantSearchConfig(
        api_key=os.getenv(
            "GOOGLE_PLACES_API_KEY",
            str(get_setting(f"{cfg_prefix}.api_key", "")),
        ).strip(),
        timeout_seconds=_parse_int(
            os.getenv("TRAVELPLANNER_RESTAURANT_TIMEOUT_SECONDS"),
            default=int(get_setting(f"{cfg_prefix}.timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
            minimum=5,
        ),
        max_results=_parse_int(
            os.getenv("TRAVELPLANNER_RESTAURANT_MAX_RESULTS"),
            default=int(get_setting(f"{cfg_prefix}.max_results", _DEFAULT_MAX_RESULTS)),
            minimum=1,
        ),
    )


class RestaurantSearchAgentState(BaseModel):
    query: str
    model_name: str
    system_state: StateContractModel | None = Field(
        default=None,
        description="Reference to global system state (optional, used when wired into a full workflow)",
    )
    agent_key: str = Field(
        default="restaurant_search",
        description="Key for storing artifacts in SystemState",
    )
    temperature: float = 0.0
    task_list: list[TaskModel] = Field(default_factory=list)
    message_history: MessageHistoryModel | None = None
    agent_artifacts: dict[str, list[AgentArtifactModel]] = Field(
        default_factory=dict,
        description="Local artifact storage when system_state is not provided",
    )


# ── Parameter extraction ────────────────────────────────────────────────────

def _build_param_extraction_prompt(task_text: str) -> str:
    return (
        "Extract restaurant search parameters from the following task.\n\n"
        f"Task: {task_text.strip()}\n\n"
        "Return strictly valid JSON with this shape:\n"
        '{"city": "Barcelona", "cuisine": "Italian", "budget": "medium", "meal_type": "dinner", '
        '"dietary_restrictions": ["vegetarian"], "min_rating": 4.0, "num_people": 2, '
        '"preferred_time": "19:30", "additional_notes": "outdoor seating preferred"}'
    )


def _extract_restaurant_params(
    task_text: str,
    model_name: str,
    temperature: float,
) -> RestaurantParamsModel:
    user_prompt = _build_param_extraction_prompt(task_text)
    structured_output, _, _ = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=_PARAM_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=RestaurantParamsModel,
    )
    return structured_output


# ── Google Places API helpers ───────────────────────────────────────────────

# Fields we request from the new Places API via field mask
_PLACES_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.types,"
    "places.rating,places.priceLevel,places.nationalPhoneNumber,places.websiteUri,"
    "places.googleMapsUri,"
    "places.regularOpeningHours.weekdayDescriptions,places.location,places.photos"
)


def _build_text_query(params: RestaurantParamsModel) -> str:
    parts = ["restaurant"]
    if params.cuisine:
        parts.append(params.cuisine)
    if params.city:
        parts.append(f"in {params.city}")
    return " ".join(parts)


def _search_restaurants(
    params: RestaurantParamsModel,
    config: RestaurantSearchConfig,
) -> dict[str, Any]:
    if not config.api_key:
        return {
            "ok": False,
            "error": "missing_api_key",
            "message": "GOOGLE_PLACES_API_KEY is not set",
        }

    text_query = _build_text_query(params)

    body: dict[str, Any] = {
        "textQuery": text_query,
        "pageSize": config.max_results,
        "languageCode": "en",
    }
    # Price levels must use the PRICE_LEVEL_* enum prefix
    if params.budget and params.budget in _BUDGET_PRICE_LEVEL:
        body["priceLevels"] = [_BUDGET_PRICE_LEVEL[params.budget]]

    try:
        response = requests.post(
            _GOOGLE_PLACES_URL,
            json=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": config.api_key,
                "X-Goog-FieldMask": _PLACES_FIELD_MASK,
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return {"ok": True, "raw": data}
    except requests.Timeout:
        return {"ok": False, "error": "timeout_error", "message": "Google Places API request timed out"}
    except requests.HTTPError as exc:
        return {"ok": False, "error": "http_error", "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": "unknown_error", "message": str(exc)}


def _extract_opening_hours(raw: dict[str, Any]) -> str | None:
    regular = raw.get("regularOpeningHours")
    if not regular:
        return None
    weekday_desc = regular.get("weekdayDescriptions")
    if isinstance(weekday_desc, list) and weekday_desc:
        return "; ".join(weekday_desc)
    return None


def _extract_photos(raw: dict[str, Any]) -> list[str]:
    photos = raw.get("photos", [])
    if not isinstance(photos, list):
        return []
    return [p.get("name", "") for p in photos if isinstance(p, dict)]


def _normalize_candidate(raw: dict[str, Any]) -> RestaurantCandidateModel:
    loc = raw.get("location")
    location_model = None
    if loc and isinstance(loc, dict):
        try:
            location_model = RestaurantLocationModel(
                lat=float(loc.get("latitude", 0)),
                lng=float(loc.get("longitude", 0)),
            )
        except (ValueError, TypeError):
            location_model = None

    display_name = raw.get("displayName")
    name_text = ""
    if isinstance(display_name, dict):
        name_text = display_name.get("text", "")
    elif isinstance(display_name, str):
        name_text = display_name

    return RestaurantCandidateModel(
        place_id=raw.get("id", ""),
        name=name_text,
        address=raw.get("formattedAddress"),
        types=raw.get("types", []),
        rating=raw.get("rating"),
        price_level=raw.get("priceLevel"),
        phone=raw.get("nationalPhoneNumber"),
        website=raw.get("websiteUri"),
        opening_hours=_extract_opening_hours(raw),
        location=location_model,
        photos=_extract_photos(raw),
        raw=raw,
    )


def _normalize_candidates(raw_data: dict[str, Any]) -> list[RestaurantCandidateModel]:
    places = raw_data.get("places", [])
    if not isinstance(places, list):
        return []
    return [_normalize_candidate(p) for p in places if isinstance(p, dict)]


# ── Error helpers ───────────────────────────────────────────────────────────

def _normalize_error(error_code: str, message: str) -> RestaurantSearchErrorModel:
    valid_codes = {
        "missing_api_key",
        "http_error",
        "timeout_error",
        "parse_error",
        "llm_error",
        "unknown_error",
        "no_results",
    }
    code = error_code if error_code in valid_codes else "unknown_error"
    return RestaurantSearchErrorModel(code=code, message=message)  # type: ignore[arg-type]


def _compute_status(ok: bool, item_count: int, errors: list[RestaurantSearchErrorModel]) -> str:
    if not ok:
        return "failed"
    if item_count == 0:
        if not errors:
            return "failed"
        return "failed" if any(e.code != "no_results" for e in errors) else "partial"
    if errors:
        return "partial"
    return "success"


# ── Candidate selection ─────────────────────────────────────────────────────

def _format_candidate_line(i: int, c: RestaurantCandidateModel) -> str:
    return (
        f"[{i}] {c.name} | {c.address or 'address unknown'} | "
        f"Rating: {c.rating or 'N/A'} | Price: {c.price_level or 'N/A'} | "
        f"Types: {', '.join(c.types) if c.types else 'N/A'}"
    )


def _select_candidate(
    candidates: list[RestaurantCandidateModel],
    params: RestaurantParamsModel,
    model_name: str,
    temperature: float,
) -> tuple[RestaurantCandidateModel, str]:
    candidate_block = "\n".join(_format_candidate_line(i, c) for i, c in enumerate(candidates))
    user_prompt = (
        f"City: {params.city}\n"
        f"Cuisine: {params.cuisine or 'any'}\n"
        f"Budget: {params.budget or 'not specified'}\n"
        f"Meal: {params.meal_type or 'not specified'}\n"
        f"Dietary restrictions: {', '.join(params.dietary_restrictions) or 'none'}\n"
        f"Minimum rating: {params.min_rating or 'not specified'}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        "Return JSON only: {\"selected_index\": int, \"selection_reason\": \"one sentence\"}"
    )
    try:
        result, _, _ = invoke_structured_model(
            model_name=model_name,
            temperature=temperature,
            system_prompt=_SELECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=RestaurantSelectionModel,
        )
        idx = max(0, min(result.selected_index, len(candidates) - 1))
        return candidates[idx], result.selection_reason
    except Exception:
        # Fallback: highest-rated candidate
        best = max(candidates, key=lambda c: c.rating or 0.0)
        return best, "Fallback to highest-rated candidate."


# ── Item construction ───────────────────────────────────────────────────────

def _build_google_maps_url(place_id: str | None, google_maps_uri: str | None) -> str | None:
    if google_maps_uri:
        return google_maps_uri
    if not place_id:
        return None
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}"


def _build_item(
    candidate: RestaurantCandidateModel | None,
    selection_reason: str | None,
    params: RestaurantParamsModel,
) -> RestaurantItemModel:
    if candidate is not None:
        return RestaurantItemModel(
            name=candidate.name,
            address=candidate.address,
            place_id=candidate.place_id,
            cuisine=params.cuisine,
            meal_type=params.meal_type,
            rating=candidate.rating,
            price_level=candidate.price_level or _BUDGET_PRICE_LEVEL.get(params.budget or "medium", "PRICE_LEVEL_MODERATE"),
            phone=candidate.phone,
            website=candidate.website,
            google_maps_url=_build_google_maps_url(
                candidate.place_id,
                candidate.raw.get("googleMapsUri") if candidate.raw else None,
            ),
            opening_hours=candidate.opening_hours,
            location=candidate.location.model_dump() if candidate.location else None,
            dietary_suitability=params.dietary_restrictions,
            selection_reason=selection_reason,
            provenance="google_places_api_new",
        )
    else:
        return RestaurantItemModel(
            name=f"Restaurant in {params.city}",
            cuisine=params.cuisine,
            meal_type=params.meal_type,
            price_level=_BUDGET_PRICE_LEVEL.get(params.budget or "medium", "PRICE_LEVEL_MODERATE"),
            dietary_suitability=params.dietary_restrictions,
            provenance="fallback_llm_suggestion",
        )


# ── Message history ─────────────────────────────────────────────────────────

def _build_message_history(
    query: str,
    task_text: str,
    user_prompt: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="restaurant_search_agent",
        model="llm",
        agent_ref="travelplanner.agents.restaurant_search_agent",
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": task_text},
            {"role": "user", "content": user_prompt},
        ],
    )


# ── Public run function ─────────────────────────────────────────────────────

def run_restaurant_search(
    params: RestaurantParamsModel,
    config: RestaurantSearchConfig,
    model_name: str,
    temperature: float,
    task_ref: str = "scratch",
    query_text: str = "",
) -> RestaurantArtifactContentModel:
    errors: list[RestaurantSearchErrorModel] = []
    items: list[RestaurantItemModel] = []

    # 1. Search Google Places API
    result = _search_restaurants(params, config)
    ok = result["ok"]

    if ok:
        raw_data = result["raw"]
        candidates = _normalize_candidates(raw_data)
        if not candidates:
            errors.append(_normalize_error("no_results", f"No restaurants found in {params.city}"))
        else:
            # 2. LLM selection
            selected, reason = _select_candidate(candidates, params, model_name, temperature)
            items.append(_build_item(selected, reason, params))
    else:
        errors.append(
            _normalize_error(
                result.get("error", "unknown_error"),
                result.get("message", "Unknown error"),
            )
        )

    status = _compute_status(ok, len(items), errors)
    return RestaurantArtifactContentModel(
        task_ref=task_ref,
        status=status,  # type: ignore[arg-type]
        provider="google_places_api_new",
        query=query_text or params.model_dump_json(),
        city=params.city,
        cuisine=params.cuisine,
        budget=params.budget,
        meal_type=params.meal_type,
        items=items,
        errors=errors,
        config={
            "api_key_set": bool(config.api_key),
            "max_results": config.max_results,
            "timeout_seconds": config.timeout_seconds,
        },
    )


# ── LangGraph node + graph ──────────────────────────────────────────────────

def make_graph():
    def restaurant_search(state: RestaurantSearchAgentState) -> dict[str, Any]:
        config = load_config_from_env()
        restaurant_tasks = [t for t in state.task_list if t.type == "restaurant" and t.is_valid]

        artifacts: list[AgentArtifactModel] = []
        last_message_history: MessageHistoryModel | None = state.message_history

        for task in restaurant_tasks:
            # Step 1: LLM extracts structured restaurant params from task text
            try:
                params = _extract_restaurant_params(
                    task.text, state.model_name, state.temperature
                )
                last_message_history = _build_message_history(
                    state.query, task.text, _build_param_extraction_prompt(task.text)
                )
            except Exception as exc:
                err_content = RestaurantArtifactContentModel(
                    task_ref=task.name,
                    status="failed",
                    provider="google_places_api_new",
                    query=task.text,
                    city="",
                    errors=[_normalize_error("llm_error", f"Parameter extraction failed: {exc}")],
                    config={"api_key_set": bool(config.api_key)},
                )
                artifacts.append(
                    AgentArtifactModel(
                        name=task.name,
                        type="restaurant_search",
                        content=err_content.model_dump(mode="json"),
                        description=f"Restaurant search failed for task '{task.name}': parameter extraction error",
                    )
                )
                continue

            # Step 2: Run the full search
            content = run_restaurant_search(
                params, config, state.model_name, state.temperature, task_ref=task.name, query_text=task.text
            )

            description = (
                f"{content.city} — {len(content.items)} restaurant(s) found, "
                f"cuisine={content.cuisine}, budget={content.budget}, status={content.status}"
            )
            artifacts.append(
                AgentArtifactModel(
                    name=task.name,
                    type="restaurant_search",
                    content=content.model_dump(mode="json"),
                    description=description,
                )
            )

        # Store artifacts (global system_state if available, otherwise local)
        if state.system_state is not None:
            existing = state.system_state.agent_artifacts.get(state.agent_key, [])
            state.system_state.agent_artifacts[state.agent_key] = existing + artifacts
            return {
                "system_state": state.system_state,
                "message_history": last_message_history,
            }
        else:
            existing = state.agent_artifacts.get(state.agent_key, [])
            state.agent_artifacts[state.agent_key] = existing + artifacts
            return {
                "agent_artifacts": state.agent_artifacts,
                "message_history": last_message_history,
            }

    graph = StateGraph(RestaurantSearchAgentState)
    graph.add_node("restaurant_search_agent", restaurant_search)
    graph.set_entry_point("restaurant_search_agent")
    graph.add_edge("restaurant_search_agent", END)
    return graph.compile()


# ── Convenience Function ────────────────────────────────────────────────────

def intelligent_restaurant_search(
    query: str,
    system_state: StateContractModel,
    model_name: str = "gpt-5-mini",
    agent_key: str = "restaurant_search",
    temperature: float = 0.0,
) -> StateContractModel:
    """Execute intelligent restaurant search from natural language query.

    Provider dispatch (handled by ``travelplanner.utils.llm``):
    * ``openrouter:<model>`` → LangChain ChatOpenAI (OpenRouter API)
    * ``ollama:<model>``     → native ``ollama.Client`` (cloud / local)
    * bare name / ``openai:*`` → LangChain ChatOpenAI (OpenAI API)

    Args:
        query: Natural language restaurant search query.
        system_state: Global system state (required).
        model_name: LLM model to use. Examples:
            - ``"gpt-5-mini"`` (default, OpenAI)
            - ``"openrouter:anthropic/claude-3.5-sonnet"``
            - ``"ollama:nemotron-3-super"``
        agent_key: Key for storing artifacts in SystemState (default: "restaurant_search").
        temperature: LLM temperature (default: 0.0).

    Returns:
        Updated StateContractModel with restaurant search artifact added.

    Example:
        >>> from travelplanner.schema.system_state import StateContractModel
        >>> system_state = StateContractModel(query="Plan Barcelona trip")
        >>> updated = intelligent_restaurant_search(
        ...     query="Italian dinner in Barcelona for 2 people, medium budget",
        ...     system_state=system_state,
        ... )
        >>> artifacts = updated.agent_artifacts.get("restaurant_search", [])
    """
    graph = make_graph()

    # Build a minimal task list from the query
    task = TaskModel(
        name="restaurant_search_task",
        type="restaurant",
        text=query,
        is_valid=True,
    )

    state = RestaurantSearchAgentState(
        query=query,
        model_name=model_name,
        system_state=system_state,
        agent_key=agent_key,
        temperature=temperature,
        task_list=[task],
    )

    result = graph.invoke(state)
    return result["system_state"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Example: direct invocation via the convenience wrapper (writes into SystemState)
    from travelplanner.schema.system_state import StateContractModel

    system_state = StateContractModel(query="Plan Barcelona trip")
    updated_state = intelligent_restaurant_search(
        query="Italian dinner in Barcelona for 2 people, medium budget",
        system_state=system_state,
        # Use ollama:nemotron-3-super for Ollama, or leave default for OpenAI (gpt-5-mini)
        model_name="ollama:nemotron-3-super",
    )
    artifacts = updated_state.agent_artifacts.get("restaurant_search", [])
    if artifacts:
        print(f"\nArtifact stored: {artifacts[0].name}")
        print(f"Description: {artifacts[0].description}")
    else:
        print("\nNo artifacts found.")
