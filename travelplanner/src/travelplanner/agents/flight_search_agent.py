from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from travelplanner.config import get_setting
from travelplanner.schema.flight_search_artifact import (
    FlightAirportModel,
    FlightLayoverModel,
    FlightLegModel,
    FlightOptionModel,
    FlightParamsModel,
    FlightPriceInsightsModel,
    FlightSearchArtifactContentModel,
    FlightSearchErrorModel,
    FlightSegmentParams,
)
from travelplanner.schema.system_state import (
    AgentArtifactModel,
    MessageHistoryModel,
    TaskModel,
)
from travelplanner.utils.llm import invoke_structured_model

# search URL specified downstream in def_search_flights() as "https://serpapi.com/search"
_SEARCH_URL = "https://serpapi.com/search"
_DEFAULT_CURRENCY = "EUR"
_DEFAULT_LANGUAGE = "en"
_DEFAULT_ADULTS = 1
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_RESULTS = 3

_PARAM_EXTRACTION_SYSTEM_PROMPT = """You are a flight parameter extractor for a travel planning assistant.

Given a natural-language flight task, extract the structured search parameters.

Rules:
- Convert city or region names to their primary IATA airport codes (e.g. "Frankfurt" → "FRA", "London" → "LHR", "New York" → "JFK").
- Dates must be in YYYY-MM-DD format.
- Determine trip_type from natural language:
    * "one way" or only a single direction with no return → 2
    * "return", "round trip", or a return date is mentioned → 1
    * Multiple sequential destinations (multi-city itinerary) → 3
- For trip_type 1 or 2: build a single-item segments list for the departure→arrival pair.
- For trip_type 3: build one segment per leg in the itinerary, in order.
- return_date is only set when trip_type == 1; otherwise omit it (null).
- If no currency is mentioned, use the default provided.
- If number of adults is not mentioned, use the default provided.
- Return JSON only — no extra text.
"""


@dataclass(frozen=True)
class FlightSearchConfig:
    api_key: str = ""
    currency: str = _DEFAULT_CURRENCY
    language: str = _DEFAULT_LANGUAGE
    default_adults: int = _DEFAULT_ADULTS
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    max_results: int = _DEFAULT_MAX_RESULTS


def _parse_int(value: str | None, *, default: int, minimum: int) -> int:
    if value is None or value.strip() == "":
        return default
    return max(int(value.strip()), minimum)


def load_config_from_env() -> FlightSearchConfig:
    cfg_prefix = "agents.flight_search"
    return FlightSearchConfig(
        api_key=os.getenv(
            "SERPAPI_API_KEY",
            str(get_setting(f"{cfg_prefix}.api_key", "")),
        ).strip(),
        currency=os.getenv(
            "TRAVELPLANNER_FLIGHT_CURRENCY",
            str(get_setting(f"{cfg_prefix}.currency", _DEFAULT_CURRENCY)),
        ).strip(),
        language=os.getenv(
            "TRAVELPLANNER_FLIGHT_LANGUAGE",
            str(get_setting(f"{cfg_prefix}.language", _DEFAULT_LANGUAGE)),
        ).strip(),
        default_adults=_parse_int(
            os.getenv("TRAVELPLANNER_FLIGHT_DEFAULT_ADULTS"),
            default=int(get_setting(f"{cfg_prefix}.default_adults", _DEFAULT_ADULTS)),
            minimum=1,
        ),
        timeout_seconds=_parse_int(
            os.getenv("TRAVELPLANNER_FLIGHT_TIMEOUT_SECONDS"),
            default=int(get_setting(f"{cfg_prefix}.timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
            minimum=5,
        ),
        max_results=_parse_int(
            os.getenv("TRAVELPLANNER_FLIGHT_MAX_RESULTS"),
            default=int(get_setting(f"{cfg_prefix}.max_results", _DEFAULT_MAX_RESULTS)),
            minimum=1,
        ),
    )


class FlightSearchAgentState(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.0
    task_list: list[TaskModel] = Field(default_factory=list)
    agent_artifacts: dict[str, list[AgentArtifactModel]] = Field(default_factory=dict)
    message_history: MessageHistoryModel | None = None


def _build_param_extraction_prompt(task_text: str, config: FlightSearchConfig) -> str:
    return "\n".join(
        [
            "Extract flight search parameters from the following task.",
            "",
            f"Task: {task_text.strip()}",
            "",
            f"Defaults if not specified: currency={config.currency}, adults={config.default_adults}",
            "",
            "Return strictly valid JSON with this shape:",
            '{"trip_type": 1, "segments": [{"departure_id": "FRA", "arrival_id": "LHR", "outbound_date": "YYYY-MM-DD"}], "return_date": "YYYY-MM-DD or null", "adults": 1, "currency": "EUR"}',
        ]
    )


def _extract_flight_params(
    task_text: str,
    model_name: str,
    temperature: float,
    config: FlightSearchConfig,
) -> FlightParamsModel:
    user_prompt = _build_param_extraction_prompt(task_text, config)
    structured_output, _, _ = invoke_structured_model(
        model_name=model_name,
        temperature=temperature,
        system_prompt=_PARAM_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=FlightParamsModel,
    )
    return structured_output


def _search_flights(
    params: FlightParamsModel,
    config: FlightSearchConfig,
) -> dict[str, Any]:
    if not config.api_key:
        return {
            "ok": False,
            "error": "missing_api_key",
            "message": "SERPAPI_API_KEY is not set",
        }

    seg = params.segments[0]
    query_params: dict[str, Any] = {
        "engine": "google_flights",
        "type": params.trip_type,
        "adults": params.adults,
        "currency": params.currency,
        "hl": config.language,
        "api_key": config.api_key,
        "departure_id": seg.departure_id,
        "arrival_id": seg.arrival_id,
        "outbound_date": seg.outbound_date,
    }
    if params.trip_type == 1:
        query_params["return_date"] = params.return_date

    try:
        response = requests.get(
            _SEARCH_URL,
            params=query_params,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return {"ok": True, "raw": data}
    except requests.Timeout:
        return {"ok": False, "error": "timeout_error", "message": "SerpAPI request timed out"}
    except requests.HTTPError as exc:
        return {"ok": False, "error": "http_error", "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": "unknown_error", "message": str(exc)}



def _normalize_leg(raw_leg: dict[str, Any]) -> FlightLegModel:
    dep = raw_leg.get("departure_airport", {})
    arr = raw_leg.get("arrival_airport", {})
    return FlightLegModel(
        departure_airport=FlightAirportModel(
            name=dep.get("name", ""),
            id=dep.get("id", ""),
            time=dep.get("time", ""),
        ),
        arrival_airport=FlightAirportModel(
            name=arr.get("name", ""),
            id=arr.get("id", ""),
            time=arr.get("time", ""),
        ),
        duration_minutes=raw_leg.get("duration", 0),
        airline=raw_leg.get("airline", ""),
        flight_number=raw_leg.get("flight_number", ""),
        airplane=raw_leg.get("airplane"),
        travel_class=raw_leg.get("travel_class", "Economy"),
        legroom=raw_leg.get("legroom"),
        extensions=raw_leg.get("extensions", []),
    )


def _normalize_flight_option(raw_option: dict[str, Any], currency: str) -> FlightOptionModel:
    legs = [_normalize_leg(leg) for leg in raw_option.get("flights", [])]
    layovers = [
        FlightLayoverModel(
            name=lv.get("name", ""),
            id=lv.get("id", ""),
            duration_minutes=lv.get("duration", 0),
        )
        for lv in raw_option.get("layovers", [])
    ]
    carbon = raw_option.get("carbon_emissions", {})
    carbon_kg = carbon.get("this_flight")
    if carbon_kg is not None:
        carbon_kg = carbon_kg // 1000

    return FlightOptionModel(
        legs=legs,
        layovers=layovers,
        total_duration_minutes=raw_option.get("total_duration", 0),
        price=float(raw_option.get("price", 0)),
        currency=currency,
        type=raw_option.get("type", ""),
        carbon_emissions_kg=carbon_kg,
        departure_token=raw_option.get("departure_token"),
    )


def _normalize_price_insights(raw: dict[str, Any] | None) -> FlightPriceInsightsModel | None:
    if not raw:
        return None
    return FlightPriceInsightsModel(
        lowest_price=raw.get("lowest_price"),
        price_level=raw.get("price_level"),
        typical_price_range=raw.get("typical_price_range", []),
    )


def _normalize_error(error_code: str, message: str) -> FlightSearchErrorModel:
    valid_codes = {"missing_api_key", "http_error", "timeout_error", "parse_error", "unknown_error"}
    code = error_code if error_code in valid_codes else "unknown_error"
    return FlightSearchErrorModel(code=code, message=message)  # type: ignore[arg-type]


def _compute_status(
    ok: bool, best_count: int, other_count: int
) -> str:
    if not ok:
        return "failed"
    if best_count > 0 or other_count > 0:
        return "success"
    return "partial"


def _build_message_history(
    query: str,
    task_text: str,
    user_prompt: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent="flight_search_agent",
        model="llm",
        agent_ref="travelplanner.agents.flight_search_agent",
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": task_text},
            {"role": "user", "content": user_prompt},
        ],
    )


def run_flight_search(
    params: FlightParamsModel,
    config: FlightSearchConfig,
    task_ref: str = "scratch",
) -> FlightSearchArtifactContentModel:
    """Execute a complete flight search for all legs and return a populated artifact.

    Handles one-way (type=2), round-trip (type=1, two calls), and multi-city
    (type=3, N independent one-way calls).  Safe to call directly from scripts or tests.
    """
    errors: list[FlightSearchErrorModel] = []
    best_flights: list[FlightOptionModel] = []
    other_flights: list[FlightOptionModel] = []
    return_flights: list[FlightOptionModel] = []
    multi_city_legs: list[list[FlightOptionModel]] = []
    price_insights: FlightPriceInsightsModel | None = None

    ok = True

    if params.trip_type == 3:
        # Multi-city: one independent one-way call per segment, no chaining
        for seg in params.segments:
            seg_params = FlightParamsModel(
                trip_type=2,
                segments=[seg],
                adults=params.adults,
                currency=params.currency,
            )
            seg_result = _search_flights(seg_params, config)
            if seg_result["ok"]:
                try:
                    leg_options = [
                        _normalize_flight_option(opt, params.currency)
                        for opt in seg_result["raw"].get("best_flights", [])[: config.max_results]
                    ]
                except Exception as exc:
                    errors.append(_normalize_error("parse_error", f"Leg normalization failed: {exc}"))
                    leg_options = []
                multi_city_legs.append(leg_options)
            else:
                ok = False
                errors.append(
                    _normalize_error(
                        seg_result.get("error", "unknown_error"),
                        seg_result.get("message", f"Error fetching {seg.departure_id}→{seg.arrival_id}"),
                    )
                )
                multi_city_legs.append([])
        total_options = sum(len(opts) for opts in multi_city_legs)
        status = _compute_status(ok, total_options, 0)
    else:
        result = _search_flights(params, config)
        ok = result["ok"]
        if ok:
            raw_data = result["raw"]
            try:
                best_flights = [
                    _normalize_flight_option(opt, params.currency)
                    for opt in raw_data.get("best_flights", [])[: config.max_results]
                ]
                other_flights = [
                    _normalize_flight_option(opt, params.currency)
                    for opt in raw_data.get("other_flights", [])[: config.max_results]
                ]
                price_insights = _normalize_price_insights(raw_data.get("price_insights"))
            except Exception as exc:
                errors.append(_normalize_error("parse_error", f"Response normalization failed: {exc}"))

            if params.trip_type == 1 and params.return_date:
                seg = params.segments[0]
                return_params = FlightParamsModel(
                    trip_type=2,
                    segments=[FlightSegmentParams(
                        departure_id=seg.arrival_id,
                        arrival_id=seg.departure_id,
                        outbound_date=params.return_date,
                    )],
                    adults=params.adults,
                    currency=params.currency,
                )
                return_result = _search_flights(return_params, config)
                if return_result["ok"]:
                    try:
                        return_flights = [
                            _normalize_flight_option(opt, params.currency)
                            for opt in return_result["raw"].get("best_flights", [])[: config.max_results]
                        ]
                    except Exception as exc:
                        errors.append(
                            _normalize_error("parse_error", f"Return-flight normalization failed: {exc}")
                        )
                else:
                    errors.append(
                        _normalize_error(
                            return_result.get("error", "unknown_error"),
                            return_result.get("message", "Error fetching return flights"),
                        )
                    )
        else:
            errors.append(
                _normalize_error(
                    result.get("error", "unknown_error"),
                    result.get("message", "Unknown error"),
                )
            )
        status = _compute_status(ok, len(best_flights), len(other_flights))

    if params.trip_type == 3:
        selected_flights = [legs[0] for legs in multi_city_legs if legs]
    else:
        selected_flights = []
        if best_flights:
            selected_flights.append(best_flights[0])
        if return_flights:
            selected_flights.append(return_flights[0])

    first_seg = params.segments[0] if params.segments else FlightSegmentParams(
        departure_id="", arrival_id="", outbound_date=""
    )
    return FlightSearchArtifactContentModel(
        task_ref=task_ref,
        status=status,  # type: ignore[arg-type]
        provider="serpapi_google_flights",
        departure_id=first_seg.departure_id,
        arrival_id=first_seg.arrival_id,
        outbound_date=first_seg.outbound_date,
        return_date=params.return_date,
        adults=params.adults,
        currency=params.currency,
        selected_flights=selected_flights,
        best_flights=best_flights,
        other_flights=other_flights,
        return_flights=return_flights,
        multi_city_legs=multi_city_legs,
        price_insights=price_insights,
        errors=errors,
        config={
            "api_key_set": bool(config.api_key),
            "trip_type": params.trip_type,
            "max_results": config.max_results,
            "timeout_seconds": config.timeout_seconds,
            "currency": config.currency,
            "language": config.language,
        },
    )


def make_graph():
    def search_node(state: FlightSearchAgentState) -> dict[str, Any]:
        config = load_config_from_env()
        flight_tasks = [t for t in state.task_list if t.type == "flight" and t.is_valid]

        artifacts: list[AgentArtifactModel] = []
        last_message_history: MessageHistoryModel | None = state.message_history

        for task in flight_tasks:
            # Step 1: LLM extracts structured flight params from task text
            try:
                params = _extract_flight_params(
                    task.text, state.model_name, state.temperature, config
                )
                last_message_history = _build_message_history(
                    state.query, task.text, _build_param_extraction_prompt(task.text, config)
                )
            except Exception as exc:
                err_content = FlightSearchArtifactContentModel(
                    task_ref=task.name,
                    status="failed",
                    provider="serpapi_google_flights",
                    departure_id="",
                    arrival_id="",
                    outbound_date="",
                    adults=config.default_adults,
                    currency=config.currency,
                    errors=[_normalize_error("parse_error", f"Parameter extraction failed: {exc}")],
                    config={"api_key_set": bool(config.api_key)},
                )
                artifacts.append(
                    AgentArtifactModel(
                        name=task.name,
                        type="flight-search-result",
                        content=err_content.model_dump(),
                        description=f"Flight search failed for task '{task.name}': parameter extraction error",
                    )
                )
                continue

            # Step 2: Run the full search (all legs/chaining)
            content = run_flight_search(params, config, task_ref=task.name)

            trip_label = {1: "round trip", 2: "one way", 3: "multi-city"}.get(params.trip_type, "")
            route = "→".join(f"{s.departure_id}→{s.arrival_id}" for s in params.segments)
            first_seg = params.segments[0]
            if params.trip_type == 3:
                legs_summary = ", ".join(
                    f"leg {i + 1}: {len(opts)} option(s)"
                    for i, opts in enumerate(content.multi_city_legs)
                )
                description = f"{route} ({trip_label}) — {legs_summary}"
            else:
                description = (
                    f"{route} on {first_seg.outbound_date} ({trip_label})"
                    + (f" / return {params.return_date}" if params.return_date else "")
                    + f" — {len(content.best_flights)} outbound, {len(content.return_flights)} return flight(s) found"
                )
            artifacts.append(
                AgentArtifactModel(
                    name=task.name,
                    type="flight-search-result",
                    content=content.model_dump(),
                    description=description,
                )
            )

        existing = state.agent_artifacts.get("flight_search_agent", [])
        return {
            "agent_artifacts": {
                **state.agent_artifacts,
                "flight_search_agent": existing + artifacts,
            },
            "message_history": last_message_history,
        }

    graph = StateGraph(FlightSearchAgentState)
    graph.add_node("flight_search_agent", search_node)
    graph.set_entry_point("flight_search_agent")
    graph.add_edge("flight_search_agent", END)
    return graph.compile()
